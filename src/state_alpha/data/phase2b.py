"""Phase 2B universe profiling and delisting-aware label panel build."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig, load_yaml
from state_alpha.data.universe import (
    add_point_in_time_universe_columns,
    apply_universe_rule,
    summarize_rule,
)
from state_alpha.labels.forward import compute_forward_labels
from state_alpha.logging_utils import configure_run_logger
from state_alpha.paths import PathPolicy


PHASE2B_SCHEMA_VERSION = "state_alpha_phase2b_panel_v1"
HORIZONS = (5, 10, 20)
SOURCE_COLUMNS = [
    "permno",
    "permco",
    "date",
    "primary_exchange",
    "trading_status",
    "security_type",
    "security_subtype",
    "share_type",
    "us_incorporated_flag",
    "price_raw",
    "market_cap_source",
    "volume",
    "shares_outstanding_source",
    "return_total",
    "return_missing_flag",
    "daily_delisting_flag",
    "market_vw_return",
]
PANEL_COLUMNS = [
    "permno",
    "permco",
    "date",
    "session_id",
    "formation_tradable",
    "lag_security_base",
    "lag_us_incorporated",
    "lag_is_previous_market_session",
    "adv_window_consecutive",
    "prior_observation_count",
    "lag_price_usd",
    "lag_market_cap_usd",
    "lag_dollar_volume_usd",
    "lag_adv20_usd",
    "lag_turnover",
    "lag_market_cap_pct_all",
    "lag_adv20_pct_all",
    "lag_market_cap_pct_us",
    "lag_adv20_pct_us",
    "universe_analysis_base",
    "universe_primary",
    "largest_lagged_mcap_share_class",
    "execution_valid",
    "execution_session_id",
    "fwd_total_return_rank_10d",
]
for _horizon in HORIZONS:
    PANEL_COLUMNS.extend(
        [
            f"label_start_session_id_{_horizon}d",
            f"label_end_session_id_{_horizon}d",
            f"fwd_total_return_{_horizon}d",
            f"fwd_market_excess_return_{_horizon}d",
            f"fwd_total_return_{_horizon}d_dlret_loss100",
            f"fwd_total_return_{_horizon}d_dlret_zero",
            f"terminal_within_{_horizon}d",
            f"missing_delisting_return_within_{_horizon}d",
            f"label_structural_valid_{_horizon}d",
        ]
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_snapshot(canonical_root: Path) -> dict[str, dict[str, int]]:
    return {
        str(path.relative_to(canonical_root)).replace("\\", "/"): {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(canonical_root.rglob("*"))
        if path.is_file()
    }


def _security_root(config: ProjectConfig) -> Path:
    root = config.project_path("data_interim") / "canonical_v1" / "security_day"
    if not root.is_dir():
        raise FileNotFoundError(f"Canonical security-day dataset missing: {root}")
    return root


def load_market_calendar(security_root: Path) -> pd.DataFrame:
    """Read and validate one value-weighted market return per CRSP session."""

    records: list[pd.DataFrame] = []
    year_dirs = sorted(security_root.glob("year=*"))
    if not year_dirs:
        raise FileNotFoundError(f"No year partitions under {security_root}")
    for year_dir in year_dirs:
        table = ds.dataset(year_dir, format="parquet").to_table(
            columns=["date", "market_vw_return"]
        )
        frame = table.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        pairs = frame.drop_duplicates(["date", "market_vw_return"])
        counts = pairs.groupby("date", observed=True).size()
        if counts.gt(1).any():
            examples = counts.loc[counts.gt(1)].head().to_dict()
            raise ValueError(
                f"Market return is not constant within date: {examples}"
            )
        records.append(
            pairs.drop_duplicates("date")[["date", "market_vw_return"]]
        )
    calendar = (
        pd.concat(records, ignore_index=True)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    if calendar["date"].duplicated().any():
        raise ValueError("Market calendar has duplicate dates across partitions")
    if calendar["market_vw_return"].isna().any():
        raise ValueError("Market calendar contains missing value-weighted returns")
    calendar.insert(0, "session_id", np.arange(len(calendar), dtype=np.int32))
    return calendar


def _read_partition(
    path: Path,
    *,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    filter_expression = None
    if cutoff_date is not None:
        filter_expression = ds.field("date") <= pa.scalar(
            cutoff_date.date(), type=pa.date32()
        )
    frame = dataset.to_table(
        columns=SOURCE_COLUMNS, filter=filter_expression
    ).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _load_year_window(
    security_root: Path,
    calendar: pd.DataFrame,
    year: int,
    *,
    include_forward: bool,
) -> pd.DataFrame:
    available = {
        int(path.name.split("=", 1)[1]): path
        for path in security_root.glob("year=*")
    }
    if year not in available:
        raise FileNotFoundError(f"Missing canonical year partition: {year}")
    frames: list[pd.DataFrame] = []
    if year - 1 in available:
        frames.append(_read_partition(available[year - 1]))
    frames.append(_read_partition(available[year]))
    if include_forward and year + 1 in available:
        current_sessions = calendar.loc[
            calendar["date"].dt.year.eq(year), "session_id"
        ]
        last_session = int(current_sessions.max())
        cutoff_session = min(last_session + max(HORIZONS) + 1, len(calendar) - 1)
        cutoff_date = calendar.loc[cutoff_session, "date"]
        frames.append(
            _read_partition(available[year + 1], cutoff_date=cutoff_date)
        )
    combined = pd.concat(frames, ignore_index=True)
    session_map = calendar.set_index("date")["session_id"]
    combined["session_id"] = combined["date"].map(session_map)
    if combined["session_id"].isna().any():
        raise ValueError("A canonical security date is absent from market calendar")
    combined["session_id"] = combined["session_id"].astype(np.int32)
    return combined.sort_values(
        ["permno", "session_id"], kind="stable"
    ).reset_index(drop=True)


def prepare_target_year(
    security_root: Path,
    calendar: pd.DataFrame,
    year: int,
    *,
    include_labels: bool,
    primary_rule: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Create target-year point-in-time variables and optional labels."""

    combined = _load_year_window(
        security_root, calendar, year, include_forward=include_labels
    )
    prepared = add_point_in_time_universe_columns(
        combined, minimum_history=252, adv_window=20
    )
    if include_labels:
        label_frame = compute_forward_labels(
            prepared,
            calendar["market_vw_return"].to_numpy(dtype=float),
            horizons=HORIZONS,
        )
        prepared = pd.concat([prepared, label_frame], axis=1)

    target_mask = prepared["date"].dt.year.eq(year)
    target = prepared.loc[target_mask].copy()
    if primary_rule is not None:
        target["universe_primary"] = apply_universe_rule(
            target, primary_rule
        )
    else:
        target["universe_primary"] = False

    target["largest_lagged_mcap_share_class"] = False
    robust_base = target["universe_analysis_base"] & target["permco"].notna()
    robust = target.loc[robust_base]
    if len(robust):
        largest_rank = robust.groupby(
            ["date", "permco"], sort=False, observed=True
        )["lag_market_cap_usd"].rank(method="first", ascending=False)
        target.loc[robust_base, "largest_lagged_mcap_share_class"] = (
            largest_rank.eq(1).to_numpy()
        )

    if include_labels:
        target["fwd_total_return_rank_10d"] = np.nan
        rank_mask = (
            target["universe_primary"]
            & target["fwd_total_return_10d"].notna()
        )
        ranked = target.loc[rank_mask].groupby(
            "date", sort=False, observed=True
        )["fwd_total_return_10d"].rank(method="average", pct=True)
        target.loc[rank_mask, "fwd_total_return_rank_10d"] = ranked - 0.5

    return target.sort_values(["date", "permno"], kind="stable").reset_index(
        drop=True
    )


def _candidate_rules(config_root: Path) -> list[dict[str, Any]]:
    universe_config = load_yaml(config_root / "configs" / "universe.yaml")
    rules = universe_config.get("candidate_rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("configs/universe.yaml must define candidate_rules")
    ids = [rule.get("id") for rule in rules]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Universe candidate rule IDs must be unique and nonempty")
    return rules


def _delisting_diagnostics(security_root: Path) -> pd.DataFrame:
    columns = [
        "permno",
        "date",
        "daily_delisting_flag",
        "return_total",
        "return_missing_flag",
        "return_duration_flag",
    ]
    frame = ds.dataset(
        security_root, format="parquet", partitioning="hive"
    ).to_table(
        columns=columns,
        filter=ds.field("daily_delisting_flag") == "Y",
    ).to_pandas()
    return (
        frame.groupby(
            ["return_missing_flag", "return_duration_flag"], dropna=False
        )
        .agg(
            rows=("permno", "size"),
            unique_permnos=("permno", "nunique"),
            missing_total_return_rows=(
                "return_total",
                lambda values: int(values.isna().sum()),
            ),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
        .sort_values("rows", ascending=False, kind="stable")
    )


def profile_phase2b_candidates(project_root: Path) -> dict[str, Path]:
    """Profile predeclared universe candidates using 2000-2012 only."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    security_root = _security_root(config)
    rules = _candidate_rules(config.root)
    tables = policy.ensure_directory(config.project_path("reports_tables"))
    memo_dir = policy.ensure_directory(config.project_path("reports_memo"))
    outputs = {
        "annual": tables / "phase2b_candidate_annual_summary.csv",
        "overall": tables / "phase2b_candidate_overall_summary.csv",
        "delisting": tables / "phase2b_delisting_semantics.csv",
        "memo": memo_dir / "PHASE2B_CANDIDATE_SELECTION.md",
    }
    for path in outputs.values():
        resolved = policy.assert_writable(path)
        if resolved.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_run_logger(
        "state_alpha.phase2b.profile",
        config.project_path("logs") / f"phase2b_profile_{timestamp}.log",
        policy,
    )
    calendar = load_market_calendar(security_root)
    annual_records: list[dict[str, Any]] = []
    for year in range(2000, 2013):
        logger.info("Profiling Phase 2B candidate rules for %d", year)
        frame = prepare_target_year(
            security_root, calendar, year, include_labels=False
        )
        for rule in rules:
            mask = apply_universe_rule(frame, rule)
            annual_records.append(
                summarize_rule(frame, str(rule["id"]), mask, year=year)
            )
    annual = pd.DataFrame(annual_records)
    annual.to_csv(outputs["annual"], index=False)

    overall_records: list[dict[str, Any]] = []
    for rule_id, group in annual.groupby("rule_id", sort=False):
        total_base = int(group["analysis_base_rows"].sum())
        total_eligible = int(group["eligible_rows"].sum())
        breadth = group.loc[
            group["daily_names_median"].gt(0), "daily_names_median"
        ]
        eligible_years = group.loc[group["daily_names_median"].gt(0)]
        overall_records.append(
            {
                "rule_id": rule_id,
                "development_analysis_base_rows": total_base,
                "development_eligible_rows": total_eligible,
                "development_eligible_share": (
                    total_eligible / total_base if total_base else np.nan
                ),
                "annual_median_daily_names_mean": float(breadth.mean()),
                "annual_median_daily_names_min": float(breadth.min()),
                "annual_median_daily_names_max": float(breadth.max()),
                "annual_median_daily_names_cv": (
                    float(breadth.std(ddof=0) / breadth.mean())
                    if len(breadth) and breadth.mean()
                    else np.nan
                ),
                "minimum_daily_names_across_years": int(
                    eligible_years["daily_names_min"].min()
                ),
                "median_selected_price_p10": float(
                    group["lag_price_usd_p10"].median()
                ),
                "median_selected_market_cap_p10": float(
                    group["lag_market_cap_usd_p10"].median()
                ),
                "median_selected_adv20_p10": float(
                    group["lag_adv20_usd_p10"].median()
                ),
            }
        )
    overall = pd.DataFrame(overall_records).sort_values(
        ["annual_median_daily_names_cv", "development_eligible_share"],
        kind="stable",
    )
    overall.to_csv(outputs["overall"], index=False)
    delisting = _delisting_diagnostics(security_root)
    delisting.to_csv(outputs["delisting"], index=False)

    memo = f"""# Phase 2B candidate-selection packet

Generated: {datetime.now().isoformat(timespec="seconds")}

## Frozen information boundary

Only 2000–2012 universe distributions were used. No technical signal, forward
return, IC, portfolio return, or 2013–2025 outcome was inspected to select a
universe rule.

## Candidate set

{len(rules)} predeclared rules compare absolute economic thresholds with
date-level capitalization and lagged-ADV20 percentiles, both with and without a
U.S.-incorporation restriction. All variables are shifted one security
observation, the lag must be the immediately preceding market session, ADV20
uses exactly 20 consecutive prior sessions, and 252 prior observations are
required.

## Selection criteria

The primary rule should retain economically meaningful price/liquidity floors,
avoid a thin cross section in every development year, and avoid severe
development-period breadth drift. These diagnostics—not future returns—are the
only permitted basis for freezing the rule.

## Delisting contract

CRSP CIZ `DlyRet` incorporates a calculable delisting return. Missing terminal
rows remain missing in the primary label. The label builder additionally
produces explicit terminal-value bounds of -100% and 0% for sensitivity; the
0% bound is not interpreted as an observed return.

- Official CIZ guide:
  https://indexes.morningstar.com/docs/guide/crsp-us-stock-databases-guide-for-flat-file-format-2-0
- Official return-missing flags:
  https://www.crsp.org/wp-content/uploads/appendix/FlagType_RM.html
- WRDS CIZ event-study implementation:
  https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/run-an-event-study-ciz-format-macro/
"""
    outputs["memo"].write_text(memo, encoding="utf-8")
    logger.info("Phase 2B candidate profile complete: %s", outputs)
    return outputs


def _resolve_primary_rule(config_root: Path) -> dict[str, Any]:
    universe_config = load_yaml(config_root / "configs" / "universe.yaml")
    rule_id = universe_config.get("frozen_primary_rule_id")
    if not rule_id:
        raise ValueError(
            "configs/universe.yaml must freeze frozen_primary_rule_id before build"
        )
    rules = {rule["id"]: rule for rule in universe_config["candidate_rules"]}
    if rule_id not in rules:
        raise ValueError(f"Frozen universe rule is not predeclared: {rule_id}")
    return rules[rule_id]


def build_phase2b_panel(project_root: Path) -> Path:
    """Build and atomically publish the full Phase 2B universe/label panel."""

    started = time.perf_counter()
    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    security_root = _security_root(config)
    canonical_root = security_root.parent
    primary_rule = _resolve_primary_rule(config.root)
    interim = policy.ensure_directory(config.project_path("data_interim"))
    final_root = policy.assert_writable(interim / "phase2b_v1")
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final_root}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_root = policy.assert_writable(
        interim / f"_phase2b_v1_build_{timestamp}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)
    logger = configure_run_logger(
        "state_alpha.phase2b.build",
        config.project_path("logs") / f"phase2b_build_{timestamp}.log",
        policy,
    )
    logger.info("Phase 2B staging root: %s", staging_root)
    snapshot_before = _canonical_snapshot(canonical_root)

    calendar = load_market_calendar(security_root)
    calendar_path = staging_root / "market_calendar.parquet"
    pq.write_table(
        pa.Table.from_pandas(calendar, preserve_index=False),
        calendar_path,
        compression="zstd",
        compression_level=6,
    )
    panel_root = staging_root / "panel"
    panel_root.mkdir()
    annual_audit: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []
    output_files.append(
        {
            "path": "market_calendar.parquet",
            "rows": len(calendar),
            "size_bytes": calendar_path.stat().st_size,
            "sha256": _sha256(calendar_path),
        }
    )

    for year in range(2000, 2026):
        logger.info("Building Phase 2B panel for %d", year)
        target = prepare_target_year(
            security_root,
            calendar,
            year,
            include_labels=True,
            primary_rule=primary_rule,
        )
        output = target.loc[target["universe_analysis_base"], PANEL_COLUMNS].copy()
        partition = panel_root / f"year={year}"
        partition.mkdir()
        file_path = partition / "part-00000.parquet"
        table = pa.Table.from_pandas(output, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"schema_version"] = PHASE2B_SCHEMA_VERSION.encode("utf-8")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(
            table,
            file_path,
            compression="zstd",
            compression_level=6,
            row_group_size=50_000,
        )
        relative = str(file_path.relative_to(staging_root)).replace("\\", "/")
        output_files.append(
            {
                "path": relative,
                "rows": len(output),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )
        annual_audit.append(
            {
                "year": year,
                "analysis_rows": int(len(output)),
                "primary_universe_rows": int(output["universe_primary"].sum()),
                "primary_unique_permnos": int(
                    output.loc[output["universe_primary"], "permno"].nunique()
                ),
                "primary_execution_invalid_rows": int(
                    (
                        output["universe_primary"]
                        & ~output["execution_valid"]
                    ).sum()
                ),
                **{
                    f"primary_valid_label_{horizon}d_rows": int(
                        (
                            output["universe_primary"]
                            & output[f"fwd_total_return_{horizon}d"].notna()
                        ).sum()
                    )
                    for horizon in HORIZONS
                },
                **{
                    f"primary_missing_delisting_return_{horizon}d_rows": int(
                        (
                            output["universe_primary"]
                            & output[
                                f"missing_delisting_return_within_{horizon}d"
                            ]
                        ).sum()
                    )
                    for horizon in HORIZONS
                },
            }
        )
        logger.info(
            "Finished %d: analysis=%d primary=%d",
            year,
            len(output),
            int(output["universe_primary"].sum()),
        )
        del target, output, table

    snapshot_after = _canonical_snapshot(canonical_root)
    canonical_unchanged = snapshot_before == snapshot_after
    if not canonical_unchanged:
        raise RuntimeError("Canonical input snapshot changed during Phase 2B build")
    manifest = {
        "schema_version": PHASE2B_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "project_root": config.root,
        "staging_root": staging_root,
        "final_root": final_root,
        "canonical_root": canonical_root,
        "canonical_snapshot": snapshot_before,
        "canonical_unchanged": canonical_unchanged,
        "primary_rule": primary_rule,
        "label_contract": {
            "execution": "t+1 close",
            "first_return_session_offset": 2,
            "horizons": list(HORIZONS),
            "valid_delisting": "compound DlyRet and hold cash to fixed horizon end",
            "missing_delisting_primary": "missing",
            "missing_delisting_sensitivity": [
                "terminal value -100%",
                "terminal value 0%",
            ],
        },
        "annual_audit": annual_audit,
        "output_files": output_files,
        "overwrite_used": False,
    }
    _write_json(staging_root / "manifest.json", manifest)
    os.replace(staging_root, final_root)
    logger.info(
        "Published Phase 2B panel to %s in %.1f seconds",
        final_root,
        time.perf_counter() - started,
    )
    return final_root
