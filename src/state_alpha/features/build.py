"""Atomic full-sample build for the frozen Gate 3 feature contract."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
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
from state_alpha.features.core import (
    CATEGORICAL_FEATURE_COLUMNS,
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    OUTPUT_IDENTIFIER_COLUMNS,
    compute_market_features,
    compute_security_features,
    finalize_target_features,
)
from state_alpha.logging_utils import configure_run_logger
from state_alpha.paths import PathPolicy


FEATURE_SCHEMA_VERSION = "state_alpha_features_v1"
LOOKBACK_SESSIONS = 320
CORRELATION_SAMPLE_PER_YEAR = 50_000
SECURITY_COLUMNS = [
    "permno",
    "date",
    "return_total",
    "return_ex_distribution",
    "price_raw",
    "volume",
    "shares_outstanding_source",
    "bid_raw",
    "ask_raw",
    "siccd",
    "primary_exchange",
]
PANEL_INPUT_COLUMNS = [
    "permno",
    "permco",
    "date",
    "session_id",
    "universe_primary",
    "largest_lagged_mcap_share_class",
    "lag_market_cap_usd",
    "lag_adv20_usd",
]
DIAGNOSTIC_NAMES = [
    "phase3_feature_annual_coverage.csv",
    "phase3_feature_stability.csv",
    "phase3_feature_correlation_development.csv",
    "phase3_feature_high_correlation_pairs_development.csv",
]


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
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(root: Path) -> dict[str, dict[str, int]]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def validate_feature_contract(config_path: Path) -> dict[str, Any]:
    """Validate the frozen list against the production column contract."""

    contract = load_yaml(config_path)
    features = contract.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("features.yaml must contain a nonempty features list")
    names = [item.get("name") for item in features]
    if any(not name for name in names):
        raise ValueError("Every feature must have a nonempty name")
    if len(names) != len(set(names)):
        raise ValueError("Frozen feature names must be unique")
    if names != FEATURE_COLUMNS:
        raise ValueError(
            "features.yaml order/names do not match production FEATURE_COLUMNS"
        )
    budget = contract.get("feature_budget", {})
    maximum = int(budget.get("maximum_primary_features", 0))
    frozen = int(budget.get("frozen_feature_count", -1))
    if len(names) != frozen or len(names) > maximum:
        raise ValueError("Frozen feature count violates the declared budget")
    required = {"role", "family", "inputs", "definition", "rationale"}
    for item in features:
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(
                f"Feature {item['name']} is missing contract fields: {missing}"
            )
    if contract.get("information_boundary", {}).get(
        "labels_in_feature_artifact"
    ):
        raise ValueError("Feature artifact contract must exclude labels")
    return contract


def _read_panel_year(
    panel_root: Path,
    year: int,
    columns: list[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(panel_root / f"year={year}", columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _read_security_year(
    security_root: Path,
    year: int,
    columns: list[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(
        security_root / f"year={year}", columns=columns
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _read_security_window(
    security_dataset: ds.Dataset,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    permnos: np.ndarray,
) -> pd.DataFrame:
    expression = (
        (ds.field("date") >= pa.scalar(start_date.date(), type=pa.date32()))
        & (ds.field("date") <= pa.scalar(end_date.date(), type=pa.date32()))
    )
    frame = security_dataset.to_table(
        columns=SECURITY_COLUMNS, filter=expression
    ).to_pandas()
    frame = frame.loc[frame["permno"].isin(permnos)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _build_cross_sectional_market_inputs(
    security_root: Path,
    panel_root: Path,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for year in range(2000, 2026):
        panel = _read_panel_year(
            panel_root,
            year,
            ["permno", "date", "universe_primary"],
        )
        keys = panel.loc[
            panel["universe_primary"], ["permno", "date"]
        ]
        if keys.empty:
            continue
        returns = _read_security_year(
            security_root, year, ["permno", "date", "return_total"]
        )
        joined = keys.merge(
            returns,
            on=["permno", "date"],
            how="left",
            validate="one_to_one",
        )
        joined["return_total"] = pd.to_numeric(
            joined["return_total"], errors="coerce"
        )
        joined["_positive"] = joined["return_total"].gt(0).astype(float)
        grouped = joined.groupby("date", sort=False, observed=True)
        daily = grouped.agg(
            primary_names=("return_total", "count"),
            market_breadth=("_positive", "mean"),
        )
        daily["cross_sectional_return_dispersion"] = grouped[
            "return_total"
        ].std(ddof=0)
        insufficient = daily["primary_names"].lt(30)
        daily.loc[
            insufficient,
            ["market_breadth", "cross_sectional_return_dispersion"],
        ] = np.nan
        records.append(daily.reset_index())
    if not records:
        raise ValueError("No frozen-primary rows available for market states")
    return (
        pd.concat(records, ignore_index=True)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def _coverage_records(
    frame: pd.DataFrame,
    year: int,
) -> list[dict[str, Any]]:
    primary = frame.loc[frame["universe_primary"]].copy()
    records: list[dict[str, Any]] = []
    for column in FEATURE_COLUMNS:
        values = primary[column]
        if column in CATEGORICAL_FEATURE_COLUMNS:
            valid = values.notna()
            quantiles = (np.nan, np.nan, np.nan)
            distinct = int(values.loc[valid].nunique())
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            valid = numeric.notna() & np.isfinite(numeric)
            finite = numeric.loc[valid]
            quantiles = (
                finite.quantile(0.01),
                finite.quantile(0.50),
                finite.quantile(0.99),
            )
            distinct = int(finite.nunique()) if len(finite) else 0
        records.append(
            {
                "year": year,
                "feature": column,
                "primary_rows": len(primary),
                "nonmissing_rows": int(valid.sum()),
                "coverage": float(valid.mean()) if len(primary) else np.nan,
                "p01": quantiles[0],
                "p50": quantiles[1],
                "p99": quantiles[2],
                "distinct_values": distinct,
            }
        )
    return records


def _stability_table(annual: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for feature, group in annual.groupby("feature", sort=False):
        total_rows = int(group["primary_rows"].sum())
        nonmissing = int(group["nonmissing_rows"].sum())
        coverage = group.loc[group["primary_rows"].gt(0), "coverage"]
        records.append(
            {
                "feature": feature,
                "primary_rows": total_rows,
                "nonmissing_rows": nonmissing,
                "overall_coverage": (
                    nonmissing / total_rows if total_rows else np.nan
                ),
                "annual_coverage_min": coverage.min(),
                "annual_coverage_median": coverage.median(),
                "annual_coverage_max": coverage.max(),
                "development_p50_min": group.loc[
                    group["year"].le(2012), "p50"
                ].min(),
                "development_p50_max": group.loc[
                    group["year"].le(2012), "p50"
                ].max(),
            }
        )
    return pd.DataFrame(records)


def _correlation_outputs(
    samples: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not samples:
        raise ValueError("No development feature sample for correlations")
    sample = pd.concat(samples, ignore_index=True)
    correlation = sample.corr(method="pearson", min_periods=1_000)
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= 0.90:
                pairs.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    pair_frame = pd.DataFrame(
        pairs,
        columns=[
            "feature_left",
            "feature_right",
            "correlation",
            "absolute_correlation",
        ],
    )
    if len(pair_frame):
        pair_frame = pair_frame.sort_values(
            "absolute_correlation", ascending=False, kind="stable"
        )
    return correlation, pair_frame


def _file_record(path: Path, root: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "rows": int(rows),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_features(project_root: Path) -> Path:
    """Build, diagnose, and atomically publish the frozen feature panel."""

    started = time.perf_counter()
    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    contract_path = config.root / "configs" / "features.yaml"
    contract = validate_feature_contract(contract_path)
    canonical_root = (
        config.project_path("data_interim") / "canonical_v1"
    )
    security_root = canonical_root / "security_day"
    phase2b_root = config.project_path("data_interim") / "phase2b_v1"
    panel_input_root = phase2b_root / "panel"
    if not security_root.is_dir() or not panel_input_root.is_dir():
        raise FileNotFoundError("Canonical and Phase 2B inputs must exist")

    interim = policy.ensure_directory(config.project_path("data_interim"))
    final_root = policy.assert_writable(interim / "features_v1")
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite feature output: {final_root}")
    table_dir = policy.ensure_directory(config.project_path("reports_tables"))
    report_targets = [table_dir / name for name in DIAGNOSTIC_NAMES]
    for path in report_targets:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_root = policy.assert_writable(
        interim / f"_features_v1_build_{timestamp}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)
    logger = configure_run_logger(
        "state_alpha.features.build",
        config.project_path("logs") / f"feature_build_{timestamp}.log",
        policy,
    )
    logger.info("Feature staging root: %s", staging_root)
    canonical_before = _snapshot(canonical_root)
    phase2b_before = _snapshot(phase2b_root)

    calendar = pd.read_parquet(phase2b_root / "market_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    logger.info("Building frozen-primary cross-sectional market inputs")
    cross_sectional = _build_cross_sectional_market_inputs(
        security_root, panel_input_root
    )
    market_features = compute_market_features(calendar, cross_sectional)
    market_path = staging_root / "market_features.parquet"
    public_market = market_features[
        ["session_id", "date", *[
            column
            for column in FEATURE_COLUMNS
            if column.startswith("market_")
            or column == "cross_sectional_return_dispersion"
        ]]
    ].copy()
    for column in public_market.columns:
        if column not in {"session_id", "date"}:
            public_market[column] = public_market[column].astype("float32")
    pq.write_table(
        pa.Table.from_pandas(public_market, preserve_index=False),
        market_path,
        compression="zstd",
        compression_level=6,
    )
    output_files = [
        _file_record(market_path, staging_root, len(public_market))
    ]

    panel_output_root = staging_root / "panel"
    diagnostics_root = staging_root / "diagnostics"
    panel_output_root.mkdir()
    diagnostics_root.mkdir()
    security_dataset = ds.dataset(
        security_root, format="parquet", partitioning="hive"
    )
    session_to_date = calendar.set_index("session_id")["date"]
    date_to_session = calendar.set_index("date")["session_id"]
    annual_audit: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    correlation_samples: list[pd.DataFrame] = []

    for year in range(2000, 2026):
        logger.info("Building Gate 3 feature panel for %d", year)
        target = _read_panel_year(
            panel_input_root, year, PANEL_INPUT_COLUMNS
        )
        if target.empty:
            output = pd.DataFrame(
                columns=[*OUTPUT_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
            )
        else:
            first_session = int(target["session_id"].min())
            last_session = int(target["session_id"].max())
            start_session = max(0, first_session - LOOKBACK_SESSIONS)
            start_date = pd.Timestamp(session_to_date.loc[start_session])
            end_date = pd.Timestamp(session_to_date.loc[last_session])
            security = _read_security_window(
                security_dataset,
                start_date=start_date,
                end_date=end_date,
                permnos=target["permno"].unique(),
            )
            security["session_id"] = security["date"].map(date_to_session)
            if security["session_id"].isna().any():
                raise ValueError("Security date is missing from market calendar")
            security["session_id"] = security["session_id"].astype(np.int32)
            calculated = compute_security_features(
                security, market_features
            )
            target_calculated = calculated.loc[
                calculated["date"].dt.year.eq(year)
            ]
            enriched = target.merge(
                target_calculated.drop(
                    columns=["siccd", "primary_exchange"]
                ),
                on=["permno", "date", "session_id"],
                how="left",
                validate="one_to_one",
            )
            # Controls are already named in calculated; retain them through the
            # merge above, while raw helper fields are discarded.
            output = finalize_target_features(enriched)
            missing_feature_rows = output[
                CONTINUOUS_FEATURE_COLUMNS
            ].isna().all(axis=1)
            if missing_feature_rows.any():
                raise ValueError(
                    f"{year} contains rows with no computed continuous features"
                )

        for column in CONTINUOUS_FEATURE_COLUMNS:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).astype("float32")
        if len(output):
            output["industry_code_2digit"] = pd.array(
                output["industry_code_2digit"], dtype="Int16"
            )
            output["exchange"] = output["exchange"].astype("string")
            output["session_id"] = output["session_id"].astype("int32")
            output["universe_primary"] = output[
                "universe_primary"
            ].astype(bool)
            output["largest_lagged_mcap_share_class"] = output[
                "largest_lagged_mcap_share_class"
            ].astype(bool)

        partition = panel_output_root / f"year={year}"
        partition.mkdir()
        file_path = partition / "part-00000.parquet"
        table = pa.Table.from_pandas(output, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"schema_version"] = FEATURE_SCHEMA_VERSION.encode("utf-8")
        metadata[b"labels_excluded"] = b"true"
        table = table.replace_schema_metadata(metadata)
        pq.write_table(
            table,
            file_path,
            compression="zstd",
            compression_level=6,
            row_group_size=50_000,
        )
        output_files.append(
            _file_record(file_path, staging_root, len(output))
        )
        primary_rows = int(output["universe_primary"].sum()) if len(output) else 0
        annual_audit.append(
            {
                "year": year,
                "analysis_rows": len(output),
                "primary_rows": primary_rows,
                "unique_permnos": int(output["permno"].nunique()) if len(output) else 0,
                "feature_count": len(FEATURE_COLUMNS),
                "label_columns": 0,
            }
        )
        coverage.extend(_coverage_records(output, year))
        if year <= 2012 and primary_rows:
            sample = output.loc[
                output["universe_primary"], CONTINUOUS_FEATURE_COLUMNS
            ]
            stride = max(
                1, int(np.ceil(len(sample) / CORRELATION_SAMPLE_PER_YEAR))
            )
            correlation_samples.append(
                sample.iloc[::stride]
                .head(CORRELATION_SAMPLE_PER_YEAR)
                .astype("float32")
            )
        logger.info(
            "Finished %d: analysis=%d primary=%d",
            year,
            len(output),
            primary_rows,
        )
        del target, output, table
        if "security" in locals():
            del security, calculated, target_calculated, enriched
        gc.collect()

    annual_coverage = pd.DataFrame(coverage)
    stability = _stability_table(annual_coverage)
    correlation, high_pairs = _correlation_outputs(correlation_samples)
    diagnostic_frames = {
        DIAGNOSTIC_NAMES[0]: annual_coverage,
        DIAGNOSTIC_NAMES[1]: stability,
        DIAGNOSTIC_NAMES[2]: correlation,
        DIAGNOSTIC_NAMES[3]: high_pairs,
    }
    for name, frame in diagnostic_frames.items():
        path = diagnostics_root / name
        frame.to_csv(path, index=name.endswith("correlation_development.csv"))
        output_files.append(_file_record(path, staging_root, len(frame)))

    canonical_after = _snapshot(canonical_root)
    phase2b_after = _snapshot(phase2b_root)
    if canonical_before != canonical_after:
        raise RuntimeError("Canonical input changed during feature build")
    if phase2b_before != phase2b_after:
        raise RuntimeError("Phase 2B input changed during feature build")
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "project_root": config.root,
        "staging_root": staging_root,
        "final_root": final_root,
        "canonical_root": canonical_root,
        "phase2b_root": phase2b_root,
        "canonical_snapshot": canonical_before,
        "phase2b_snapshot": phase2b_before,
        "canonical_unchanged": True,
        "phase2b_unchanged": True,
        "feature_contract_sha256": _sha256(contract_path),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "feature_contract_version": contract["version"],
        "lookback_sessions_loaded": LOOKBACK_SESSIONS,
        "price_representation": contract["price_representation"],
        "information_boundary": contract["information_boundary"],
        "correlation_diagnostic": {
            "period": "2000-2012",
            "sampling": (
                f"deterministic stride, cap {CORRELATION_SAMPLE_PER_YEAR} "
                "primary rows per year"
            ),
            "label_columns_used": False,
        },
        "annual_audit": annual_audit,
        "output_files": output_files,
        "overwrite_used": False,
    }
    _write_json(staging_root / "manifest.json", manifest)
    os.replace(staging_root, final_root)
    for name, target_path in zip(DIAGNOSTIC_NAMES, report_targets, strict=True):
        shutil.copy2(final_root / "diagnostics" / name, target_path)
    logger.info(
        "Published Gate 3 features to %s in %.1f seconds",
        final_root,
        time.perf_counter() - started,
    )
    return final_root
