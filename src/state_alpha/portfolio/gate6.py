"""Atomic Gate 6 portfolio translation and research evidence build."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_alpha.config import ProjectConfig, load_yaml
from state_alpha.logging_utils import configure_run_logger
from state_alpha.models.gate4 import (
    _file_records,
    _sha256,
    _snapshot,
    _write_json,
    _write_parquet,
)
from state_alpha.models.statistics import newey_west_mean
from state_alpha.paths import PathPolicy
from state_alpha.portfolio.core import (
    EXPOSURE_COLUMNS,
    INDUSTRY_DIVISIONS,
    performance_summary,
    realized_return_paths,
    turnover_between,
    variant_catalog,
    weight_matrix,
)
from state_alpha.portfolio.scoring import (
    load_formation_year,
    score_formation_year,
)


GATE6_SCHEMA_VERSION = "state_alpha_gate6_portfolios_v1"
REPORT_TABLES = [
    "gate6_variant_catalog.csv",
    "gate6_formation_domain_audit.csv",
    "gate6_performance_summary.csv",
    "gate6_cost_stress.csv",
    "gate6_performance_by_year.csv",
    "gate6_regime_summary.csv",
    "gate6_exposure_summary.csv",
    "gate6_liquidity_bucket_summary.csv",
    "gate6_participation_summary.csv",
    "gate6_missing_return_sensitivity.csv",
]
REPORT_MEMO = "GATE6_PORTFOLIO_EVIDENCE.md"


def validate_gate6_contract(
    contract: dict[str, Any],
    gate5_manifest: dict[str, Any],
) -> None:
    """Fail closed on any portfolio-governance or score drift."""

    if contract.get("status") != (
        "frozen_after_gate5_before_gate6_pnl"
    ):
        raise ValueError("Gate 6 contract is not frozen")
    governance = contract["governance"]
    required_true = [
        "gate4_and_gate5_oos_already_observed",
        "repeated_oos_tuning_prohibited",
        "model_feature_hyperparameter_changes_prohibited",
        "winner_selection_prohibited",
        "carry_every_score_forward",
        "label_filtered_gate5_predictions_prohibited_for_formation",
    ]
    if not all(governance.get(name) is True for name in required_true):
        raise ValueError("Gate 6 governance contract is incomplete")
    if governance.get("confirmatory_untouched_oos_claim_allowed") is not False:
        raise ValueError("Gate 6 cannot claim untouched confirmatory OOS")
    if governance.get("strict_oos_used_for_portfolio_selection") is not False:
        raise ValueError("Gate 6 cannot select portfolios on OOS")
    if contract["scores"] != gate5_manifest.get("scores"):
        raise ValueError("Gate 6 scores drifted from Gate 5")
    if int(contract["formation"]["holding_return_sessions"]) != 10:
        raise ValueError("Gate 6 holding period drifted")
    if float(
        contract["formation"]["fixed_capital_fraction_per_cohort"]
    ) != 0.5:
        raise ValueError("Gate 6 sleeve allocation drifted")


def _assert_source_hashes(
    project_root: Path,
    contract: dict[str, Any],
) -> None:
    governance = contract["governance"]
    paths = {
        "source_gate5_config_sha256": project_root
        / "configs"
        / "gate5.yaml",
        "source_models_config_sha256": project_root
        / "configs"
        / "models.yaml",
        "source_gate5_manifest_sha256": project_root
        / "data"
        / "processed"
        / "gate5_v1"
        / "manifest.json",
        "source_gate4_manifest_sha256": project_root
        / "data"
        / "processed"
        / "gate4_v1"
        / "manifest.json",
        "source_features_manifest_sha256": project_root
        / "data"
        / "interim"
        / "features_v1"
        / "manifest.json",
        "source_phase2b_manifest_sha256": project_root
        / "data"
        / "interim"
        / "phase2b_v1"
        / "manifest.json",
        "source_canonical_manifest_sha256": project_root
        / "data"
        / "interim"
        / "canonical_v1"
        / "manifest.json",
    }
    for key, path in paths.items():
        if _sha256(path) != governance[key]:
            raise ValueError(f"Gate 6 frozen source hash changed: {key}")


def _formation_sessions(
    calendar: pd.DataFrame,
    contract: dict[str, Any],
) -> list[int]:
    anchor = int(contract["formation"]["anchor_session_id"])
    last = int(
        calendar.loc[
            pd.to_datetime(calendar["date"]).dt.year.eq(2025),
            "session_id",
        ].max()
    )
    final_offset = int(
        contract["formation"]["last_return_offset_sessions"]
    )
    sessions = list(range(anchor, last - final_offset + 1, 5))
    if not sessions or sessions[-1] + final_offset > last:
        raise ValueError("Gate 6 formation calendar is invalid")
    return sessions


def _load_returns_by_session(
    canonical_root: Path,
    years: list[int],
    date_to_session: pd.Series,
) -> dict[int, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for year in years:
        path = (
            canonical_root
            / "security_day"
            / f"year={year}"
            / "part-00000.parquet"
        )
        if path.is_file():
            frames.append(
                pd.read_parquet(
                    path,
                    columns=[
                        "permno",
                        "date",
                        "return_total",
                        "daily_delisting_flag",
                    ],
                )
            )
    if not frames:
        raise ValueError("Gate 6 canonical return source is empty")
    returns = pd.concat(frames, ignore_index=True)
    returns["date"] = pd.to_datetime(returns["date"], errors="raise")
    returns["session_id"] = returns["date"].map(date_to_session)
    if returns["session_id"].isna().any():
        raise ValueError("Canonical return date is outside market calendar")
    output: dict[int, pd.DataFrame] = {}
    for session_id, group in returns.groupby(
        "session_id", sort=False, observed=True
    ):
        if group["permno"].duplicated().any():
            raise ValueError("Duplicate canonical security-session return")
        output[int(session_id)] = group.set_index("permno")[
            ["return_total", "daily_delisting_flag"]
        ]
    return output


def _cohort_diagnostics(
    base: pd.DataFrame,
    *,
    formation_date: pd.Timestamp,
    formation_session_id: int,
    execution_session_id: int,
    exit_session_id: int,
    weights: np.ndarray,
    filled_weights: np.ndarray,
    fill_mask: np.ndarray,
    alive_end: np.ndarray,
    missing_names: np.ndarray,
    continuous: np.ndarray,
    industries: np.ndarray,
    lag_adv: np.ndarray,
    contract: dict[str, Any],
) -> pd.DataFrame:
    output = base.copy()
    sleeve = float(
        contract["formation"]["fixed_capital_fraction_per_cohort"]
    )
    absolute = np.abs(filled_weights)
    gross = absolute.sum(axis=1)
    output["formation_date"] = formation_date
    output["formation_session_id"] = formation_session_id
    output["execution_session_id"] = execution_session_id
    output["exit_session_id"] = exit_session_id
    output["formation_names"] = len(fill_mask)
    output["execution_valid_names"] = int(fill_mask.sum())
    output["postfill_net_weight"] = filled_weights.sum(axis=1)
    output["postfill_gross_weight"] = gross
    output["cash_weight_for_self_financing"] = (
        -filled_weights.sum(axis=1) * sleeve
    )
    output["scheduled_exit_gross_weight"] = (
        np.abs(filled_weights * alive_end).sum(axis=1)
    )
    exposure = filled_weights @ continuous
    for index, name in enumerate(EXPOSURE_COLUMNS):
        output[name] = exposure[:, index]
    industry_net = filled_weights @ industries
    output["industry_one_way_imbalance"] = (
        0.5 * np.abs(industry_net).sum(axis=1)
    )
    squared = np.square(absolute).sum(axis=1)
    output["cohort_hhi"] = np.divide(
        squared,
        np.square(gross),
        out=np.full(len(gross), np.nan),
        where=gross > 0,
    )
    output["cohort_effective_names"] = np.divide(
        1.0,
        output["cohort_hhi"],
        out=np.full(len(gross), np.nan),
        where=output["cohort_hhi"].to_numpy(dtype=float) > 0,
    )
    output["cohort_max_absolute_weight"] = absolute.max(axis=1)
    output["missing_bound_names"] = (
        (absolute > 0) & missing_names[None, :]
    ).sum(axis=1)
    output["missing_bound_gross_weight"] = (
        absolute * missing_names[None, :]
    ).sum(axis=1)

    valid_adv = np.isfinite(lag_adv) & (lag_adv > 0)
    base_participation = np.divide(
        sleeve * absolute,
        lag_adv[None, :],
        out=np.full_like(absolute, np.nan),
        where=valid_adv[None, :] & (absolute > 0),
    )
    for nav in contract["exposures"]["adv_reference_nav_usd"]:
        label = f"{int(nav / 1_000_000)}m"
        participation = base_participation * float(nav)
        output[f"entry_participation_p95_{label}"] = np.nanquantile(
            participation, 0.95, axis=1
        )
        output[f"entry_participation_max_{label}"] = np.nanmax(
            participation, axis=1
        )
        for threshold in contract["exposures"][
            "participation_thresholds"
        ]:
            active = np.isfinite(participation)
            numerator = (
                active & (participation > float(threshold))
            ).sum(axis=1)
            denominator = active.sum(axis=1)
            output[
                f"entry_fraction_gt_{int(float(threshold) * 100)}pct_"
                f"{label}"
            ] = np.divide(
                numerator,
                denominator,
                out=np.zeros(len(output), dtype=float),
                where=denominator > 0,
            )
    return output


def _periods() -> dict[str, tuple[int, int]]:
    return {
        "full_secondary_oos": (2013, 2025),
        "transition_sensitivity": (2013, 2013),
        "boundary_clean_primary": (2014, 2025),
    }


def _cost_stress(
    daily: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    trading_days = int(contract["statistics"]["trading_days_per_year"])
    hac_lags = int(contract["statistics"]["hac_lags"])
    for bps in contract["turnover_and_costs"]["one_way_cost_bps"]:
        column = f"net_return_{int(bps)}bps"
        summary = performance_summary(
            daily,
            value_column=column,
            group_columns=["variant_id"],
            periods=_periods(),
            hac_lags=hac_lags,
            trading_days=trading_days,
        )
        summary["one_way_cost_bps"] = int(bps)
        outputs.append(summary)
    return pd.concat(outputs, ignore_index=True)


def _year_summary(
    daily: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for year in range(2013, 2026):
        summary = performance_summary(
            daily,
            value_column="gross_return_loss100",
            group_columns=["variant_id"],
            periods={str(year): (year, year)},
            hac_lags=int(contract["statistics"]["hac_lags"]),
            trading_days=int(
                contract["statistics"]["trading_days_per_year"]
            ),
        )
        summary["year"] = year
        outputs.append(summary.drop(columns="period"))
    return pd.concat(outputs, ignore_index=True)


def _regime_summary(
    daily: pd.DataFrame,
    regimes: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    merged = daily.merge(
        regimes[
            ["date", "trend_regime", "volatility_regime"]
        ],
        on="date",
        how="left",
        validate="many_to_one",
    )
    merged = merged.loc[merged["year"].between(2014, 2025)]
    records: list[dict[str, Any]] = []
    for keys, group in merged.groupby(
        ["variant_id", "trend_regime", "volatility_regime"],
        sort=False,
        observed=True,
        dropna=False,
    ):
        stats = newey_west_mean(
            group["gross_return_loss100"],
            int(contract["statistics"]["hac_lags"]),
        )
        values = group["gross_return_loss100"].to_numpy(dtype=float)
        annualized_return = float(
            np.nanmean(values)
            * int(contract["statistics"]["trading_days_per_year"])
        )
        annualized_volatility = float(
            np.nanstd(values, ddof=1)
            * np.sqrt(
                int(contract["statistics"]["trading_days_per_year"])
            )
        )
        records.append(
            {
                "variant_id": keys[0],
                "trend_regime": keys[1],
                "volatility_regime": keys[2],
                **stats,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe": (
                    annualized_return / annualized_volatility
                    if annualized_volatility > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _turnover_period_summary(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for period, (start_year, end_year) in _periods().items():
        sample = daily.loc[daily["year"].between(start_year, end_year)]
        for variant_id, group in sample.groupby(
            "variant_id", sort=False, observed=True
        ):
            gross = float(group["gross_return_loss100"].mean())
            turnover = float(group["one_way_turnover"].mean())
            records.append(
                {
                    "variant_id": variant_id,
                    "period": period,
                    "mean_daily_one_way_turnover": turnover,
                    "annualized_one_way_turnover": turnover * 252.0,
                    "break_even_one_way_bps": (
                        10000.0 * gross / (2.0 * turnover)
                        if turnover > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _exposure_summary(
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    sample = cohorts.loc[
        pd.to_datetime(cohorts["formation_date"]).dt.year.between(2014, 2025)
    ]
    records: list[dict[str, Any]] = []
    for variant_id, group in sample.groupby(
        "variant_id", sort=False, observed=True
    ):
        record: dict[str, Any] = {
            "variant_id": variant_id,
            "cohorts": len(group),
            "mean_prefill_net_weight": float(
                group["prefill_net_weight"].mean()
            ),
            "maximum_prefill_abs_net_weight": float(
                group["prefill_net_weight"].abs().max()
            ),
            "mean_prefill_gross_weight": float(
                group["prefill_gross_weight"].mean()
            ),
            "mean_postfill_gross_weight": float(
                group["postfill_gross_weight"].mean()
            ),
            "mean_absolute_postfill_net_weight": float(
                group["postfill_net_weight"].abs().mean()
            ),
            "mean_prefill_max_abs_constraint": float(
                group["prefill_max_abs_constraint"].mean()
            ),
            "maximum_prefill_constraint": float(
                group["prefill_max_abs_constraint"].max()
            ),
            "mean_industry_one_way_imbalance": float(
                group["industry_one_way_imbalance"].mean()
            ),
            "mean_cohort_hhi": float(group["cohort_hhi"].mean()),
            "mean_cohort_effective_names": float(
                group["cohort_effective_names"].mean()
            ),
            "maximum_cohort_weight": float(
                group["cohort_max_absolute_weight"].max()
            ),
            "mean_missing_bound_gross_weight": float(
                group["missing_bound_gross_weight"].mean()
            ),
        }
        for column in EXPOSURE_COLUMNS:
            record[f"mean_{column}"] = float(group[column].mean())
            record[f"mean_absolute_{column}"] = float(
                group[column].abs().mean()
            )
        records.append(record)
    return pd.DataFrame(records)


def _participation_summary(
    cohorts: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    sample = cohorts.loc[
        pd.to_datetime(cohorts["formation_date"]).dt.year.between(2014, 2025)
    ]
    columns: list[str] = []
    for nav in contract["exposures"]["adv_reference_nav_usd"]:
        label = f"{int(nav / 1_000_000)}m"
        columns.extend(
            [
                f"entry_participation_p95_{label}",
                f"entry_participation_max_{label}",
            ]
        )
        for threshold in contract["exposures"][
            "participation_thresholds"
        ]:
            columns.append(
                f"entry_fraction_gt_{int(float(threshold) * 100)}pct_"
                f"{label}"
            )
    records: list[dict[str, Any]] = []
    for variant_id, group in sample.groupby(
        "variant_id", sort=False, observed=True
    ):
        record: dict[str, Any] = {
            "variant_id": variant_id,
            "cohorts": len(group),
        }
        for column in columns:
            record[f"mean_{column}"] = float(group[column].mean())
            record[f"p95_{column}"] = float(group[column].quantile(0.95))
        records.append(record)
    return pd.DataFrame(records)


def _formation_audit(
    year_records: list[dict[str, Any]],
) -> pd.DataFrame:
    output = pd.DataFrame(year_records)
    total = {
        "year": "total",
        "formation_dates": int(output["formation_dates"].sum()),
        "point_in_time_primary_rows": int(
            output["point_in_time_primary_rows"].sum()
        ),
        "execution_valid_rows": int(
            output["execution_valid_rows"].sum()
        ),
        "gate5_label_valid_rows": int(
            output["gate5_label_valid_rows"].sum()
        ),
        "label_filter_excluded_rows": int(
            output["label_filter_excluded_rows"].sum()
        ),
    }
    return pd.concat([output, pd.DataFrame([total])], ignore_index=True)


def _memo_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Score | Ann. gross | Vol. | Sharpe | NW t | Max DD | "
        "Break-even bps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| `{row.signal}` | {row.annualized_return:.2%} | "
            f"{row.annualized_volatility:.2%} | {row.sharpe:.2f} | "
            f"{row.newey_west_t:.2f} | {row.maximum_drawdown:.2%} | "
            f"{row.break_even_one_way_bps:.1f} |"
        )
    return "\n".join(lines)


def _write_evidence_memo(
    path: Path,
    *,
    performance: pd.DataFrame,
    exposure: pd.DataFrame,
    formation_audit: pd.DataFrame,
    missing_sensitivity: pd.DataFrame,
    config_hash: str,
) -> None:
    rank_raw = performance.loc[
        performance["period"].eq("boundary_clean_primary")
        & performance["weighting"].eq("rank_weight")
        & performance["neutralization"].eq("raw")
        & performance["bucket"].eq("all")
    ].sort_values("signal", kind="stable")
    rank_neutral = performance.loc[
        performance["period"].eq("boundary_clean_primary")
        & performance["weighting"].eq("rank_weight")
        & performance["neutralization"].eq("risk_neutralized")
        & performance["bucket"].eq("all")
    ].sort_values("signal", kind="stable")
    neutral_exposure = exposure.loc[
        exposure["neutralization"].eq("risk_neutralized")
        & exposure["bucket"].eq("all")
    ]
    maximum_constraint = (
        float(neutral_exposure["maximum_prefill_constraint"].max())
        if not neutral_exposure.empty
        else np.nan
    )
    total = formation_audit.iloc[-1]
    maximum_missing_gap = float(
        missing_sensitivity.loc[
            missing_sensitivity["period"].eq(
                "boundary_clean_primary"
            ),
            "annualized_return_difference_zero_minus_loss100",
        ].abs().max()
    )
    path.write_text(
        f"""# Gate 6 portfolio evidence

Date: {datetime.now().date().isoformat()}  
Classification: secondary locked after Gate 5  
Frozen config SHA-256: `{config_hash}`

## Formation-domain correction

Gate 5 predictions were label-evaluation rows, not a valid formation domain.
Gate 6 therefore rescored every frozen annual model on {int(total['point_in_time_primary_rows']):,}
point-in-time primary-universe observations across
{int(total['formation_dates']):,} formation dates. The {int(total['label_filter_excluded_rows']):,}
rows excluded by the Gate 5 label-valid table were restored. Future labels are
absent from the Gate 6 formation-score artifact. Failed t+1 executions are not
used for reranking or redistribution; unmatched notional stays in cash.

## Boundary-clean 2014-2025 raw rank-weight portfolios

{_memo_table(rank_raw)}

## Boundary-clean 2014-2025 risk-neutralized rank-weight portfolios

{_memo_table(rank_neutral)}

Risk-neutralized weights use a support-preserving projection against beta,
size, and industry-division constraints before execution. The maximum
pre-fill constraint residual across those boundary-clean cohorts was
{maximum_constraint:.3e}. Execution failures are not followed by an
outcome-aware reoptimization, so post-fill exposure diagnostics remain
reported.

## Cost and missing-return interpretation

The reported break-even and 5/10/20/30 bps results are one-way cost stress
tests, not calibrated realized costs. The maximum absolute difference in
annualized return between the conservative -100% missing-terminal path and the
0% sensitivity path was {maximum_missing_gap:.2%} across boundary-clean
variants. Valid delisting returns are included and then held as cash.

## Research conclusion boundary

All ten scores and all preregistered portfolio variants are reported. No
winner is selected from Gate 5 or Gate 6 OOS outcomes. These results are
secondary evidence because earlier OOS model evidence had already been
observed. Reference-NAV participation is a diagnostic, not a capacity
estimate, and cost stresses are not a net-tradability claim.

## Interview narrative

The research contribution is the audit trail as much as the model: a
label-valid prediction panel was caught before it could become an investable
universe, frozen annual models were redeployed on the point-in-time domain,
overlapping labels were translated into delayed staggered daily P&L, and
performance was separated from exposure, cost, liquidity, and missing-terminal
assumptions.

## Reproducibility checklist

- frozen Gate 6 contract and source hashes;
- complete point-in-time formation domain, with no forward-label columns;
- every fifth market-session formation and t+1-close execution;
- two fixed half-capital sleeves and ten return sessions;
- entry/exit turnover netted by security;
- conservative and zero terminal-value paths;
- all 100 preregistered variants, with no outcome-based selection;
- atomic no-overwrite artifact and independent hash validation.
""",
        encoding="utf-8",
    )


def build_gate6(project_root: Path) -> Path:
    """Build the frozen Gate 6 portfolio artifact atomically."""

    started = time.perf_counter()
    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    contract_path = config.root / "configs" / "gate6.yaml"
    gate5_contract_path = config.root / "configs" / "gate5.yaml"
    contract = load_yaml(contract_path)
    gate5_contract = load_yaml(gate5_contract_path)

    processed = config.project_path("data_processed")
    interim = config.project_path("data_interim")
    gate4_root = processed / "gate4_v1"
    gate5_root = processed / "gate5_v1"
    features_root = interim / "features_v1"
    phase2b_root = interim / "phase2b_v1"
    canonical_root = interim / "canonical_v1"
    source_roots = {
        "gate4": gate4_root,
        "gate5": gate5_root,
        "features": features_root,
        "phase2b": phase2b_root,
        "canonical": canonical_root,
    }
    if not all(path.is_dir() for path in source_roots.values()):
        raise FileNotFoundError("Gate 6 requires all frozen source artifacts")
    gate5_manifest = json.loads(
        (gate5_root / "manifest.json").read_text(encoding="utf-8")
    )
    validate_gate6_contract(contract, gate5_manifest)
    _assert_source_hashes(config.root, contract)
    gate5_validation = json.loads(
        (
            config.project_path("reports_tables")
            / "gate5_validation.json"
        ).read_text(encoding="utf-8")
    )
    if gate5_validation.get("passed") is not True:
        raise ValueError("Gate 5 independent validation did not pass")

    final_root = policy.assert_writable(processed / "gate6_v1")
    if final_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Gate 6 output: {final_root}"
        )
    report_table_dir = policy.ensure_directory(
        config.project_path("reports_tables")
    )
    report_memo_dir = policy.ensure_directory(
        config.project_path("reports_memo")
    )
    report_targets = [report_table_dir / name for name in REPORT_TABLES]
    memo_target = report_memo_dir / REPORT_MEMO
    for path in [*report_targets, memo_target]:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite Gate 6 report: {path}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_root = policy.assert_writable(
        processed / f"_gate6_v1_build_{timestamp}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)
    logger = configure_run_logger(
        "state_alpha.gate6.build",
        config.project_path("logs") / f"gate6_build_{timestamp}.log",
        policy,
    )
    logger.info("Gate 6 staging root: %s", staging_root)
    source_before = {
        name: _snapshot(path) for name, path in source_roots.items()
    }
    config_hash = _sha256(contract_path)

    formation_root = staging_root / "formation_scores"
    evidence_root = staging_root / "evidence"
    diagnostics_root = staging_root / "diagnostics"
    for path in [formation_root, evidence_root, diagnostics_root]:
        path.mkdir()

    calendar = pd.read_parquet(phase2b_root / "market_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    calendar = calendar.sort_values("session_id", kind="stable")
    formation_sessions = _formation_sessions(calendar, contract)
    formation_set = set(formation_sessions)
    formation_by_year = (
        calendar.loc[calendar["session_id"].isin(formation_set)]
        .assign(year=lambda value: value["date"].dt.year)
        .groupby("year", observed=True)["session_id"]
        .agg(lambda value: set(value.astype(int)))
        .to_dict()
    )
    variants = variant_catalog(contract)
    _write_parquet(
        variants,
        evidence_root / "variant_catalog.parquet",
        schema_version=GATE6_SCHEMA_VERSION,
    )

    year_audit: list[dict[str, Any]] = []
    scored_rows = 0
    feature_panel = features_root / "panel"
    phase2b_panel = phase2b_root / "panel"
    for year in range(2013, 2026):
        sessions = formation_by_year.get(year, set())
        formation = load_formation_year(
            feature_panel,
            phase2b_panel,
            year,
            sessions,
            load_yaml(config.root / "configs" / "models.yaml"),
        )
        scored = score_formation_year(
            formation,
            year=year,
            gate4_root=gate4_root,
            gate5_root=gate5_root,
            gate5_contract=gate5_contract,
        )
        gate5_evaluation = pd.read_parquet(
            gate5_root
            / "predictions"
            / f"year={year}"
            / "part-00000.parquet",
            columns=["session_id"],
        )
        gate5_rows = int(
            gate5_evaluation["session_id"].isin(sessions).sum()
        )
        partition = formation_root / f"year={year}"
        partition.mkdir()
        _write_parquet(
            scored,
            partition / "part-00000.parquet",
            schema_version=GATE6_SCHEMA_VERSION,
        )
        year_audit.append(
            {
                "year": year,
                "formation_dates": int(scored["session_id"].nunique()),
                "point_in_time_primary_rows": len(scored),
                "execution_valid_rows": int(
                    scored["execution_valid"].astype(bool).sum()
                ),
                "gate5_label_valid_rows": gate5_rows,
                "label_filter_excluded_rows": len(scored) - gate5_rows,
            }
        )
        scored_rows += len(scored)
        logger.info(
            "Gate 6 rescored %d: dates=%d rows=%d Gate5-label-rows=%d",
            year,
            scored["session_id"].nunique(),
            len(scored),
            gate5_rows,
        )
        del formation, scored, gate5_evaluation
        gc.collect()

    formation_audit = _formation_audit(year_audit)
    expected = contract["governance"]["pre_pnl_formation_domain_audit"]
    total_audit = formation_audit.iloc[-1]
    if int(total_audit["formation_dates"]) != int(
        expected["formation_dates"]
    ):
        raise ValueError("Gate 6 formation-date audit changed")
    if int(total_audit["point_in_time_primary_rows"]) != int(
        expected["point_in_time_primary_rows"]
    ):
        raise ValueError("Gate 6 point-in-time formation rows changed")
    if int(total_audit["execution_valid_rows"]) != int(
        expected["execution_valid_rows"]
    ):
        raise ValueError("Gate 6 execution-valid audit changed")
    if int(total_audit["gate5_label_valid_rows"]) != int(
        expected["gate5_label_valid_rows"]
    ):
        raise ValueError("Gate 6 Gate 5 label-domain audit changed")

    oos_calendar = calendar.loc[
        calendar["session_id"].between(
            int(contract["formation"]["anchor_session_id"]),
            int(
                calendar.loc[
                    calendar["date"].dt.year.eq(2025), "session_id"
                ].max()
            ),
        )
    ].copy()
    oos_dates = oos_calendar["date"].to_numpy()
    oos_sessions = oos_calendar["session_id"].to_numpy(dtype=int)
    session_to_index = {
        int(session): index
        for index, session in enumerate(oos_sessions)
    }
    date_to_session = calendar.set_index("date")["session_id"]
    daily_lower = np.zeros((len(oos_calendar), len(variants)), dtype=float)
    daily_upper = np.zeros_like(daily_lower)
    daily_turnover = np.zeros_like(daily_lower)
    cohort_frames: list[pd.DataFrame] = []
    rolling: deque[dict[str, Any]] = deque()
    sleeve = float(
        contract["formation"]["fixed_capital_fraction_per_cohort"]
    )
    holding_count = int(
        contract["formation"]["holding_return_sessions"]
    )

    for year in range(2013, 2026):
        scores = pd.read_parquet(formation_root / f"year={year}")
        returns_by_session = _load_returns_by_session(
            canonical_root,
            [year, year + 1],
            date_to_session,
        )
        for formation_session, group in scores.groupby(
            "session_id", sort=True, observed=True
        ):
            frame = group.sort_values("permno", kind="stable").reset_index(
                drop=True
            )
            formation_session = int(formation_session)
            execution_session = formation_session + 1
            holding_sessions = list(
                range(formation_session + 2, formation_session + 12)
            )
            if len(holding_sessions) != holding_count:
                raise ValueError("Gate 6 holding-session count changed")
            exit_session = holding_sessions[-1]
            weights, base_diagnostics, continuous, industries = (
                weight_matrix(frame, variants, contract)
            )
            fill_mask = frame["execution_valid"].astype(bool).to_numpy()
            filled_weights = weights * fill_mask[None, :]
            lower, upper, alive_end, _ = realized_return_paths(
                frame["permno"].to_numpy(dtype=np.int64),
                fill_mask,
                holding_sessions,
                returns_by_session,
            )
            return_indices = [
                session_to_index[session] for session in holding_sessions
            ]
            daily_lower[return_indices, :] += (
                sleeve * (filled_weights @ lower)
            ).T
            daily_upper[return_indices, :] += (
                sleeve * (filled_weights @ upper)
            ).T

            entry_weights = sleeve * filled_weights
            if len(rolling) >= 2:
                exiting = rolling.popleft()
                if int(exiting["exit_session"]) != execution_session:
                    raise ValueError("Gate 6 cohort roll calendar changed")
                event_turnover = turnover_between(
                    frame["permno"].to_numpy(dtype=np.int64),
                    entry_weights,
                    exiting["permnos"],
                    exiting["exit_weights"],
                )
            else:
                event_turnover = turnover_between(
                    frame["permno"].to_numpy(dtype=np.int64),
                    entry_weights,
                )
            daily_turnover[
                session_to_index[execution_session], :
            ] += event_turnover
            exit_weights = sleeve * (
                filled_weights * alive_end[None, :]
            )
            rolling.append(
                {
                    "exit_session": exit_session,
                    "permnos": frame["permno"].to_numpy(dtype=np.int64),
                    "exit_weights": exit_weights,
                }
            )
            missing_names = np.any(
                np.abs(lower - upper) > 0, axis=1
            )
            cohort_frames.append(
                _cohort_diagnostics(
                    base_diagnostics,
                    formation_date=pd.Timestamp(frame["date"].iloc[0]),
                    formation_session_id=formation_session,
                    execution_session_id=execution_session,
                    exit_session_id=exit_session,
                    weights=weights,
                    filled_weights=filled_weights,
                    fill_mask=fill_mask,
                    alive_end=alive_end,
                    missing_names=missing_names,
                    continuous=continuous,
                    industries=industries,
                    lag_adv=pd.to_numeric(
                        frame["lag_adv20_usd"], errors="coerce"
                    ).to_numpy(dtype=float),
                    contract=contract,
                )
            )
        logger.info("Gate 6 constructed cohorts through %d", year)
        del scores, returns_by_session
        gc.collect()

    while rolling:
        exiting = rolling.popleft()
        daily_turnover[
            session_to_index[int(exiting["exit_session"])], :
        ] += 0.5 * np.abs(exiting["exit_weights"]).sum(axis=1)

    cohorts = pd.concat(cohort_frames, ignore_index=True)
    _write_parquet(
        cohorts,
        evidence_root / "cohort_diagnostics.parquet",
        schema_version=GATE6_SCHEMA_VERSION,
    )
    daily = pd.DataFrame(
        {
            "date": np.repeat(oos_dates, len(variants)),
            "session_id": np.repeat(oos_sessions, len(variants)),
            "variant_id": np.tile(
                variants["variant_id"].to_numpy(), len(oos_calendar)
            ),
            "gross_return_loss100": daily_lower.reshape(-1),
            "gross_return_zero": daily_upper.reshape(-1),
            "one_way_turnover": daily_turnover.reshape(-1),
        }
    )
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily["year"] = daily["date"].dt.year
    for bps in contract["turnover_and_costs"]["one_way_cost_bps"]:
        daily[f"net_return_{int(bps)}bps"] = (
            daily["gross_return_loss100"]
            - 2.0
            * daily["one_way_turnover"]
            * float(bps)
            / 10000.0
        )
    _write_parquet(
        daily,
        evidence_root / "daily_portfolios.parquet",
        schema_version=GATE6_SCHEMA_VERSION,
    )

    performance = performance_summary(
        daily,
        value_column="gross_return_loss100",
        group_columns=["variant_id"],
        periods=_periods(),
        hac_lags=int(contract["statistics"]["hac_lags"]),
        trading_days=int(
            contract["statistics"]["trading_days_per_year"]
        ),
    )
    performance = performance.merge(
        _turnover_period_summary(daily),
        on=["variant_id", "period"],
        how="left",
        validate="one_to_one",
    ).merge(
        variants,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    cost_stress = _cost_stress(daily, contract).merge(
        variants,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    by_year = _year_summary(daily, contract).merge(
        variants,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    regimes = pd.read_parquet(gate4_root / "market_regimes.parquet")
    regimes["date"] = pd.to_datetime(regimes["date"], errors="raise")
    regime_summary = _regime_summary(
        daily, regimes, contract
    ).merge(
        variants,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    exposure = _exposure_summary(cohorts).merge(
        variants,
        on="variant_id",
        how="left",
        validate="one_to_one",
    )
    participation = _participation_summary(
        cohorts, contract
    ).merge(
        variants,
        on="variant_id",
        how="left",
        validate="one_to_one",
    )
    liquidity = performance.loc[
        performance["period"].eq("boundary_clean_primary")
        & performance["bucket"].ne("all")
    ].copy()
    upper_performance = performance_summary(
        daily,
        value_column="gross_return_zero",
        group_columns=["variant_id"],
        periods=_periods(),
        hac_lags=int(contract["statistics"]["hac_lags"]),
        trading_days=int(
            contract["statistics"]["trading_days_per_year"]
        ),
    )
    missing_sensitivity = performance[
        [
            "variant_id",
            "period",
            "annualized_return",
            "sharpe",
            "cumulative_return",
        ]
    ].rename(
        columns={
            "annualized_return": "loss100_annualized_return",
            "sharpe": "loss100_sharpe",
            "cumulative_return": "loss100_cumulative_return",
        }
    ).merge(
        upper_performance[
            [
                "variant_id",
                "period",
                "annualized_return",
                "sharpe",
                "cumulative_return",
            ]
        ].rename(
            columns={
                "annualized_return": "zero_annualized_return",
                "sharpe": "zero_sharpe",
                "cumulative_return": "zero_cumulative_return",
            }
        ),
        on=["variant_id", "period"],
        how="left",
        validate="one_to_one",
    ).merge(
        variants,
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    missing_sensitivity[
        "annualized_return_difference_zero_minus_loss100"
    ] = (
        missing_sensitivity["zero_annualized_return"]
        - missing_sensitivity["loss100_annualized_return"]
    )

    diagnostic_frames = {
        REPORT_TABLES[0]: variants,
        REPORT_TABLES[1]: formation_audit,
        REPORT_TABLES[2]: performance,
        REPORT_TABLES[3]: cost_stress,
        REPORT_TABLES[4]: by_year,
        REPORT_TABLES[5]: regime_summary,
        REPORT_TABLES[6]: exposure,
        REPORT_TABLES[7]: liquidity,
        REPORT_TABLES[8]: participation,
        REPORT_TABLES[9]: missing_sensitivity,
    }
    for name, frame in diagnostic_frames.items():
        frame.to_csv(diagnostics_root / name, index=False)
    _write_evidence_memo(
        diagnostics_root / REPORT_MEMO,
        performance=performance,
        exposure=exposure,
        formation_audit=formation_audit,
        missing_sensitivity=missing_sensitivity,
        config_hash=config_hash,
    )

    source_after = {
        name: _snapshot(path) for name, path in source_roots.items()
    }
    if source_before != source_after:
        raise RuntimeError("A frozen Gate 6 source changed during build")
    output_files = _file_records(staging_root)
    manifest = {
        "schema_version": GATE6_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "project_root": config.root,
        "staging_root": staging_root,
        "final_root": final_root,
        "source_roots": source_roots,
        "source_snapshots": source_before,
        "sources_unchanged": True,
        "gate6_config_sha256": config_hash,
        "gate5_config_sha256": _sha256(gate5_contract_path),
        "source_manifest_hashes": {
            name: _sha256(path / "manifest.json")
            for name, path in source_roots.items()
        },
        "analysis_classification": contract["governance"][
            "analysis_classification"
        ],
        "confirmatory_untouched_oos_claim_allowed": False,
        "strict_oos_used_for_portfolio_selection": False,
        "winner_selection_used": False,
        "formation_domain_uses_future_labels": False,
        "formation_dates": len(formation_sessions),
        "formation_score_rows": scored_rows,
        "portfolio_variants": len(variants),
        "daily_portfolio_rows": len(daily),
        "cohort_diagnostic_rows": len(cohorts),
        "boundary_clean_period": "2014-2025",
        "transition_sensitivity_period": "2013",
        "missing_return_primary": "loss100",
        "missing_return_sensitivity": "zero",
        "output_files": output_files,
        "overwrite_used": False,
    }
    _write_json(staging_root / "manifest.json", manifest)
    os.replace(staging_root, final_root)
    for name, target in zip(REPORT_TABLES, report_targets, strict=True):
        shutil.copy2(final_root / "diagnostics" / name, target)
    shutil.copy2(
        final_root / "diagnostics" / REPORT_MEMO,
        memo_target,
    )
    logger.info(
        "Published Gate 6 to %s in %.1f seconds",
        final_root,
        time.perf_counter() - started,
    )
    return final_root
