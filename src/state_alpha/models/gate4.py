"""Atomic Gate 4 build: classical evidence and explicit linear interactions."""

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
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig, load_yaml
from state_alpha.features.core import OUTPUT_IDENTIFIER_COLUMNS
from state_alpha.logging_utils import configure_run_logger
from state_alpha.models.linear import (
    coefficient_table,
    fit_selected_models,
    predict_models,
    select_hyperparameters,
)
from state_alpha.models.matrix import (
    FAMILY_COLUMNS,
    alpha_contract,
    classical_sample,
    load_modeling_year,
    modeling_sample,
    period_name,
)
from state_alpha.models.statistics import (
    daily_double_sorts,
    daily_fama_macbeth,
    daily_rank_ic,
    daily_sorts,
    newey_west_mean,
    summarize_by_year,
    summarize_daily,
)
from state_alpha.paths import PathPolicy


GATE4_SCHEMA_VERSION = "state_alpha_gate4_baselines_v1"
REPORT_TABLES = [
    "gate4_univariate_ic_summary.csv",
    "gate4_univariate_ic_by_year.csv",
    "gate4_sort_summary.csv",
    "gate4_sort_by_year.csv",
    "gate4_double_sort_summary.csv",
    "gate4_fama_macbeth_summary.csv",
    "gate4_linear_hyperparameter_selection.csv",
    "gate4_linear_model_ic_summary.csv",
    "gate4_linear_model_ic_by_year.csv",
    "gate4_linear_model_sort_summary.csv",
    "gate4_market_regime_summary.csv",
    "gate4_model_refits.csv",
    "gate4_model_coefficients.csv",
]
REPORT_MEMO = "GATE4_BASELINE_EVIDENCE.md"


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


def _write_parquet(
    frame: pd.DataFrame,
    path: Path,
    *,
    schema_version: str = GATE4_SCHEMA_VERSION,
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = schema_version.encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        row_group_size=50_000,
    )


def _market_regimes(
    market_features_path: Path,
    contract: dict[str, Any],
) -> pd.DataFrame:
    market = pd.read_parquet(market_features_path)
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    market = market.sort_values("session_id", kind="stable").reset_index(
        drop=True
    )
    config = contract["market_regimes"]
    lookback = int(config["lookback_sessions"])
    minimum = int(config["minimum_history_sessions"])
    trend = config["trend_variable"]
    volatility = config["volatility_variable"]
    market["_trend_threshold"] = (
        market[trend]
        .shift(1)
        .rolling(lookback, min_periods=minimum)
        .median()
    )
    market["_volatility_threshold"] = (
        market[volatility]
        .shift(1)
        .rolling(lookback, min_periods=minimum)
        .median()
    )
    market["trend_regime"] = np.where(
        market["_trend_threshold"].isna(),
        None,
        np.where(
            market[trend].ge(market["_trend_threshold"]),
            "up",
            "down",
        ),
    )
    market["volatility_regime"] = np.where(
        market["_volatility_threshold"].isna(),
        None,
        np.where(
            market[volatility].ge(market["_volatility_threshold"]),
            "high",
            "low",
        ),
    )
    return market[
        [
            "session_id",
            "date",
            trend,
            volatility,
            "_trend_threshold",
            "_volatility_threshold",
            "trend_regime",
            "volatility_regime",
        ]
    ]


def _load_samples(
    root: Path,
    years: range | list[int],
) -> pd.DataFrame:
    frames = [
        pd.read_parquet(root / f"year={year}")
        for year in years
        if (root / f"year={year}").is_dir()
    ]
    if not frames:
        raise ValueError("No optimizer samples found")
    output = pd.concat(frames, ignore_index=True)
    output["date"] = pd.to_datetime(output["date"], errors="raise")
    return output


def _purge_and_embargo_training(
    frame: pd.DataFrame,
    evaluation_start_session_id: int,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    """Keep only labels ending strictly before the embargo cutoff."""

    embargo = int(
        contract["linear_models"]["expanding_oos"][
            "embargo_trading_days"
        ]
    )
    cutoff = int(evaluation_start_session_id) - embargo
    output = frame.loc[
        frame["label_end_session_id_10d"].lt(cutoff)
    ].copy()
    if output.empty:
        raise ValueError(
            "Purging and embargo removed the full training sample"
        )
    return output, cutoff


def _write_model_state(
    root: Path,
    *,
    state_id: str,
    model_fit_id: str,
    preprocessor: Any,
    models: dict[str, Any],
) -> str:
    """Persist transparent reconstruction state without pickle/joblib."""

    model_state = root / state_id
    model_state.mkdir(parents=False, exist_ok=False)
    preprocessing = preprocessor.to_dict()
    preprocessing["model_fit_id"] = model_fit_id
    _write_json(
        model_state / "preprocessing.json",
        preprocessing,
    )
    _write_json(
        model_state / "linear_models.json",
        {
            name: {
                "class": type(model).__name__,
                "parameters": model.get_params(deep=False),
                "intercept": float(model.intercept_),
                "coefficients": np.asarray(
                    model.coef_, dtype=float
                ).tolist(),
            }
            for name, model in models.items()
        },
    )
    return f"model_states/{state_id}"


def _summarize_double_sorts(
    frame: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["period"] = work["year"].map(period_name)
    records: list[dict[str, Any]] = []
    groups = ["family", "state_feature", "state_bin", "period"]
    for keys, group in work.groupby(groups, sort=False, observed=True):
        record = dict(zip(groups, keys, strict=True))
        record.update(newey_west_mean(group["spread_return"], hac_lags))
        records.append(record)
    return pd.DataFrame(records)


def _summarize_fama_macbeth(
    frame: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["period"] = work["year"].map(period_name)
    records: list[dict[str, Any]] = []
    groups = ["specification", "term", "period"]
    for keys, group in work.groupby(groups, sort=False, observed=True):
        record = dict(zip(groups, keys, strict=True))
        record.update(newey_west_mean(group["coefficient"], hac_lags))
        record["mean_daily_observations"] = float(
            group["observations"].mean()
        )
        record["median_condition_number"] = float(
            group["condition_number"].median()
        )
        records.append(record)
    return pd.DataFrame(records)


def _add_turnover_to_sort_summary(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    work = daily.copy()
    work["period"] = work["year"].map(period_name)
    turnover = (
        work.groupby(
            ["signal", "quantiles", "period"],
            sort=False,
            observed=True,
        )["one_way_turnover"]
        .mean()
        .rename("mean_one_way_turnover")
        .reset_index()
    )
    return summary.merge(
        turnover,
        on=["signal", "quantiles", "period"],
        how="left",
        validate="one_to_one",
    )


def _regime_summary(
    daily_ic: pd.DataFrame,
    daily_sorts_frame: pd.DataFrame,
    regimes: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    regime_columns = ["trend_regime", "volatility_regime"]
    for evidence, frame, value, signal_columns in [
        ("rank_ic", daily_ic, "rank_ic", ["signal"]),
        (
            "sort_spread",
            daily_sorts_frame.loc[
                daily_sorts_frame["quantiles"].eq(10)
            ],
            "spread_return",
            ["signal", "quantiles"],
        ),
    ]:
        joined = frame.merge(
            regimes[["date", *regime_columns]],
            on="date",
            how="left",
            validate="many_to_one",
        )
        joined["period"] = joined["year"].map(period_name)
        for regime_column in regime_columns:
            groups = [*signal_columns, "period", regime_column]
            for keys, group in joined.dropna(
                subset=[regime_column]
            ).groupby(groups, sort=False, observed=True):
                record = dict(zip(groups, keys, strict=True))
                record["evidence"] = evidence
                record["regime_dimension"] = regime_column
                record["regime"] = record.pop(regime_column)
                record.update(newey_west_mean(group[value], hac_lags))
                records.append(record)
    return pd.DataFrame(records)


def _prediction_evidence(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_map: dict[str, str] = {}
    for score in [
        "base_family_score",
        "ridge_score",
        "elastic_net_score",
    ]:
        rank_column = f"rank__{score}"
        frame[rank_column] = frame.groupby(
            "date", sort=False, observed=True
        )[score].rank(method="average", pct=True)
        rank_map[score] = rank_column
    ic = pd.concat(
        [
            daily_rank_ic(frame, {signal: rank_column})
            for signal, rank_column in rank_map.items()
        ],
        ignore_index=True,
    )
    quantiles = {signal: [10] for signal in rank_map}
    sorts = daily_sorts(
        frame,
        rank_map,
        quantiles,
        turnover_signals=set(rank_map),
    )
    return ic, sorts


def _prediction_frame(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    model_fit_id: str,
) -> pd.DataFrame:
    output = frame[
        [
            "permno",
            "permco",
            "date",
            "session_id",
            "label_end_session_id_10d",
            "fwd_total_return_10d",
            "fwd_total_return_rank_10d",
            "base_family_score",
        ]
    ].copy()
    output["ridge_score"] = predictions["ridge"]
    output["elastic_net_score"] = predictions["elastic_net"]
    output["model_fit_id"] = model_fit_id
    return output


def _file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        rows: int | None = None
        if path.suffix == ".parquet":
            rows = pq.ParquetFile(path).metadata.num_rows
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8") as stream:
                rows = max(sum(1 for _ in stream) - 1, 0)
        records.append(
            {
                "path": relative,
                "rows": rows,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _write_evidence_memo(
    path: Path,
    *,
    ic_summary: pd.DataFrame,
    sort_summary: pd.DataFrame,
    model_ic_summary: pd.DataFrame,
    selected: dict[str, Any],
    validation_rows: int,
    oos_rows: int,
    validation_base_score_rows: int,
    oos_base_score_rows: int,
) -> None:
    def _line(
        table: pd.DataFrame,
        mask: pd.Series,
        mean_column: str = "mean",
    ) -> str:
        selected_rows = table.loc[mask]
        if selected_rows.empty:
            return "not available"
        row = selected_rows.iloc[0]
        return (
            f"{float(row[mean_column]):.6f} "
            f"(NW t={float(row['newey_west_t']):.2f}, "
            f"n={int(row['observations'])})"
        )

    oos_models = model_ic_summary["period"].eq("strict_oos")
    model_lines = []
    for signal in [
        "base_family_score",
        "ridge_score",
        "elastic_net_score",
    ]:
        model_lines.append(
            f"- `{signal}` strict-OOS rank IC: "
            + _line(
                model_ic_summary,
                oos_models & model_ic_summary["signal"].eq(signal),
            )
        )
    best_univariate = ic_summary.loc[
        ic_summary["period"].eq("strict_oos")
    ].sort_values("mean", ascending=False, kind="stable").head(1)
    best_sort = sort_summary.loc[
        sort_summary["period"].eq("strict_oos")
        & sort_summary["quantiles"].eq(10)
    ].sort_values("mean", ascending=False, kind="stable").head(1)
    univariate_text = (
        "not available"
        if best_univariate.empty
        else (
            f"`{best_univariate.iloc[0]['signal']}` at "
            f"{best_univariate.iloc[0]['mean']:.6f}"
        )
    )
    sort_text = (
        "not available"
        if best_sort.empty
        else (
            f"`{best_sort.iloc[0]['signal']}` at "
            f"{best_sort.iloc[0]['mean']:.6f} per 10-session label"
        )
    )
    path.write_text(
        f"""# Gate 4 baseline evidence

Generated: {datetime.now().isoformat(timespec="seconds")}

## Governance

The feature list and hyperparameter grids were frozen before this run. Ridge
and Elastic Net penalties were selected on 2010–2012 only. The 2013–2025
strict OOS period was evaluated once by the frozen annual expanding-window
procedure. No tree model was fit.

## Coverage

- Full validation prediction rows: {validation_rows:,}.
- Full strict-OOS prediction rows: {oos_rows:,}.
- Nonmissing base-family score rows: {validation_base_score_rows:,}
  validation and {oos_base_score_rows:,} strict OOS.
- Optimizer sampling affects fitting only; all reported predictions and
  linear-model evaluation statistics use the full valid primary-universe
  rows. Base-family evidence uses its explicitly reported nonmissing rows.

## Compact results

- Best individual strict-OOS univariate rank IC: {univariate_text}.
- Best individual/family strict-OOS decile spread: {sort_text}.
{chr(10).join(model_lines)}

Selected penalties:

- Ridge alpha: {selected['ridge']['alpha']}.
- Elastic Net alpha: {selected['elastic_net']['alpha']}.
- Elastic Net l1 ratio: {selected['elastic_net']['l1_ratio']}.

## Interpretation boundary

Forward 10-session sort spreads overlap in time and are diagnostic, not a
standalone daily tradable backtest. Newey–West inference uses 20 lags.
Turnover is an equal-weight membership diagnostic. Cost stress tests,
portfolio accounting, exposure constraints, and break-even costs remain Gate
6 work.
""",
        encoding="utf-8",
    )


def build_gate4(project_root: Path) -> Path:
    """Build full Gate 4 evidence with validation-only tuning and annual OOS."""

    started = time.perf_counter()
    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    contract_path = config.root / "configs" / "models.yaml"
    contract = load_yaml(contract_path)
    if contract.get("status") != "frozen_before_gate4_outcomes":
        raise ValueError("models.yaml is not frozen for Gate 4")
    if contract["gate4_scope"]["prohibited"] != [
        "tree_model",
        "feature_revision_using_strict_oos",
        "repeated_strict_oos_tuning",
        "random_stock_date_split",
    ]:
        raise ValueError("Gate 4 prohibition contract changed")

    features_root = config.project_path("data_interim") / "features_v1"
    feature_panel_root = features_root / "panel"
    phase2b_root = config.project_path("data_interim") / "phase2b_v1"
    phase2b_panel_root = phase2b_root / "panel"
    if not feature_panel_root.is_dir() or not phase2b_panel_root.is_dir():
        raise FileNotFoundError("Gate 3 and Phase 2B artifacts are required")
    processed = policy.ensure_directory(config.project_path("data_processed"))
    final_root = policy.assert_writable(processed / "gate4_v1")
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite Gate 4 output: {final_root}")

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
            raise FileExistsError(f"Refusing to overwrite Gate 4 report: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_root = policy.assert_writable(
        processed / f"_gate4_v1_build_{timestamp}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)
    logger = configure_run_logger(
        "state_alpha.gate4.build",
        config.project_path("logs") / f"gate4_build_{timestamp}.log",
        policy,
    )
    logger.info("Gate 4 staging root: %s", staging_root)
    features_before = _snapshot(features_root)
    phase2b_before = _snapshot(phase2b_root)

    samples_root = staging_root / "optimizer_samples"
    evidence_root = staging_root / "evidence"
    predictions_root = staging_root / "predictions"
    diagnostics_root = staging_root / "diagnostics"
    model_states_root = staging_root / "model_states"
    for path in [
        samples_root,
        evidence_root,
        predictions_root,
        diagnostics_root,
        model_states_root,
    ]:
        path.mkdir()

    regimes = _market_regimes(
        features_root / "market_features.parquet", contract
    )
    _write_parquet(regimes, staging_root / "market_regimes.parquet")
    alpha_names, _, _ = alpha_contract(contract)
    individual_rank_map = {
        name: f"rank__{name}" for name in alpha_names
    }
    family_rank_map = {
        family: f"rank__{family}" for family in FAMILY_COLUMNS
    }
    rank_map = {**individual_rank_map, **family_rank_map}
    quantiles = {
        **{name: [10] for name in individual_rank_map},
        **{name: [5, 10] for name in family_rank_map},
    }
    hac_lags = int(contract["classical_baselines"]["hac_lags"])
    maximum_sample = int(
        contract["linear_models"]["optimizer_sample"][
            "maximum_rows_per_year"
        ]
    )
    annual_audit: list[dict[str, Any]] = []
    all_ic: list[pd.DataFrame] = []
    all_sorts: list[pd.DataFrame] = []
    all_double: list[pd.DataFrame] = []
    all_fmb: list[pd.DataFrame] = []

    for year in range(2001, 2026):
        logger.info("Preparing full Gate 4 classical evidence for %d", year)
        frame = load_modeling_year(
            feature_panel_root, phase2b_panel_root, year, contract
        )
        if frame.empty:
            raise ValueError(f"No valid primary Gate 4 rows in {year}")
        classical_frame = classical_sample(frame, contract)
        if classical_frame.empty:
            raise ValueError(
                f"No valid classical complete-case rows in {year}"
            )
        ic = daily_rank_ic(classical_frame, rank_map)
        sorts = daily_sorts(
            classical_frame,
            rank_map,
            quantiles,
            turnover_signals=set(FAMILY_COLUMNS),
        )
        double = daily_double_sorts(
            classical_frame,
            list(
                contract["classical_baselines"]["double_sort"][
                    "state_features"
                ]
            ),
        )
        fmb = daily_fama_macbeth(classical_frame, contract)
        sample = modeling_sample(frame, maximum_sample)
        partition = samples_root / f"year={year}"
        partition.mkdir()
        _write_parquet(sample, partition / "part-00000.parquet")
        all_ic.append(ic)
        all_sorts.append(sorts)
        all_double.append(double)
        all_fmb.append(fmb)
        annual_audit.append(
            {
                "year": year,
                "valid_primary_rows": len(frame),
                "classical_complete_case_rows": len(classical_frame),
                "classical_dates": int(
                    classical_frame["date"].nunique()
                ),
                "minimum_classical_names_per_date": int(
                    classical_frame.groupby(
                        "date", sort=False, observed=True
                    ).size().min()
                ),
                "optimizer_sample_rows": len(sample),
                "dates": int(frame["date"].nunique()),
                "minimum_session_id": int(frame["session_id"].min()),
                "maximum_session_id": int(frame["session_id"].max()),
                "maximum_label_end_session_id": int(
                    frame["label_end_session_id_10d"].max()
                ),
            }
        )
        logger.info(
            "Finished classical %d: primary=%d complete=%d sample=%d",
            year,
            len(frame),
            len(classical_frame),
            len(sample),
        )
        del frame, classical_frame, ic, sorts, double, fmb, sample
        gc.collect()

    daily_ic_frame = pd.concat(all_ic, ignore_index=True)
    daily_sort_frame = pd.concat(all_sorts, ignore_index=True)
    daily_double_frame = pd.concat(all_double, ignore_index=True)
    daily_fmb_frame = pd.concat(all_fmb, ignore_index=True)
    _write_parquet(daily_ic_frame, evidence_root / "daily_rank_ic.parquet")
    _write_parquet(
        daily_sort_frame, evidence_root / "daily_sort_spreads.parquet"
    )
    _write_parquet(
        daily_double_frame, evidence_root / "daily_double_sorts.parquet"
    )
    _write_parquet(
        daily_fmb_frame, evidence_root / "daily_fama_macbeth.parquet"
    )

    ic_summary = summarize_daily(
        daily_ic_frame,
        value_column="rank_ic",
        group_columns=["signal"],
        hac_lags=hac_lags,
        add_icir=True,
    )
    ic_by_year = summarize_by_year(
        daily_ic_frame,
        value_column="rank_ic",
        group_columns=["signal"],
        hac_lags=hac_lags,
    )
    sort_summary = summarize_daily(
        daily_sort_frame,
        value_column="spread_return",
        group_columns=["signal", "quantiles"],
        hac_lags=hac_lags,
    )
    sort_summary = _add_turnover_to_sort_summary(
        sort_summary, daily_sort_frame
    )
    sort_by_year = summarize_by_year(
        daily_sort_frame,
        value_column="spread_return",
        group_columns=["signal", "quantiles"],
        hac_lags=hac_lags,
    )
    double_summary = _summarize_double_sorts(
        daily_double_frame, hac_lags
    )
    fmb_summary = _summarize_fama_macbeth(daily_fmb_frame, hac_lags)

    calendar = pd.read_parquet(phase2b_root / "market_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    validation_start_session = int(
        calendar.loc[calendar["date"].dt.year.eq(2010), "session_id"].min()
    )
    training = _load_samples(samples_root, range(2001, 2010))
    training, validation_embargo_cutoff = _purge_and_embargo_training(
        training, validation_start_session, contract
    )
    validation_sample = _load_samples(samples_root, range(2010, 2013))
    logger.info(
        "Selecting linear penalties: train=%d validation_sample=%d",
        len(training),
        len(validation_sample),
    )
    selected, hyperparameter_results = select_hyperparameters(
        training, validation_sample, contract
    )
    logger.info("Selected Gate 4 penalties: %s", selected)

    prediction_ic: list[pd.DataFrame] = []
    prediction_sorts: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    refits: list[dict[str, Any]] = []
    validation_rows = 0
    oos_rows = 0
    validation_base_score_rows = 0
    oos_base_score_rows = 0

    validation_preprocessor, validation_models = fit_selected_models(
        training, selected, contract
    )
    validation_fit_id = "validation_train_2001_2009"
    validation_model_state_path = _write_model_state(
        model_states_root,
        state_id="validation",
        model_fit_id=validation_fit_id,
        preprocessor=validation_preprocessor,
        models=validation_models,
    )
    coefficients.append(
        coefficient_table(
            validation_models,
            validation_preprocessor,
            evaluation_year=2010,
            train_end_session_id=int(training["session_id"].max()),
        ).assign(model_fit_id=validation_fit_id)
    )
    refits.append(
        {
            "model_fit_id": validation_fit_id,
            "model_state_path": validation_model_state_path,
            "evaluation_start_year": 2010,
            "evaluation_end_year": 2012,
            "training_rows": len(training),
            "training_max_session_id": int(training["session_id"].max()),
            "training_max_label_end_session_id": int(
                training["label_end_session_id_10d"].max()
            ),
            "evaluation_start_session_id": validation_start_session,
            "embargo_sessions": int(
                contract["linear_models"]["expanding_oos"][
                    "embargo_trading_days"
                ]
            ),
            "embargo_cutoff_session_id": validation_embargo_cutoff,
            "purge_passed": bool(
                training["label_end_session_id_10d"].max()
                < validation_start_session
            ),
            "embargo_passed": bool(
                training["label_end_session_id_10d"].max()
                < validation_embargo_cutoff
            ),
            "strict_oos": False,
        }
    )
    for year in range(2010, 2013):
        frame = load_modeling_year(
            feature_panel_root, phase2b_panel_root, year, contract
        )
        predictions = predict_models(
            frame, validation_preprocessor, validation_models
        )
        output = _prediction_frame(
            frame,
            predictions,
            model_fit_id=validation_fit_id,
        )
        partition = predictions_root / f"year={year}"
        partition.mkdir()
        _write_parquet(output, partition / "part-00000.parquet")
        ic, sorts = _prediction_evidence(output)
        prediction_ic.append(ic)
        prediction_sorts.append(sorts)
        validation_rows += len(output)
        validation_base_score_rows += int(
            output["base_family_score"].notna().sum()
        )
        logger.info("Predicted full validation %d: rows=%d", year, len(output))
        del frame, predictions, output, ic, sorts
        gc.collect()
    del validation_models, validation_preprocessor

    for year in range(2013, 2026):
        evaluation_start_session = int(
            calendar.loc[
                calendar["date"].dt.year.eq(year), "session_id"
            ].min()
        )
        annual_training = _load_samples(
            samples_root, range(2001, year)
        )
        annual_training, embargo_cutoff = _purge_and_embargo_training(
            annual_training, evaluation_start_session, contract
        )
        preprocessor, models = fit_selected_models(
            annual_training, selected, contract
        )
        fit_id = f"strict_oos_train_through_{year - 1}"
        model_state_path = _write_model_state(
            model_states_root,
            state_id=f"oos_{year}",
            model_fit_id=fit_id,
            preprocessor=preprocessor,
            models=models,
        )
        coefficients.append(
            coefficient_table(
                models,
                preprocessor,
                evaluation_year=year,
                train_end_session_id=int(
                    annual_training["session_id"].max()
                ),
            ).assign(model_fit_id=fit_id)
        )
        refits.append(
            {
                "model_fit_id": fit_id,
                "model_state_path": model_state_path,
                "evaluation_start_year": year,
                "evaluation_end_year": year,
                "training_rows": len(annual_training),
                "training_max_session_id": int(
                    annual_training["session_id"].max()
                ),
                "training_max_label_end_session_id": int(
                    annual_training["label_end_session_id_10d"].max()
                ),
                "evaluation_start_session_id": evaluation_start_session,
                "embargo_sessions": int(
                    contract["linear_models"]["expanding_oos"][
                        "embargo_trading_days"
                    ]
                ),
                "embargo_cutoff_session_id": embargo_cutoff,
                "purge_passed": bool(
                    annual_training["label_end_session_id_10d"].max()
                    < evaluation_start_session
                ),
                "embargo_passed": bool(
                    annual_training["label_end_session_id_10d"].max()
                    < embargo_cutoff
                ),
                "strict_oos": True,
            }
        )
        frame = load_modeling_year(
            feature_panel_root, phase2b_panel_root, year, contract
        )
        predictions = predict_models(frame, preprocessor, models)
        output = _prediction_frame(
            frame, predictions, model_fit_id=fit_id
        )
        partition = predictions_root / f"year={year}"
        partition.mkdir()
        _write_parquet(output, partition / "part-00000.parquet")
        ic, sorts = _prediction_evidence(output)
        prediction_ic.append(ic)
        prediction_sorts.append(sorts)
        oos_rows += len(output)
        oos_base_score_rows += int(
            output["base_family_score"].notna().sum()
        )
        logger.info(
            "Predicted strict OOS %d: train=%d evaluation=%d",
            year,
            len(annual_training),
            len(output),
        )
        del (
            annual_training,
            preprocessor,
            models,
            frame,
            predictions,
            output,
            ic,
            sorts,
        )
        gc.collect()

    prediction_ic_frame = pd.concat(prediction_ic, ignore_index=True)
    prediction_sort_frame = pd.concat(prediction_sorts, ignore_index=True)
    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    refit_frame = pd.DataFrame(refits)
    _write_parquet(
        prediction_ic_frame,
        evidence_root / "daily_linear_model_rank_ic.parquet",
    )
    _write_parquet(
        prediction_sort_frame,
        evidence_root / "daily_linear_model_sorts.parquet",
    )

    model_ic_summary = summarize_daily(
        prediction_ic_frame,
        value_column="rank_ic",
        group_columns=["signal"],
        hac_lags=hac_lags,
        add_icir=True,
    )
    model_ic_by_year = summarize_by_year(
        prediction_ic_frame,
        value_column="rank_ic",
        group_columns=["signal"],
        hac_lags=hac_lags,
    )
    model_sort_summary = summarize_daily(
        prediction_sort_frame,
        value_column="spread_return",
        group_columns=["signal", "quantiles"],
        hac_lags=hac_lags,
    )
    model_sort_summary = _add_turnover_to_sort_summary(
        model_sort_summary, prediction_sort_frame
    )
    regime_summary = _regime_summary(
        prediction_ic_frame,
        prediction_sort_frame,
        regimes,
        hac_lags,
    )

    diagnostic_frames = {
        REPORT_TABLES[0]: ic_summary,
        REPORT_TABLES[1]: ic_by_year,
        REPORT_TABLES[2]: sort_summary,
        REPORT_TABLES[3]: sort_by_year,
        REPORT_TABLES[4]: double_summary,
        REPORT_TABLES[5]: fmb_summary,
        REPORT_TABLES[6]: hyperparameter_results,
        REPORT_TABLES[7]: model_ic_summary,
        REPORT_TABLES[8]: model_ic_by_year,
        REPORT_TABLES[9]: model_sort_summary,
        REPORT_TABLES[10]: regime_summary,
        REPORT_TABLES[11]: refit_frame,
        REPORT_TABLES[12]: coefficient_frame,
    }
    for name, frame in diagnostic_frames.items():
        frame.to_csv(diagnostics_root / name, index=False)
    memo_path = diagnostics_root / REPORT_MEMO
    _write_evidence_memo(
        memo_path,
        ic_summary=ic_summary,
        sort_summary=sort_summary,
        model_ic_summary=model_ic_summary,
        selected=selected,
        validation_rows=validation_rows,
        oos_rows=oos_rows,
        validation_base_score_rows=validation_base_score_rows,
        oos_base_score_rows=oos_base_score_rows,
    )

    features_after = _snapshot(features_root)
    phase2b_after = _snapshot(phase2b_root)
    if features_before != features_after:
        raise RuntimeError("Gate 3 input changed during Gate 4")
    if phase2b_before != phase2b_after:
        raise RuntimeError("Phase 2B input changed during Gate 4")
    output_files = _file_records(staging_root)
    manifest = {
        "schema_version": GATE4_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "project_root": config.root,
        "staging_root": staging_root,
        "final_root": final_root,
        "features_root": features_root,
        "phase2b_root": phase2b_root,
        "features_snapshot": features_before,
        "phase2b_snapshot": phase2b_before,
        "features_unchanged": True,
        "phase2b_unchanged": True,
        "models_config_sha256": _sha256(contract_path),
        "features_config_sha256": _sha256(
            config.root / "configs" / "features.yaml"
        ),
        "labels_config_sha256": _sha256(
            config.root / "configs" / "labels.yaml"
        ),
        "selected_hyperparameters": selected,
        "selection_period": "2010-2012",
        "strict_oos_period": "2013-2025",
        "strict_oos_used_for_selection": False,
        "tree_model_fit": False,
        "validation_prediction_rows": validation_rows,
        "strict_oos_prediction_rows": oos_rows,
        "validation_base_score_nonmissing_rows": (
            validation_base_score_rows
        ),
        "strict_oos_base_score_nonmissing_rows": oos_base_score_rows,
        "annual_audit": annual_audit,
        "model_refits": refits,
        "output_files": output_files,
        "overwrite_used": False,
    }
    _write_json(staging_root / "manifest.json", manifest)
    os.replace(staging_root, final_root)
    for name, target in zip(REPORT_TABLES, report_targets, strict=True):
        shutil.copy2(final_root / "diagnostics" / name, target)
    shutil.copy2(final_root / "diagnostics" / REPORT_MEMO, memo_target)
    logger.info(
        "Published Gate 4 to %s in %.1f seconds",
        final_root,
        time.perf_counter() - started,
    )
    return final_root
