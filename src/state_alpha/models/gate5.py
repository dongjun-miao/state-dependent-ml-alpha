"""Atomic Gate 5 secondary locked nested-ablation and tree build."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from state_alpha.config import ProjectConfig, load_yaml
from state_alpha.logging_utils import configure_run_logger
from state_alpha.models.gate4 import (
    _add_turnover_to_sort_summary,
    _file_records,
    _load_samples,
    _market_regimes,
    _sha256,
    _snapshot,
    _write_json,
    _write_parquet,
)
from state_alpha.models.matrix import load_modeling_year, period_name
from state_alpha.models.nested import (
    FrozenGate4Preprocessor,
    add_capped_gates,
    daily_mechanism_attribution,
    deterministic_cap,
    fit_ridge_variants,
    fit_tree_variants,
    predict_nested_models,
    ridge_coefficient_table,
    ridge_variant_contract,
    select_tree_candidate,
    tree_variant_contract,
    validate_gate5_contract,
)
from state_alpha.models.statistics import (
    daily_rank_ic,
    daily_sorts,
    newey_west_mean,
    summarize_by_year,
    summarize_daily,
)
from state_alpha.paths import PathPolicy


GATE5_SCHEMA_VERSION = "state_alpha_gate5_secondary_locked_v1"
SCORE_COLUMNS = [
    "base_family_score",
    "ridge_alpha_only",
    "ridge_state_only",
    "ridge_main_effects",
    "ridge_explicit_interactions",
    "tree_alpha_only",
    "tree_state_only",
    "tree_alpha_state",
    "linear_capped_gate",
    "tree_capped_gate",
]
REPORT_TABLES = [
    "gate5_tree_model_selection.csv",
    "gate5_model_ic_summary.csv",
    "gate5_model_ic_by_year.csv",
    "gate5_model_sort_summary.csv",
    "gate5_paired_comparisons.csv",
    "gate5_mechanism_summary.csv",
    "gate5_mechanism_by_year.csv",
    "gate5_market_regime_summary.csv",
    "gate5_model_refits.csv",
    "gate5_ridge_coefficients.csv",
    "gate5_boundary_clean_2014_2025.csv",
]
REPORT_MEMO = "GATE5_SECONDARY_LOCKED_EVIDENCE.md"


def _purge_training(
    frame: pd.DataFrame,
    evaluation_start_session_id: int,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    embargo = int(contract["periods"]["embargo_trading_days"])
    cutoff = int(evaluation_start_session_id) - embargo
    output = frame.loc[
        frame["label_end_session_id_10d"].lt(cutoff)
    ].copy()
    if output.empty:
        raise ValueError("Gate 5 purge and embargo removed all training rows")
    return output, cutoff


def _gate4_preprocessor_path(
    gate4_root: Path,
    evaluation_year: int,
) -> Path:
    state_id = (
        "validation"
        if evaluation_year == 2010
        else f"oos_{evaluation_year}"
    )
    return gate4_root / "model_states" / state_id / "preprocessing.json"


def _load_gate4_prediction(
    gate4_root: Path,
    year: int,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        gate4_root
        / "predictions"
        / f"year={year}"
        / "part-00000.parquet"
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _assert_gate4_alignment(
    modeling_frame: pd.DataFrame,
    gate4_prediction: pd.DataFrame,
    year: int,
) -> None:
    columns = [
        "permno",
        "permco",
        "date",
        "session_id",
        "label_end_session_id_10d",
        "fwd_total_return_10d",
        "fwd_total_return_rank_10d",
        "base_family_score",
    ]
    try:
        pd.testing.assert_frame_equal(
            modeling_frame[columns].reset_index(drop=True),
            gate4_prediction[columns].reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(
            f"Gate 5 / Gate 4 prediction alignment failed for {year}"
        ) from exc


def _prediction_frame(
    modeling_frame: pd.DataFrame,
    gate4_prediction: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    model_fit_id: str,
) -> pd.DataFrame:
    output = modeling_frame[
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
    for name, values in predictions.items():
        output[name] = values
    output["ridge_explicit_interactions"] = pd.to_numeric(
        gate4_prediction["ridge_score"], errors="raise"
    ).to_numpy(dtype=np.float32)
    output = add_capped_gates(output)
    output["model_fit_id"] = model_fit_id
    missing = sorted(set(SCORE_COLUMNS).difference(output.columns))
    if missing:
        raise ValueError(f"Gate 5 prediction scores missing: {missing}")
    return output


def _score_evidence(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_map: dict[str, str] = {}
    for score in SCORE_COLUMNS:
        rank_column = f"rank__{score}"
        frame[rank_column] = frame.groupby(
            "date", sort=False, observed=True
        )[score].rank(method="average", pct=True)
        rank_map[score] = rank_column
    ic = pd.concat(
        [
            daily_rank_ic(frame, {score: rank_column})
            for score, rank_column in rank_map.items()
        ],
        ignore_index=True,
    )
    sorts = daily_sorts(
        frame,
        rank_map,
        {score: [10] for score in SCORE_COLUMNS},
        turnover_signals=set(SCORE_COLUMNS),
    )
    return ic, sorts


def _paired_comparisons(
    daily_ic: pd.DataFrame,
    daily_sorts_frame: pd.DataFrame,
    contract: dict[str, Any],
    hac_lags: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for evidence, frame, value in [
        ("rank_ic", daily_ic, "rank_ic"),
        (
            "decile_spread",
            daily_sorts_frame.loc[
                daily_sorts_frame["quantiles"].eq(10)
            ],
            "spread_return",
        ),
    ]:
        for challenger, benchmark in contract["evaluation"][
            "paired_comparisons"
        ]:
            left = frame.loc[
                frame["signal"].eq(challenger), ["date", "year", value]
            ].rename(columns={value: "challenger_value"})
            right = frame.loc[
                frame["signal"].eq(benchmark), ["date", value]
            ].rename(columns={value: "benchmark_value"})
            paired = left.merge(
                right,
                on="date",
                how="inner",
                validate="one_to_one",
            )
            paired["difference"] = (
                paired["challenger_value"] - paired["benchmark_value"]
            )
            paired["period"] = paired["year"].map(period_name)
            for period, group in paired.groupby(
                "period", sort=False, observed=True
            ):
                result = {
                    "evidence": evidence,
                    "challenger": challenger,
                    "benchmark": benchmark,
                    "period": period,
                    "challenger_mean": float(
                        group["challenger_value"].mean()
                    ),
                    "benchmark_mean": float(
                        group["benchmark_value"].mean()
                    ),
                }
                result.update(
                    newey_west_mean(group["difference"], hac_lags)
                )
                records.append(result)
    return pd.DataFrame(records)


MECHANISM_METRICS = [
    "suppressed_fraction",
    "mean_gate_strength",
    "raw_signed_payoff",
    "gated_signed_payoff",
    "adverse_state_loss_avoided",
    "favorable_state_gain_sacrificed",
    "net_suppression_effect",
    "retained_signal_payoff",
]


def _summarize_mechanism(
    daily: pd.DataFrame,
    *,
    by_year: bool,
    hac_lags: int,
) -> pd.DataFrame:
    work = daily.copy()
    work["period"] = work["year"].map(period_name)
    groups = ["gate_model", "year"] if by_year else [
        "gate_model",
        "period",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in work.groupby(groups, sort=False, observed=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(groups, key_values, strict=True))
        for metric in MECHANISM_METRICS:
            record = {**base, "metric": metric}
            record.update(newey_west_mean(group[metric], hac_lags))
            record["mean_daily_observations"] = float(
                group["observations"].mean()
            )
            records.append(record)
    return pd.DataFrame(records)


def _regime_summary(
    daily_ic: pd.DataFrame,
    daily_sorts_frame: pd.DataFrame,
    regimes: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    regime_columns = ["trend_regime", "volatility_regime"]
    for evidence, frame, value in [
        ("rank_ic", daily_ic, "rank_ic"),
        (
            "decile_spread",
            daily_sorts_frame.loc[
                daily_sorts_frame["quantiles"].eq(10)
            ],
            "spread_return",
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
            groups = ["signal", "period", regime_column]
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


def _boundary_clean_sensitivity(
    daily_ic: pd.DataFrame,
    daily_sorts_frame: pd.DataFrame,
    daily_mechanism: pd.DataFrame,
    contract: dict[str, Any],
    hac_lags: int,
) -> pd.DataFrame:
    """Summarize 2014-2025 separately after the Gate 4 boundary audit."""

    records: list[dict[str, Any]] = []
    sample_name = "boundary_clean_2014_2025"
    ic = daily_ic.loc[daily_ic["year"].between(2014, 2025)]
    for signal, group in ic.groupby("signal", sort=False, observed=True):
        record = {
            "sample": sample_name,
            "evidence": "rank_ic",
            "signal": signal,
            "benchmark": "",
            "metric": "rank_ic",
        }
        record.update(newey_west_mean(group["rank_ic"], hac_lags))
        records.append(record)
    sorts = daily_sorts_frame.loc[
        daily_sorts_frame["year"].between(2014, 2025)
        & daily_sorts_frame["quantiles"].eq(10)
    ]
    for signal, group in sorts.groupby(
        "signal", sort=False, observed=True
    ):
        record = {
            "sample": sample_name,
            "evidence": "decile_spread",
            "signal": signal,
            "benchmark": "",
            "metric": "spread_return",
        }
        record.update(
            newey_west_mean(group["spread_return"], hac_lags)
        )
        records.append(record)
    for evidence, frame, value in [
        ("rank_ic_difference", ic, "rank_ic"),
        ("decile_spread_difference", sorts, "spread_return"),
    ]:
        for challenger, benchmark in contract["evaluation"][
            "paired_comparisons"
        ]:
            left = frame.loc[
                frame["signal"].eq(challenger), ["date", value]
            ].rename(columns={value: "challenger"})
            right = frame.loc[
                frame["signal"].eq(benchmark), ["date", value]
            ].rename(columns={value: "benchmark_value"})
            paired = left.merge(
                right,
                on="date",
                how="inner",
                validate="one_to_one",
            )
            difference = paired["challenger"] - paired["benchmark_value"]
            record = {
                "sample": sample_name,
                "evidence": evidence,
                "signal": challenger,
                "benchmark": benchmark,
                "metric": "difference",
            }
            record.update(newey_west_mean(difference, hac_lags))
            records.append(record)
    mechanism = daily_mechanism.loc[
        daily_mechanism["year"].between(2014, 2025)
    ]
    for gate_model, group in mechanism.groupby(
        "gate_model", sort=False, observed=True
    ):
        for metric in MECHANISM_METRICS:
            record = {
                "sample": sample_name,
                "evidence": "mechanism",
                "signal": gate_model,
                "benchmark": "",
                "metric": metric,
            }
            record.update(newey_west_mean(group[metric], hac_lags))
            records.append(record)
    return pd.DataFrame(records)


def _write_model_state(
    state_root: Path,
    *,
    model_fit_id: str,
    evaluation_year: int,
    preprocessor_path: Path,
    ridge_models: dict[str, Any],
    tree_models: dict[str, Any],
    selected_candidate: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    state_root.mkdir(parents=False, exist_ok=False)
    ridge_contract = ridge_variant_contract(contract)
    ridge_state = {
        name: {
            "feature_names": ridge_contract[name]["feature_names"],
            "parameters": model.get_params(deep=False),
            "intercept": float(model.intercept_),
            "coefficients": np.asarray(
                model.coef_, dtype=float
            ).tolist(),
        }
        for name, model in ridge_models.items()
    }
    _write_json(state_root / "ridge_models.json", ridge_state)
    _write_json(
        state_root / "metadata.json",
        {
            "model_fit_id": model_fit_id,
            "evaluation_year": evaluation_year,
            "source_gate4_preprocessor": preprocessor_path,
            "source_gate4_preprocessor_sha256": _sha256(
                preprocessor_path
            ),
            "selected_tree_candidate": selected_candidate,
            "tree_features": tree_variant_contract(contract),
            "sklearn_serialization": "joblib",
        },
    )
    joblib.dump(
        tree_models,
        state_root / "tree_models.joblib",
        compress=3,
    )


def _write_evidence_memo(
    path: Path,
    *,
    ic_summary: pd.DataFrame,
    sort_summary: pd.DataFrame,
    paired: pd.DataFrame,
    mechanism: pd.DataFrame,
    selected_candidate: dict[str, Any],
    validation_rows: int,
    oos_rows: int,
    boundary_clean: pd.DataFrame,
) -> None:
    def _metric(
        table: pd.DataFrame,
        *,
        signal: str,
        period: str = "strict_oos",
    ) -> str:
        row = table.loc[
            table["signal"].eq(signal) & table["period"].eq(period)
        ]
        if row.empty:
            return "not available"
        value = row.iloc[0]
        return (
            f"{float(value['mean']):.6f} "
            f"(NW t={float(value['newey_west_t']):.2f}, "
            f"n={int(value['observations'])})"
        )

    interaction_delta = paired.loc[
        paired["evidence"].eq("rank_ic")
        & paired["period"].eq("strict_oos")
        & paired["challenger"].eq("ridge_explicit_interactions")
        & paired["benchmark"].eq("ridge_main_effects")
    ]
    tree_delta = paired.loc[
        paired["evidence"].eq("rank_ic")
        & paired["period"].eq("strict_oos")
        & paired["challenger"].eq("tree_alpha_state")
        & paired["benchmark"].eq("tree_alpha_only")
    ]

    def _delta_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "not available"
        row = frame.iloc[0]
        return (
            f"{float(row['mean']):.6f} "
            f"(NW t={float(row['newey_west_t']):.2f})"
        )

    mechanism_oos = mechanism.loc[
        mechanism["period"].eq("strict_oos")
        & mechanism["metric"].eq("net_suppression_effect")
    ]
    mechanism_lines = [
        (
            f"- `{row.gate_model}` net suppression effect: "
            f"{float(row.mean):.6f} "
            f"(NW t={float(row.newey_west_t):.2f})."
        )
        for row in mechanism_oos.itertuples()
    ]
    boundary_ic = boundary_clean.loc[
        boundary_clean["evidence"].eq("rank_ic")
    ]

    def _boundary_metric(signal: str) -> str:
        row = boundary_ic.loc[boundary_ic["signal"].eq(signal)]
        if row.empty:
            return "not available"
        value = row.iloc[0]
        return (
            f"{float(value['mean']):.6f} "
            f"(NW t={float(value['newey_west_t']):.2f})"
        )

    path.write_text(
        f"""# Gate 5 secondary locked evidence

Generated: {datetime.now().isoformat(timespec="seconds")}

## Governance

Gate 4 strict-OOS outcomes were already observed before the exact Gate 5 tree
grid was frozen. Gate 5 is therefore a secondary locked analysis, not a new
untouched confirmatory test. The tree grid and feature groups were frozen
before any Gate 5 model fit. Selection used 2010–2012 only, after removing
validation labels that crossed the 2013 boundary plus a 20-session embargo.

The pre-fit audit found that the Gate 4 validation selector had retained 1,065
labels reaching 2013. Gate 4's 2013 source score is therefore a transition
sensitivity. Gate 5 reports both 2013–2025 secondary OOS and a boundary-clean
2014–2025 sensitivity.

## Coverage

- Full validation prediction rows: {validation_rows:,}.
- Full strict-OOS prediction rows: {oos_rows:,}.
- All annual evaluation uses full valid primary rows.

## Selected constrained tree

`{selected_candidate['candidate_id']}` with
{selected_candidate['max_leaf_nodes']} leaves,
{selected_candidate['max_iter']} iterations, and learning rate
{selected_candidate['learning_rate']}.

## Strict-OOS rank IC

- Ridge alpha-only: {_metric(ic_summary, signal='ridge_alpha_only')}.
- Ridge state-only: {_metric(ic_summary, signal='ridge_state_only')}.
- Ridge main effects: {_metric(ic_summary, signal='ridge_main_effects')}.
- Ridge explicit interactions: {_metric(ic_summary, signal='ridge_explicit_interactions')}.
- Tree alpha-only: {_metric(ic_summary, signal='tree_alpha_only')}.
- Tree state-only: {_metric(ic_summary, signal='tree_state_only')}.
- Tree alpha+state: {_metric(ic_summary, signal='tree_alpha_state')}.

Boundary-clean 2014–2025 rank IC:

- Ridge explicit interactions: {_boundary_metric('ridge_explicit_interactions')}.
- Tree alpha+state: {_boundary_metric('tree_alpha_state')}.

Paired daily rank-IC increments:

- Ridge interactions minus main effects: {_delta_text(interaction_delta)}.
- Tree alpha+state minus alpha-only: {_delta_text(tree_delta)}.

## Suppression diagnostic

{chr(10).join(mechanism_lines)}

The capped gates can reduce but never amplify the unconditional base exposure.
This attribution uses high-conviction observations and is not self-financing
portfolio P&L.

## Interpretation boundary

Decile spreads use overlapping 10-session labels. They are forecast
diagnostics, not annualizable daily returns. Portfolio construction, exposure
neutralization, staggered-cohort accounting, cost stress, break-even costs,
drawdown, and capacity remain Gate 6 work.
""",
        encoding="utf-8",
    )


def build_gate5(project_root: Path) -> Path:
    """Build the frozen Gate 5 secondary locked analysis atomically."""

    started = time.perf_counter()
    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    gate4_contract_path = config.root / "configs" / "models.yaml"
    gate5_contract_path = config.root / "configs" / "gate5.yaml"
    gate4_contract = load_yaml(gate4_contract_path)
    contract = load_yaml(gate5_contract_path)
    validate_gate5_contract(contract, gate4_contract)

    gate4_root = config.project_path("data_processed") / "gate4_v1"
    if not gate4_root.is_dir():
        raise FileNotFoundError("Gate 4 artifact is required for Gate 5")
    gate4_manifest = json.loads(
        (gate4_root / "manifest.json").read_text(encoding="utf-8")
    )
    if gate4_manifest.get("schema_version") != contract["governance"][
        "source_gate4_schema"
    ]:
        raise ValueError("Gate 4 schema differs from Gate 5 contract")
    current_models_hash = _sha256(gate4_contract_path)
    if current_models_hash != contract["governance"][
        "source_models_config_sha256"
    ]:
        raise ValueError("Gate 4 model contract changed before Gate 5")
    validation_report = json.loads(
        (
            config.project_path("reports_tables")
            / "gate4_validation.json"
        ).read_text(encoding="utf-8")
    )
    if validation_report.get("passed") is not True:
        raise ValueError("Gate 4 independent validation did not pass")

    features_root = Path(gate4_manifest["features_root"])
    phase2b_root = Path(gate4_manifest["phase2b_root"])
    feature_panel_root = features_root / "panel"
    phase2b_panel_root = phase2b_root / "panel"
    processed = policy.ensure_directory(config.project_path("data_processed"))
    final_root = policy.assert_writable(processed / "gate5_v1")
    if final_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Gate 5 output: {final_root}"
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
                f"Refusing to overwrite Gate 5 report: {path}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_root = policy.assert_writable(
        processed / f"_gate5_v1_build_{timestamp}"
    )
    staging_root.mkdir(parents=False, exist_ok=False)
    logger = configure_run_logger(
        "state_alpha.gate5.build",
        config.project_path("logs") / f"gate5_build_{timestamp}.log",
        policy,
    )
    logger.info("Gate 5 staging root: %s", staging_root)
    gate4_before = _snapshot(gate4_root)
    features_before = _snapshot(features_root)
    phase2b_before = _snapshot(phase2b_root)

    evidence_root = staging_root / "evidence"
    predictions_root = staging_root / "predictions"
    model_states_root = staging_root / "model_states"
    diagnostics_root = staging_root / "diagnostics"
    for path in [
        evidence_root,
        predictions_root,
        model_states_root,
        diagnostics_root,
    ]:
        path.mkdir()

    sample_root = gate4_root / "optimizer_samples"
    calendar = pd.read_parquet(phase2b_root / "market_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise")
    validation_start_session = int(
        calendar.loc[calendar["date"].dt.year.eq(2010), "session_id"].min()
    )
    validation_training = _load_samples(
        sample_root, range(2001, 2010)
    )
    validation_training, validation_cutoff = _purge_training(
        validation_training, validation_start_session, contract
    )
    validation_sample = _load_samples(sample_root, range(2010, 2013))
    strict_oos_start_session = int(
        calendar.loc[calendar["date"].dt.year.eq(2013), "session_id"].min()
    )
    selection_embargo_cutoff = (
        strict_oos_start_session
        - int(contract["periods"]["embargo_trading_days"])
    )
    validation_selection_sample = validation_sample.loc[
        validation_sample["label_end_session_id_10d"].lt(
            selection_embargo_cutoff
        )
    ].copy()
    if validation_selection_sample.empty:
        raise ValueError(
            "Gate 5 validation-to-OOS purge removed all selection rows"
        )
    maximum_tree_rows = int(
        contract["tree"]["optimizer_sample"][
            "maximum_expanding_training_rows"
        ]
    )
    tree_selection_training = deterministic_cap(
        validation_training, maximum_tree_rows
    )
    validation_preprocessor_path = _gate4_preprocessor_path(
        gate4_root, 2010
    )
    validation_preprocessor = FrozenGate4Preprocessor.load(
        validation_preprocessor_path
    )
    selected_tree, selection_results = select_tree_candidate(
        tree_selection_training,
        validation_selection_sample,
        validation_preprocessor,
        contract,
    )
    logger.info("Selected Gate 5 tree candidate: %s", selected_tree)

    all_ic: list[pd.DataFrame] = []
    all_sorts: list[pd.DataFrame] = []
    all_mechanism: list[pd.DataFrame] = []
    all_coefficients: list[pd.DataFrame] = []
    refits: list[dict[str, Any]] = []
    validation_rows = 0
    oos_rows = 0

    for evaluation_year in [2010, *range(2013, 2026)]:
        strict_oos = evaluation_year >= 2013
        evaluation_start = (
            validation_start_session
            if not strict_oos
            else int(
                calendar.loc[
                    calendar["date"].dt.year.eq(evaluation_year),
                    "session_id",
                ].min()
            )
        )
        if strict_oos:
            ridge_training = _load_samples(
                sample_root, range(2001, evaluation_year)
            )
            ridge_training, embargo_cutoff = _purge_training(
                ridge_training, evaluation_start, contract
            )
            evaluation_years = [evaluation_year]
            model_fit_id = (
                f"secondary_locked_train_through_{evaluation_year - 1}"
            )
            state_id = f"oos_{evaluation_year}"
        else:
            ridge_training = validation_training
            embargo_cutoff = validation_cutoff
            evaluation_years = [2010, 2011, 2012]
            model_fit_id = "validation_train_2001_2009"
            state_id = "validation"
        tree_training = deterministic_cap(
            ridge_training, maximum_tree_rows
        )
        preprocessor_path = _gate4_preprocessor_path(
            gate4_root, evaluation_year
        )
        preprocessor = FrozenGate4Preprocessor.load(preprocessor_path)
        if preprocessor.model_fit_id != (
            "validation_train_2001_2009"
            if not strict_oos
            else f"strict_oos_train_through_{evaluation_year - 1}"
        ):
            raise ValueError("Gate 4 preprocessing identity mismatch")
        ridge_models = fit_ridge_variants(
            ridge_training, preprocessor, contract
        )
        tree_models = fit_tree_variants(
            tree_training, preprocessor, selected_tree, contract
        )
        _write_model_state(
            model_states_root / state_id,
            model_fit_id=model_fit_id,
            evaluation_year=evaluation_year,
            preprocessor_path=preprocessor_path,
            ridge_models=ridge_models,
            tree_models=tree_models,
            selected_candidate=selected_tree,
            contract=contract,
        )
        all_coefficients.append(
            ridge_coefficient_table(
                ridge_models,
                contract,
                model_fit_id=model_fit_id,
                evaluation_year=evaluation_year,
            )
        )
        refits.append(
            {
                "model_fit_id": model_fit_id,
                "model_state_path": f"model_states/{state_id}",
                "evaluation_start_year": evaluation_years[0],
                "evaluation_end_year": evaluation_years[-1],
                "ridge_training_rows": len(ridge_training),
                "tree_training_rows": len(tree_training),
                "training_max_session_id": int(
                    ridge_training["session_id"].max()
                ),
                "training_max_label_end_session_id": int(
                    ridge_training["label_end_session_id_10d"].max()
                ),
                "evaluation_start_session_id": evaluation_start,
                "embargo_sessions": int(
                    contract["periods"]["embargo_trading_days"]
                ),
                "embargo_cutoff_session_id": embargo_cutoff,
                "purge_passed": bool(
                    ridge_training["label_end_session_id_10d"].max()
                    < evaluation_start
                ),
                "embargo_passed": bool(
                    ridge_training["label_end_session_id_10d"].max()
                    < embargo_cutoff
                ),
                "strict_oos": strict_oos,
            }
        )
        for year in evaluation_years:
            frame = load_modeling_year(
                feature_panel_root,
                phase2b_panel_root,
                year,
                gate4_contract,
            )
            gate4_prediction = _load_gate4_prediction(gate4_root, year)
            _assert_gate4_alignment(frame, gate4_prediction, year)
            predictions = predict_nested_models(
                frame,
                preprocessor,
                ridge_models,
                tree_models,
                contract,
            )
            output = _prediction_frame(
                frame,
                gate4_prediction,
                predictions,
                model_fit_id=model_fit_id,
            )
            partition = predictions_root / f"year={year}"
            partition.mkdir()
            _write_parquet(
                output,
                partition / "part-00000.parquet",
                schema_version=GATE5_SCHEMA_VERSION,
            )
            ic, sorts = _score_evidence(output)
            mechanism = daily_mechanism_attribution(output)
            all_ic.append(ic)
            all_sorts.append(sorts)
            all_mechanism.append(mechanism)
            if year <= 2012:
                validation_rows += len(output)
            else:
                oos_rows += len(output)
            logger.info(
                "Gate 5 predicted %d: ridge_train=%d tree_train=%d rows=%d",
                year,
                len(ridge_training),
                len(tree_training),
                len(output),
            )
            del (
                frame,
                gate4_prediction,
                predictions,
                output,
                ic,
                sorts,
                mechanism,
            )
            gc.collect()
        del (
            ridge_training,
            tree_training,
            preprocessor,
            ridge_models,
            tree_models,
        )
        gc.collect()

    daily_ic_frame = pd.concat(all_ic, ignore_index=True)
    daily_sort_frame = pd.concat(all_sorts, ignore_index=True)
    daily_mechanism_frame = pd.concat(all_mechanism, ignore_index=True)
    coefficient_frame = pd.concat(all_coefficients, ignore_index=True)
    refit_frame = pd.DataFrame(refits)
    _write_parquet(
        daily_ic_frame,
        evidence_root / "daily_model_rank_ic.parquet",
        schema_version=GATE5_SCHEMA_VERSION,
    )
    _write_parquet(
        daily_sort_frame,
        evidence_root / "daily_model_sorts.parquet",
        schema_version=GATE5_SCHEMA_VERSION,
    )
    _write_parquet(
        daily_mechanism_frame,
        evidence_root / "daily_mechanism_attribution.parquet",
        schema_version=GATE5_SCHEMA_VERSION,
    )

    hac_lags = 20
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
    paired = _paired_comparisons(
        daily_ic_frame, daily_sort_frame, contract, hac_lags
    )
    mechanism_summary = _summarize_mechanism(
        daily_mechanism_frame, by_year=False, hac_lags=hac_lags
    )
    mechanism_by_year = _summarize_mechanism(
        daily_mechanism_frame, by_year=True, hac_lags=hac_lags
    )
    boundary_clean = _boundary_clean_sensitivity(
        daily_ic_frame,
        daily_sort_frame,
        daily_mechanism_frame,
        contract,
        hac_lags,
    )
    regimes = _market_regimes(
        features_root / "market_features.parquet",
        gate4_contract,
    )
    regime_summary = _regime_summary(
        daily_ic_frame, daily_sort_frame, regimes, hac_lags
    )

    diagnostic_frames = {
        REPORT_TABLES[0]: selection_results,
        REPORT_TABLES[1]: ic_summary,
        REPORT_TABLES[2]: ic_by_year,
        REPORT_TABLES[3]: sort_summary,
        REPORT_TABLES[4]: paired,
        REPORT_TABLES[5]: mechanism_summary,
        REPORT_TABLES[6]: mechanism_by_year,
        REPORT_TABLES[7]: regime_summary,
        REPORT_TABLES[8]: refit_frame,
        REPORT_TABLES[9]: coefficient_frame,
        REPORT_TABLES[10]: boundary_clean,
    }
    for name, frame in diagnostic_frames.items():
        frame.to_csv(diagnostics_root / name, index=False)
    memo_path = diagnostics_root / REPORT_MEMO
    _write_evidence_memo(
        memo_path,
        ic_summary=ic_summary,
        sort_summary=sort_summary,
        paired=paired,
        mechanism=mechanism_summary,
        selected_candidate=selected_tree,
        validation_rows=validation_rows,
        oos_rows=oos_rows,
        boundary_clean=boundary_clean,
    )

    gate4_after = _snapshot(gate4_root)
    features_after = _snapshot(features_root)
    phase2b_after = _snapshot(phase2b_root)
    if gate4_before != gate4_after:
        raise RuntimeError("Gate 4 input changed during Gate 5")
    if features_before != features_after:
        raise RuntimeError("Gate 3 input changed during Gate 5")
    if phase2b_before != phase2b_after:
        raise RuntimeError("Phase 2B input changed during Gate 5")

    output_files = _file_records(staging_root)
    manifest = {
        "schema_version": GATE5_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "project_root": config.root,
        "staging_root": staging_root,
        "final_root": final_root,
        "gate4_root": gate4_root,
        "features_root": features_root,
        "phase2b_root": phase2b_root,
        "gate4_snapshot": gate4_before,
        "features_snapshot": features_before,
        "phase2b_snapshot": phase2b_before,
        "gate4_unchanged": True,
        "features_unchanged": True,
        "phase2b_unchanged": True,
        "gate5_config_sha256": _sha256(gate5_contract_path),
        "models_config_sha256": current_models_hash,
        "source_gate4_manifest_sha256": _sha256(
            gate4_root / "manifest.json"
        ),
        "analysis_classification": (
            "secondary_locked_post_gate4"
        ),
        "gate4_strict_oos_already_observed": True,
        "confirmatory_untouched_oos_claim_allowed": False,
        "selection_period": "2010-2012",
        "strict_oos_period": "2013-2025",
        "strict_oos_used_for_selection": False,
        "tree_backend": contract["tree"]["backend"],
        "selected_tree_candidate": selected_tree,
        "tree_selection_variant": contract["tree"][
            "hyperparameter_selection"
        ]["fit_variant"],
        "tree_selection_training_rows": len(tree_selection_training),
        "tree_selection_validation_rows": len(
            validation_selection_sample
        ),
        "tree_selection_validation_max_label_end_session_id": int(
            validation_selection_sample[
                "label_end_session_id_10d"
            ].max()
        ),
        "tree_selection_oos_start_session_id": (
            strict_oos_start_session
        ),
        "tree_selection_embargo_cutoff_session_id": (
            selection_embargo_cutoff
        ),
        "tree_selection_boundary_passed": bool(
            validation_selection_sample[
                "label_end_session_id_10d"
            ].max()
            < selection_embargo_cutoff
        ),
        "gate4_2013_classification": "transition_sensitivity",
        "boundary_clean_period": "2014-2025",
        "validation_prediction_rows": validation_rows,
        "strict_oos_prediction_rows": oos_rows,
        "scores": SCORE_COLUMNS,
        "model_refits": refits,
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
        "Published Gate 5 to %s in %.1f seconds",
        final_root,
        time.perf_counter() - started,
    )
    return final_root
