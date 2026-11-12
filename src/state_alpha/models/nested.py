"""Frozen Gate 5 nested ablations and constrained histogram trees."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from state_alpha.features.core import CONTINUOUS_FEATURE_COLUMNS
from state_alpha.models.linear import _mean_daily_spearman
from state_alpha.models.matrix import FAMILY_COLUMNS, alpha_contract


CATEGORICAL_FEATURE_NAMES = [
    *[f"industry_{value:02d}" for value in range(100)],
    "industry_missing",
    "exchange_N",
    "exchange_Q",
    "exchange_A",
    "exchange_missing",
]


def validate_gate5_contract(
    contract: dict[str, Any],
    gate4_contract: dict[str, Any],
) -> None:
    """Fail closed if the frozen feature groups or governance drift."""

    if contract.get("status") != "frozen_after_gate4_before_gate5_fit":
        raise ValueError("Gate 5 contract is not frozen")
    governance = contract["governance"]
    required_true = [
        "gate4_strict_oos_already_observed",
        "feature_groups_derived_only_from_frozen_gate3_roles",
        "tree_grid_may_not_use_gate4_oos_results",
        "repeated_oos_tuning_prohibited",
        "random_stock_date_split_prohibited",
        "new_features_prohibited",
        "volume_as_primary_alpha_prohibited",
    ]
    if not all(governance.get(name) is True for name in required_true):
        raise ValueError("Gate 5 governance contract is incomplete")
    if governance.get("confirmatory_untouched_oos_claim_allowed") is not False:
        raise ValueError("Gate 5 cannot claim a new untouched OOS test")
    if governance.get("strict_oos_used_for_selection") is not False:
        raise ValueError("Gate 5 strict OOS cannot be used for selection")

    alpha_names, _, _ = alpha_contract(gate4_contract)
    groups = contract["feature_groups"]
    if groups["alpha_primitives"] != alpha_names:
        raise ValueError("Gate 5 alpha primitives drifted from Gate 4")
    if groups["alpha_family_composites"] != FAMILY_COLUMNS:
        raise ValueError("Gate 5 family composites drifted from Gate 4")
    states = list(groups["state_continuous"])
    expected_states = [
        name
        for name in CONTINUOUS_FEATURE_COLUMNS
        if name not in alpha_names
    ]
    if states != expected_states:
        raise ValueError("Gate 5 state group drifted from Gate 3 roles")
    all_continuous = [
        *groups["alpha_primitives"],
        *groups["state_continuous"],
        *groups["alpha_family_composites"],
    ]
    if set(all_continuous) != set(
        [*CONTINUOUS_FEATURE_COLUMNS, *FAMILY_COLUMNS]
    ):
        raise ValueError("Gate 5 continuous feature partition is incomplete")
    if len(all_continuous) != len(set(all_continuous)):
        raise ValueError("Gate 5 feature groups overlap")
    if any(
        token in name
        for name in groups["alpha_primitives"]
        for token in ["volume", "turnover", "adv", "amihud"]
    ):
        raise ValueError("Volume/liquidity variable entered the alpha group")
    candidates = contract["tree"]["candidate_grid"]
    if not 1 <= len(candidates) <= 4:
        raise ValueError("Gate 5 tree grid is not constrained")
    if len({item["candidate_id"] for item in candidates}) != len(candidates):
        raise ValueError("Gate 5 tree candidate ids are not unique")


@dataclass(frozen=True)
class FrozenGate4Preprocessor:
    """Read-only reconstruction of a Gate 4 training preprocessor."""

    model_fit_id: str
    continuous_columns: list[str]
    lower: np.ndarray
    upper: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "FrozenGate4Preprocessor":
        value = json.loads(path.read_text(encoding="utf-8"))
        columns = list(value["continuous_columns"])
        arrays = {
            name: np.asarray(value[name], dtype=float)
            for name in ["lower", "upper", "median", "mean", "scale"]
        }
        if any(len(array) != len(columns) for array in arrays.values()):
            raise ValueError("Gate 4 preprocessing state has wrong dimensions")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Gate 4 preprocessing state is not finite")
        if (arrays["scale"] <= 0).any():
            raise ValueError("Gate 4 preprocessing scale is not positive")
        return cls(
            model_fit_id=str(value["model_fit_id"]),
            continuous_columns=columns,
            **arrays,
        )

    @property
    def column_index(self) -> dict[str, int]:
        return {
            name: number
            for number, name in enumerate(self.continuous_columns)
        }

    def standardized_all(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.continuous_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        values = np.clip(values, self.lower, self.upper)
        values = np.where(np.isfinite(values), values, self.median)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def indices(self, columns: list[str]) -> list[int]:
        index = self.column_index
        missing = sorted(set(columns).difference(index))
        if missing:
            raise ValueError(
                f"Gate 5 columns missing from Gate 4 state: {missing}"
            )
        return [index[name] for name in columns]


def categorical_matrix(frame: pd.DataFrame) -> sparse.csr_matrix:
    industry = pd.to_numeric(
        frame["industry_code_2digit"], errors="coerce"
    ).to_numpy(dtype=float)
    industry_code = np.full(len(frame), 100, dtype=np.int64)
    valid = np.isfinite(industry) & (industry >= 0) & (industry <= 99)
    industry_code[valid] = industry[valid].astype(np.int64)
    exchange_code = (
        frame["exchange"]
        .map({"N": 0, "Q": 1, "A": 2})
        .fillna(3)
        .to_numpy(dtype=np.int64)
    )
    rows = np.arange(len(frame), dtype=np.int64)
    return sparse.csr_matrix(
        (
            np.ones(len(frame) * 2, dtype=np.float32),
            (
                np.concatenate([rows, rows]),
                np.concatenate([industry_code, 101 + exchange_code]),
            ),
        ),
        shape=(len(frame), 105),
    )


def group_columns(
    contract: dict[str, Any],
    group_names: list[str],
) -> list[str]:
    groups = contract["feature_groups"]
    output: list[str] = []
    for group in group_names:
        output.extend(groups[group])
    if len(output) != len(set(output)):
        raise ValueError("Requested Gate 5 groups overlap")
    return output


def ridge_variant_contract(
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, value in contract["linear_ablation"]["variants"].items():
        if "source" in value:
            continue
        columns = group_columns(
            contract, list(value["continuous_groups"])
        )
        include_categorical = bool(
            value["include_categorical_controls"]
        )
        output[name] = {
            "continuous_columns": columns,
            "include_categorical_controls": include_categorical,
            "feature_names": [
                *columns,
                *(
                    CATEGORICAL_FEATURE_NAMES
                    if include_categorical
                    else []
                ),
            ],
        }
    return output


def tree_variant_contract(
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "continuous_columns": group_columns(
                contract, list(value["continuous_groups"])
            )
        }
        for name, value in contract["tree"]["variants"].items()
    }


def deterministic_cap(
    frame: pd.DataFrame,
    maximum_rows: int,
) -> pd.DataFrame:
    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    if len(frame) <= maximum_rows:
        return frame.reset_index(drop=True)
    stride = int(np.ceil(len(frame) / maximum_rows))
    return frame.iloc[::stride].head(maximum_rows).reset_index(drop=True)


def _ridge_matrix(
    standardized: np.ndarray,
    preprocessor: FrozenGate4Preprocessor,
    columns: list[str],
    categories: sparse.csr_matrix | None,
) -> sparse.csr_matrix:
    numeric = sparse.csr_matrix(
        standardized[:, preprocessor.indices(columns)]
    )
    if categories is None:
        return numeric
    return sparse.hstack([numeric, categories], format="csr")


def fit_ridge_variants(
    training: pd.DataFrame,
    preprocessor: FrozenGate4Preprocessor,
    contract: dict[str, Any],
) -> dict[str, Ridge]:
    standardized = preprocessor.standardized_all(training)
    variants = ridge_variant_contract(contract)
    categories = categorical_matrix(training)
    target = pd.to_numeric(
        training["fwd_total_return_rank_10d"], errors="raise"
    ).to_numpy(dtype=float)
    models: dict[str, Ridge] = {}
    for name, value in variants.items():
        categorical = (
            categories
            if value["include_categorical_controls"]
            else None
        )
        matrix = _ridge_matrix(
            standardized,
            preprocessor,
            value["continuous_columns"],
            categorical,
        )
        model = Ridge(
            alpha=float(contract["linear_ablation"]["alpha"]),
            solver=str(contract["linear_ablation"]["solver"]),
            fit_intercept=True,
        )
        model.fit(matrix, target)
        models[name] = model
    return models


def _tree_parameters(
    contract: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    common = dict(contract["tree"]["common_parameters"])
    common.update(
        {
            key: value
            for key, value in candidate.items()
            if key != "candidate_id"
        }
    )
    return common


def select_tree_candidate(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    preprocessor: FrozenGate4Preprocessor,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    primary = str(
        contract["tree"]["hyperparameter_selection"]["fit_variant"]
    )
    columns = tree_variant_contract(contract)[primary][
        "continuous_columns"
    ]
    indices = preprocessor.indices(columns)
    x_train = preprocessor.standardized_all(training)[:, indices]
    x_validation = preprocessor.standardized_all(validation)[:, indices]
    y_train = pd.to_numeric(
        training["fwd_total_return_rank_10d"], errors="raise"
    ).to_numpy(dtype=float)
    records: list[dict[str, Any]] = []
    for order, candidate in enumerate(contract["tree"]["candidate_grid"]):
        model = HistGradientBoostingRegressor(
            **_tree_parameters(contract, candidate)
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_validation)
        metric, dates = _mean_daily_spearman(
            predictions,
            validation["fwd_total_return_rank_10d"],
            validation["date"],
        )
        records.append(
            {
                "candidate_order": order,
                **candidate,
                "fit_variant": primary,
                "validation_mean_daily_rank_ic": metric,
                "validation_dates": dates,
                "fitted_iterations": int(model.n_iter_),
                "strict_oos_used": False,
            }
        )
    results = pd.DataFrame(records)
    ranked = results.sort_values(
        [
            "validation_mean_daily_rank_ic",
            "max_leaf_nodes",
            "max_iter",
            "learning_rate",
            "candidate_order",
        ],
        ascending=[False, True, True, True, True],
        kind="stable",
    )
    if ranked.empty or not np.isfinite(
        float(ranked.iloc[0]["validation_mean_daily_rank_ic"])
    ):
        raise ValueError("No valid Gate 5 tree candidate")
    selected_id = str(ranked.iloc[0]["candidate_id"])
    selected = next(
        dict(candidate)
        for candidate in contract["tree"]["candidate_grid"]
        if candidate["candidate_id"] == selected_id
    )
    return selected, results


def fit_tree_variants(
    training: pd.DataFrame,
    preprocessor: FrozenGate4Preprocessor,
    selected_candidate: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, HistGradientBoostingRegressor]:
    standardized = preprocessor.standardized_all(training)
    target = pd.to_numeric(
        training["fwd_total_return_rank_10d"], errors="raise"
    ).to_numpy(dtype=float)
    models: dict[str, HistGradientBoostingRegressor] = {}
    for name, value in tree_variant_contract(contract).items():
        matrix = standardized[
            :, preprocessor.indices(value["continuous_columns"])
        ]
        model = HistGradientBoostingRegressor(
            **_tree_parameters(contract, selected_candidate)
        )
        model.fit(matrix, target)
        models[name] = model
    return models


def predict_nested_models(
    frame: pd.DataFrame,
    preprocessor: FrozenGate4Preprocessor,
    ridge_models: dict[str, Ridge],
    tree_models: dict[str, HistGradientBoostingRegressor],
    contract: dict[str, Any],
    *,
    chunk_size: int = 100_000,
) -> dict[str, np.ndarray]:
    output = {
        name: np.empty(len(frame), dtype=np.float32)
        for name in [*ridge_models, *tree_models]
    }
    ridge_contract = ridge_variant_contract(contract)
    tree_contract = tree_variant_contract(contract)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        chunk = frame.iloc[start:stop]
        standardized = preprocessor.standardized_all(chunk)
        categories = categorical_matrix(chunk)
        for name, model in ridge_models.items():
            value = ridge_contract[name]
            categorical = (
                categories
                if value["include_categorical_controls"]
                else None
            )
            matrix = _ridge_matrix(
                standardized,
                preprocessor,
                value["continuous_columns"],
                categorical,
            )
            output[name][start:stop] = model.predict(matrix).astype(
                np.float32
            )
        for name, model in tree_models.items():
            columns = tree_contract[name]["continuous_columns"]
            matrix = standardized[:, preprocessor.indices(columns)]
            output[name][start:stop] = model.predict(matrix).astype(
                np.float32
            )
    return output


def ridge_coefficient_table(
    models: dict[str, Ridge],
    contract: dict[str, Any],
    *,
    model_fit_id: str,
    evaluation_year: int,
) -> pd.DataFrame:
    variants = ridge_variant_contract(contract)
    records: list[dict[str, Any]] = []
    for name, model in models.items():
        names = variants[name]["feature_names"]
        values = np.asarray(model.coef_, dtype=float)
        if len(names) != len(values):
            raise ValueError("Gate 5 Ridge coefficient dimensions differ")
        for term, coefficient in zip(names, values, strict=True):
            records.append(
                {
                    "model_fit_id": model_fit_id,
                    "evaluation_year": evaluation_year,
                    "model": name,
                    "term": term,
                    "coefficient": coefficient,
                }
            )
        records.append(
            {
                "model_fit_id": model_fit_id,
                "evaluation_year": evaluation_year,
                "model": name,
                "term": "intercept",
                "coefficient": float(model.intercept_),
            }
        )
    return pd.DataFrame(records)


def add_capped_gates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add preregistered linear/tree capped-gate diagnostics."""

    output = frame.copy()
    grouped = output.groupby("date", sort=False, observed=True)
    base_rank = grouped["base_family_score"].rank(
        method="average", pct=True
    )
    output["base_exposure"] = base_rank - 0.5
    definitions = {
        "linear": (
            output["ridge_explicit_interactions"]
            - output["ridge_main_effects"]
        ),
        "tree": (
            output["tree_alpha_state"] - output["tree_alpha_only"]
        ),
    }
    for name, component in definitions.items():
        rank = component.groupby(output["date"], sort=False).rank(
            method="average", pct=True
        )
        centered = rank - 0.5
        aligned = np.sign(centered).eq(np.sign(output["base_exposure"]))
        strength = (
            2.0 * centered.abs()
        ).where(aligned, 0.0).clip(0.0, 1.0)
        strength = strength.where(output["base_exposure"].notna())
        output[f"{name}_gate_strength"] = strength.astype(np.float32)
        output[f"{name}_capped_gate"] = (
            output["base_exposure"] * strength
        ).astype(np.float32)
    return output


def daily_mechanism_attribution(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Daily high-conviction suppression accounting, not portfolio P&L."""

    work = frame.loc[
        frame["base_exposure"].abs().ge(0.40)
        & frame["fwd_total_return_10d"].notna()
    ].copy()
    records: list[dict[str, Any]] = []
    for date, date_frame in work.groupby(
        "date", sort=True, observed=True
    ):
        signed_payoff = (
            np.sign(date_frame["base_exposure"].to_numpy(dtype=float))
            * date_frame["fwd_total_return_10d"].to_numpy(dtype=float)
        )
        for name in ["linear", "tree"]:
            gate = date_frame[f"{name}_gate_strength"].to_numpy(
                dtype=float
            )
            valid = np.isfinite(signed_payoff) & np.isfinite(gate)
            if not valid.any():
                continue
            payoff = signed_payoff[valid]
            gate = gate[valid]
            removed = 1.0 - gate
            avoided = removed * np.maximum(-payoff, 0.0)
            sacrificed = removed * np.maximum(payoff, 0.0)
            gated = gate * payoff
            records.append(
                {
                    "date": date,
                    "year": pd.Timestamp(date).year,
                    "gate_model": f"{name}_capped_gate",
                    "observations": int(len(payoff)),
                    "suppressed_fraction": float((gate == 0).mean()),
                    "mean_gate_strength": float(gate.mean()),
                    "raw_signed_payoff": float(payoff.mean()),
                    "gated_signed_payoff": float(gated.mean()),
                    "adverse_state_loss_avoided": float(avoided.mean()),
                    "favorable_state_gain_sacrificed": float(
                        sacrificed.mean()
                    ),
                    "net_suppression_effect": float(
                        avoided.mean() - sacrificed.mean()
                    ),
                    "retained_signal_payoff": (
                        float(gated.sum() / gate.sum())
                        if gate.sum() > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)
