"""Point-in-time Gate 6 rescoring from frozen Gate 4 and Gate 5 states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from state_alpha.features.core import OUTPUT_IDENTIFIER_COLUMNS
from state_alpha.models.gate5 import SCORE_COLUMNS
from state_alpha.models.linear import Gate4Preprocessor
from state_alpha.models.matrix import FAMILY_COLUMNS, alpha_contract
from state_alpha.models.nested import (
    FrozenGate4Preprocessor,
    add_capped_gates,
    predict_nested_models,
    ridge_variant_contract,
)


FORMATION_POINT_IN_TIME_COLUMNS = [
    "execution_valid",
    "execution_session_id",
    "lag_price_usd",
    "lag_market_cap_usd",
    "lag_adv20_usd",
]


class FrozenLinearPredictor:
    """Minimal read-only predictor for serialized linear coefficients."""

    def __init__(self, coefficients: list[float], intercept: float) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)

    def predict(self, matrix: Any) -> np.ndarray:
        return (
            np.asarray(matrix @ self.coefficients).reshape(-1)
            + self.intercept
        )


def _family_scores(
    frame: pd.DataFrame,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    alpha_names, directions, families = alpha_contract(model_contract)
    grouped = output.groupby("date", sort=False, observed=True)
    for name in alpha_names:
        output[f"rank__{name}"] = grouped[name].rank(
            method="average",
            pct=True,
            ascending=directions[name] > 0,
        )
    family_name_map = {
        "relative_strength": "family_relative_strength",
        "price_trend": "family_price_trend",
        "oversold_recovery": "family_oversold_recovery",
    }
    for family, members in families.items():
        destination = family_name_map[family]
        rank_names = [f"rank__{name}" for name in members]
        minimum = int(
            model_contract["alpha_families"][family][
                "minimum_nonmissing_members"
            ]
        )
        counts = output[rank_names].notna().sum(axis=1)
        score = output[rank_names].sub(0.5).mean(axis=1, skipna=True)
        output[destination] = score.where(counts.ge(minimum))
    output["base_family_score"] = output[FAMILY_COLUMNS].mean(
        axis=1, skipna=False
    )
    return output


def load_formation_year(
    feature_panel_root: Path,
    phase2b_panel_root: Path,
    year: int,
    formation_sessions: set[int],
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    """Load the complete point-in-time primary universe without labels."""

    features = pd.read_parquet(feature_panel_root / f"year={year}")
    point_in_time = pd.read_parquet(
        phase2b_panel_root / f"year={year}",
        columns=[
            *OUTPUT_IDENTIFIER_COLUMNS,
            *FORMATION_POINT_IN_TIME_COLUMNS,
        ],
    )
    try:
        pd.testing.assert_frame_equal(
            features[OUTPUT_IDENTIFIER_COLUMNS].reset_index(drop=True),
            point_in_time[OUTPUT_IDENTIFIER_COLUMNS].reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(
            f"Gate 6 Gate 3 / Phase 2B identifiers differ in {year}"
        ) from exc
    for column in FORMATION_POINT_IN_TIME_COLUMNS:
        features[column] = point_in_time[column].to_numpy()
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    valid = (
        features["universe_primary"].astype(bool)
        & features["session_id"].isin(formation_sessions)
    )
    output = features.loc[valid].copy()
    if output.empty:
        raise ValueError(f"No Gate 6 formation rows in {year}")
    output = _family_scores(output, model_contract)
    return output.sort_values(
        ["session_id", "permno"], kind="stable"
    ).reset_index(drop=True)


def _load_gate4_preprocessor(path: Path) -> Gate4Preprocessor:
    value = json.loads(path.read_text(encoding="utf-8"))
    return Gate4Preprocessor(
        continuous_columns=list(value["continuous_columns"]),
        family_columns=list(value["family_columns"]),
        interaction_state_columns=list(
            value["interaction_state_columns"]
        ),
        lower=np.asarray(value["lower"], dtype=float),
        upper=np.asarray(value["upper"], dtype=float),
        median=np.asarray(value["median"], dtype=float),
        mean=np.asarray(value["mean"], dtype=float),
        scale=np.asarray(value["scale"], dtype=float),
    )


def _gate4_ridge_prediction(
    frame: pd.DataFrame,
    state_root: Path,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    preprocessor = _load_gate4_preprocessor(
        state_root / "preprocessing.json"
    )
    state = json.loads(
        (state_root / "linear_models.json").read_text(encoding="utf-8")
    )["ridge"]
    if len(state["coefficients"]) != len(preprocessor.feature_names):
        raise ValueError("Gate 4 Ridge coefficient dimensions changed")
    predictor = FrozenLinearPredictor(
        state["coefficients"], state["intercept"]
    )
    output = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        output[start:stop] = predictor.predict(
            preprocessor.transform(frame.iloc[start:stop])
        ).astype(np.float32)
    return output


def _gate5_ridge_models(
    state_root: Path,
    gate5_contract: dict[str, Any],
) -> dict[str, FrozenLinearPredictor]:
    state = json.loads(
        (state_root / "ridge_models.json").read_text(encoding="utf-8")
    )
    expected = ridge_variant_contract(gate5_contract)
    models: dict[str, FrozenLinearPredictor] = {}
    for name, value in state.items():
        if list(value["feature_names"]) != expected[name]["feature_names"]:
            raise ValueError(f"Gate 5 Ridge feature names changed: {name}")
        models[name] = FrozenLinearPredictor(
            value["coefficients"], value["intercept"]
        )
    if set(models) != set(expected):
        raise ValueError("Gate 5 Ridge state is incomplete")
    return models


def score_formation_year(
    frame: pd.DataFrame,
    *,
    year: int,
    gate4_root: Path,
    gate5_root: Path,
    gate5_contract: dict[str, Any],
) -> pd.DataFrame:
    """Apply the frozen annual models to the unfiltered formation domain."""

    gate4_state = gate4_root / "model_states" / f"oos_{year}"
    gate5_state = gate5_root / "model_states" / f"oos_{year}"
    metadata = json.loads(
        (gate5_state / "metadata.json").read_text(encoding="utf-8")
    )
    preprocessor = FrozenGate4Preprocessor.load(
        gate4_state / "preprocessing.json"
    )
    ridge_models = _gate5_ridge_models(gate5_state, gate5_contract)
    tree_models = joblib.load(gate5_state / "tree_models.joblib")
    predictions = predict_nested_models(
        frame,
        preprocessor,
        ridge_models,
        tree_models,
        gate5_contract,
    )
    output = frame.copy()
    for name, values in predictions.items():
        output[name] = values
    output["ridge_explicit_interactions"] = _gate4_ridge_prediction(
        frame, gate4_state
    )
    output = add_capped_gates(output)
    output["model_fit_id"] = str(metadata["model_fit_id"])
    missing = sorted(set(SCORE_COLUMNS).difference(output.columns))
    if missing:
        raise ValueError(f"Gate 6 formation scores missing: {missing}")
    if any(
        column.startswith("fwd_") or column.startswith("label_")
        for column in output.columns
    ):
        raise ValueError("Future-label column entered Gate 6 formation scores")
    keep = [
        "permno",
        "permco",
        "date",
        "session_id",
        "execution_valid",
        "execution_session_id",
        "lag_price_usd",
        "lag_market_cap_usd",
        "lag_adv20_usd",
        "beta_252d",
        "log_lagged_market_cap",
        "log_price",
        "log_adv20_usd",
        "amihud_illiquidity_20d",
        "quoted_spread_proxy",
        "industry_code_2digit",
        *SCORE_COLUMNS,
        "model_fit_id",
    ]
    return output[keep].sort_values(
        ["session_id", "permno"], kind="stable"
    ).reset_index(drop=True)
