from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from state_alpha.config import load_yaml
from state_alpha.models.linear import Gate4Preprocessor
from state_alpha.models.matrix import load_modeling_year
from state_alpha.models.nested import (
    FrozenGate4Preprocessor,
    add_capped_gates,
    daily_mechanism_attribution,
    deterministic_cap,
    fit_ridge_variants,
    fit_tree_variants,
    predict_nested_models,
    ridge_variant_contract,
    select_tree_candidate,
    tree_variant_contract,
    validate_gate5_contract,
)
from test_gate4 import _write_modeling_inputs


def _contracts() -> tuple[dict, dict]:
    root = Path(__file__).resolve().parents[1]
    return (
        load_yaml(root / "configs" / "gate5.yaml"),
        load_yaml(root / "configs" / "models.yaml"),
    )


def _preprocessor_state(
    tmp_path: Path,
    frame: pd.DataFrame,
    gate4_contract: dict,
) -> FrozenGate4Preprocessor:
    fitted = Gate4Preprocessor.fit(frame, gate4_contract)
    value = fitted.to_dict()
    value["model_fit_id"] = "synthetic_fit"
    path = tmp_path / "preprocessing.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return FrozenGate4Preprocessor.load(path)


def _small_tree_contract(contract: dict) -> dict:
    value = deepcopy(contract)
    value["tree"]["candidate_grid"] = [
        {
            "candidate_id": "small_3",
            "learning_rate": 0.05,
            "max_iter": 12,
            "max_leaf_nodes": 3,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        },
        {
            "candidate_id": "small_7",
            "learning_rate": 0.05,
            "max_iter": 12,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 50,
            "l2_regularization": 3.0,
        },
    ]
    return value


def test_gate5_contract_and_nested_dimensions(tmp_path: Path) -> None:
    contract, gate4_contract = _contracts()
    validate_gate5_contract(contract, gate4_contract)
    feature_root, label_root = _write_modeling_inputs(tmp_path)
    frame = load_modeling_year(
        feature_root, label_root, 2005, gate4_contract
    )
    preprocessor = _preprocessor_state(
        tmp_path, frame, gate4_contract
    )
    ridge = ridge_variant_contract(contract)
    assert len(ridge["ridge_alpha_only"]["feature_names"]) == 17
    assert len(ridge["ridge_state_only"]["feature_names"]) == 127
    assert len(ridge["ridge_main_effects"]["feature_names"]) == 144
    tree = tree_variant_contract(contract)
    assert len(tree["tree_alpha_only"]["continuous_columns"]) == 17
    assert len(tree["tree_state_only"]["continuous_columns"]) == 22
    assert len(tree["tree_alpha_state"]["continuous_columns"]) == 39
    standardized = preprocessor.standardized_all(frame)
    assert standardized.shape == (len(frame), 39)
    assert np.isfinite(standardized).all()


def test_gate5_models_selection_prediction_and_cap(
    tmp_path: Path,
) -> None:
    contract, gate4_contract = _contracts()
    contract = _small_tree_contract(contract)
    feature_root, label_root = _write_modeling_inputs(
        tmp_path, dates=10, names=300
    )
    frame = load_modeling_year(
        feature_root, label_root, 2005, gate4_contract
    )
    preprocessor = _preprocessor_state(
        tmp_path, frame, gate4_contract
    )
    selected, results = select_tree_candidate(
        frame, frame, preprocessor, contract
    )
    assert selected["candidate_id"] in {"small_3", "small_7"}
    assert len(results) == 2
    assert not results["strict_oos_used"].astype(bool).any()
    ridge_models = fit_ridge_variants(
        frame, preprocessor, contract
    )
    tree_models = fit_tree_variants(
        frame, preprocessor, selected, contract
    )
    predictions = predict_nested_models(
        frame,
        preprocessor,
        ridge_models,
        tree_models,
        contract,
        chunk_size=700,
    )
    assert set(predictions) == {
        "ridge_alpha_only",
        "ridge_state_only",
        "ridge_main_effects",
        "tree_alpha_only",
        "tree_state_only",
        "tree_alpha_state",
    }
    assert all(
        len(value) == len(frame) and np.isfinite(value).all()
        for value in predictions.values()
    )

    output = frame[
        [
            "date",
            "base_family_score",
            "fwd_total_return_10d",
        ]
    ].copy()
    for name, values in predictions.items():
        output[name] = values
    output["ridge_explicit_interactions"] = (
        output["ridge_main_effects"]
        + 0.2 * output["base_family_score"].fillna(0)
    )
    output = add_capped_gates(output)
    for name in ["linear", "tree"]:
        strength = output[f"{name}_gate_strength"].dropna()
        assert strength.between(0.0, 1.0).all()
        assert (
            output[f"{name}_capped_gate"].abs()
            <= output["base_exposure"].abs() + 1e-7
        ).dropna().all()
    mechanism = daily_mechanism_attribution(output)
    assert set(mechanism["gate_model"]) == {
        "linear_capped_gate",
        "tree_capped_gate",
    }
    assert mechanism["suppressed_fraction"].between(0, 1).all()


def test_deterministic_tree_cap_is_chronological() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=101),
            "value": np.arange(101),
        }
    )
    capped = deterministic_cap(frame, 20)
    assert len(capped) <= 20
    assert capped["date"].is_monotonic_increasing
    assert capped.iloc[0]["value"] == 0
