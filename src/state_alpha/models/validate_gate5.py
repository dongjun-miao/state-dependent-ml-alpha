"""Independent validation of Gate 5 secondary locked evidence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig, load_yaml
from state_alpha.models.gate4 import _sha256, _snapshot
from state_alpha.models.gate5 import (
    GATE5_SCHEMA_VERSION,
    SCORE_COLUMNS,
)
from state_alpha.models.nested import (
    ridge_variant_contract,
    tree_variant_contract,
)
from state_alpha.paths import PathPolicy


def validate_gate5(
    project_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify hashes, nested-model governance, coverage, and immutability."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    root = config.project_path("data_processed") / "gate5_v1"
    if not root.is_dir():
        raise FileNotFoundError(f"Gate 5 root missing: {root}")
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    contract = load_yaml(config.root / "configs" / "gate5.yaml")
    errors: list[str] = []
    if manifest.get("schema_version") != GATE5_SCHEMA_VERSION:
        errors.append("manifest schema version mismatch")
    if manifest.get("analysis_classification") != (
        "secondary_locked_post_gate4"
    ):
        errors.append("wrong Gate 5 analysis classification")
    if manifest.get("gate4_strict_oos_already_observed") is not True:
        errors.append("Gate 4 OOS observation was not disclosed")
    if manifest.get(
        "confirmatory_untouched_oos_claim_allowed"
    ) is not False:
        errors.append("Gate 5 incorrectly permits untouched-OOS claims")
    if manifest.get("strict_oos_used_for_selection") is not False:
        errors.append("strict OOS was used for Gate 5 selection")
    if manifest.get("selection_period") != "2010-2012":
        errors.append("wrong Gate 5 selection period")
    if manifest.get("strict_oos_period") != "2013-2025":
        errors.append("wrong Gate 5 OOS period")
    if manifest.get("scores") != SCORE_COLUMNS:
        errors.append("Gate 5 score contract differs")
    if manifest.get("tree_selection_boundary_passed") is not True:
        errors.append("Gate 5 validation-to-OOS boundary did not pass")
    if manifest.get(
        "tree_selection_validation_max_label_end_session_id", np.inf
    ) >= manifest.get(
        "tree_selection_embargo_cutoff_session_id", -np.inf
    ):
        errors.append("Gate 5 selection label crosses OOS embargo cutoff")
    if manifest.get("gate4_2013_classification") != (
        "transition_sensitivity"
    ):
        errors.append("Gate 4 2013 boundary issue was not disclosed")
    if manifest.get("boundary_clean_period") != "2014-2025":
        errors.append("Gate 5 boundary-clean period is wrong")

    files_verified = 0
    for item in manifest.get("output_files", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing output: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        if _sha256(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {item['path']}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            metadata = parquet.schema_arrow.metadata or {}
            if (
                metadata.get(b"schema_version", b"").decode()
                != GATE5_SCHEMA_VERSION
            ):
                errors.append(f"Parquet schema mismatch: {item['path']}")
            if item["rows"] != parquet.metadata.num_rows:
                errors.append(f"Parquet row mismatch: {item['path']}")
        files_verified += 1

    selection = pd.read_csv(
        root / "diagnostics" / "gate5_tree_model_selection.csv"
    )
    if selection.empty:
        errors.append("tree selection table is empty")
        selected_id = None
    else:
        if selection["strict_oos_used"].astype(bool).any():
            errors.append("tree selection table marks strict OOS use")
        ranked = selection.sort_values(
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
        selected_id = str(ranked.iloc[0]["candidate_id"])
        if selected_id != manifest.get(
            "selected_tree_candidate", {}
        ).get("candidate_id"):
            errors.append("selected tree is not the validation winner")
        if set(selection["fit_variant"]) != {"tree_alpha_state"}:
            errors.append("tree grid was selected on the wrong variant")

    refits = pd.DataFrame(manifest.get("model_refits", []))
    if len(refits) != 14:
        errors.append("expected one validation plus 13 OOS Gate 5 refits")
    if len(refits) and not refits["purge_passed"].astype(bool).all():
        errors.append("one or more Gate 5 refits failed purge")
    if len(refits) and not refits["embargo_passed"].astype(bool).all():
        errors.append("one or more Gate 5 refits failed embargo")
    if len(refits) and (
        refits["training_max_label_end_session_id"]
        >= refits["embargo_cutoff_session_id"]
    ).any():
        errors.append("a Gate 5 training label crosses the embargo")
    strict = refits.loc[refits["strict_oos"].astype(bool)]
    if strict["evaluation_start_year"].tolist() != list(range(2013, 2026)):
        errors.append("Gate 5 OOS refit years are incomplete")
    maximum_tree_rows = int(
        contract["tree"]["optimizer_sample"][
            "maximum_expanding_training_rows"
        ]
    )
    if len(refits) and refits["tree_training_rows"].gt(
        maximum_tree_rows
    ).any():
        errors.append("Gate 5 tree training cap was exceeded")

    ridge_contract = ridge_variant_contract(contract)
    tree_contract = tree_variant_contract(contract)
    model_states_verified = 0
    discovered = (
        sorted(
            str(path.relative_to(root)).replace("\\", "/")
            for path in (root / "model_states").iterdir()
            if path.is_dir()
        )
        if (root / "model_states").is_dir()
        else []
    )
    expected = (
        sorted(refits["model_state_path"].astype(str).tolist())
        if len(refits)
        else []
    )
    if discovered != expected:
        errors.append("Gate 5 model-state directories are incomplete")
    for row in refits.to_dict("records"):
        state_root = root / str(row["model_state_path"])
        metadata_path = state_root / "metadata.json"
        ridge_path = state_root / "ridge_models.json"
        tree_path = state_root / "tree_models.joblib"
        if not (
            metadata_path.is_file()
            and ridge_path.is_file()
            and tree_path.is_file()
        ):
            errors.append(f"incomplete model state: {row['model_fit_id']}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ridge = json.loads(ridge_path.read_text(encoding="utf-8"))
        if metadata.get("model_fit_id") != row["model_fit_id"]:
            errors.append(f"model-state identity mismatch: {row['model_fit_id']}")
        source_preprocessor = Path(metadata["source_gate4_preprocessor"])
        if not source_preprocessor.is_file() or _sha256(
            source_preprocessor
        ) != metadata.get("source_gate4_preprocessor_sha256"):
            errors.append(
                f"source preprocessor mismatch: {row['model_fit_id']}"
            )
        if set(ridge) != set(ridge_contract):
            errors.append(f"wrong Ridge variants: {row['model_fit_id']}")
        else:
            for name, value in ridge.items():
                expected_dimension = len(
                    ridge_contract[name]["feature_names"]
                )
                coefficients = np.asarray(
                    value.get("coefficients", []), dtype=float
                )
                if (
                    len(coefficients) != expected_dimension
                    or not np.isfinite(coefficients).all()
                ):
                    errors.append(
                        f"invalid Ridge state {name}: {row['model_fit_id']}"
                    )
        try:
            trees = joblib.load(tree_path)
        except Exception as exc:  # pragma: no cover - corruption path
            errors.append(
                f"tree state cannot be loaded: {row['model_fit_id']} {exc}"
            )
            continue
        if set(trees) != set(tree_contract):
            errors.append(f"wrong tree variants: {row['model_fit_id']}")
        elif any(
            type(model).__name__ != "HistGradientBoostingRegressor"
            for model in trees.values()
        ):
            errors.append(f"wrong tree class: {row['model_fit_id']}")
        model_states_verified += 1

    prediction_rows = {"validation": 0, "strict_oos": 0}
    gate4_root = Path(manifest["gate4_root"])
    for year in range(2010, 2026):
        gate5_path = (
            root / "predictions" / f"year={year}" / "part-00000.parquet"
        )
        gate4_path = (
            gate4_root
            / "predictions"
            / f"year={year}"
            / "part-00000.parquet"
        )
        if not gate5_path.is_file() or not gate4_path.is_file():
            errors.append(f"missing prediction input/output for {year}")
            continue
        frame = pd.read_parquet(gate5_path)
        gate4 = pd.read_parquet(gate4_path)
        required = {
            "permno",
            "permco",
            "date",
            "session_id",
            "label_end_session_id_10d",
            "fwd_total_return_10d",
            "fwd_total_return_rank_10d",
            "base_family_score",
            *SCORE_COLUMNS,
            "base_exposure",
            "linear_gate_strength",
            "tree_gate_strength",
            "model_fit_id",
        }
        if not required.issubset(frame.columns):
            errors.append(f"Gate 5 prediction schema incomplete for {year}")
            continue
        if frame.duplicated(["permno", "date"]).any():
            errors.append(f"duplicate Gate 5 prediction keys for {year}")
        identity = [
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
                frame[identity].reset_index(drop=True),
                gate4[identity].reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError:
            errors.append(f"Gate 5 / Gate 4 row mismatch for {year}")
        if not np.array_equal(
            frame["ridge_explicit_interactions"].to_numpy(),
            gate4["ridge_score"].to_numpy(),
            equal_nan=True,
        ):
            errors.append(f"Gate 4 Ridge score was not reused for {year}")
        required_finite = [
            name
            for name in SCORE_COLUMNS
            if name
            not in {
                "base_family_score",
                "linear_capped_gate",
                "tree_capped_gate",
            }
        ]
        numeric = frame[required_finite].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any() or np.isinf(
            numeric.to_numpy(dtype=float)
        ).any():
            errors.append(f"invalid Gate 5 model score for {year}")
        for gate in ["linear", "tree"]:
            strength = pd.to_numeric(
                frame[f"{gate}_gate_strength"], errors="coerce"
            )
            valid = strength.dropna()
            if not valid.between(0.0, 1.0).all():
                errors.append(f"invalid capped gate strength for {year}")
        key = "validation" if year <= 2012 else "strict_oos"
        prediction_rows[key] += len(frame)

    if prediction_rows["validation"] != manifest.get(
        "validation_prediction_rows"
    ):
        errors.append("Gate 5 validation row total mismatch")
    if prediction_rows["strict_oos"] != manifest.get(
        "strict_oos_prediction_rows"
    ):
        errors.append("Gate 5 OOS row total mismatch")

    gate4_unchanged = (
        _snapshot(Path(manifest["gate4_root"]))
        == manifest["gate4_snapshot"]
    )
    features_unchanged = (
        _snapshot(Path(manifest["features_root"]))
        == manifest["features_snapshot"]
    )
    phase2b_unchanged = (
        _snapshot(Path(manifest["phase2b_root"]))
        == manifest["phase2b_snapshot"]
    )
    if not gate4_unchanged:
        errors.append("Gate 4 source artifact changed")
    if not features_unchanged:
        errors.append("Gate 3 source artifact changed")
    if not phase2b_unchanged:
        errors.append("Phase 2B source artifact changed")
    config_checks = {
        "gate5_config_unchanged": (
            _sha256(config.root / "configs" / "gate5.yaml")
            == manifest.get("gate5_config_sha256")
        ),
        "models_config_unchanged": (
            _sha256(config.root / "configs" / "models.yaml")
            == manifest.get("models_config_sha256")
        ),
        "gate4_manifest_unchanged": (
            _sha256(Path(manifest["gate4_root"]) / "manifest.json")
            == manifest.get("source_gate4_manifest_sha256")
        ),
    }
    for name, passed in config_checks.items():
        if not passed:
            errors.append(name)

    result = {
        "schema_version": GATE5_SCHEMA_VERSION,
        "validated_at": datetime.now().isoformat(),
        "gate5_root": str(root),
        "passed": not errors,
        "errors": errors,
        "output_files_verified": files_verified,
        "selected_tree_candidate": selected_id,
        "tree_selection_validation_rows": manifest.get(
            "tree_selection_validation_rows"
        ),
        "tree_selection_boundary_passed": manifest.get(
            "tree_selection_boundary_passed"
        ),
        "gate4_2013_classification": manifest.get(
            "gate4_2013_classification"
        ),
        "boundary_clean_period": manifest.get(
            "boundary_clean_period"
        ),
        "validation_prediction_rows": prediction_rows["validation"],
        "strict_oos_prediction_rows": prediction_rows["strict_oos"],
        "model_refits_verified": len(refits),
        "strict_oos_annual_refits_verified": len(strict),
        "model_states_verified": model_states_verified,
        "all_refits_purged": (
            bool(refits["purge_passed"].astype(bool).all())
            if len(refits)
            else False
        ),
        "all_refits_embargoed": (
            bool(refits["embargo_passed"].astype(bool).all())
            if len(refits)
            else False
        ),
        "strict_oos_used_for_selection": manifest.get(
            "strict_oos_used_for_selection"
        ),
        "analysis_classification": manifest.get(
            "analysis_classification"
        ),
        "confirmatory_untouched_oos_claim_allowed": manifest.get(
            "confirmatory_untouched_oos_claim_allowed"
        ),
        "gate4_input_unchanged": gate4_unchanged,
        "features_input_unchanged": features_unchanged,
        "phase2b_input_unchanged": phase2b_unchanged,
        **config_checks,
        "overwrite_used": manifest.get("overwrite_used"),
    }
    report_path = (
        output_path
        if output_path is not None
        else config.project_path("reports_tables")
        / "gate5_validation.json"
    )
    report_path = policy.assert_writable(report_path)
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Gate 5 validation: {report_path}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise ValueError(f"Gate 5 validation failed: {errors[:8]}")
    return result
