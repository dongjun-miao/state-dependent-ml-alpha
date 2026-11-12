"""Independent validation of Gate 4 evidence and expanding OOS governance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig
from state_alpha.models.gate4 import GATE4_SCHEMA_VERSION
from state_alpha.paths import PathPolicy


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


def validate_gate4(
    project_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify hashes, prediction coverage, purging, configs, and inputs."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    root = config.project_path("data_processed") / "gate4_v1"
    if not root.is_dir():
        raise FileNotFoundError(f"Gate 4 root missing: {root}")
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    if manifest.get("schema_version") != GATE4_SCHEMA_VERSION:
        errors.append("manifest schema version mismatch")
    if manifest.get("strict_oos_used_for_selection") is not False:
        errors.append("strict OOS was marked as used for model selection")
    if manifest.get("tree_model_fit") is not False:
        errors.append("tree model was fit inside Gate 4")
    if manifest.get("selection_period") != "2010-2012":
        errors.append("wrong model-selection period")
    if manifest.get("strict_oos_period") != "2013-2025":
        errors.append("wrong strict-OOS period")

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
                != GATE4_SCHEMA_VERSION
            ):
                errors.append(f"Parquet schema mismatch: {item['path']}")
            if item["rows"] != parquet.metadata.num_rows:
                errors.append(f"Parquet row mismatch: {item['path']}")
        files_verified += 1

    refits = pd.DataFrame(manifest.get("model_refits", []))
    if len(refits) != 14:
        errors.append("expected one validation fit plus 13 strict-OOS refits")
    if len(refits) and not refits["purge_passed"].astype(bool).all():
        errors.append("one or more model refits failed the purge contract")
    if len(refits) and not refits["embargo_passed"].astype(bool).all():
        errors.append("one or more model refits failed the embargo contract")
    if len(refits) and (
        refits["training_max_label_end_session_id"]
        >= refits["evaluation_start_session_id"]
    ).any():
        errors.append("training label overlaps evaluation start")
    if len(refits) and (
        refits["training_max_label_end_session_id"]
        >= refits["embargo_cutoff_session_id"]
    ).any():
        errors.append("training label crosses the embargo cutoff")
    strict = refits.loc[refits["strict_oos"].astype(bool)]
    if strict["evaluation_start_year"].tolist() != list(range(2013, 2026)):
        errors.append("strict-OOS annual refit years are incomplete")

    prediction_rows = {"validation": 0, "strict_oos": 0}
    base_score_rows = {"validation": 0, "strict_oos": 0}
    phase2b_panel = Path(manifest["phase2b_root"]) / "panel"
    for year in range(2010, 2026):
        path = root / "predictions" / f"year={year}" / "part-00000.parquet"
        if not path.is_file():
            errors.append(f"missing prediction partition for {year}")
            continue
        frame = pd.read_parquet(path)
        required = {
            "permno",
            "permco",
            "date",
            "session_id",
            "label_end_session_id_10d",
            "fwd_total_return_10d",
            "fwd_total_return_rank_10d",
            "base_family_score",
            "ridge_score",
            "elastic_net_score",
            "model_fit_id",
        }
        if not required.issubset(frame.columns):
            errors.append(f"prediction schema incomplete for {year}")
            continue
        if frame.duplicated(["permno", "date"]).any():
            errors.append(f"duplicate prediction keys for {year}")
        scores = frame[["ridge_score", "elastic_net_score"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if scores.isna().any().any() or np.isinf(scores.to_numpy()).any():
            errors.append(f"invalid linear-model score for {year}")
        base_score = pd.to_numeric(
            frame["base_family_score"], errors="coerce"
        )
        if np.isinf(base_score.to_numpy(dtype=float)).any():
            errors.append(f"infinite base-family score for {year}")
        labels = pd.read_parquet(
            phase2b_panel / f"year={year}",
            columns=[
                "permno",
                "permco",
                "date",
                "session_id",
                "universe_primary",
                "execution_valid",
                "label_end_session_id_10d",
                "fwd_total_return_10d",
                "fwd_total_return_rank_10d",
            ],
        )
        expected = labels.loc[
            labels["universe_primary"].astype(bool)
            & labels["execution_valid"].astype(bool)
            & labels["fwd_total_return_10d"].notna()
            & labels["fwd_total_return_rank_10d"].notna(),
            [
                "permno",
                "permco",
                "date",
                "session_id",
                "label_end_session_id_10d",
                "fwd_total_return_10d",
                "fwd_total_return_rank_10d",
            ],
        ].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                frame[
                    [
                        "permno",
                        "permco",
                        "date",
                        "session_id",
                        "label_end_session_id_10d",
                        "fwd_total_return_10d",
                        "fwd_total_return_rank_10d",
                    ]
                ].reset_index(drop=True),
                expected,
                check_dtype=False,
            )
        except AssertionError:
            errors.append(f"prediction/Phase 2B row mismatch for {year}")
        key = "validation" if year <= 2012 else "strict_oos"
        prediction_rows[key] += len(frame)
        base_score_rows[key] += int(base_score.notna().sum())

    if prediction_rows["validation"] != manifest.get(
        "validation_prediction_rows"
    ):
        errors.append("validation prediction total mismatch")
    if prediction_rows["strict_oos"] != manifest.get(
        "strict_oos_prediction_rows"
    ):
        errors.append("strict-OOS prediction total mismatch")
    if base_score_rows["validation"] != manifest.get(
        "validation_base_score_nonmissing_rows"
    ):
        errors.append("validation base-score coverage mismatch")
    if base_score_rows["strict_oos"] != manifest.get(
        "strict_oos_base_score_nonmissing_rows"
    ):
        errors.append("strict-OOS base-score coverage mismatch")

    model_state_paths = (
        sorted(refits["model_state_path"].astype(str).tolist())
        if len(refits)
        else []
    )
    discovered_state_paths = (
        sorted(
            str(path.relative_to(root)).replace("\\", "/")
            for path in (root / "model_states").iterdir()
            if path.is_dir()
        )
        if (root / "model_states").is_dir()
        else []
    )
    if discovered_state_paths != model_state_paths:
        errors.append("transparent model-state directories are incomplete")
    model_states_verified = 0
    for row in refits.to_dict("records"):
        model_fit_id = str(row["model_fit_id"])
        state_root = root / str(row["model_state_path"])
        preprocessing_path = state_root / "preprocessing.json"
        models_path = state_root / "linear_models.json"
        if not preprocessing_path.is_file() or not models_path.is_file():
            errors.append(f"incomplete model state: {model_fit_id}")
            continue
        preprocessing = json.loads(
            preprocessing_path.read_text(encoding="utf-8")
        )
        models = json.loads(models_path.read_text(encoding="utf-8"))
        if preprocessing.get("model_fit_id") != model_fit_id:
            errors.append(f"model-state identity mismatch: {model_fit_id}")
        if len(preprocessing.get("feature_names", [])) != 174:
            errors.append(f"wrong feature dimension: {model_fit_id}")
        moments = [
            *preprocessing.get("lower", []),
            *preprocessing.get("upper", []),
            *preprocessing.get("median", []),
            *preprocessing.get("mean", []),
            *preprocessing.get("scale", []),
        ]
        if not moments or not np.isfinite(
            np.asarray(moments, dtype=float)
        ).all():
            errors.append(f"invalid preprocessing state: {model_fit_id}")
        if set(models) != {"ridge", "elastic_net"}:
            errors.append(f"wrong linear models: {model_fit_id}")
        elif any(
            len(value.get("coefficients", [])) != 174
            for value in models.values()
        ):
            errors.append(f"wrong coefficient dimension: {model_fit_id}")
        model_states_verified += 1

    features_unchanged = (
        _snapshot(Path(manifest["features_root"]))
        == manifest["features_snapshot"]
    )
    phase2b_unchanged = (
        _snapshot(Path(manifest["phase2b_root"]))
        == manifest["phase2b_snapshot"]
    )
    if not features_unchanged:
        errors.append("Gate 3 input snapshot changed")
    if not phase2b_unchanged:
        errors.append("Phase 2B input snapshot changed")
    config_checks = {
        "models_config_unchanged": (
            _sha256(config.root / "configs" / "models.yaml")
            == manifest.get("models_config_sha256")
        ),
        "features_config_unchanged": (
            _sha256(config.root / "configs" / "features.yaml")
            == manifest.get("features_config_sha256")
        ),
        "labels_config_unchanged": (
            _sha256(config.root / "configs" / "labels.yaml")
            == manifest.get("labels_config_sha256")
        ),
    }
    for name, passed in config_checks.items():
        if not passed:
            errors.append(name)

    result = {
        "schema_version": GATE4_SCHEMA_VERSION,
        "validated_at": datetime.now().isoformat(),
        "gate4_root": str(root),
        "passed": not errors,
        "errors": errors,
        "output_files_verified": files_verified,
        "validation_prediction_rows": prediction_rows["validation"],
        "strict_oos_prediction_rows": prediction_rows["strict_oos"],
        "validation_base_score_nonmissing_rows": (
            base_score_rows["validation"]
        ),
        "strict_oos_base_score_nonmissing_rows": (
            base_score_rows["strict_oos"]
        ),
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
        "tree_model_fit": manifest.get("tree_model_fit"),
        "features_input_unchanged": features_unchanged,
        "phase2b_input_unchanged": phase2b_unchanged,
        **config_checks,
        "overwrite_used": manifest.get("overwrite_used"),
    }
    report_path = (
        output_path
        if output_path is not None
        else config.project_path("reports_tables")
        / "gate4_validation.json"
    )
    report_path = policy.assert_writable(report_path)
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Gate 4 validation: {report_path}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise ValueError(f"Gate 4 validation failed: {errors[:8]}")
    return result
