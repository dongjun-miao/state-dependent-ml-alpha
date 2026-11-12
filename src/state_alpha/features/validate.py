"""Independent validation for the Gate 3 feature artifact."""

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
from state_alpha.features.build import FEATURE_SCHEMA_VERSION
from state_alpha.features.core import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MARKET_PUBLIC_COLUMNS,
    OUTPUT_IDENTIFIER_COLUMNS,
)
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


def validate_features(
    project_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify hashes, keys, schema, market joins, finiteness, and input snapshots."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    root = config.project_path("data_interim") / "features_v1"
    if not root.is_dir():
        raise FileNotFoundError(f"Gate 3 feature root missing: {root}")
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        errors.append("manifest schema version mismatch")
    if manifest.get("feature_columns") != FEATURE_COLUMNS:
        errors.append("manifest feature-column contract mismatch")
    if manifest.get("feature_count") != len(FEATURE_COLUMNS):
        errors.append("manifest feature count mismatch")
    if manifest.get("information_boundary", {}).get(
        "labels_in_feature_artifact"
    ):
        errors.append("manifest permits labels in feature artifact")

    files_verified = 0
    panel_files = 0
    for item in manifest.get("output_files", []):
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing output: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        if _sha256(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {item['path']}")
        files_verified += 1

    market_path = root / "market_features.parquet"
    market = pd.read_parquet(market_path)
    if market["session_id"].tolist() != list(range(len(market))):
        errors.append("market feature session IDs are not contiguous")
    if market["date"].duplicated().any():
        errors.append("market features contain duplicate dates")
    market_lookup = market.set_index("session_id")

    phase2b_panel = Path(manifest["phase2b_root"]) / "panel"
    annual_actual: list[dict[str, int]] = []
    expected_columns = [*OUTPUT_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]
    forbidden_prefixes = ("fwd_", "label_", "target_")
    for year in range(2000, 2026):
        file_path = (
            root / "panel" / f"year={year}" / "part-00000.parquet"
        )
        if not file_path.is_file():
            errors.append(f"missing panel partition for {year}")
            continue
        panel_files += 1
        parquet = pq.ParquetFile(file_path)
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b"schema_version", b"").decode() != FEATURE_SCHEMA_VERSION:
            errors.append(f"Parquet schema version mismatch for {year}")
        if metadata.get(b"labels_excluded", b"").decode() != "true":
            errors.append(f"labels-excluded metadata missing for {year}")
        frame = parquet.read().to_pandas()
        if list(frame.columns) != expected_columns:
            errors.append(f"column contract mismatch for {year}")
            continue
        if any(
            column.startswith(forbidden_prefixes)
            for column in frame.columns
        ):
            errors.append(f"forbidden label-like column in {year}")
        if len(frame) != next(
            item["rows"]
            for item in manifest["output_files"]
            if item["path"].endswith(
                f"panel/year={year}/part-00000.parquet"
            )
        ):
            errors.append(f"manifest row count mismatch for {year}")
        if frame.duplicated(["permno", "date"]).any():
            errors.append(f"duplicate security-date keys for {year}")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if len(frame) and not dates.dt.year.eq(year).all():
            errors.append(f"wrong-year rows for {year}")

        input_keys = pd.read_parquet(
            phase2b_panel / f"year={year}",
            columns=OUTPUT_IDENTIFIER_COLUMNS,
        )
        try:
            pd.testing.assert_frame_equal(
                frame[OUTPUT_IDENTIFIER_COLUMNS].reset_index(drop=True),
                input_keys.reset_index(drop=True),
                check_dtype=False,
                check_like=False,
            )
        except AssertionError:
            errors.append(f"Phase 2B identifier mismatch for {year}")

        if len(frame):
            numeric = frame[CONTINUOUS_FEATURE_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            )
            if np.isinf(numeric.to_numpy()).any():
                errors.append(f"infinite feature value for {year}")
            primary = frame["universe_primary"].astype(bool)
            rank = pd.to_numeric(
                frame["rs_percentile_60d"], errors="coerce"
            )
            if not rank.loc[primary].dropna().between(
                0.0, 1.0, inclusive="both"
            ).all():
                errors.append(f"relative-strength rank out of bounds for {year}")
            if rank.loc[~primary].notna().any():
                errors.append(f"non-primary rank is populated for {year}")
            for column in MARKET_PUBLIC_COLUMNS:
                expected = frame["session_id"].map(market_lookup[column])
                actual = pd.to_numeric(frame[column], errors="coerce")
                if not np.allclose(
                    actual.to_numpy(dtype=float),
                    expected.to_numpy(dtype=float),
                    equal_nan=True,
                    rtol=2e-6,
                    atol=2e-7,
                ):
                    errors.append(f"market-state join mismatch: {year} {column}")
            industry = pd.to_numeric(
                frame["industry_code_2digit"], errors="coerce"
            ).dropna()
            if len(industry) and not industry.between(
                0, 99, inclusive="both"
            ).all():
                errors.append(f"industry control out of range for {year}")
            exchange = frame["exchange"].dropna()
            if len(exchange) and not exchange.isin(["N", "Q", "A"]).all():
                errors.append(f"exchange control out of range for {year}")

        annual_actual.append(
            {
                "year": year,
                "analysis_rows": len(frame),
                "primary_rows": (
                    int(frame["universe_primary"].sum()) if len(frame) else 0
                ),
                "unique_permnos": (
                    int(frame["permno"].nunique()) if len(frame) else 0
                ),
                "feature_count": len(FEATURE_COLUMNS),
                "label_columns": 0,
            }
        )

    if annual_actual != manifest.get("annual_audit"):
        errors.append("annual audit does not reconcile")
    canonical_unchanged = (
        _snapshot(Path(manifest["canonical_root"]))
        == manifest["canonical_snapshot"]
    )
    phase2b_unchanged = (
        _snapshot(Path(manifest["phase2b_root"]))
        == manifest["phase2b_snapshot"]
    )
    if not canonical_unchanged:
        errors.append("canonical input snapshot changed")
    if not phase2b_unchanged:
        errors.append("Phase 2B input snapshot changed")
    contract_hash = _sha256(config.root / "configs" / "features.yaml")
    if contract_hash != manifest.get("feature_contract_sha256"):
        errors.append("feature contract changed after build")

    result = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "validated_at": datetime.now().isoformat(),
        "feature_root": str(root),
        "passed": not errors,
        "errors": errors,
        "feature_count": len(FEATURE_COLUMNS),
        "label_columns": 0,
        "output_files_verified": files_verified,
        "panel_files_verified": panel_files,
        "analysis_rows": sum(row["analysis_rows"] for row in annual_actual),
        "primary_rows": sum(row["primary_rows"] for row in annual_actual),
        "canonical_input_unchanged": canonical_unchanged,
        "phase2b_input_unchanged": phase2b_unchanged,
        "feature_contract_unchanged": (
            contract_hash == manifest.get("feature_contract_sha256")
        ),
        "overwrite_used": manifest.get("overwrite_used"),
        "performance_or_label_diagnostics_used": False,
    }
    report_path = (
        output_path
        if output_path is not None
        else config.project_path("reports_tables")
        / "phase3_features_validation.json"
    )
    report_path = policy.assert_writable(report_path)
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation report: {report_path}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise ValueError(f"Gate 3 validation failed: {errors[:8]}")
    return result
