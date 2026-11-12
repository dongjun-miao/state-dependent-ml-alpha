"""Independent validation of the published canonical Parquet layer."""

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
from state_alpha.data.build import source_snapshot
from state_alpha.data.schema import (
    DISTRIBUTION_ARROW_SCHEMA,
    SCHEMA_VERSION,
    SECURITY_ARROW_SCHEMA,
)
from state_alpha.paths import PathPolicy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_keys(frame: pd.DataFrame) -> np.ndarray:
    permno = pd.to_numeric(frame["permno"], errors="raise").astype(np.int64)
    dates = pd.to_datetime(frame["date"], errors="raise")
    date_number = (
        dates.dt.year.to_numpy(dtype=np.int64) * 10_000
        + dates.dt.month.to_numpy(dtype=np.int64) * 100
        + dates.dt.day.to_numpy(dtype=np.int64)
    )
    return permno.to_numpy() * 100_000_000 + date_number


def validate_canonical(project_root: Path) -> tuple[Path, dict[str, Any]]:
    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    root = config.project_path("data_interim") / "canonical_v1"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path = policy.assert_writable(
        config.project_path("reports_tables")
        / "phase2a_canonical_validation.json"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation: {output_path}")

    errors: list[str] = []
    security_rows = 0
    raw_record_count_sum = 0
    distribution_record_count_sum = 0
    delisting_rows = 0
    delisting_missing_return_rows = 0
    unique_permnos: set[int] = set()
    dates: set[str] = set()
    parquet_hashes_verified = 0
    distribution_rows = 0
    distribution_provenance: set[tuple[str, int]] = set()

    file_manifest = {
        item["path"]: item for item in manifest["output_files"]
    }
    for relative, expected in sorted(file_manifest.items()):
        path = root / Path(relative)
        if not path.is_file():
            errors.append(f"missing parquet: {relative}")
            continue
        if path.stat().st_size != expected["size_bytes"]:
            errors.append(f"size mismatch: {relative}")
        if _sha256(path) != expected["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")
        else:
            parquet_hashes_verified += 1
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows != expected["rows"]:
            errors.append(f"metadata row mismatch: {relative}")
        dataset = relative.split("/", 1)[0]
        expected_schema = (
            SECURITY_ARROW_SCHEMA
            if dataset == "security_day"
            else DISTRIBUTION_ARROW_SCHEMA
        )
        if parquet_file.schema_arrow.names != expected_schema.names:
            errors.append(f"schema names mismatch: {relative}")
        schema_version = (
            parquet_file.schema_arrow.metadata or {}
        ).get(b"schema_version", b"").decode("utf-8")
        if schema_version != SCHEMA_VERSION:
            errors.append(f"schema version mismatch: {relative}")

        if dataset == "security_day":
            frame = parquet_file.read(
                columns=[
                    "permno",
                    "date",
                    "daily_delisting_flag",
                    "return_total",
                    "source_record_count",
                    "distribution_record_count",
                ]
            ).to_pandas()
            year = int(Path(relative).parts[1].split("=", 1)[1])
            parsed_dates = pd.to_datetime(frame["date"], errors="raise")
            if not parsed_dates.dt.year.eq(year).all():
                errors.append(f"partition-year mismatch: {relative}")
            keys = _encode_keys(frame)
            keys.sort()
            duplicates = int((keys[1:] == keys[:-1]).sum())
            if duplicates:
                errors.append(f"duplicate security-day keys in {relative}: {duplicates}")
            security_rows += len(frame)
            raw_record_count_sum += int(frame["source_record_count"].sum())
            distribution_record_count_sum += int(
                frame["distribution_record_count"].sum()
            )
            flagged = frame["daily_delisting_flag"].eq("Y")
            delisting_rows += int(flagged.sum())
            delisting_missing_return_rows += int(
                (flagged & frame["return_total"].isna()).sum()
            )
            unique_permnos.update(
                int(value) for value in frame["permno"].dropna().unique()
            )
            dates.update(str(value) for value in frame["date"].dropna().unique())
        elif dataset == "distributions":
            frame = parquet_file.read(
                columns=["source_file", "source_row_number"]
            ).to_pandas()
            distribution_rows += len(frame)
            for source_file, row_number in zip(
                frame["source_file"],
                frame["source_row_number"],
                strict=False,
            ):
                key = (str(source_file), int(row_number))
                if key in distribution_provenance:
                    errors.append(f"duplicate distribution provenance: {key}")
                    break
                distribution_provenance.add(key)
        else:
            errors.append(f"unknown dataset in manifest: {dataset}")

    expected_audit = manifest["audit"]
    equations = {
        "security_rows_match_manifest": (
            security_rows == expected_audit["security_day_rows"]
        ),
        "raw_record_sum_matches_manifest": (
            raw_record_count_sum == expected_audit["raw_rows"]
        ),
        "raw_minus_duplicates_equals_security": (
            expected_audit["raw_rows"]
            - expected_audit["duplicate_rows_beyond_first"]
            == security_rows
        ),
        "distribution_rows_match_manifest": (
            distribution_rows == expected_audit["distribution_rows"]
        ),
        "distribution_counts_match_dataset": (
            distribution_record_count_sum == distribution_rows
        ),
        "delisting_rows_match_manifest": (
            delisting_rows == expected_audit["delisting_rows"]
        ),
        "delisting_missing_return_matches_manifest": (
            delisting_missing_return_rows
            == expected_audit["delisting_rows_missing_return"]
        ),
        "all_manifest_parquet_hashes_verified": (
            parquet_hashes_verified == len(file_manifest)
        ),
        "no_build_conflicts": not expected_audit["conflict_examples"],
    }
    for name, passed in equations.items():
        if not passed:
            errors.append(f"failed equation: {name}")

    snapshots_now = {
        name: source_snapshot(Path(snapshot["path"]))
        for name, snapshot in manifest["source_snapshots"].items()
    }
    raw_unchanged = snapshots_now == manifest["source_snapshots"]
    if not raw_unchanged:
        errors.append("raw source snapshot changed after build")

    result = {
        "validated_at": datetime.now().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "passed": not errors,
        "errors": errors,
        "security_day_rows": security_rows,
        "distribution_rows": distribution_rows,
        "raw_record_count_sum": raw_record_count_sum,
        "duplicate_rows_beyond_first": expected_audit[
            "duplicate_rows_beyond_first"
        ],
        "unique_permnos": len(unique_permnos),
        "unique_dates": len(dates),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "delisting_rows": delisting_rows,
        "delisting_missing_return_rows": delisting_missing_return_rows,
        "parquet_files_verified": parquet_hashes_verified,
        "raw_sources_unchanged": raw_unchanged,
        "equations": equations,
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError(f"Canonical validation failed: {errors}")
    return output_path, result

