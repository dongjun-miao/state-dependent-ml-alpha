"""Independent validation for the Phase 2B universe and label panel."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig
from state_alpha.data.phase2b import HORIZONS, PHASE2B_SCHEMA_VERSION
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


def validate_phase2b(
    project_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate hashes, row equations, timing, terminal bounds, and inputs."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    root = config.project_path("data_interim") / "phase2b_v1"
    if not root.is_dir():
        raise FileNotFoundError(f"Phase 2B root missing: {root}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("schema_version") != PHASE2B_SCHEMA_VERSION:
        errors.append("manifest schema version mismatch")

    calendar_path = root / "market_calendar.parquet"
    calendar = pq.read_table(calendar_path).to_pandas()
    if calendar["session_id"].tolist() != list(range(len(calendar))):
        errors.append("market calendar session IDs are not contiguous from zero")
    if calendar["date"].duplicated().any():
        errors.append("market calendar contains duplicate dates")
    if calendar["date"].is_monotonic_increasing is False:
        errors.append("market calendar dates are not increasing")
    if calendar["market_vw_return"].isna().any():
        errors.append("market calendar contains missing returns")

    files_verified = 0
    panel_files = 0
    totals = {
        "analysis_rows": 0,
        "primary_universe_rows": 0,
        "primary_execution_invalid_rows": 0,
    }
    for horizon in HORIZONS:
        totals[f"primary_valid_label_{horizon}d_rows"] = 0
        totals[f"primary_missing_delisting_return_{horizon}d_rows"] = 0

    for item in manifest["output_files"]:
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing output file: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        if _sha256(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {item['path']}")
        files_verified += 1
        if not item["path"].startswith("panel/"):
            continue
        panel_files += 1
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b"schema_version", b"").decode() != PHASE2B_SCHEMA_VERSION:
            errors.append(f"Parquet schema metadata mismatch: {item['path']}")
        frame = parquet.read().to_pandas()
        if len(frame) != item["rows"]:
            errors.append(f"row-count mismatch: {item['path']}")
        if frame.duplicated(["permno", "date"]).any():
            errors.append(f"duplicate (permno,date): {item['path']}")
        year = int(Path(item["path"]).parent.name.split("=", 1)[1])
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if len(frame) and not dates.dt.year.eq(year).all():
            errors.append(f"wrong-year rows: {item['path']}")
        if len(frame) and not frame["universe_analysis_base"].all():
            errors.append(f"non-analysis-base row published: {item['path']}")
        primary = frame["universe_primary"].astype(bool)
        totals["analysis_rows"] += len(frame)
        totals["primary_universe_rows"] += int(primary.sum())
        totals["primary_execution_invalid_rows"] += int(
            (primary & ~frame["execution_valid"]).sum()
        )

        rank = frame.loc[primary, "fwd_total_return_rank_10d"].dropna()
        if len(rank) and not rank.between(-0.5, 0.5, inclusive="both").all():
            errors.append(f"10-day rank outside [-0.5,0.5]: {item['path']}")
        for horizon in HORIZONS:
            start = frame[f"label_start_session_id_{horizon}d"]
            end = frame[f"label_end_session_id_{horizon}d"]
            if len(frame) and not start.eq(frame["session_id"] + 2).all():
                errors.append(f"wrong {horizon}d label start: {item['path']}")
            if len(frame) and not end.eq(
                frame["session_id"] + horizon + 1
            ).all():
                errors.append(f"wrong {horizon}d label end: {item['path']}")
            label = frame[f"fwd_total_return_{horizon}d"]
            valid = label.notna()
            if (valid & ~frame["execution_valid"]).any():
                errors.append(
                    f"{horizon}d label exists without valid execution: "
                    f"{item['path']}"
                )
            if (valid & ~frame[f"label_structural_valid_{horizon}d"]).any():
                errors.append(
                    f"{horizon}d label exists without structural validity: "
                    f"{item['path']}"
                )
            if len(label.dropna()) and label.dropna().lt(-1.0).any():
                errors.append(f"{horizon}d label below -100%: {item['path']}")
            missing_delist = frame[
                f"missing_delisting_return_within_{horizon}d"
            ].astype(bool)
            if (missing_delist & label.notna()).any():
                errors.append(
                    f"primary {horizon}d label fills missing delisting return: "
                    f"{item['path']}"
                )
            lower = frame[
                f"fwd_total_return_{horizon}d_dlret_loss100"
            ]
            if (
                missing_delist
                & lower.notna()
                & ~lower.eq(-1.0)
            ).any():
                errors.append(
                    f"{horizon}d -100% bound is not -1: {item['path']}"
                )
            totals[f"primary_valid_label_{horizon}d_rows"] += int(
                (primary & valid).sum()
            )
            totals[
                f"primary_missing_delisting_return_{horizon}d_rows"
            ] += int((primary & missing_delist).sum())

    expected_totals = {
        key: sum(int(row[key]) for row in manifest["annual_audit"])
        for key in totals
    }
    equations = {
        f"{key}_matches_manifest": totals[key] == expected_totals[key]
        for key in totals
    }
    for name, passed in equations.items():
        if not passed:
            errors.append(name)

    canonical_root = Path(manifest["canonical_root"])
    canonical_unchanged = _snapshot(canonical_root) == manifest[
        "canonical_snapshot"
    ]
    if not canonical_unchanged:
        errors.append("canonical input snapshot changed")
    result = {
        "schema_version": PHASE2B_SCHEMA_VERSION,
        "validated_at": datetime.now().isoformat(),
        "phase2b_root": str(root),
        "passed": not errors,
        "errors": errors,
        "market_calendar_rows": len(calendar),
        "market_calendar_date_min": str(calendar["date"].min()),
        "market_calendar_date_max": str(calendar["date"].max()),
        "output_files_verified": files_verified,
        "panel_files_verified": panel_files,
        "totals": totals,
        "equations": equations,
        "canonical_input_unchanged": canonical_unchanged,
        "overwrite_used": manifest.get("overwrite_used"),
        "primary_rule": manifest.get("primary_rule"),
    }
    report_path = (
        output_path
        if output_path is not None
        else config.project_path("reports_tables")
        / "phase2b_panel_validation.json"
    )
    report_path = policy.assert_writable(report_path)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if errors:
        raise ValueError(f"Phase 2B validation failed: {errors[:5]}")
    return result
