"""Build the versioned, delisting-aware CRSP CIZ canonical Parquet layer."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from state_alpha.config import ProjectConfig
from state_alpha.data.ciz import (
    SOURCE_ROW_COLUMN,
    ContiguousRunBuffer,
    ReconcileResult,
    reconcile_security_days,
)
from state_alpha.data.schema import (
    DISTRIBUTION_ARROW_SCHEMA,
    SCHEMA_VERSION,
    SECURITY_ARROW_SCHEMA,
    SOURCE_COLUMNS,
)
from state_alpha.logging_utils import configure_run_logger
from state_alpha.paths import PathPolicy


class BuildContractError(RuntimeError):
    """Raised when source or output state violates the canonical build contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def source_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1:
            raise BuildContractError(
                f"Expected one CSV member in {path}, found {len(members)}"
            )
        member = members[0]
        with archive.open(member) as stream:
            header = pd.read_csv(stream, nrows=0).columns.tolist()
    missing = sorted(set(SOURCE_COLUMNS).difference(header))
    if missing:
        raise BuildContractError(f"{path.name} is missing source columns: {missing}")
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "member_name": member.filename,
        "member_crc32": f"{member.CRC:08x}",
        "member_uncompressed_bytes": member.file_size,
        "member_compressed_bytes": member.compress_size,
        "source_column_count": len(header),
        "source_columns_sha256": hashlib.sha256(
            "\n".join(header).encode("utf-8")
        ).hexdigest(),
    }


def preflight_source_keys(
    path: Path,
    *,
    chunksize: int = 500_000,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Prove exact duplicates equal contiguous duplicates for streaming collapse."""

    batches: list[np.ndarray] = []
    rows = 0
    adjacent_duplicates = 0
    previous_key: int | None = None
    date_min: int | None = None
    date_max: int | None = None
    for chunk_number, frame in enumerate(
        pd.read_csv(
            path,
            compression="zip",
            usecols=["PERMNO", "DlyCalDt"],
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        permno = pd.to_numeric(frame["PERMNO"], errors="raise").astype(np.int64)
        date = pd.to_numeric(
            frame["DlyCalDt"].astype("string").str.replace("-", "", regex=False),
            errors="raise",
        ).astype(np.int64)
        keys = permno.to_numpy() * 100_000_000 + date.to_numpy()
        if len(keys):
            adjacent_duplicates += int(np.sum(keys[1:] == keys[:-1]))
            if previous_key is not None:
                adjacent_duplicates += int(keys[0] == previous_key)
            previous_key = int(keys[-1])
            local_min = int(date.min())
            local_max = int(date.max())
            date_min = local_min if date_min is None else min(date_min, local_min)
            date_max = local_max if date_max is None else max(date_max, local_max)
        batches.append(keys)
        rows += len(keys)
        if logger and chunk_number % 10 == 0:
            logger.info("Preflight %s: %s rows", path.name, f"{rows:,}")
    all_keys = np.concatenate(batches) if batches else np.array([], dtype=np.int64)
    all_keys.sort()
    duplicate_mask = all_keys[1:] == all_keys[:-1]
    exact_duplicate_rows = int(duplicate_mask.sum())
    if exact_duplicate_rows:
        _, counts = np.unique(all_keys, return_counts=True)
        duplicate_counts = counts[counts > 1]
        duplicate_keys = int(len(duplicate_counts))
        max_multiplicity = int(duplicate_counts.max())
    else:
        duplicate_keys = 0
        max_multiplicity = 1 if rows else 0
    contiguous = exact_duplicate_rows == adjacent_duplicates
    if not contiguous:
        raise BuildContractError(
            f"{path.name}: exact duplicate rows {exact_duplicate_rows} do not "
            f"equal adjacent duplicate rows {adjacent_duplicates}"
        )
    return {
        "rows": rows,
        "date_min": str(date_min) if date_min is not None else None,
        "date_max": str(date_max) if date_max is not None else None,
        "duplicate_keys": duplicate_keys,
        "duplicate_rows_beyond_first": exact_duplicate_rows,
        "max_multiplicity": max_multiplicity,
        "adjacent_duplicate_rows": adjacent_duplicates,
        "duplicates_contiguous": contiguous,
    }


class PartitionedParquetWriters:
    """Maintain one ParquetWriter per dataset/year."""

    def __init__(
        self,
        root: Path,
        *,
        compression: str,
        compression_level: int,
        path_policy: PathPolicy,
    ) -> None:
        self.root = path_policy.assert_writable(root)
        self.compression = compression
        self.compression_level = compression_level
        self.path_policy = path_policy
        self._writers: dict[tuple[str, int], pq.ParquetWriter] = {}
        self.rows: Counter[str] = Counter()

    def _writer(
        self,
        dataset: str,
        year: int,
        schema: pa.Schema,
    ) -> pq.ParquetWriter:
        key = (dataset, year)
        if key not in self._writers:
            path = self.path_policy.assert_writable(
                self.root
                / dataset
                / f"year={year}"
                / "part-00000.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[key] = pq.ParquetWriter(
                path,
                schema=schema,
                compression=self.compression,
                compression_level=self.compression_level,
                use_dictionary=True,
                write_statistics=True,
            )
        return self._writers[key]

    def write(
        self,
        dataset: str,
        frame: pd.DataFrame,
        schema: pa.Schema,
    ) -> None:
        if frame.empty:
            return
        years = pd.to_datetime(frame["date"], errors="raise").dt.year
        for year, index in years.groupby(years).groups.items():
            batch = frame.loc[index].reset_index(drop=True)
            table = pa.Table.from_pandas(
                batch,
                schema=schema,
                preserve_index=False,
                safe=True,
            )
            self._writer(dataset, int(year), schema).write_table(
                table,
                row_group_size=min(len(table), 250_000),
            )
            self.rows[f"{dataset}:{int(year)}"] += len(table)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


@dataclass
class BuildAudit:
    raw_rows: int = 0
    security_day_rows: int = 0
    distribution_rows: int = 0
    duplicate_keys: int = 0
    duplicate_rows_beyond_first: int = 0
    max_multiplicity: int = 0
    delisting_rows: int = 0
    delisting_rows_missing_return: int = 0
    conflict_examples: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, result: ReconcileResult, file_name: str) -> None:
        self.raw_rows += result.raw_rows
        self.security_day_rows += result.security_day_rows
        self.distribution_rows += result.distribution_rows
        self.duplicate_keys += result.duplicate_keys
        self.duplicate_rows_beyond_first += result.duplicate_rows_beyond_first
        self.max_multiplicity = max(
            self.max_multiplicity, result.max_multiplicity
        )
        self.delisting_rows += result.delisting_rows
        self.delisting_rows_missing_return += (
            result.delisting_rows_missing_return
        )
        self.conflict_examples.extend(result.conflict_examples)
        local = self.files.setdefault(
            file_name,
            {
                "raw_rows": 0,
                "security_day_rows": 0,
                "distribution_rows": 0,
                "duplicate_keys": 0,
                "duplicate_rows_beyond_first": 0,
                "max_multiplicity": 0,
                "delisting_rows": 0,
                "delisting_rows_missing_return": 0,
            },
        )
        local["raw_rows"] += result.raw_rows
        local["security_day_rows"] += result.security_day_rows
        local["distribution_rows"] += result.distribution_rows
        local["duplicate_keys"] += result.duplicate_keys
        local["duplicate_rows_beyond_first"] += result.duplicate_rows_beyond_first
        local["max_multiplicity"] = max(
            local["max_multiplicity"], result.max_multiplicity
        )
        local["delisting_rows"] += result.delisting_rows
        local["delisting_rows_missing_return"] += (
            result.delisting_rows_missing_return
        )


def _process_emitted(
    emitted: pd.DataFrame,
    *,
    source_file: str,
    audit: BuildAudit,
    writers: PartitionedParquetWriters,
) -> None:
    if emitted.empty:
        return
    result = reconcile_security_days(
        emitted,
        source_file=source_file,
        fail_on_conflict=True,
    )
    audit.add(result, source_file)
    writers.write("security_day", result.security_days, SECURITY_ARROW_SCHEMA)
    writers.write(
        "distributions",
        result.distributions,
        DISTRIBUTION_ARROW_SCHEMA,
    )


def _output_manifest(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.rglob("*.parquet")):
        metadata = pq.read_metadata(path)
        results.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "sha256": _sha256(path),
            }
        )
    return results


def build_canonical(
    project_root: Path,
    *,
    source_files: Iterable[Path] | None = None,
    run_preflight: bool = True,
) -> Path:
    """Execute an atomic, no-overwrite canonical build and return final root."""

    config = ProjectConfig.load(project_root)
    policy = PathPolicy.from_config(config)
    contract = config.data_contract
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.project_path("logs") / f"phase2a_build_{timestamp}.log"
    logger = configure_run_logger("state_alpha.phase2a", log_path, policy)

    raw_dir = policy.assert_read_only_source(config.external_path("crsp_raw_dir"))
    paths = (
        [policy.assert_read_only_source(Path(path)) for path in source_files]
        if source_files is not None
        else sorted(raw_dir.glob(contract["source_glob"]))
    )
    if not paths:
        raise BuildContractError(f"No CRSP sources found in {raw_dir}")

    output_name = contract["outputs"]["root_name"]
    final_root = policy.assert_writable(
        config.project_path("data_interim") / output_name
    )
    if final_root.exists():
        raise FileExistsError(
            f"Canonical output already exists; refusing overwrite: {final_root}"
        )
    staging_root = policy.assert_writable(
        config.project_path("data_interim")
        / f"_{output_name}_build_{timestamp}"
    )
    if staging_root.exists():
        raise FileExistsError(f"Staging output already exists: {staging_root}")
    staging_root.mkdir(parents=True)

    snapshots_before = {path.name: source_snapshot(path) for path in paths}
    preflight: dict[str, Any] = {}
    if run_preflight:
        for path in paths:
            logger.info("Preflight source keys: %s", path)
            preflight[path.name] = preflight_source_keys(path, logger=logger)
            logger.info("Preflight complete: %s %s", path.name, preflight[path.name])
        _write_json(staging_root / "source_key_preflight.json", preflight)

    writers = PartitionedParquetWriters(
        staging_root,
        compression=contract["outputs"]["parquet_compression"],
        compression_level=int(
            contract["outputs"]["parquet_compression_level"]
        ),
        path_policy=policy,
    )
    audit = BuildAudit()
    started = time.time()
    try:
        for source_path in paths:
            logger.info("Building from %s", source_path)
            buffer = ContiguousRunBuffer()
            source_row = 0
            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    source_path,
                    compression="zip",
                    usecols=SOURCE_COLUMNS,
                    chunksize=int(contract["chunk_size"]),
                    low_memory=False,
                ),
                start=1,
            ):
                count = len(chunk)
                chunk[SOURCE_ROW_COLUMN] = np.arange(
                    source_row + 1,
                    source_row + count + 1,
                    dtype=np.int64,
                )
                source_row += count
                emitted = buffer.push(chunk)
                _process_emitted(
                    emitted,
                    source_file=source_path.name,
                    audit=audit,
                    writers=writers,
                )
                if chunk_number % 10 == 0:
                    logger.info(
                        "%s: read %s raw rows; emitted %s security days",
                        source_path.name,
                        f"{source_row:,}",
                        f"{audit.files.get(source_path.name, {}).get(
                            'security_day_rows', 0
                        ):,}",
                    )
            _process_emitted(
                buffer.finish(),
                source_file=source_path.name,
                audit=audit,
                writers=writers,
            )
            expected = preflight.get(source_path.name, {}).get("rows")
            if expected is not None and source_row != expected:
                raise BuildContractError(
                    f"{source_path.name}: build read {source_row}, "
                    f"preflight read {expected}"
                )
            logger.info(
                "Completed source %s: %s rows",
                source_path.name,
                f"{source_row:,}",
            )
    finally:
        writers.close()

    snapshots_after = {path.name: source_snapshot(path) for path in paths}
    if snapshots_before != snapshots_after:
        raise BuildContractError("One or more raw source snapshots changed during build")
    if audit.raw_rows - audit.duplicate_rows_beyond_first != audit.security_day_rows:
        raise BuildContractError(
            "Row reconciliation failed: raw - duplicate extras != security days"
        )
    if run_preflight:
        expected_duplicates = sum(
            value["duplicate_rows_beyond_first"] for value in preflight.values()
        )
        if audit.duplicate_rows_beyond_first != expected_duplicates:
            raise BuildContractError(
                f"Build duplicate extras {audit.duplicate_rows_beyond_first} "
                f"!= preflight {expected_duplicates}"
            )

    output_files = _output_manifest(staging_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "elapsed_seconds": time.time() - started,
        "project_root": str(config.root),
        "staging_root": str(staging_root),
        "final_root": str(final_root),
        "source_snapshots": snapshots_before,
        "source_key_preflight": preflight,
        "audit": asdict(audit),
        "partition_rows": dict(writers.rows),
        "output_files": output_files,
        "raw_sources_unchanged": True,
        "overwrite_used": False,
    }
    _write_json(staging_root / "manifest.json", manifest)
    staging_root.rename(final_root)
    logger.info(
        "Canonical build complete: raw=%s security_day=%s distributions=%s",
        f"{audit.raw_rows:,}",
        f"{audit.security_day_rows:,}",
        f"{audit.distribution_rows:,}",
    )
    logger.info("Final output: %s", final_root)
    return final_root
