"""Streaming reconciliation of CRSP CIZ security-day and distribution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from state_alpha.data.schema import (
    DISTRIBUTION_DATE_FIELDS,
    DISTRIBUTION_DETECTION_COLUMNS,
    DISTRIBUTION_FLOAT_FIELDS,
    DISTRIBUTION_INTEGER_FIELDS,
    DISTRIBUTION_SOURCE_TO_CANONICAL,
    SECURITY_DATE_FIELDS,
    SECURITY_FLOAT_FIELDS,
    SECURITY_INTEGER_FIELDS,
    SECURITY_SOURCE_TO_CANONICAL,
)


KEY_COLUMNS = ["PERMNO", "DlyCalDt"]
SOURCE_ROW_COLUMN = "_source_row_number"


class DuplicateConflictError(ValueError):
    """Raised when duplicate security-day rows disagree on daily fields."""


@dataclass
class ReconcileResult:
    security_days: pd.DataFrame
    distributions: pd.DataFrame
    raw_rows: int
    security_day_rows: int
    duplicate_keys: int
    duplicate_rows_beyond_first: int
    max_multiplicity: int
    distribution_rows: int
    delisting_rows: int
    delisting_rows_missing_return: int
    conflict_examples: list[dict[str, Any]] = field(default_factory=list)


class ContiguousRunBuffer:
    """Hold the trailing key run so duplicates spanning chunks remain together."""

    def __init__(self) -> None:
        self._carry: pd.DataFrame | None = None

    def push(self, chunk: pd.DataFrame) -> pd.DataFrame:
        if chunk.empty:
            return chunk
        combined = (
            pd.concat([self._carry, chunk], ignore_index=True)
            if self._carry is not None
            else chunk.reset_index(drop=True)
        )
        last_permno = combined["PERMNO"].iloc[-1]
        last_date = combined["DlyCalDt"].iloc[-1]
        same_as_last = (
            combined["PERMNO"].eq(last_permno)
            & combined["DlyCalDt"].eq(last_date)
        ).to_numpy()
        not_same = np.flatnonzero(~same_as_last)
        start = int(not_same[-1] + 1) if len(not_same) else 0
        emitted = combined.iloc[:start].copy()
        self._carry = combined.iloc[start:].copy()
        return emitted

    def finish(self) -> pd.DataFrame:
        if self._carry is None:
            return pd.DataFrame()
        result = self._carry
        self._carry = None
        return result


def _coerce_canonical(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    date_fields: set[str],
    integer_fields: set[str],
    float_fields: set[str],
) -> pd.DataFrame:
    canonical = frame[list(mapping)].rename(columns=mapping).copy()
    for column in canonical:
        if column in date_fields:
            canonical[column] = pd.to_datetime(
                canonical[column], errors="coerce"
            ).dt.date
        elif column in integer_fields:
            canonical[column] = pd.to_numeric(
                canonical[column], errors="coerce"
            ).astype("Int64")
        elif column in float_fields:
            canonical[column] = pd.to_numeric(
                canonical[column], errors="coerce"
            ).astype("float64")
        else:
            canonical[column] = canonical[column].astype("string")
    return canonical


def _distribution_mask(frame: pd.DataFrame) -> pd.Series:
    return frame[list(DISTRIBUTION_DETECTION_COLUMNS)].notna().any(axis=1)


def _first_nonmissing(series: pd.Series) -> Any:
    values = series.dropna()
    return values.iloc[0] if not values.empty else pd.NA


def _conflicting_columns(group: pd.DataFrame) -> list[str]:
    conflicts: list[str] = []
    for column in SECURITY_SOURCE_TO_CANONICAL:
        values = group[column].dropna()
        if values.empty:
            continue
        if values.astype("string").nunique(dropna=True) > 1:
            conflicts.append(column)
    return conflicts


def reconcile_security_days(
    frame: pd.DataFrame,
    *,
    source_file: str,
    fail_on_conflict: bool = True,
) -> ReconcileResult:
    """Collapse security-day duplicates while preserving distribution records."""

    if frame.empty:
        return ReconcileResult(
            security_days=pd.DataFrame(),
            distributions=pd.DataFrame(),
            raw_rows=0,
            security_day_rows=0,
            duplicate_keys=0,
            duplicate_rows_beyond_first=0,
            max_multiplicity=0,
            distribution_rows=0,
            delisting_rows=0,
            delisting_rows_missing_return=0,
        )
    required = {
        *SECURITY_SOURCE_TO_CANONICAL,
        *DISTRIBUTION_SOURCE_TO_CANONICAL,
        SOURCE_ROW_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Source frame is missing required columns: {missing}")
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError("PERMNO and DlyCalDt must be nonmissing")

    group_sizes = frame.groupby(KEY_COLUMNS, sort=False, dropna=False).size()
    duplicate_sizes = group_sizes[group_sizes > 1]
    duplicate_keys = int(len(duplicate_sizes))
    duplicate_rows = int((duplicate_sizes - 1).sum())
    max_multiplicity = int(group_sizes.max())
    distribution_mask = _distribution_mask(frame)
    distribution_counts = distribution_mask.groupby(
        [frame["PERMNO"], frame["DlyCalDt"]], sort=False
    ).sum()

    duplicate_mask = frame.duplicated(KEY_COLUMNS, keep=False)
    singles = frame.loc[~duplicate_mask].copy()
    singles["source_record_count"] = np.int16(1)
    if not singles.empty:
        single_index = pd.MultiIndex.from_frame(singles[KEY_COLUMNS])
        singles["distribution_record_count"] = (
            distribution_counts.reindex(single_index).to_numpy(dtype=np.int16)
        )

    collapsed_groups: list[pd.Series] = []
    conflicts: list[dict[str, Any]] = []
    for key, group in frame.loc[duplicate_mask].groupby(
        KEY_COLUMNS, sort=False, dropna=False
    ):
        conflict_columns = _conflicting_columns(group)
        if conflict_columns:
            example = {
                "permno": int(key[0]),
                "date": str(key[1]),
                "columns": conflict_columns,
                "source_rows": [
                    int(value) for value in group[SOURCE_ROW_COLUMN].tolist()
                ],
            }
            conflicts.append(example)
            if fail_on_conflict:
                raise DuplicateConflictError(
                    f"Conflicting daily fields for {key}: {conflict_columns}"
                )
        record = group.iloc[0].copy()
        for column in SECURITY_SOURCE_TO_CANONICAL:
            record[column] = _first_nonmissing(group[column])
        record[SOURCE_ROW_COLUMN] = int(group[SOURCE_ROW_COLUMN].min())
        record["source_record_count"] = np.int16(len(group))
        record["distribution_record_count"] = np.int16(
            distribution_mask.loc[group.index].sum()
        )
        collapsed_groups.append(record)

    collapsed = (
        pd.DataFrame(collapsed_groups)
        if collapsed_groups
        else frame.iloc[:0].copy()
    )
    day_source = pd.concat([singles, collapsed], ignore_index=True, sort=False)
    day_source = day_source.sort_values(SOURCE_ROW_COLUMN, kind="stable")
    security_days = _coerce_canonical(
        day_source,
        SECURITY_SOURCE_TO_CANONICAL,
        SECURITY_DATE_FIELDS,
        SECURITY_INTEGER_FIELDS,
        SECURITY_FLOAT_FIELDS,
    )
    security_days["source_file"] = pd.array(
        [source_file] * len(security_days), dtype="string"
    )
    security_days["source_row_number"] = day_source[SOURCE_ROW_COLUMN].astype(
        "int64"
    ).to_numpy()
    security_days["source_record_count"] = day_source[
        "source_record_count"
    ].astype("int16").to_numpy()
    security_days["distribution_record_count"] = day_source[
        "distribution_record_count"
    ].astype("int16").to_numpy()

    distribution_source = frame.loc[distribution_mask].copy()
    distributions = _coerce_canonical(
        distribution_source,
        DISTRIBUTION_SOURCE_TO_CANONICAL,
        DISTRIBUTION_DATE_FIELDS,
        DISTRIBUTION_INTEGER_FIELDS,
        DISTRIBUTION_FLOAT_FIELDS,
    )
    distributions["source_file"] = pd.array(
        [source_file] * len(distributions), dtype="string"
    )
    distributions["source_row_number"] = distribution_source[
        SOURCE_ROW_COLUMN
    ].astype("int64").to_numpy()
    if not distribution_source.empty:
        distribution_index = pd.MultiIndex.from_frame(
            distribution_source[KEY_COLUMNS]
        )
        distributions["security_day_record_count"] = (
            group_sizes.reindex(distribution_index).to_numpy(dtype=np.int16)
        )
    else:
        distributions["security_day_record_count"] = pd.Series(dtype="int16")

    delisting = security_days["daily_delisting_flag"].eq("Y")
    return ReconcileResult(
        security_days=security_days.reset_index(drop=True),
        distributions=distributions.reset_index(drop=True),
        raw_rows=len(frame),
        security_day_rows=len(security_days),
        duplicate_keys=duplicate_keys,
        duplicate_rows_beyond_first=duplicate_rows,
        max_multiplicity=max_multiplicity,
        distribution_rows=len(distributions),
        delisting_rows=int(delisting.sum()),
        delisting_rows_missing_return=int(
            (delisting & security_days["return_total"].isna()).sum()
        ),
        conflict_examples=conflicts,
    )
