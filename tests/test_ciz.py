from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from state_alpha.data.ciz import (
    SOURCE_ROW_COLUMN,
    ContiguousRunBuffer,
    DuplicateConflictError,
    reconcile_security_days,
)
from state_alpha.data.schema import SOURCE_COLUMNS


def source_row(**updates: Any) -> dict[str, Any]:
    row = {column: None for column in SOURCE_COLUMNS}
    row.update(
        {
            "PERMNO": 10001,
            "PERMCO": 9001,
            "DlyCalDt": "2020-01-02",
            "PrimaryExch": "N",
            "ConditionalType": "RW",
            "TradingStatusFlg": "A",
            "SecurityType": "EQTY",
            "SecuritySubType": "COM",
            "ShareType": "NS",
            "USIncFlg": "Y",
            "IssuerType": "CORP",
            "DlyDelFlg": "N",
            "DlyPrc": 10.0,
            "DlyRet": 0.01,
            "DlyRetx": 0.01,
            "DlyFacPrc": 1.0,
            "DlyVol": 1000.0,
            "DlyClose": 10.0,
            "DlyLow": 9.5,
            "DlyHigh": 10.2,
            "DlyOpen": 9.8,
            "ShrOut": 100.0,
            "vwretd": 0.002,
            "vwretx": 0.002,
            "ewretd": 0.001,
            "ewretx": 0.001,
            "sprtrn": 0.002,
        }
    )
    row.update(updates)
    return row


def test_duplicate_security_day_preserves_distributions() -> None:
    rows = [
        source_row(
            PERMNO=10002,
            PERMCO=9002,
            DlyCalDt="2020-01-03",
            DlyDelFlg="Y",
            TradingStatusFlg="D",
            SecuritySubType="UNK",
            DlyRet=None,
        ),
        source_row(
            DisExDt="2020-01-02",
            DisSeqNbr=1,
            DisType="CD",
            DisDivAmt=0.1,
        ),
        source_row(
            DisExDt="2020-01-02",
            DisSeqNbr=2,
            DisType="SD",
            DisFacShr=0.05,
        ),
    ]
    frame = pd.DataFrame(rows)
    frame[SOURCE_ROW_COLUMN] = [1, 2, 3]
    result = reconcile_security_days(frame, source_file="sample.zip")

    assert result.raw_rows == 3
    assert result.security_day_rows == 2
    assert result.duplicate_keys == 1
    assert result.duplicate_rows_beyond_first == 1
    assert result.distribution_rows == 2
    assert result.delisting_rows == 1
    assert result.delisting_rows_missing_return == 1
    day = result.security_days.loc[result.security_days["permno"].eq(10001)].iloc[0]
    assert day["source_record_count"] == 2
    assert day["distribution_record_count"] == 2
    assert result.distributions["distribution_sequence_number"].tolist() == [1, 2]
    assert result.distributions["source_file"].isna().sum() == 0


def test_conflicting_daily_values_fail_closed() -> None:
    frame = pd.DataFrame([source_row(DlyPrc=10.0), source_row(DlyPrc=11.0)])
    frame[SOURCE_ROW_COLUMN] = [1, 2]
    with pytest.raises(DuplicateConflictError):
        reconcile_security_days(frame, source_file="sample.zip")


def test_contiguous_run_buffer_keeps_boundary_group() -> None:
    first = pd.DataFrame(
        [
            source_row(PERMNO=1, DlyCalDt="2020-01-02"),
            source_row(PERMNO=2, DlyCalDt="2020-01-03"),
        ]
    )
    second = pd.DataFrame(
        [
            source_row(PERMNO=2, DlyCalDt="2020-01-03"),
            source_row(PERMNO=3, DlyCalDt="2020-01-06"),
        ]
    )
    buffer = ContiguousRunBuffer()
    emitted_1 = buffer.push(first)
    emitted_2 = buffer.push(second)
    final = buffer.finish()

    assert emitted_1["PERMNO"].tolist() == [1]
    assert emitted_2["PERMNO"].tolist() == [2, 2]
    assert final["PERMNO"].tolist() == [3]
