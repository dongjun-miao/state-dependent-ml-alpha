from __future__ import annotations

import numpy as np
import pandas as pd

from state_alpha.data.universe import (
    add_point_in_time_universe_columns,
    apply_universe_rule,
)


def _universe_fixture() -> pd.DataFrame:
    rows = []
    for permno in [10001, 10002]:
        for session in range(8):
            rows.append(
                {
                    "permno": permno,
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=session),
                    "session_id": session,
                    "primary_exchange": "N",
                    "trading_status": "A",
                    "security_type": "EQTY",
                    "security_subtype": "COM",
                    "share_type": "NS",
                    "us_incorporated_flag": "Y",
                    "price_raw": 10.0 if permno == 10001 else 20.0,
                    "market_cap_source": 100_000.0
                    if permno == 10001
                    else 200_000.0,
                    "volume": 100.0,
                    "shares_outstanding_source": 1_000.0,
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["permno", "session_id"], kind="stable"
    )
    frame.loc[
        (frame["permno"] == 10001) & (frame["session_id"] == 3), "volume"
    ] = 10_000.0
    return frame.reset_index(drop=True)


def test_universe_metrics_are_lagged_and_adv_excludes_formation_day() -> None:
    result = add_point_in_time_universe_columns(
        _universe_fixture(), minimum_history=3, adv_window=3
    )
    row_3 = result.loc[
        (result["permno"] == 10001) & (result["session_id"] == 3)
    ].iloc[0]
    row_4 = result.loc[
        (result["permno"] == 10001) & (result["session_id"] == 4)
    ].iloc[0]

    assert row_3["lag_adv20_usd"] == 1_000.0
    assert row_3["lag_dollar_volume_usd"] == 1_000.0
    assert row_4["lag_dollar_volume_usd"] == 100_000.0
    assert row_4["lag_adv20_usd"] == 34_000.0
    assert bool(row_3["universe_analysis_base"])


def test_universe_rule_uses_only_lagged_columns_and_date_ranks() -> None:
    result = add_point_in_time_universe_columns(
        _universe_fixture(), minimum_history=3, adv_window=3
    )
    rule = {
        "kind": "absolute",
        "minimum_price_usd": 5.0,
        "minimum_market_cap_usd": 150_000_000.0,
        "minimum_adv20_usd": 500.0,
        "require_us_incorporated": True,
    }
    mask_before = apply_universe_rule(result, rule)

    changed = _universe_fixture()
    changed.loc[
        (changed["permno"] == 10002) & (changed["session_id"] == 4),
        ["price_raw", "market_cap_source", "volume"],
    ] = [0.01, 1.0, 0.0]
    result_changed = add_point_in_time_universe_columns(
        changed, minimum_history=3, adv_window=3
    )
    mask_after = apply_universe_rule(result_changed, rule)
    target = (result["permno"] == 10002) & (result["session_id"] == 4)

    assert bool(mask_before.loc[target].iloc[0])
    assert bool(mask_after.loc[target].iloc[0])
    assert np.isclose(
        result.loc[target, "lag_market_cap_pct_us"].iloc[0], 1.0
    )


def test_stale_previous_observation_fails_point_in_time_base() -> None:
    fixture = _universe_fixture()
    fixture = fixture.loc[
        ~((fixture["permno"] == 10001) & (fixture["session_id"] == 2))
    ].reset_index(drop=True)
    result = add_point_in_time_universe_columns(
        fixture, minimum_history=2, adv_window=2
    )
    row = result.loc[
        (result["permno"] == 10001) & (result["session_id"] == 3)
    ].iloc[0]
    assert not bool(row["lag_is_previous_market_session"])
    assert not bool(row["universe_analysis_base"])
