from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from state_alpha.features.build import validate_feature_contract
from state_alpha.features.core import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    compute_market_features,
    compute_security_features,
    finalize_target_features,
)


def make_market(length: int = 330) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=length)
    calendar = pd.DataFrame(
        {
            "session_id": np.arange(length, dtype=np.int32),
            "date": dates,
            "market_vw_return": np.linspace(-0.001, 0.001, length),
        }
    )
    cross = pd.DataFrame(
        {
            "date": dates,
            "cross_sectional_return_dispersion": 0.02,
            "market_breadth": 0.55,
        }
    )
    return compute_market_features(calendar, cross)


def make_security(length: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=length)
    rows: list[pd.DataFrame] = []
    for permno, shift in [(10001, 0.0), (10002, 0.0002)]:
        total = 0.001 + shift + np.sin(np.arange(length) / 17.0) * 0.002
        retx = total - 0.0001
        rows.append(
            pd.DataFrame(
                {
                    "permno": permno,
                    "date": dates,
                    "session_id": np.arange(length, dtype=np.int32),
                    "return_total": total,
                    "return_ex_distribution": retx,
                    "price_raw": 20.0
                    * np.cumprod(np.r_[1.0, 1.0 + retx[1:]]),
                    "volume": 1_000_000.0,
                    "shares_outstanding_source": 100_000.0,
                    "bid_raw": 19.99,
                    "ask_raw": 20.01,
                    "siccd": 3571,
                    "primary_exchange": "N",
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["permno", "session_id"], kind="stable"
    ).reset_index(drop=True)


def test_frozen_contract_is_unique_and_within_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = validate_feature_contract(root / "configs" / "features.yaml")
    assert contract["feature_budget"]["frozen_feature_count"] == 38
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 38


def test_dlyretx_price_chain_ignores_raw_price_discontinuity() -> None:
    market = make_market()
    security = make_security()
    altered = security.copy()
    altered.loc[
        altered["session_id"].ge(150), "price_raw"
    ] /= 2.0
    base_features = compute_security_features(security, market)
    altered_features = compute_security_features(altered, market)
    price_chain_columns = [
        "price_distance_prior_high_126d",
        "price_distance_ma_20d",
        "trend_distance_vol_normalized_60d",
        "drawdown_20d",
        "drawdown_60d",
        "distance_52w_high",
        "recovery_above_ma_10d",
    ]
    np.testing.assert_allclose(
        base_features[price_chain_columns],
        altered_features[price_chain_columns],
        equal_nan=True,
    )
    assert not np.allclose(
        base_features["log_price"],
        altered_features["log_price"],
    )


def test_future_observation_cannot_change_formation_features() -> None:
    market = make_market()
    security = make_security()
    changed = security.copy()
    mask = (
        changed["permno"].eq(10001)
        & changed["session_id"].eq(281)
    )
    changed.loc[mask, "return_total"] = 0.70
    changed.loc[mask, "return_ex_distribution"] = 0.65
    base = compute_security_features(security, market)
    alternative = compute_security_features(changed, market)
    row = base["permno"].eq(10001) & base["session_id"].eq(280)
    calculated_continuous = [
        column
        for column in CONTINUOUS_FEATURE_COLUMNS
        if column in base.columns
    ]
    np.testing.assert_allclose(
        base.loc[row, calculated_continuous].to_numpy(dtype=float),
        alternative.loc[
            row, calculated_continuous
        ].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_session_gap_invalidates_and_then_releases_rolling_window() -> None:
    market = make_market()
    security = make_security().loc[
        lambda frame: frame["permno"].eq(10001)
    ].copy()
    security.loc[security["session_id"].ge(250), "session_id"] += 1
    features = compute_security_features(security, market)
    at_255 = features.loc[features["session_id"].eq(256)].iloc[0]
    at_270 = features.loc[features["session_id"].eq(271)].iloc[0]
    assert pd.isna(at_255["turnover_20d"])
    assert pd.isna(at_255["realized_volatility_20d"])
    assert pd.notna(at_270["turnover_20d"])
    assert pd.notna(at_270["realized_volatility_20d"])


def test_target_rank_is_primary_only_and_artifact_has_no_labels() -> None:
    market = make_market()
    security = make_security()
    calculated = compute_security_features(security, market)
    target = calculated.loc[calculated["session_id"].eq(280)].copy()
    target["permco"] = target["permno"] + 10_000
    target["universe_primary"] = [True, False]
    target["largest_lagged_mcap_share_class"] = True
    target["lag_market_cap_usd"] = 500_000_000.0
    target["lag_adv20_usd"] = 5_000_000.0
    output = finalize_target_features(target)
    assert output.loc[output["universe_primary"], "rs_percentile_60d"].eq(1.0).all()
    assert output.loc[
        ~output["universe_primary"], "rs_percentile_60d"
    ].isna().all()
    assert list(output.columns[6:]) == FEATURE_COLUMNS
    assert not any(
        column.startswith(("fwd_", "label_", "target_"))
        for column in output.columns
    )


def test_invalid_quote_is_missing_not_zero() -> None:
    market = make_market()
    security = make_security()
    mask = security["permno"].eq(10001) & security["session_id"].eq(280)
    security.loc[mask, ["bid_raw", "ask_raw"]] = [21.0, 20.0]
    output = compute_security_features(security, market)
    value = output.loc[
        output["permno"].eq(10001) & output["session_id"].eq(280),
        "quoted_spread_proxy",
    ].iloc[0]
    assert pd.isna(value)
