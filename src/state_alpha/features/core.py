"""Core Gate 3 feature calculations.

Every rolling security feature ends at formation close ``t`` and requires an
unbroken CRSP market-session window. The corporate-action-safe synthetic price
chain is constructed from CIZ ``DlyRetx`` rather than raw OHLC levels.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


OUTPUT_IDENTIFIER_COLUMNS = [
    "permno",
    "permco",
    "date",
    "session_id",
    "universe_primary",
    "largest_lagged_mcap_share_class",
]

FEATURE_COLUMNS = [
    # Alpha signals: relative strength.
    "rs_excess_return_20d",
    "rs_excess_return_60d",
    "rs_excess_return_126d",
    "rs_percentile_60d",
    "rs_acceleration_20_minus_60d",
    # Alpha signals: price/trend.
    "price_distance_prior_high_126d",
    "price_distance_ma_20d",
    "trend_distance_vol_normalized_60d",
    # Alpha signals: oversold/recovery.
    "reversal_5d",
    "drawdown_20d",
    "drawdown_60d",
    "distance_52w_high",
    "momentum_slope_reversal",
    "recovery_above_ma_10d",
    # Stock states.
    "log_lagged_market_cap",
    "beta_252d",
    "realized_volatility_20d",
    "realized_volatility_60d",
    "idiosyncratic_volatility_60d",
    "momentum_126_21d",
    # Liquidity states.
    "log_adv20_usd",
    "turnover_20d",
    "amihud_illiquidity_20d",
    "zero_return_frequency_20d",
    "log_price",
    "quoted_spread_proxy",
    # Attention/crowding states.
    "abnormal_turnover_20d",
    "extreme_absolute_return",
    "maximum_daily_return_20d",
    "volatility_expansion_20_60d",
    # Market states.
    "market_return_20d",
    "market_return_60d",
    "market_realized_volatility_20d",
    "market_drawdown_126d",
    "cross_sectional_return_dispersion",
    "market_breadth_20d",
    # Controls.
    "industry_code_2digit",
    "exchange",
]

CATEGORICAL_FEATURE_COLUMNS = ["industry_code_2digit", "exchange"]
CONTINUOUS_FEATURE_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURE_COLUMNS
]

SECURITY_REQUIRED_COLUMNS = {
    "permno",
    "date",
    "session_id",
    "return_total",
    "return_ex_distribution",
    "price_raw",
    "volume",
    "shares_outstanding_source",
    "bid_raw",
    "ask_raw",
    "siccd",
    "primary_exchange",
}

MARKET_PUBLIC_COLUMNS = [
    "market_return_20d",
    "market_return_60d",
    "market_realized_volatility_20d",
    "market_drawdown_126d",
    "cross_sectional_return_dispersion",
    "market_breadth_20d",
]


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    context: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric))


def _rolling_sums(
    values: pd.Series,
    keys: pd.Series,
    sessions: pd.Series,
    windows: Iterable[int],
) -> dict[int, pd.Series]:
    """Exact consecutive-window sums with no missing or infinite members."""

    numeric = _finite_series(values)
    finite = numeric.notna()
    filled = numeric.fillna(0.0)
    cumulative = filled.groupby(keys, sort=False, observed=True).cumsum()
    counts = (
        finite.astype(np.int32)
        .groupby(keys, sort=False, observed=True)
        .cumsum()
    )
    position = (
        keys.groupby(keys, sort=False, observed=True).cumcount().astype(np.int32)
        + 1
    )
    grouped_cumulative = cumulative.groupby(
        keys, sort=False, observed=True
    )
    grouped_counts = counts.groupby(keys, sort=False, observed=True)
    grouped_sessions = sessions.groupby(keys, sort=False, observed=True)
    results: dict[int, pd.Series] = {}
    for window in sorted(set(int(value) for value in windows)):
        prior_sum = grouped_cumulative.shift(window).fillna(0.0)
        prior_count = grouped_counts.shift(window).fillna(0)
        window_sum = cumulative - prior_sum
        window_count = counts - prior_count
        first_session = grouped_sessions.shift(window - 1)
        consecutive = sessions.sub(first_session).eq(window - 1)
        valid = position.ge(window) & window_count.eq(window) & consecutive
        results[window] = window_sum.where(valid)
    return results


def _rolling_max(
    values: pd.Series,
    keys: pd.Series,
    sessions: pd.Series,
    window: int,
) -> pd.Series:
    numeric = _finite_series(values)
    rolled = (
        numeric.groupby(keys, sort=False, observed=True)
        .rolling(window, min_periods=window)
        .max()
        .reset_index(level=0, drop=True)
        .reindex(numeric.index)
    )
    first_session = sessions.groupby(
        keys, sort=False, observed=True
    ).shift(window - 1)
    return rolled.where(sessions.sub(first_session).eq(window - 1))


def _rolling_mean(
    values: pd.Series,
    keys: pd.Series,
    sessions: pd.Series,
    window: int,
) -> pd.Series:
    return _rolling_sums(values, keys, sessions, [window])[window] / window


def _variance_from_sums(
    sums: pd.Series,
    squared_sums: pd.Series,
    window: int,
) -> pd.Series:
    mean = sums / window
    variance = squared_sums / window - mean * mean
    return variance.clip(lower=0.0)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.where(denominator.gt(0))
    return result.where(np.isfinite(result))


def _calendar_rolling_sum(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=window).sum()


def compute_market_features(
    calendar: pd.DataFrame,
    cross_sectional: pd.DataFrame,
) -> pd.DataFrame:
    """Create continuous market states without using forward observations."""

    _require_columns(
        calendar,
        {"session_id", "date", "market_vw_return"},
        context="market calendar",
    )
    _require_columns(
        cross_sectional,
        {"date", "cross_sectional_return_dispersion", "market_breadth"},
        context="cross-sectional market input",
    )
    work = calendar[["session_id", "date", "market_vw_return"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work = work.sort_values("session_id", kind="stable").reset_index(drop=True)
    expected = np.arange(len(work), dtype=np.int64)
    if not np.array_equal(work["session_id"].to_numpy(), expected):
        raise ValueError("Market calendar session_id must be contiguous from zero")

    market_return = _finite_series(work["market_vw_return"])
    safe_log = np.log1p(market_return.where(market_return.gt(-1.0)))
    for window in (20, 60, 126):
        work[f"_market_log_sum_{window}d"] = _calendar_rolling_sum(
            safe_log, window
        )
    work["market_return_20d"] = np.expm1(work["_market_log_sum_20d"])
    work["market_return_60d"] = np.expm1(work["_market_log_sum_60d"])

    sum20 = _calendar_rolling_sum(market_return, 20)
    sumsq20 = _calendar_rolling_sum(market_return * market_return, 20)
    variance20 = _variance_from_sums(sum20, sumsq20, 20)
    work["market_realized_volatility_20d"] = np.sqrt(variance20 * 252.0)

    log_index = safe_log.fillna(0.0).cumsum()
    price_index = np.exp(log_index)
    high126 = price_index.rolling(126, min_periods=126).max()
    work["market_drawdown_126d"] = _safe_ratio(
        price_index, high126
    ) - 1.0

    cross = cross_sectional.copy()
    cross["date"] = pd.to_datetime(cross["date"], errors="raise")
    work = work.merge(
        cross[
            [
                "date",
                "cross_sectional_return_dispersion",
                "market_breadth",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
    work["market_breadth_20d"] = work["market_breadth"].rolling(
        20, min_periods=20
    ).mean()
    return work[
        [
            "session_id",
            "date",
            "market_vw_return",
            "_market_log_sum_20d",
            "_market_log_sum_60d",
            "_market_log_sum_126d",
            *MARKET_PUBLIC_COLUMNS,
        ]
    ]


def compute_security_features(
    frame: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute security features on a lookback window ending at formation day."""

    _require_columns(
        frame, SECURITY_REQUIRED_COLUMNS, context="security feature input"
    )
    _require_columns(
        market_features,
        {
            "session_id",
            "market_vw_return",
            "_market_log_sum_20d",
            "_market_log_sum_60d",
            "_market_log_sum_126d",
            *MARKET_PUBLIC_COLUMNS,
        },
        context="market feature input",
    )
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    work = work.sort_values(
        ["permno", "session_id"], kind="stable"
    ).reset_index(drop=True)
    if work.duplicated(["permno", "session_id"]).any():
        raise ValueError("Security input contains duplicate security-sessions")

    keys = work["permno"]
    sessions = work["session_id"].astype(np.int64)
    total_return = _finite_series(work["return_total"])
    price_return = _finite_series(work["return_ex_distribution"])
    total_log = np.log1p(total_return.where(total_return.gt(-1.0)))
    price_log = np.log1p(price_return.where(price_return.gt(-1.0)))

    market_lookup = market_features.set_index("session_id")
    market_return = sessions.map(market_lookup["market_vw_return"])
    market_log_sums = {
        window: sessions.map(
            market_lookup[f"_market_log_sum_{window}d"]
        )
        for window in (20, 60, 126)
    }
    total_log_sums = _rolling_sums(
        total_log, keys, sessions, [5, 20, 21, 60, 126]
    )
    total_sums = _rolling_sums(
        total_return, keys, sessions, [20, 60, 252]
    )
    total_squared_sums = _rolling_sums(
        total_return * total_return, keys, sessions, [20, 60, 252]
    )

    output = work[
        [
            "permno",
            "date",
            "session_id",
            "siccd",
            "primary_exchange",
        ]
    ].copy()
    for window in (20, 60, 126):
        output[f"rs_excess_return_{window}d"] = np.expm1(
            total_log_sums[window] - market_log_sums[window]
        )
    output["rs_acceleration_20_minus_60d"] = (
        output["rs_excess_return_20d"]
        - output["rs_excess_return_60d"]
    )

    # DlyRetx is the authoritative dividend-excluded, corporate-action-aware
    # daily price return. Raw OHLC is deliberately absent from this chain.
    price_log_sums = _rolling_sums(
        price_log, keys, sessions, [10, 20, 60, 127, 252]
    )
    log_price_index = price_log.fillna(0.0).groupby(
        keys, sort=False, observed=True
    ).cumsum()
    price_index = np.exp(log_price_index)
    prior_price = price_index.groupby(
        keys, sort=False, observed=True
    ).shift(1)
    prior_high126 = _rolling_max(
        prior_price, keys, sessions, 126
    )
    prior_link = sessions.groupby(
        keys, sort=False, observed=True
    ).shift(126)
    prior_high126 = prior_high126.where(
        sessions.sub(prior_link).eq(126)
        & price_log_sums[127].notna()
    )
    output["price_distance_prior_high_126d"] = (
        _safe_ratio(price_index, prior_high126) - 1.0
    )

    ma10 = _rolling_mean(price_index, keys, sessions, 10)
    ma20 = _rolling_mean(price_index, keys, sessions, 20)
    ma60 = _rolling_mean(price_index, keys, sessions, 60)
    ma10 = ma10.where(price_log_sums[10].notna())
    ma20 = ma20.where(price_log_sums[20].notna())
    ma60 = ma60.where(price_log_sums[60].notna())
    output["price_distance_ma_20d"] = (
        _safe_ratio(price_index, ma20) - 1.0
    )
    output["recovery_above_ma_10d"] = (
        _safe_ratio(price_index, ma10) - 1.0
    )

    variance20 = _variance_from_sums(
        total_sums[20], total_squared_sums[20], 20
    )
    variance60 = _variance_from_sums(
        total_sums[60], total_squared_sums[60], 60
    )
    variance252 = _variance_from_sums(
        total_sums[252], total_squared_sums[252], 252
    )
    daily_vol60 = np.sqrt(variance60)
    output["realized_volatility_20d"] = np.sqrt(variance20 * 252.0)
    output["realized_volatility_60d"] = np.sqrt(variance60 * 252.0)
    output["trend_distance_vol_normalized_60d"] = _safe_ratio(
        _safe_ratio(price_index, ma60) - 1.0,
        daily_vol60 * np.sqrt(60.0),
    )

    return5 = np.expm1(total_log_sums[5])
    return20 = np.expm1(total_log_sums[20])
    output["reversal_5d"] = -return5
    output["momentum_slope_reversal"] = return5 - return20 / 4.0
    output["momentum_126_21d"] = np.expm1(
        total_log_sums[126] - total_log_sums[21]
    )

    for window, column in [
        (20, "drawdown_20d"),
        (60, "drawdown_60d"),
        (252, "distance_52w_high"),
    ]:
        rolling_high = _rolling_max(price_index, keys, sessions, window)
        rolling_high = rolling_high.where(price_log_sums[window].notna())
        output[column] = _safe_ratio(price_index, rolling_high) - 1.0

    market_sum60 = _rolling_sums(
        market_return, keys, sessions, [60, 252]
    )
    market_squared_sum60 = _rolling_sums(
        market_return * market_return, keys, sessions, [60, 252]
    )
    cross_sum60 = _rolling_sums(
        total_return * market_return, keys, sessions, [60, 252]
    )
    for window in (60, 252):
        mean_stock = total_sums[window] / window
        mean_market = market_sum60[window] / window
        covariance = (
            cross_sum60[window] / window - mean_stock * mean_market
        )
        market_variance = _variance_from_sums(
            market_sum60[window],
            market_squared_sum60[window],
            window,
        )
        if window == 252:
            output["beta_252d"] = _safe_ratio(
                covariance, market_variance
            )
        else:
            residual_variance = (
                variance60 - covariance * covariance / market_variance
            ).clip(lower=0.0)
            output["idiosyncratic_volatility_60d"] = np.sqrt(
                residual_variance * 252.0
            )

    price = _finite_series(work["price_raw"]).abs()
    volume = _finite_series(work["volume"])
    shares = _finite_series(work["shares_outstanding_source"]) * 1_000.0
    turnover = _safe_ratio(volume, shares)
    dollar_volume = price * volume
    output["turnover_20d"] = _rolling_mean(
        turnover, keys, sessions, 20
    )
    amihud = _safe_ratio(total_return.abs(), dollar_volume) * 1_000_000.0
    output["amihud_illiquidity_20d"] = _rolling_mean(
        amihud, keys, sessions, 20
    )
    zero_return = total_return.eq(0.0).astype(float).where(
        total_return.notna()
    )
    output["zero_return_frequency_20d"] = _rolling_mean(
        zero_return, keys, sessions, 20
    )
    output["log_price"] = np.log(price.where(price.gt(0)))

    bid = _finite_series(work["bid_raw"])
    ask = _finite_series(work["ask_raw"])
    midpoint = (bid + ask) / 2.0
    valid_quote = bid.gt(0) & ask.ge(bid) & midpoint.gt(0)
    output["quoted_spread_proxy"] = (
        (ask - bid) / midpoint
    ).where(valid_quote)

    prior_turnover = turnover.groupby(
        keys, sort=False, observed=True
    ).shift(1)
    prior_turnover_mean20 = _rolling_mean(
        prior_turnover, keys, sessions, 20
    )
    prior_turnover_link = sessions.groupby(
        keys, sort=False, observed=True
    ).shift(20)
    prior_turnover_mean20 = prior_turnover_mean20.where(
        sessions.sub(prior_turnover_link).eq(20)
    )
    turnover_ratio = _safe_ratio(turnover, prior_turnover_mean20)
    output["abnormal_turnover_20d"] = np.log(
        turnover_ratio.where(turnover_ratio.gt(0))
    )

    sum61 = _rolling_sums(
        total_return, keys, sessions, [61]
    )[61]
    squared_sum61 = _rolling_sums(
        total_return * total_return, keys, sessions, [61]
    )[61]
    prior_sum60 = sum61 - total_return
    prior_squared_sum60 = squared_sum61 - total_return * total_return
    prior_variance60 = _variance_from_sums(
        prior_sum60, prior_squared_sum60, 60
    )
    output["extreme_absolute_return"] = _safe_ratio(
        total_return.abs(), np.sqrt(prior_variance60)
    )
    output["maximum_daily_return_20d"] = _rolling_max(
        total_return, keys, sessions, 20
    )
    output["volatility_expansion_20_60d"] = _safe_ratio(
        output["realized_volatility_20d"],
        output["realized_volatility_60d"],
    ) - 1.0

    for column in MARKET_PUBLIC_COLUMNS:
        output[column] = sessions.map(market_lookup[column])

    sic = pd.to_numeric(work["siccd"], errors="coerce")
    industry = np.floor(sic / 100.0)
    output["industry_code_2digit"] = industry.where(
        industry.between(0, 99, inclusive="both")
    )
    output["exchange"] = work["primary_exchange"].where(
        work["primary_exchange"].isin(["N", "Q", "A"])
    )

    numeric_columns = [
        column
        for column in output.columns
        if column not in {"date", "exchange"}
    ]
    for column in numeric_columns:
        if column in {"permno", "session_id", "siccd"}:
            continue
        output[column] = _finite_series(output[column])
    return output


def finalize_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach target-only universe states and same-day primary-universe rank."""

    _require_columns(
        frame,
        {
            *OUTPUT_IDENTIFIER_COLUMNS,
            "lag_market_cap_usd",
            "lag_adv20_usd",
            "rs_excess_return_60d",
        },
        context="target feature frame",
    )
    output = frame.copy()
    lagged_cap = _finite_series(output["lag_market_cap_usd"])
    lagged_adv = _finite_series(output["lag_adv20_usd"])
    output["log_lagged_market_cap"] = np.log(
        lagged_cap.where(lagged_cap.gt(0))
    )
    output["log_adv20_usd"] = np.log(lagged_adv.where(lagged_adv.gt(0)))
    output["rs_percentile_60d"] = np.nan
    primary = output["universe_primary"].astype(bool)
    eligible = primary & output["rs_excess_return_60d"].notna()
    ranked = output.loc[eligible].groupby(
        "date", sort=False, observed=True
    )["rs_excess_return_60d"].rank(method="average", pct=True)
    output.loc[eligible, "rs_percentile_60d"] = ranked
    return output[[*OUTPUT_IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]]
