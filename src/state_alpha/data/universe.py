"""Point-in-time universe variables and frozen-rule evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_UNIVERSE_COLUMNS = {
    "permno",
    "date",
    "session_id",
    "primary_exchange",
    "trading_status",
    "security_type",
    "security_subtype",
    "share_type",
    "us_incorporated_flag",
    "price_raw",
    "market_cap_source",
    "volume",
    "shares_outstanding_source",
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required universe columns: {missing}")


def add_point_in_time_universe_columns(
    frame: pd.DataFrame,
    *,
    minimum_history: int = 252,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Add lagged eligibility variables without using formation-day filters.

    The input must contain one row per security-session and be sorted by
    ``(permno, session_id)``. Price, capitalization, dollar volume, security
    descriptors, and exchange eligibility are shifted one security observation.
    The previous observation must also be the immediately preceding market
    session, so stale records cannot pass merely because they are adjacent.
    """

    _require_columns(frame, REQUIRED_UNIVERSE_COLUMNS)
    work = frame.copy()
    if not work[["permno", "session_id"]].equals(
        work[["permno", "session_id"]].sort_values(
            ["permno", "session_id"], kind="stable"
        )
    ):
        raise ValueError("Universe input must be sorted by (permno, session_id)")
    if work.duplicated(["permno", "session_id"]).any():
        raise ValueError("Universe input contains duplicate security-sessions")

    price = pd.to_numeric(work["price_raw"], errors="coerce").abs()
    cap_source = pd.to_numeric(work["market_cap_source"], errors="coerce")
    volume = pd.to_numeric(work["volume"], errors="coerce")
    shares_source = pd.to_numeric(
        work["shares_outstanding_source"], errors="coerce"
    )
    source_base = (
        work["security_type"].eq("EQTY")
        & work["security_subtype"].eq("COM")
        & work["share_type"].eq("NS")
        & work["primary_exchange"].isin(["N", "Q", "A"])
        & work["trading_status"].eq("A")
    )
    current_tradable = (
        work["primary_exchange"].isin(["N", "Q", "A"])
        & work["trading_status"].eq("A")
        & price.gt(0)
    )
    daily_dollar_volume = price * volume
    turnover = volume / (shares_source * 1_000.0)

    work["_source_base"] = source_base
    work["_source_us_incorporated"] = work["us_incorporated_flag"].eq("Y")
    work["_price_usd"] = price
    work["_market_cap_usd"] = cap_source * 1_000.0
    work["_daily_dollar_volume_usd"] = daily_dollar_volume
    work["_turnover"] = turnover
    work["formation_tradable"] = current_tradable

    grouped = work.groupby("permno", sort=False, observed=True)
    work["lag_security_base"] = (
        grouped["_source_base"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    work["lag_us_incorporated"] = (
        grouped["_source_us_incorporated"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    work["lag_price_usd"] = grouped["_price_usd"].shift(1)
    work["lag_market_cap_usd"] = grouped["_market_cap_usd"].shift(1)
    work["lag_dollar_volume_usd"] = grouped[
        "_daily_dollar_volume_usd"
    ].shift(1)
    work["lag_turnover"] = grouped["_turnover"].shift(1)
    work["prior_observation_count"] = grouped.cumcount().astype("int32")

    lag_session_1 = grouped["session_id"].shift(1)
    lag_session_window = grouped["session_id"].shift(adv_window)
    work["lag_is_previous_market_session"] = (
        lag_session_1.eq(work["session_id"] - 1)
    )
    work["adv_window_consecutive"] = (
        lag_session_1 - lag_session_window
    ).eq(adv_window - 1)

    lagged_dollar_volume = work["lag_dollar_volume_usd"]
    work["lag_adv20_usd"] = (
        lagged_dollar_volume.groupby(work["permno"], sort=False, observed=True)
        .rolling(adv_window, min_periods=adv_window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    work["universe_analysis_base"] = (
        work["formation_tradable"]
        & work["lag_security_base"]
        & work["lag_is_previous_market_session"]
        & work["adv_window_consecutive"]
        & work["prior_observation_count"].ge(minimum_history)
        & work["lag_price_usd"].gt(0)
        & work["lag_market_cap_usd"].gt(0)
        & work["lag_adv20_usd"].gt(0)
        & np.isfinite(work["lag_turnover"])
        & work["lag_turnover"].ge(0)
    )

    work["lag_market_cap_pct_all"] = np.nan
    work["lag_adv20_pct_all"] = np.nan
    work["lag_market_cap_pct_us"] = np.nan
    work["lag_adv20_pct_us"] = np.nan
    base = work["universe_analysis_base"]
    us_base = base & work["lag_us_incorporated"]
    for mask, suffix in [(base, "all"), (us_base, "us")]:
        eligible = work.loc[mask]
        work.loc[mask, f"lag_market_cap_pct_{suffix}"] = eligible.groupby(
            "date", sort=False, observed=True
        )["lag_market_cap_usd"].rank(method="average", pct=True)
        work.loc[mask, f"lag_adv20_pct_{suffix}"] = eligible.groupby(
            "date", sort=False, observed=True
        )["lag_adv20_usd"].rank(method="average", pct=True)

    return work.drop(
        columns=[
            "_source_base",
            "_source_us_incorporated",
            "_price_usd",
            "_market_cap_usd",
            "_daily_dollar_volume_usd",
            "_turnover",
        ]
    )


def apply_universe_rule(
    frame: pd.DataFrame,
    rule: dict[str, Any],
) -> pd.Series:
    """Evaluate an absolute or date-percentile rule on lagged columns."""

    required = {
        "universe_analysis_base",
        "lag_us_incorporated",
        "lag_price_usd",
        "lag_market_cap_usd",
        "lag_adv20_usd",
    }
    _require_columns(frame, required)
    mask = frame["universe_analysis_base"].astype(bool).copy()
    require_us = bool(rule.get("require_us_incorporated", False))
    if require_us:
        mask &= frame["lag_us_incorporated"]

    kind = rule.get("kind")
    minimum_price = float(rule.get("minimum_price_usd", 0.0))
    mask &= frame["lag_price_usd"].ge(minimum_price)
    if kind == "absolute":
        mask &= frame["lag_market_cap_usd"].ge(
            float(rule["minimum_market_cap_usd"])
        )
        mask &= frame["lag_adv20_usd"].ge(float(rule["minimum_adv20_usd"]))
    elif kind == "percentile":
        suffix = "us" if require_us else "all"
        cap_column = f"lag_market_cap_pct_{suffix}"
        adv_column = f"lag_adv20_pct_{suffix}"
        _require_columns(frame, {cap_column, adv_column})
        mask &= frame[cap_column].gt(
            float(rule["exclude_bottom_market_cap_percent"]) / 100.0
        )
        mask &= frame[adv_column].gt(
            float(rule["exclude_bottom_adv20_percent"]) / 100.0
        )
    else:
        raise ValueError(f"Unknown universe rule kind: {kind!r}")
    return mask.fillna(False)


def summarize_rule(
    frame: pd.DataFrame,
    rule_id: str,
    mask: pd.Series,
    *,
    year: int,
) -> dict[str, Any]:
    """Create development-only breadth and economic-scale diagnostics."""

    dates = pd.to_datetime(frame["date"], errors="coerce")
    selected = frame.loc[mask].copy()
    daily = selected.groupby(dates.loc[mask], observed=True).size()
    base_rows = int(frame["universe_analysis_base"].sum())
    result: dict[str, Any] = {
        "year": year,
        "rule_id": rule_id,
        "analysis_base_rows": base_rows,
        "eligible_rows": int(mask.sum()),
        "eligible_share_of_analysis_base": (
            float(mask.sum() / base_rows) if base_rows else float("nan")
        ),
        "unique_permnos": int(selected["permno"].nunique()),
        "daily_names_p10": float(daily.quantile(0.10)) if len(daily) else np.nan,
        "daily_names_median": float(daily.median()) if len(daily) else np.nan,
        "daily_names_p90": float(daily.quantile(0.90)) if len(daily) else np.nan,
        "daily_names_min": int(daily.min()) if len(daily) else 0,
    }
    for column in [
        "lag_price_usd",
        "lag_market_cap_usd",
        "lag_adv20_usd",
        "lag_turnover",
    ]:
        values = pd.to_numeric(selected[column], errors="coerce")
        result[f"{column}_p10"] = float(values.quantile(0.10))
        result[f"{column}_median"] = float(values.median())
    return result
