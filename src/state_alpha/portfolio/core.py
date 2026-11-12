"""Pure Gate 6 portfolio construction, return-path, and summary helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from state_alpha.models.gate5 import SCORE_COLUMNS
from state_alpha.models.statistics import newey_west_mean


INDUSTRY_DIVISIONS = [f"industry_division_{value}" for value in range(10)]
INDUSTRY_DIVISIONS.append("industry_division_missing")
EXPOSURE_COLUMNS = [
    "beta_raw",
    "beta_z",
    "size_z",
    "price_z",
    "adv_z",
    "amihud_z",
    "spread_z",
]


def variant_catalog(contract: dict[str, Any]) -> pd.DataFrame:
    """Return the fully preregistered, deterministic portfolio catalog."""

    records: list[dict[str, str]] = []
    main = contract["portfolio_variants"]["main"]
    for signal in contract["scores"]:
        for weighting in main["weighting"]:
            for neutralization in main["neutralization"]:
                records.append(
                    {
                        "variant_id": (
                            f"{signal}__{weighting}__"
                            f"{neutralization}__all"
                        ),
                        "signal": signal,
                        "weighting": weighting,
                        "neutralization": neutralization,
                        "bucket": "all",
                    }
                )
    buckets = [
        *contract["portfolio_variants"]["liquidity_buckets"][
            "adv_terciles"
        ],
        *contract["portfolio_variants"]["liquidity_buckets"][
            "amihud_terciles"
        ],
    ]
    for signal in contract["scores"]:
        for bucket in buckets:
            records.append(
                {
                    "variant_id": (
                        f"{signal}__equal_weight__raw__{bucket}"
                    ),
                    "signal": signal,
                    "weighting": "equal_weight",
                    "neutralization": "raw",
                    "bucket": bucket,
                }
            )
    output = pd.DataFrame(records)
    if list(contract["scores"]) != SCORE_COLUMNS:
        raise ValueError("Gate 6 scores drifted from Gate 5")
    if output["variant_id"].duplicated().any():
        raise ValueError("Gate 6 variant ids are not unique")
    return output


def _filled_zscore(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    if not finite.any():
        return np.zeros(len(numeric), dtype=float)
    median = float(np.nanmedian(numeric))
    filled = np.where(finite, numeric, median)
    standard_deviation = float(filled.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        return np.zeros(len(numeric), dtype=float)
    return (filled - float(filled.mean())) / standard_deviation


def exposure_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build formation-time continuous, industry, and constraint matrices."""

    beta_raw = pd.to_numeric(
        frame["beta_252d"], errors="coerce"
    ).to_numpy(dtype=float)
    beta_median = (
        float(np.nanmedian(beta_raw))
        if np.isfinite(beta_raw).any()
        else 0.0
    )
    beta_raw = np.where(np.isfinite(beta_raw), beta_raw, beta_median)
    beta_z = _filled_zscore(frame["beta_252d"])
    size_z = _filled_zscore(frame["log_lagged_market_cap"])
    continuous = np.column_stack(
        [
            beta_raw,
            beta_z,
            size_z,
            _filled_zscore(frame["log_price"]),
            _filled_zscore(frame["log_adv20_usd"]),
            _filled_zscore(frame["amihud_illiquidity_20d"]),
            _filled_zscore(frame["quoted_spread_proxy"]),
        ]
    )

    industry = pd.to_numeric(
        frame["industry_code_2digit"], errors="coerce"
    ).to_numpy(dtype=float)
    division = np.full(len(frame), 10, dtype=np.int64)
    valid = np.isfinite(industry) & (industry >= 0) & (industry <= 99)
    division[valid] = (
        industry[valid].astype(np.int64) // 10
    )
    rows = np.arange(len(frame), dtype=np.int64)
    industry_matrix = np.zeros((len(frame), 11), dtype=float)
    industry_matrix[rows, division] = 1.0
    constraints = np.column_stack(
        [beta_z, size_z, industry_matrix]
    )
    return continuous, industry_matrix, constraints


def bucket_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Assign same-date ADV and Amihud terciles."""

    adv_rank = frame["log_adv20_usd"].rank(
        method="average", pct=True
    ).to_numpy(dtype=float)
    amihud_rank = frame["amihud_illiquidity_20d"].rank(
        method="average", pct=True
    ).to_numpy(dtype=float)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "adv_low": np.isfinite(adv_rank) & (adv_rank <= 1.0 / 3.0),
        "adv_mid": (
            np.isfinite(adv_rank)
            & (adv_rank > 1.0 / 3.0)
            & (adv_rank <= 2.0 / 3.0)
        ),
        "adv_high": np.isfinite(adv_rank) & (adv_rank > 2.0 / 3.0),
        "amihud_liquid": (
            np.isfinite(amihud_rank) & (amihud_rank <= 1.0 / 3.0)
        ),
        "amihud_mid": (
            np.isfinite(amihud_rank)
            & (amihud_rank > 1.0 / 3.0)
            & (amihud_rank <= 2.0 / 3.0)
        ),
        "amihud_illiquid": (
            np.isfinite(amihud_rank) & (amihud_rank > 2.0 / 3.0)
        ),
    }


def raw_long_short_weights(
    scores: pd.Series,
    *,
    weighting: str,
    eligible: np.ndarray,
    zero_is_cash: bool,
) -> tuple[np.ndarray, int, int]:
    """Construct one gross, zero net top-minus-bottom decile weights."""

    values = pd.to_numeric(scores, errors="coerce")
    valid = eligible & values.notna().to_numpy()
    ranks = pd.Series(
        np.where(valid, values.to_numpy(dtype=float), np.nan),
        index=scores.index,
    ).rank(method="average", pct=True)
    rank_values = ranks.to_numpy(dtype=float)
    numeric = values.to_numpy(dtype=float)
    long_mask = valid & (rank_values > 0.90)
    short_mask = valid & (rank_values <= 0.10)
    if zero_is_cash:
        long_mask &= numeric > 0
        short_mask &= numeric < 0
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    if long_count == 0 or short_count == 0:
        raise ValueError("Portfolio decile has an empty long or short leg")

    weights = np.zeros(len(scores), dtype=float)
    if weighting == "equal_weight":
        weights[long_mask] = 0.5 / long_count
        weights[short_mask] = -0.5 / short_count
    elif weighting == "rank_weight":
        long_strength = np.maximum(rank_values[long_mask] - 0.90, 0.0)
        short_strength = np.maximum(0.10 - rank_values[short_mask], 0.0)
        if long_strength.sum() <= 0 or short_strength.sum() <= 0:
            raise ValueError("Rank-weighted decile has zero strength")
        weights[long_mask] = 0.5 * long_strength / long_strength.sum()
        weights[short_mask] = -0.5 * short_strength / short_strength.sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    return weights, long_count, short_count


def neutralize_weights(
    weights: np.ndarray,
    constraints: np.ndarray,
) -> np.ndarray:
    """Project within the selected support and restore one gross."""

    output = np.asarray(weights, dtype=float).copy()
    support = np.flatnonzero(np.abs(output) > 0)
    if len(support) < 3:
        raise ValueError("Neutralization support is too small")
    design = constraints[support]
    fitted = design @ np.linalg.lstsq(
        design, output[support], rcond=None
    )[0]
    output[support] -= fitted
    gross = float(np.abs(output).sum())
    if not np.isfinite(gross) or gross <= 1e-10:
        raise ValueError("Neutralization removed the entire portfolio")
    output /= gross
    return output


def weight_matrix(
    frame: pd.DataFrame,
    variants: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Construct every frozen variant for one formation date."""

    masks = bucket_masks(frame)
    continuous, industries, constraints = exposure_matrix(frame)
    zero_is_cash = set(contract["construction"]["zero_is_cash_signals"])
    raw_cache: dict[tuple[str, str, str], tuple[np.ndarray, int, int]] = {}
    rows: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for variant in variants.itertuples(index=False):
        key = (variant.signal, variant.weighting, variant.bucket)
        if key not in raw_cache:
            raw_cache[key] = raw_long_short_weights(
                frame[variant.signal],
                weighting=variant.weighting,
                eligible=masks[variant.bucket],
                zero_is_cash=variant.signal in zero_is_cash,
            )
        raw, long_count, short_count = raw_cache[key]
        weights = raw
        if variant.neutralization == "risk_neutralized":
            weights = neutralize_weights(raw, constraints)
        rows.append(weights)
        constraint_exposure = weights @ constraints
        diagnostics.append(
            {
                "variant_id": variant.variant_id,
                "long_names": long_count,
                "short_names": short_count,
                "prefill_net_weight": float(weights.sum()),
                "prefill_gross_weight": float(np.abs(weights).sum()),
                "prefill_max_abs_constraint": float(
                    np.abs(constraint_exposure).max()
                ),
            }
        )
    return (
        np.vstack(rows),
        pd.DataFrame(diagnostics),
        continuous,
        industries,
    )


def realized_return_paths(
    permnos: np.ndarray,
    fill_mask: np.ndarray,
    holding_sessions: list[int],
    returns_by_session: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create conservative and zero-bound daily paths without future filtering."""

    count = len(permnos)
    horizon = len(holding_sessions)
    lower = np.zeros((count, horizon), dtype=float)
    upper = np.zeros((count, horizon), dtype=float)
    active_before = np.zeros((count, horizon), dtype=bool)
    alive = np.asarray(fill_mask, dtype=bool).copy()
    for column, session_id in enumerate(holding_sessions):
        active_before[:, column] = alive
        daily = returns_by_session.get(session_id)
        if daily is None:
            values = np.full(count, np.nan)
            terminal = np.zeros(count, dtype=bool)
        else:
            aligned = daily.reindex(permnos)
            values = pd.to_numeric(
                aligned["return_total"], errors="coerce"
            ).to_numpy(dtype=float)
            terminal = aligned["daily_delisting_flag"].eq(
                "Y"
            ).to_numpy(dtype=bool)
        valid = np.isfinite(values) & (values >= -1.0)
        observed = alive & valid
        missing = alive & ~valid
        lower[observed, column] = values[observed]
        upper[observed, column] = values[observed]
        lower[missing, column] = -1.0
        upper[missing, column] = 0.0
        alive &= ~(missing | (terminal & observed))
    return lower, upper, alive, active_before


def turnover_between(
    entry_permnos: np.ndarray,
    entry_weights: np.ndarray,
    exit_permnos: np.ndarray | None = None,
    exit_weights: np.ndarray | None = None,
) -> np.ndarray:
    """One-way turnover after security-level netting at one close."""

    if exit_permnos is None or exit_weights is None:
        return 0.5 * np.abs(entry_weights).sum(axis=1)
    union = np.union1d(entry_permnos, exit_permnos)
    delta = np.zeros((entry_weights.shape[0], len(union)), dtype=float)
    delta[:, np.searchsorted(union, entry_permnos)] += entry_weights
    delta[:, np.searchsorted(union, exit_permnos)] -= exit_weights
    return 0.5 * np.abs(delta).sum(axis=1)


def performance_summary(
    daily: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
    periods: dict[str, tuple[int, int]],
    hac_lags: int,
    trading_days: int,
) -> pd.DataFrame:
    """Summarize daily portfolio returns over predeclared periods."""

    records: list[dict[str, Any]] = []
    for period, (start_year, end_year) in periods.items():
        sample = daily.loc[
            daily["year"].between(start_year, end_year)
        ]
        for keys, group in sample.groupby(
            group_columns, sort=False, observed=True
        ):
            key_values = keys if isinstance(keys, tuple) else (keys,)
            record = dict(zip(group_columns, key_values, strict=True))
            values = pd.to_numeric(
                group[value_column], errors="coerce"
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            stats = newey_west_mean(values, hac_lags)
            annualized_return = (
                float(values.mean() * trading_days)
                if len(values)
                else np.nan
            )
            annualized_volatility = (
                float(values.std(ddof=1) * np.sqrt(trading_days))
                if len(values) > 1
                else np.nan
            )
            sharpe = (
                annualized_return / annualized_volatility
                if np.isfinite(annualized_volatility)
                and annualized_volatility > 0
                else np.nan
            )
            wealth = (
                np.cumprod(1.0 + values)
                if len(values)
                else np.asarray([], dtype=float)
            )
            running_peak = np.maximum.accumulate(wealth)
            drawdown = (
                wealth / running_peak - 1.0
                if len(wealth)
                else np.asarray([], dtype=float)
            )
            record.update(
                {
                    "period": period,
                    **stats,
                    "annualized_return": annualized_return,
                    "annualized_volatility": annualized_volatility,
                    "sharpe": sharpe,
                    "maximum_drawdown": (
                        float(drawdown.min()) if len(drawdown) else np.nan
                    ),
                    "cumulative_return": (
                        float(wealth[-1] - 1.0)
                        if len(wealth)
                        else np.nan
                    ),
                }
            )
            records.append(record)
    return pd.DataFrame(records)
