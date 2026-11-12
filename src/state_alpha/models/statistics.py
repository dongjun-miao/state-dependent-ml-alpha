"""Transparent Gate 4 IC, sort, double-sort, and Fama-MacBeth statistics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from state_alpha.models.matrix import FAMILY_COLUMNS, period_name


def newey_west_mean(
    values: Iterable[float],
    lags: int = 20,
) -> dict[str, float | int]:
    """Bartlett-kernel HAC inference for the mean of an ordered series."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    count = len(array)
    if count == 0:
        return {
            "observations": 0,
            "mean": np.nan,
            "standard_deviation": np.nan,
            "newey_west_standard_error": np.nan,
            "newey_west_t": np.nan,
        }
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if count > 1 else np.nan
    centered = array - mean
    long_variance = float(np.dot(centered, centered) / count)
    maximum_lag = min(int(lags), count - 1)
    for lag in range(1, maximum_lag + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
        weight = 1.0 - lag / (maximum_lag + 1.0)
        long_variance += 2.0 * weight * covariance
    long_variance = max(long_variance, 0.0)
    standard_error = float(np.sqrt(long_variance / count))
    t_value = mean / standard_error if standard_error > 0 else np.nan
    return {
        "observations": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "newey_west_standard_error": standard_error,
        "newey_west_t": t_value,
    }


def _correlation_by_group(
    frame: pd.DataFrame,
    x_columns: list[str],
    y_column: str,
    group_column: str = "date",
) -> pd.DataFrame:
    """Vectorized within-group correlations on a common complete-case sample."""

    columns = [group_column, y_column, *x_columns]
    work = frame[columns].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=[group_column, *x_columns])
    grouped = work.groupby(group_column, sort=False, observed=True)
    count = grouped.size().astype(float)
    sum_y = grouped[y_column].sum()
    sum_y2 = work[y_column].pow(2).groupby(
        work[group_column], sort=False, observed=True
    ).sum()
    sum_x = grouped[x_columns].sum()
    sum_x2 = work[x_columns].pow(2).groupby(
        work[group_column], sort=False, observed=True
    ).sum()
    xy = work[x_columns].mul(work[y_column], axis=0)
    sum_xy = xy.groupby(
        work[group_column], sort=False, observed=True
    ).sum()
    numerator = sum_xy.sub(
        sum_x.mul(sum_y / count, axis=0), axis=0
    )
    x_ss = sum_x2.sub(sum_x.pow(2).div(count, axis=0), axis=0)
    y_ss = sum_y2 - sum_y.pow(2) / count
    denominator = np.sqrt(x_ss.mul(y_ss, axis=0))
    correlation = numerator / denominator.where(denominator.gt(0))
    correlation.index.name = group_column
    return correlation.reset_index()


def daily_rank_ic(
    frame: pd.DataFrame,
    rank_columns: dict[str, str],
) -> pd.DataFrame:
    correlations = _correlation_by_group(
        frame,
        list(rank_columns.values()),
        "fwd_total_return_rank_10d",
    )
    reverse = {value: key for key, value in rank_columns.items()}
    output = correlations.melt(
        id_vars="date", var_name="_column", value_name="rank_ic"
    )
    output["signal"] = output["_column"].map(reverse)
    output["year"] = pd.to_datetime(output["date"]).dt.year
    return output[["date", "year", "signal", "rank_ic"]]


def _membership_turnover(
    previous: set[int] | None,
    current: set[int],
) -> float:
    if previous is None or not previous or not current:
        return np.nan
    common = len(previous.intersection(current))
    exited = len(previous) - common
    entered = len(current) - common
    return 0.5 * (
        common * abs(1.0 / len(current) - 1.0 / len(previous))
        + exited / len(previous)
        + entered / len(current)
    )


def _turnover_by_date(
    frame: pd.DataFrame,
    bin_column: str,
    quantiles: int,
) -> pd.Series:
    long_sets = (
        frame.loc[frame[bin_column].eq(quantiles)]
        .groupby("date", sort=True, observed=True)["permno"]
        .agg(lambda values: set(int(value) for value in values))
    )
    short_sets = (
        frame.loc[frame[bin_column].eq(1)]
        .groupby("date", sort=True, observed=True)["permno"]
        .agg(lambda values: set(int(value) for value in values))
    )
    dates = sorted(set(long_sets.index).intersection(short_sets.index))
    values: dict[pd.Timestamp, float] = {}
    previous_long: set[int] | None = None
    previous_short: set[int] | None = None
    for date in dates:
        current_long = long_sets.loc[date]
        current_short = short_sets.loc[date]
        long_turnover = _membership_turnover(previous_long, current_long)
        short_turnover = _membership_turnover(previous_short, current_short)
        legs = np.asarray([long_turnover, short_turnover], dtype=float)
        values[date] = (
            float(legs[np.isfinite(legs)].mean())
            if np.isfinite(legs).any()
            else np.nan
        )
        previous_long = current_long
        previous_short = current_short
    return pd.Series(values, name="one_way_turnover")


def daily_sorts(
    frame: pd.DataFrame,
    rank_columns: dict[str, str],
    quantiles_by_signal: dict[str, list[int]],
    turnover_signals: set[str],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for signal, rank_column in rank_columns.items():
        for quantiles in quantiles_by_signal[signal]:
            work = frame[
                [
                    "permno",
                    "date",
                    rank_column,
                    "fwd_total_return_10d",
                ]
            ].dropna()
            if work.empty:
                continue
            bin_column = "_sort_bin"
            work[bin_column] = np.ceil(
                work[rank_column] * quantiles
            ).clip(1, quantiles).astype(np.int8)
            grouped = work.groupby(
                ["date", bin_column], sort=False, observed=True
            )["fwd_total_return_10d"].agg(["mean", "size"])
            mean = grouped["mean"].unstack(bin_column)
            size = grouped["size"].unstack(bin_column)
            valid = mean.columns.isin([1, quantiles])
            if valid.sum() < 2:
                continue
            daily = pd.DataFrame(
                {
                    "date": mean.index,
                    "short_return": mean[1],
                    "long_return": mean[quantiles],
                    "spread_return": mean[quantiles] - mean[1],
                    "short_names": size[1],
                    "long_names": size[quantiles],
                }
            )
            daily["one_way_turnover"] = np.nan
            if signal in turnover_signals and quantiles == 10:
                turnover = _turnover_by_date(
                    work, bin_column, quantiles
                )
                daily["one_way_turnover"] = daily["date"].map(turnover)
            daily["signal"] = signal
            daily["quantiles"] = quantiles
            daily["year"] = pd.to_datetime(daily["date"]).dt.year
            outputs.append(daily)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def daily_double_sorts(
    frame: pd.DataFrame,
    state_features: list[str],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for state in state_features:
        state_rank = frame.groupby(
            "date", sort=False, observed=True
        )[state].rank(method="average", pct=True)
        state_bin = np.ceil(state_rank * 5).clip(1, 5)
        for family in FAMILY_COLUMNS:
            work = frame[
                [
                    "date",
                    "fwd_total_return_10d",
                    family,
                ]
            ].copy()
            work["_state_bin"] = state_bin
            work = work.dropna()
            if work.empty:
                continue
            work["_alpha_pct"] = work.groupby(
                ["date", "_state_bin"], sort=False, observed=True
            )[family].rank(method="average", pct=True)
            work["_alpha_bin"] = np.ceil(
                work["_alpha_pct"] * 5
            ).clip(1, 5).astype(np.int8)
            means = work.groupby(
                ["date", "_state_bin", "_alpha_bin"],
                sort=False,
                observed=True,
            )["fwd_total_return_10d"].mean().unstack("_alpha_bin")
            if 1 not in means or 5 not in means:
                continue
            daily = pd.DataFrame(
                {
                    "date": means.index.get_level_values("date"),
                    "state_bin": means.index.get_level_values("_state_bin"),
                    "spread_return": means[5] - means[1],
                }
            )
            daily["family"] = family
            daily["state_feature"] = state
            daily["year"] = pd.to_datetime(daily["date"]).dt.year
            outputs.append(daily)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def _fixed_effect_columns(frame: pd.DataFrame) -> pd.DataFrame:
    industry = pd.to_numeric(
        frame["industry_code_2digit"], errors="coerce"
    )
    division = np.floor(industry / 10.0)
    output = pd.DataFrame(index=frame.index)
    for value in range(1, 10):
        output[f"industry_division_{value}"] = division.eq(value).astype(float)
    for value in ["Q", "A"]:
        output[f"exchange_{value}"] = frame["exchange"].eq(value).astype(float)
    return output


def daily_fama_macbeth(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    specification = contract["classical_baselines"]["fama_macbeth"]
    family_columns = list(specification["family_predictors"])
    state_columns = list(specification["state_predictors"])
    control_columns = [
        value
        for value in specification["controls"]
        if value
        not in {
            "industry_division_fixed_effects",
            "exchange_fixed_effects",
        }
    ]
    rank_sources = [*family_columns, *state_columns, *control_columns]
    ranked = frame.groupby(
        "date", sort=False, observed=True
    )[rank_sources].rank(method="average", pct=True).sub(0.5)
    continuous_names = [*family_columns, *state_columns, *control_columns]
    continuous = ranked[continuous_names]
    interactions = pd.DataFrame(index=frame.index)
    for family in family_columns:
        for state in state_columns:
            interactions[f"{family}_x_{state}"] = (
                ranked[family] * ranked[state]
            )
    fixed_effects = _fixed_effect_columns(frame)
    y = pd.to_numeric(
        frame[specification["dependent"]], errors="coerce"
    )
    minimum = int(specification["minimum_daily_observations"])
    outputs: list[dict[str, Any]] = []
    for date, index in frame.groupby(
        "date", sort=True, observed=True
    ).groups.items():
        for spec_name, extra in [
            ("base", None),
            ("explicit_interactions", interactions),
        ]:
            x_parts = [continuous.loc[index], fixed_effects.loc[index]]
            if extra is not None:
                x_parts.insert(1, extra.loc[index])
            x_frame = pd.concat(x_parts, axis=1)
            valid = y.loc[index].notna() & x_frame.notna().all(axis=1)
            if int(valid.sum()) < minimum:
                continue
            x_valid = x_frame.loc[valid].to_numpy(dtype=float)
            x_valid = np.column_stack(
                [np.ones(len(x_valid), dtype=float), x_valid]
            )
            y_valid = y.loc[index].loc[valid].to_numpy(dtype=float)
            coefficients, _, rank, singular = np.linalg.lstsq(
                x_valid, y_valid, rcond=None
            )
            terms = ["intercept", *x_frame.columns]
            condition = (
                float(singular.max() / singular.min())
                if len(singular) and singular.min() > 0
                else np.inf
            )
            for term, coefficient in zip(terms, coefficients, strict=True):
                outputs.append(
                    {
                        "date": date,
                        "year": pd.Timestamp(date).year,
                        "specification": spec_name,
                        "term": term,
                        "coefficient": coefficient,
                        "observations": int(valid.sum()),
                        "matrix_rank": int(rank),
                        "condition_number": condition,
                    }
                )
    return pd.DataFrame(outputs)


def summarize_daily(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
    hac_lags: int,
    add_icir: bool = False,
) -> pd.DataFrame:
    work = frame.copy()
    work["period"] = work["year"].map(period_name)
    outputs: list[dict[str, Any]] = []
    for keys, group in work.groupby(
        [*group_columns, "period"], sort=False, observed=True
    ):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip([*group_columns, "period"], key_values, strict=True))
        stats = newey_west_mean(group[value_column], hac_lags)
        record.update(stats)
        numeric = pd.to_numeric(
            group[value_column], errors="coerce"
        ).to_numpy(dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        record["positive_fraction"] = (
            float((numeric > 0).mean()) if len(numeric) else np.nan
        )
        if add_icir:
            standard_deviation = stats["standard_deviation"]
            record["annualized_icir"] = (
                float(stats["mean"] / standard_deviation * np.sqrt(252.0))
                if pd.notna(standard_deviation) and standard_deviation > 0
                else np.nan
            )
        outputs.append(record)
    return pd.DataFrame(outputs)


def summarize_by_year(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
    hac_lags: int,
) -> pd.DataFrame:
    outputs: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        [*group_columns, "year"], sort=False, observed=True
    ):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip([*group_columns, "year"], key_values, strict=True))
        record.update(newey_west_mean(group[value_column], hac_lags))
        outputs.append(record)
    return pd.DataFrame(outputs)
