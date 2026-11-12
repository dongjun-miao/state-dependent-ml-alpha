"""Delisting-aware forward labels under a delayed t+1-close execution rule."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_LABEL_COLUMNS = {
    "permno",
    "session_id",
    "return_total",
    "daily_delisting_flag",
    "formation_tradable",
}


def _group_layout(permno: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.flatnonzero(
        np.r_[True, permno[1:] != permno[:-1]]
    ).astype(np.int64)
    ends = np.r_[starts[1:] - 1, len(permno) - 1].astype(np.int64)
    lengths = ends - starts + 1
    end_for_row = np.repeat(ends, lengths)
    return starts, ends, end_for_row


def _group_cumulative(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(values)
    offsets = np.where(starts == 0, 0, cumulative[np.maximum(starts - 1, 0)])
    return cumulative - np.repeat(offsets, lengths)


def _range_sum(
    cumulative: np.ndarray,
    end_indices: np.ndarray,
    before_start_indices: np.ndarray,
) -> np.ndarray:
    return cumulative[end_indices] - cumulative[before_start_indices]


def compute_forward_labels(
    frame: pd.DataFrame,
    market_returns: np.ndarray,
    *,
    horizons: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Compute labels from intervals after execution at t+1 close.

    For formation session ``s``, the first included CRSP daily return is stored
    at session ``s+2`` and the h-session label ends at ``s+h+1``. A valid
    delisting return terminates the stock leg and the proceeds are held as cash
    through the fixed label end. Missing delisting returns remain missing in the
    primary label; explicit -100% and 0% terminal-value bounds are returned.
    """

    missing = sorted(REQUIRED_LABEL_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required label columns: {missing}")
    work = frame
    if not work[["permno", "session_id"]].equals(
        work[["permno", "session_id"]].sort_values(
            ["permno", "session_id"], kind="stable"
        )
    ):
        raise ValueError("Label input must be sorted by (permno, session_id)")
    if work.duplicated(["permno", "session_id"]).any():
        raise ValueError("Label input contains duplicate security-sessions")

    horizons_tuple = tuple(sorted({int(value) for value in horizons}))
    if not horizons_tuple or horizons_tuple[0] <= 0:
        raise ValueError("Horizons must contain positive integers")

    permno = work["permno"].to_numpy()
    sessions = work["session_id"].to_numpy(dtype=np.int64)
    returns = pd.to_numeric(work["return_total"], errors="coerce").to_numpy(
        dtype=float
    )
    terminal = work["daily_delisting_flag"].eq("Y").to_numpy(dtype=bool)
    tradable = work["formation_tradable"].fillna(False).to_numpy(dtype=bool)
    n_rows = len(work)
    positions = np.arange(n_rows, dtype=np.int64)
    starts, ends, group_end = _group_layout(permno)
    lengths = ends - starts + 1

    invalid_return = np.isnan(returns) | (returns < -1.0)
    total_loss = returns == -1.0
    log_component = np.zeros(n_rows, dtype=float)
    valid_log = ~(invalid_return | total_loss)
    log_component[valid_log] = np.log1p(returns[valid_log])
    cumulative_log = _group_cumulative(log_component, starts, lengths)
    cumulative_missing = _group_cumulative(
        invalid_return.astype(np.int64), starts, lengths
    )
    cumulative_loss = _group_cumulative(
        total_loss.astype(np.int64), starts, lengths
    )

    sentinel = n_rows
    terminal_positions = np.where(terminal, positions, sentinel)
    next_terminal = np.full(n_rows, sentinel, dtype=np.int64)
    for start, end in zip(starts, ends, strict=True):
        next_terminal[start : end + 1] = np.minimum.accumulate(
            terminal_positions[start : end + 1][::-1]
        )[::-1]

    next_index = np.minimum(positions + 1, n_rows - 1)
    execution_valid = (
        (positions + 1 <= group_end)
        & (sessions[next_index] == sessions + 1)
        & tradable[next_index]
    )
    result = pd.DataFrame(
        {
            "execution_valid": execution_valid,
            "execution_session_id": sessions + 1,
        },
        index=work.index,
    )

    market = np.asarray(market_returns, dtype=float)
    market_invalid = np.isnan(market) | (market <= -1.0)
    market_log = np.zeros(len(market), dtype=float)
    market_log[~market_invalid] = np.log1p(market[~market_invalid])
    market_log_cumulative = np.cumsum(market_log)
    market_invalid_cumulative = np.cumsum(market_invalid.astype(np.int64))

    for horizon in horizons_tuple:
        fixed_end_session = sessions + horizon + 1
        normal_end = positions + horizon + 1
        safe_normal_end = np.minimum(normal_end, n_rows - 1)
        terminal_index_safe = np.minimum(next_terminal, n_rows - 1)
        terminal_session = sessions[terminal_index_safe]
        terminal_in_window = (
            (next_terminal < sentinel)
            & (next_terminal <= group_end)
            & (terminal_session >= sessions + 2)
            & (terminal_session <= fixed_end_session)
        )
        end_index = np.where(terminal_in_window, next_terminal, normal_end)
        safe_end = np.minimum(end_index, n_rows - 1)
        start_index = positions + 2
        safe_start = np.minimum(start_index, n_rows - 1)

        within_group = (
            (start_index <= group_end)
            & (end_index <= group_end)
            & (sessions[safe_start] == sessions + 2)
        )
        consecutive = (
            safe_end - safe_start
            == sessions[safe_end] - sessions[safe_start]
        )
        normal_reaches_fixed_end = terminal_in_window | (
            (normal_end <= group_end)
            & (sessions[safe_normal_end] == fixed_end_session)
        )
        structural_valid = (
            execution_valid
            & within_group
            & consecutive
            & normal_reaches_fixed_end
            & (fixed_end_session < len(market))
        )

        before_start = np.minimum(positions + 1, n_rows - 1)
        missing_count = _range_sum(
            cumulative_missing, safe_end, before_start
        )
        loss_count = _range_sum(cumulative_loss, safe_end, before_start)
        log_sum = _range_sum(cumulative_log, safe_end, before_start)
        primary_valid = structural_valid & (missing_count == 0)
        primary_return = np.full(n_rows, np.nan)
        primary_growth = np.expm1(log_sum)
        primary_growth[loss_count > 0] = -1.0
        primary_return[primary_valid] = primary_growth[primary_valid]

        terminal_missing = (
            terminal_in_window
            & invalid_return[terminal_index_safe]
        )
        bound_valid = structural_valid & (
            (missing_count == 0)
            | (terminal_missing & (missing_count == 1))
        )
        lower_return = primary_return.copy()
        lower_return[bound_valid & terminal_missing] = -1.0
        upper_return = primary_return.copy()
        upper_growth = np.expm1(log_sum)
        upper_growth[loss_count > 0] = -1.0
        upper_return[bound_valid & terminal_missing] = upper_growth[
            bound_valid & terminal_missing
        ]

        market_start = sessions + 2
        market_end = fixed_end_session
        market_range_valid = (
            (market_start >= 0)
            & (market_end < len(market))
        )
        safe_market_start_before = np.clip(
            market_start - 1, 0, len(market) - 1
        )
        safe_market_end = np.minimum(market_end, len(market) - 1)
        market_log_sum = (
            market_log_cumulative[safe_market_end]
            - market_log_cumulative[safe_market_start_before]
        )
        market_missing_count = (
            market_invalid_cumulative[safe_market_end]
            - market_invalid_cumulative[safe_market_start_before]
        )
        market_valid = market_range_valid & (market_missing_count == 0)
        market_growth = np.full(n_rows, np.nan)
        market_growth[market_valid] = np.expm1(market_log_sum[market_valid])

        excess = np.full(n_rows, np.nan)
        excess_valid = primary_valid & market_valid
        excess[excess_valid] = (
            (1.0 + primary_return[excess_valid])
            / (1.0 + market_growth[excess_valid])
            - 1.0
        )

        result[f"label_start_session_id_{horizon}d"] = market_start
        result[f"label_end_session_id_{horizon}d"] = market_end
        result[f"fwd_total_return_{horizon}d"] = primary_return
        result[f"fwd_market_excess_return_{horizon}d"] = excess
        result[f"fwd_total_return_{horizon}d_dlret_loss100"] = lower_return
        result[f"fwd_total_return_{horizon}d_dlret_zero"] = upper_return
        result[f"terminal_within_{horizon}d"] = terminal_in_window
        result[f"missing_delisting_return_within_{horizon}d"] = terminal_missing
        result[f"label_structural_valid_{horizon}d"] = structural_valid

    return result
