from __future__ import annotations

import numpy as np
import pandas as pd

from state_alpha.labels.forward import compute_forward_labels


def _security(
    permno: int,
    sessions: range,
    *,
    terminal_session: int | None = None,
    missing_terminal_return: bool = False,
    untradable_sessions: set[int] | None = None,
) -> pd.DataFrame:
    untradable = untradable_sessions or set()
    rows = []
    for session in sessions:
        terminal = session == terminal_session
        value = np.nan if terminal and missing_terminal_return else (
            -0.5 if terminal else 0.01
        )
        rows.append(
            {
                "permno": permno,
                "session_id": session,
                "return_total": value,
                "daily_delisting_flag": "Y" if terminal else "N",
                "formation_tradable": session not in untradable,
            }
        )
    return pd.DataFrame(rows)


def test_label_starts_after_t_plus_1_close_and_has_exact_endpoint() -> None:
    frame = _security(10001, range(0, 10))
    result = compute_forward_labels(frame, np.zeros(20), horizons=(5,))
    expected = 1.01**5 - 1.0

    assert result.loc[0, "execution_session_id"] == 1
    assert result.loc[0, "label_start_session_id_5d"] == 2
    assert result.loc[0, "label_end_session_id_5d"] == 6
    assert np.isclose(result.loc[0, "fwd_total_return_5d"], expected)

    changed = frame.copy()
    changed.loc[0, "return_total"] = 5.0
    changed.loc[1, "return_total"] = -0.9
    changed_result = compute_forward_labels(
        changed, np.zeros(20), horizons=(5,)
    )
    assert np.isclose(changed_result.loc[0, "fwd_total_return_5d"], expected)


def test_valid_delisting_terminates_stock_leg_and_holds_cash() -> None:
    frame = _security(10002, range(0, 5), terminal_session=4)
    result = compute_forward_labels(frame, np.zeros(20), horizons=(5,))
    expected = (1.01**2) * 0.5 - 1.0

    assert bool(result.loc[0, "terminal_within_5d"])
    assert bool(result.loc[0, "label_structural_valid_5d"])
    assert np.isclose(result.loc[0, "fwd_total_return_5d"], expected)


def test_missing_delisting_return_stays_missing_with_explicit_bounds() -> None:
    frame = _security(
        10003,
        range(0, 5),
        terminal_session=4,
        missing_terminal_return=True,
    )
    result = compute_forward_labels(frame, np.zeros(20), horizons=(5,))

    assert bool(result.loc[0, "missing_delisting_return_within_5d"])
    assert np.isnan(result.loc[0, "fwd_total_return_5d"])
    assert result.loc[0, "fwd_total_return_5d_dlret_loss100"] == -1.0
    assert np.isclose(
        result.loc[0, "fwd_total_return_5d_dlret_zero"],
        1.01**2 - 1.0,
    )


def test_unavailable_t_plus_1_close_invalidates_execution() -> None:
    frame = _security(
        10004,
        range(0, 8),
        terminal_session=1,
        untradable_sessions={1},
    )
    result = compute_forward_labels(frame, np.zeros(20), horizons=(5,))
    assert not bool(result.loc[0, "execution_valid"])
    assert np.isnan(result.loc[0, "fwd_total_return_5d"])


def test_missing_market_session_cannot_be_bridged_by_security_row_shift() -> None:
    frame = _security(10005, range(0, 10))
    frame = frame.loc[frame["session_id"].ne(3)].reset_index(drop=True)
    result = compute_forward_labels(frame, np.zeros(20), horizons=(5,))
    assert not bool(result.loc[0, "label_structural_valid_5d"])
    assert np.isnan(result.loc[0, "fwd_total_return_5d"])


def test_dataset_end_returns_missing_instead_of_indexing_past_calendar() -> None:
    frame = _security(10006, range(0, 8))
    result = compute_forward_labels(frame, np.zeros(8), horizons=(5,))
    assert np.isclose(result.loc[0, "fwd_total_return_5d"], 1.01**5 - 1.0)
    assert np.isnan(result.loc[2, "fwd_total_return_5d"])
    assert np.isnan(result.loc[7, "fwd_total_return_5d"])

    terminal = _security(10007, range(0, 8), terminal_session=7)
    terminal_result = compute_forward_labels(
        terminal, np.zeros(8), horizons=(5,)
    )
    assert bool(terminal_result.loc[2, "terminal_within_5d"])
    assert not bool(
        terminal_result.loc[2, "label_structural_valid_5d"]
    )
    assert np.isnan(terminal_result.loc[2, "fwd_total_return_5d"])
