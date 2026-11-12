"""Calendar-index leakage guards for expanding-window validation."""

from __future__ import annotations

import numpy as np


def purged_training_mask(
    formation_session_ids: np.ndarray,
    label_end_session_ids: np.ndarray,
    validation_start_session_id: int,
) -> np.ndarray:
    """Keep only training labels that end strictly before validation starts."""

    formation = np.asarray(formation_session_ids)
    label_end = np.asarray(label_end_session_ids)
    if formation.shape != label_end.shape:
        raise ValueError("formation and label-end arrays must have equal shape")
    return (
        (formation < validation_start_session_id)
        & (label_end < validation_start_session_id)
    )


def embargo_allows(
    candidate_start_session_id: int,
    prior_evaluation_end_session_id: int,
    embargo_trading_days: int,
) -> bool:
    """Return whether a later fold starts after the full embargo interval."""

    if embargo_trading_days < 0:
        raise ValueError("embargo_trading_days must be non-negative")
    return candidate_start_session_id > (
        prior_evaluation_end_session_id + embargo_trading_days
    )
