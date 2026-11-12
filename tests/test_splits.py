from __future__ import annotations

import numpy as np
import pytest

from state_alpha.validation.splits import (
    embargo_allows,
    purged_training_mask,
)


def test_purge_removes_training_labels_that_touch_validation() -> None:
    formation = np.array([70, 75, 80, 95])
    label_end = np.array([90, 99, 100, 115])
    mask = purged_training_mask(formation, label_end, 100)
    assert mask.tolist() == [True, True, False, False]


def test_embargo_requires_full_gap_after_prior_evaluation() -> None:
    assert not embargo_allows(120, 100, 20)
    assert embargo_allows(121, 100, 20)
    with pytest.raises(ValueError):
        embargo_allows(121, 100, -1)
