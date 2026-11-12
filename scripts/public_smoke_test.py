"""Synthetic-only import and feature contract smoke test; no backtest evidence."""
from pathlib import Path
import numpy as np
import yaml
from state_alpha.validation.splits import purged_training_mask, embargo_allows

spec = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/features.yaml").read_text())
assert spec["feature_budget"]["frozen_feature_count"] == 38
formation = np.array([1, 2, 3, 4])
label_end = np.array([3, 4, 6, 8])
assert purged_training_mask(formation, label_end, 5).tolist() == [True, True, False, False]
assert embargo_allows(31, 10, 20)
assert not embargo_allows(30, 10, 20)
print("Synthetic/config smoke test passed; no empirical performance computed.")
