from __future__ import annotations

from pathlib import Path
import shutil

from state_alpha.models.gate4 import build_gate4
from state_alpha.models.gate5 import build_gate5
from state_alpha.models.validate_gate4 import validate_gate4
from state_alpha.models.validate_gate5 import validate_gate5
from test_gate4_integration import _synthetic_project


def test_gate5_atomic_build_and_independent_validation(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    project = _synthetic_project(tmp_path)
    shutil.copy2(
        source_root / "configs" / "gate5.yaml",
        project / "configs" / "gate5.yaml",
    )
    build_gate4(project)
    validate_gate4(project)

    output = build_gate5(project)
    assert output == project / "data" / "processed" / "gate5_v1"
    assert output.is_dir()
    assert not list(
        (project / "data" / "processed").glob("_gate5_v1_build_*")
    )
    result = validate_gate5(project)
    assert result["passed"], result["errors"]
    assert result["model_refits_verified"] == 14
    assert result["strict_oos_annual_refits_verified"] == 13
    assert result["model_states_verified"] == 14
    assert result["all_refits_purged"] is True
    assert result["all_refits_embargoed"] is True
    assert result["validation_prediction_rows"] == 3 * 3 * 210
    assert result["strict_oos_prediction_rows"] == 13 * 3 * 210
    assert result["strict_oos_used_for_selection"] is False
    assert (
        result["analysis_classification"]
        == "secondary_locked_post_gate4"
    )
    assert result["confirmatory_untouched_oos_claim_allowed"] is False
