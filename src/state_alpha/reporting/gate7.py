"""Small dependency-light helpers for the Gate 7 presentation package."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


SCORE_ORDER = [
    "base_family_score",
    "ridge_alpha_only",
    "ridge_state_only",
    "ridge_main_effects",
    "ridge_explicit_interactions",
    "tree_alpha_only",
    "tree_state_only",
    "tree_alpha_state",
    "linear_capped_gate",
    "tree_capped_gate",
]

SCORE_LABELS = {
    "base_family_score": "Base family",
    "ridge_alpha_only": "Ridge alpha",
    "ridge_state_only": "Ridge state",
    "ridge_main_effects": "Ridge main",
    "ridge_explicit_interactions": "Ridge interactions",
    "tree_alpha_only": "Tree alpha",
    "tree_state_only": "Tree state",
    "tree_alpha_state": "Tree alpha + state",
    "linear_capped_gate": "Linear capped gate",
    "tree_capped_gate": "Tree capped gate",
}

FIGURE_FILES = [
    "figure_01_research_pipeline.png",
    "figure_02_boundary_clean_rank_ic.png",
    "figure_03_raw_rank_cost_stress.png",
    "figure_04_neutral_rank_cost_stress.png",
    "figure_05_role_based_wealth_paths.png",
    "figure_06_exposure_neutralization.png",
    "figure_07_liquidity_and_capacity.png",
]


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_sources(project_root: Path, config: dict[str, Any]) -> dict[str, str]:
    """Verify every presentation input and return its observed digest."""

    observed: dict[str, str] = {}
    failures: list[str] = []
    for entry in config["locked_sources"]:
        relative = str(entry["path"])
        source = project_root / Path(relative)
        if not source.is_file():
            failures.append(f"missing: {relative}")
            continue
        digest = sha256_file(source)
        observed[relative] = digest
        if digest != str(entry["sha256"]).lower():
            failures.append(
                f"hash mismatch: {relative}: expected {entry['sha256']}, observed {digest}"
            )
    if failures:
        raise ValueError("Gate 7 locked-source verification failed: " + "; ".join(failures))
    return observed
