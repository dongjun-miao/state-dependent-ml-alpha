"""CLI for the full Phase 2B point-in-time universe and label build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_alpha.data.phase2b import build_phase2b_panel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(build_phase2b_panel(args.project_root))


if __name__ == "__main__":
    main()
