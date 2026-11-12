"""CLI for the descriptive Phase 2A universe profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_alpha.data.profile import profile_universe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    for name, path in profile_universe(args.project_root).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

