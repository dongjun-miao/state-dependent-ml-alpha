"""CLI for independent Phase 2B panel validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_alpha.data.validate_phase2b import validate_phase2b  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_phase2b(args.project_root),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
