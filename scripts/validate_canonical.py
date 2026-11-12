"""CLI for independent Phase 2A canonical validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_alpha.data.validate import validate_canonical  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    output, result = validate_canonical(args.project_root)
    print(output)
    print(f"passed={result['passed']}")


if __name__ == "__main__":
    main()

