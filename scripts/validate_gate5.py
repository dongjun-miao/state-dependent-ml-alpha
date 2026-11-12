"""Independently validate the Gate 5 secondary locked artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from state_alpha.config import discover_project_root
from state_alpha.models.validate_gate5 import validate_gate5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    root = (
        args.project_root.resolve()
        if args.project_root is not None
        else discover_project_root()
    )
    result = validate_gate5(root)
    print(f"passed={result['passed']}")
    print(
        f"strict_oos_prediction_rows="
        f"{result['strict_oos_prediction_rows']}"
    )


if __name__ == "__main__":
    main()
