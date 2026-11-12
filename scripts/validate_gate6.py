"""Independently validate the Gate 6 portfolio artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from state_alpha.config import discover_project_root
from state_alpha.portfolio.validate_gate6 import validate_gate6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    root = (
        args.project_root.resolve()
        if args.project_root is not None
        else discover_project_root()
    )
    result = validate_gate6(root)
    print(f"passed={result['passed']}")
    print(f"portfolio_variants={result['portfolio_variants']}")


if __name__ == "__main__":
    main()
