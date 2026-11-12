"""CLI for the Phase 2A CRSP canonical build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_alpha.data.build import build_canonical  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip exact duplicate-contiguity preflight (not recommended).",
    )
    args = parser.parse_args()
    result = build_canonical(
        args.project_root,
        run_preflight=not args.skip_preflight,
    )
    print(result)


if __name__ == "__main__":
    main()

