"""Presentation-layer utilities for frozen research evidence."""

from .gate7 import FIGURE_FILES, SCORE_LABELS, SCORE_ORDER, sha256_file, verify_locked_sources

__all__ = [
    "FIGURE_FILES",
    "SCORE_LABELS",
    "SCORE_ORDER",
    "sha256_file",
    "verify_locked_sources",
]

