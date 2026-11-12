"""File and console logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from state_alpha.paths import PathPolicy


def configure_run_logger(
    name: str,
    log_path: Path,
    path_policy: PathPolicy,
) -> logging.Logger:
    """Create an idempotent UTF-8 file/console logger."""

    resolved = path_policy.assert_writable(log_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(resolved, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

