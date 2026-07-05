"""Logging setup: console + a detailed per-run log file under output/ (gitignored,
per CLAUDE.md §6 -- only the small metrics JSON in results/ is meant to be tracked).
Append mode so a resumed run keeps a continuous log rather than truncating history.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: str | Path, run_name: str) -> logging.Logger:
    log_dir = Path(output_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    logger = logging.getLogger(f"training.{run_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # avoid duplicate handlers on repeated setup calls (e.g. tests)
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
