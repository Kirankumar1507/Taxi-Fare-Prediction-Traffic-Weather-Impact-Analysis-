"""Shared utilities for the taxi fare prediction pipeline."""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
