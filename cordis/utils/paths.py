"""
cordis/utils/paths.py
=====================
Project root resolution for the CORDIS simulation framework.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_MARKERS = [
    "configs/default.json",
    "setup.py",
    "pyproject.toml",
    ".git",
]

_MAX_LEVELS = 10


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Return the absolute path to the CORDIS project root directory,
    regardless of what working directory PyCharm sets at runtime.

    Walks upward from this file's location until it finds a directory
    containing a known project marker (configs/default.json, .git, etc.).
    Result is cached after the first call.
    """
    candidate = Path(__file__).resolve().parent

    for _ in range(_MAX_LEVELS):
        for marker in _MARKERS:
            if (candidate / marker).exists():
                return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent

    raise RuntimeError(
        f"Could not locate the CORDIS project root.\n"
        f"Searched upward from: {Path(__file__).resolve().parent}\n"
        f"Looked for markers: {_MARKERS}\n"
        f"Make sure 'configs/default.json' exists at the project root."
    )


def project_path(*parts: str) -> Path:
    """
    Build an absolute path by joining parts onto the project root.

    Example:
        project_path("configs", "default.json")
        # → /your/project/CORDIS/configs/default.json
    """
    root = get_project_root()
    result = root
    for part in parts:
        result = result / part
    return result
