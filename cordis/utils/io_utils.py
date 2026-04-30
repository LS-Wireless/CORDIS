"""
cordis/utils/io_utils.py
========================
Input/output utilities for the CORDIS simulation framework.

Responsibilities
----------------
- Saving and loading ``SimResult`` objects (NumPy ``.npz`` + JSON sidecar)
- Generating deterministic, reproducible output filenames
- Seed file management (committed to the repo for reproducibility)
- Snapshotting the current git commit hash into saved results

File format
-----------
Each simulation run produces two files in the output directory:

``<tag>_<timestamp>.npz``
    NumPy archive containing all raw per-trial arrays
    (SINR, SCNR, fronthaul counts, convergence curves, …).

``<tag>_<timestamp>_meta.json``
    JSON sidecar with the full config snapshot, git hash, Python version,
    timestamp, and any scalar summary statistics.

This paired format lets you load the metadata cheaply (JSON) and the
heavy arrays lazily (npz).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from cordis.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Git helpers
# =============================================================================

def get_git_hash() -> str:
    """
    Return the short git commit hash of the current HEAD.

    Returns ``"unknown"`` if the working directory is not a git repo or
    git is not installed.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# =============================================================================
# Filename helpers
# =============================================================================

def make_filename(
    algorithm: str,
    tag: str = "",
    extension: str = ".npz",
) -> str:
    """
    Generate a timestamped, human-readable filename.

    Pattern:  ``<algorithm>[_<tag>]_<YYYYMMDD_HHMMSS><extension>``

    Parameters
    ----------
    algorithm : str
        Short algorithm identifier, e.g. ``"cordis_admm"`` or ``"split"``.
    tag : str
        Optional free-form label, e.g. ``"snr_sweep"`` or ``"pareto"``.
    extension : str
        File extension including the dot.  Default ``".npz"``.

    Returns
    -------
    str
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [algorithm]
    if tag:
        parts.append(tag)
    parts.append(ts)
    return "_".join(parts) + extension


# =============================================================================
# Save / load
# =============================================================================

def save_results(
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    save_dir: Union[str, Path],
    algorithm: str,
    tag: str = "",
) -> Path:
    """
    Persist simulation results to disk.

    Saves:
    - A ``.npz`` file containing all raw arrays.
    - A ``_meta.json`` sidecar file containing metadata + scalar summaries.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Mapping from result name (e.g. ``"sinr_db"``) to a NumPy array.
        All arrays are saved verbatim.
    metadata : dict
        Scalar metadata: config snapshot, git hash, timing, etc.  Must be
        JSON-serializable.
    save_dir : str or Path
        Directory in which the output files are written.  Created if absent.
    algorithm : str
        Short algorithm name used in the filename (e.g. ``"cordis_admm"``).
    tag : str
        Optional experiment label appended to the filename.

    Returns
    -------
    Path
        Path to the saved ``.npz`` file (the ``.json`` sidecar is at the
        same path with ``_meta.json`` suffix replacing ``.npz``).

    Examples
    --------
    >>> arrays = {"sinr_db": np.array([10.2, 11.5, 9.8])}
    >>> meta   = {"n_trials": 3, "seed": 42}
    >>> path   = save_results(arrays, meta, "results/", "cordis_admm")
    """
    from cordis.utils.paths import get_project_root
    out_dir = Path(save_dir)
    if not out_dir.is_absolute():
        out_dir = get_project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = make_filename(algorithm, tag, extension="")
    npz_path  = out_dir / (stem + ".npz")
    json_path = out_dir / (stem + "_meta.json")

    # ── Save arrays ───────────────────────────────────────────────────────
    np.savez_compressed(npz_path, **arrays)
    logger.info("Arrays saved  → %s", npz_path)

    # ── Augment metadata with environment info ────────────────────────────
    full_meta = {
        "algorithm": algorithm,
        "tag": tag,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_hash": get_git_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "array_keys": list(arrays.keys()),
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        **metadata,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2, default=_json_serialiser)
    logger.info("Metadata saved → %s", json_path)

    return npz_path


def load_results(
    npz_path: Union[str, Path],
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load a saved simulation result from a ``.npz`` / ``_meta.json`` pair.

    Parameters
    ----------
    npz_path : str or Path
        Path to the ``.npz`` file.  The sidecar is expected at the same
        stem with ``_meta.json``.

    Returns
    -------
    arrays : dict[str, np.ndarray]
    metadata : dict
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Result file not found: {npz_path}")

    json_path = npz_path.parent / (npz_path.stem + "_meta.json")

    arrays: Dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as f:
        for key in f.files:
            arrays[key] = f[key]

    metadata: Dict[str, Any] = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        logger.warning("Metadata sidecar not found: %s", json_path)

    logger.info("Loaded %d arrays from %s", len(arrays), npz_path)
    return arrays, metadata


# =============================================================================
# Seed management
# =============================================================================

def save_seed(
    seed: int,
    rng_state: Dict[str, Any],
    label: str,
    seeds_dir: Union[str, Path] = "results/seeds",
) -> Path:
    """
    Persist a random seed and the full NumPy RNG state to a JSON file.

    This allows exact reproduction of any specific Monte Carlo trial.

    Parameters
    ----------
    seed : int
        Master seed value.
    rng_state : dict
        State dictionary from ``np.random.default_rng(seed).bit_generator.state``.
    label : str
        Human-readable identifier (e.g. ``"topology_circle_seed42"``).
    seeds_dir : str or Path
        Directory for seed files.  These files **should be committed** to git.

    Returns
    -------
    Path
        Path to the saved seed JSON.
    """
    seeds_dir = Path(seeds_dir)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Convert uint32/uint64 arrays to lists for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    payload = {
        "seed": seed,
        "label": label,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "rng_state": json.loads(json.dumps(rng_state, default=_convert)),
    }

    path = seeds_dir / f"{label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.debug("Seed saved → %s", path)
    return path


def load_seed(path: Union[str, Path]) -> tuple[int, Dict[str, Any]]:
    """
    Load a seed file saved by :func:`save_seed`.

    Returns
    -------
    seed : int
    rng_state : dict
        Pass to ``rng.bit_generator.state = rng_state`` to restore exact state.
    """
    with open(Path(path), encoding="utf-8") as f:
        payload = json.load(f)
    return payload["seed"], payload["rng_state"]


def make_rng(seed: int) -> np.random.Generator:
    """
    Create a NumPy ``Generator`` from an integer seed.

    Using ``np.random.default_rng`` (PCG64) rather than the legacy
    ``np.random.seed`` gives independent, reproducible streams that are
    safe to split across parallel workers.

    Parameters
    ----------
    seed : int

    Returns
    -------
    np.random.Generator
    """
    return np.random.default_rng(seed)


def child_rng(parent_rng: np.random.Generator) -> np.random.Generator:
    """
    Spawn a statistically independent child generator from *parent_rng*.

    Used to create per-trial RNGs in the Monte Carlo runner without
    reseeding from a fixed integer (which would correlate streams).
    """
    return np.random.default_rng(parent_rng.integers(0, 2**63 - 1))


# =============================================================================
# JSON serialisation helper
# =============================================================================

def _json_serialiser(obj: Any) -> Any:
    """
    Custom JSON serialiser for types not handled by the stdlib encoder.
    Handles NumPy scalars and arrays, Paths, and dataclasses.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    # dataclass → dict
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# =============================================================================
# Directory helpers
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_results(
    results_dir: Union[str, Path],
    algorithm: Optional[str] = None,
) -> list[Path]:
    """
    List all ``.npz`` result files in *results_dir*, optionally filtered
    by algorithm name prefix.

    Parameters
    ----------
    results_dir : str or Path
    algorithm : str or None
        If given, only return files whose name starts with this string.

    Returns
    -------
    list[Path]
        Sorted list of matching ``.npz`` paths.
    """
    d = Path(results_dir)
    if not d.is_dir():
        return []
    pattern = f"{algorithm}*.npz" if algorithm else "*.npz"
    return sorted(d.glob(pattern))

