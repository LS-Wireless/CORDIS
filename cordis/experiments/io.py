"""
Stage 8a — Output path conventions for experiments and figures.

Two root directories, both overridable by environment variables so the
same code runs on a laptop or on a SLURM compute node with scratch
storage:

* ``$CORDIS_RESULTS_DIR`` — defaults to ``./results``
* ``$CORDIS_FIGURES_DIR`` — defaults to ``./figures``

Per-experiment results live in
``$CORDIS_RESULTS_DIR/exp_<name>/<timestamp>/`` with a ``latest``
symlink updated on each :func:`experiment_dir` call; figures in
``$CORDIS_FIGURES_DIR/exp_<name>/``; per-run logs in
``$CORDIS_RESULTS_DIR/exp_<name>/<timestamp>/logs/``.

The ``exp_`` prefix is added automatically by both
:func:`experiment_dir` and :func:`figure_dir` so the .gitignore can
use a single pattern (``exp_*/``) to catch every regenerated output
directory, while committed reference content (``figures/examples/``,
``figures/.gitkeep``) sits outside that pattern.  The prefix is
idempotent — passing ``"exp_foo"`` does not produce ``exp_exp_foo``.

Public API
~~~~~~~~~~
:func:`experiment_dir`, :func:`figure_dir`, :func:`latest_result`,
:func:`log_dir`.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


_RESULTS_ENV          = "CORDIS_RESULTS_DIR"
_FIGURES_ENV          = "CORDIS_FIGURES_DIR"
_DEFAULT_RESULTS_ROOT = Path("results")
_DEFAULT_FIGURES_ROOT = Path("figures")
_TIMESTAMP_FORMAT     = "%Y%m%d_%H%M%S"
_EXP_PREFIX           = "exp_"


def _root_or_env(
    env_var: str,
    default: Path,
    override: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve root directory: explicit override > env var > default."""
    if override is not None:
        return Path(override)
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)
    return default


def _validate_name(name: str) -> None:
    """Reject names that would corrupt the path hierarchy."""
    if not name:
        raise ValueError("experiment name must be non-empty")
    if any(c in name for c in r"/\:"):
        raise ValueError(f"experiment name {name!r} contains a path separator")
    if any(c.isspace() for c in name):
        raise ValueError(f"experiment name {name!r} contains whitespace")
    if name in {".", ".."}:
        raise ValueError(f"experiment name {name!r} is reserved")


def _wrap_name(name: str) -> str:
    """Prepend the canonical ``exp_`` prefix if not already present.

    Idempotent: ``_wrap_name("foo") == "exp_foo"`` and
    ``_wrap_name("exp_foo") == "exp_foo"``.
    """
    return name if name.startswith(_EXP_PREFIX) else _EXP_PREFIX + name


def experiment_dir(
    name: str,
    timestamp: Optional[str] = None,
    root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create ``<root>/exp_<name>/<timestamp>/`` and update the ``latest``
    symlink alongside it.

    Parameters
    ----------
    name : str
        Experiment identifier — must be non-empty and contain no
        path separators or whitespace.  The ``exp_`` prefix is added
        automatically (idempotent).
    timestamp : str, optional
        ``YYYYMMDD_HHMMSS``.  Defaults to the current time.
    root : str | Path, optional
        Override ``$CORDIS_RESULTS_DIR`` for this call.

    Returns
    -------
    Path
        The created leaf directory.

    Notes
    -----
    The ``latest`` symlink is best-effort — on filesystems that do not
    support symlinks (rare in Linux clusters, common on Windows mounts)
    we log a warning but otherwise succeed.
    """
    _validate_name(name)
    r = _root_or_env(_RESULTS_ENV, _DEFAULT_RESULTS_ROOT, root)
    ts = timestamp or _dt.datetime.now().strftime(_TIMESTAMP_FORMAT)
    exp_name = _wrap_name(name)
    leaf = r / exp_name / ts
    leaf.mkdir(parents=True, exist_ok=True)

    latest = r / exp_name / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        # Relative symlink target so the tree is portable when copied.
        latest.symlink_to(ts)
    except OSError as e:
        logger.warning("Could not update 'latest' symlink at %s: %s",
                       latest, e)
    return leaf


def figure_dir(
    name: str,
    root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create ``<root>/exp_<name>/`` and return it.

    Parameters
    ----------
    name : str
        Experiment identifier (same rules as :func:`experiment_dir`).
        The ``exp_`` prefix is added automatically (idempotent).
    root : str | Path, optional
        Override ``$CORDIS_FIGURES_DIR`` for this call.
    """
    _validate_name(name)
    r = _root_or_env(_FIGURES_ENV, _DEFAULT_FIGURES_ROOT, root)
    d = r / _wrap_name(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_result(
    name: str,
    root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve ``<root>/exp_<name>/latest`` to the directory it points to.

    Raises
    ------
    FileNotFoundError
        If the experiment does not exist or has no ``latest`` symlink.
    """
    _validate_name(name)
    r = _root_or_env(_RESULTS_ENV, _DEFAULT_RESULTS_ROOT, root)
    latest = r / _wrap_name(name) / "latest"
    if not latest.exists():
        raise FileNotFoundError(
            f"No 'latest' for experiment {name!r}: {latest} does not exist."
        )
    return latest.resolve()


def log_dir(experiment_path: Union[str, Path]) -> Path:
    """
    Create and return ``<experiment_path>/logs/`` for per-run logs.

    Parameters
    ----------
    experiment_path : str | Path
        Path returned by :func:`experiment_dir` (or any directory
        you wish to host a ``logs/`` subdirectory).

    Returns
    -------
    Path
        The created ``logs/`` directory.

    Notes
    -----
    Co-locating logs with results means a single experiment directory
    (results + logs together) is fully self-contained for sharing or
    archiving.  The ``latest`` symlink at the parent level brings the
    most recent log along automatically.
    """
    d = Path(experiment_path) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

