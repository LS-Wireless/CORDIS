"""
notebooks/_playground_helpers.py
================================

Shared boilerplate for the playground notebooks (Stage 12).

The four notebooks under ``notebooks/playground_*.ipynb`` each focus
on one result *kind* (single / sweep / trace / table).  They all need
the same three things:

    1.  Make the in-tree ``cordis`` package importable when the
        notebook is launched from ``notebooks/`` (Jupyter's cwd
        defaults to the notebook directory, not the repo root).
    2.  Apply the IEEE paper rcParams via
        :func:`cordis.plotting.apply_paper_style`.
    3.  Locate and load the most recent
        ``results/exp_<name>/<ts>/`` run for a chosen experiment.

This module centralises those steps so each notebook can start with a
two-line import block.

Usage from a notebook cell:

>>> from _playground_helpers import (
...     setup_paper_style, load_latest_result, load_run, latest_run_dir,
... )
>>> setup_paper_style()
>>> result = load_latest_result("sinr_cdf")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# ── Path setup ──────────────────────────────────────────────────────────────
# Notebooks live in <repo>/notebooks/.  We want <repo> on sys.path so
# `from cordis... import ...` resolves to the in-tree package.
_HERE      = Path(__file__).resolve().parent           # <repo>/notebooks/
_REPO_ROOT = _HERE.parent                              # <repo>/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def setup_paper_style(use_latex: bool = True) -> None:
    """Apply CORDIS paper-style mpl rcParams.

    Falls back to mathtext (no system LaTeX needed) when ``pdflatex``
    is not on PATH — see :func:`cordis.plotting.style.apply_paper_style`.

    Parameters
    ----------
    use_latex : bool, default True
        Pass through to ``apply_paper_style``.  Set False to force
        mathtext rendering even when LaTeX is installed (useful for
        quick iteration when you don't need the polished fonts).
    """
    # Import here so this module itself stays cheap to import even
    # when matplotlib is slow to load.
    from cordis.plotting import apply_paper_style
    apply_paper_style(use_latex=use_latex)


def latest_run_dir(experiment_name: str,
                   results_root: Union[str, Path] = "results") -> Path:
    """Return the path of the most recent ``results/exp_<name>/<ts>/``.

    Parameters
    ----------
    experiment_name : str
        Experiment registry name, e.g. ``"sinr_cdf"``.
    results_root : str | Path, default ``"results"``
        Root directory containing ``exp_<name>/`` subdirs.  Resolved
        relative to the repo root if relative.

    Raises
    ------
    FileNotFoundError
        If ``results/exp_<name>/`` doesn't exist, or contains no
        timestamped run directories.
    """
    root = Path(results_root)
    if not root.is_absolute():
        root = _REPO_ROOT / root
    exp_dir = root / f"exp_{experiment_name}"
    if not exp_dir.is_dir():
        raise FileNotFoundError(
            f"No results directory at {exp_dir}.  "
            f"Have you run `make {experiment_name}` yet?"
        )
    runs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(
            f"No timestamped runs found under {exp_dir}."
        )
    return runs[-1]


def load_latest_result(experiment_name: str,
                       results_root: Union[str, Path] = "results"):
    """Load the most recent :class:`ExperimentResult` for ``experiment_name``.

    Echoes which run directory was loaded so notebook cells stay
    self-documenting.
    """
    from cordis.experiments.result import ExperimentResult
    run = latest_run_dir(experiment_name, results_root=results_root)
    print(f"Loaded: {run}")
    return ExperimentResult.load(run)


def load_run(path: Union[str, Path]):
    """Load a specific run directory (e.g. for multi-run overlay)."""
    from cordis.experiments.result import ExperimentResult
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return ExperimentResult.load(p)


def load_result(experiment_name: str,
                exp_dir: Optional[Union[str, Path]] = None,
                results_root: Union[str, Path] = "results"):
    """Load an :class:`ExperimentResult` — latest run by default, or a
    specific run directory if ``exp_dir`` is given.

    Convenience dispatcher for notebook cells:

    .. code-block:: python

        # Load most recent run (default).
        result = load_result('sinr_cdf')

        # Load a specific timestamped run for paper-grade plots or
        # re-rendering an older comparison.
        result = load_result('sinr_cdf',
                             exp_dir='results/exp_sinr_cdf/20260520_113500')

    Parameters
    ----------
    experiment_name : str
        Bare experiment name (without ``exp_`` prefix), e.g. ``'sinr_cdf'``.
        Ignored when ``exp_dir`` is given.
    exp_dir : str or Path, optional
        Path to a specific run directory.  If provided, loads that
        directory directly; otherwise loads the most recent run of
        ``experiment_name`` under ``results_root``.
    results_root : str or Path
        Root directory containing per-experiment subfolders.
        Defaults to ``"results"``.
    """
    if exp_dir is None:
        return load_latest_result(experiment_name, results_root=results_root)
    return load_run(exp_dir)


def summarize(result) -> None:
    """Print a one-paragraph summary of an :class:`ExperimentResult`.

    Useful as a sanity check after loading: confirms the experiment
    name, kind, and headline metadata before you build a figure.
    """
    print(f"Experiment:   {result.name}")
    print(f"Kind:         {result.kind}")
    if getattr(result, "metadata", None):
        for key in ("n_drops", "n_realizations", "spec_set"):
            if key in result.metadata:
                print(f"{key+':':<14}{result.metadata[key]}")
        if "cfg_summary" in result.metadata:
            print("cfg_summary:")
            for k, v in result.metadata["cfg_summary"].items():
                print(f"  {k:<22} = {v}")
    # Algorithm list (where applicable).
    if result.kind == "single" and result.sim_result is not None:
        names = list(getattr(result.sim_result, "algorithm_names", []) or [])
        if names:
            print(f"Algorithms:   {', '.join(names)}")
    elif result.kind == "sweep" and result.sweep_axis is not None:
        ax = result.sweep_axis
        print(f"Sweep axis:   {ax.name} = {ax.values}")


__all__ = [
    "setup_paper_style",
    "latest_run_dir",
    "load_latest_result",
    "load_run",
    "summarize",
]

