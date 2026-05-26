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


# ─────────────────────────────────────────────────────────────────────
# Stage 21 — array-result helpers
# ─────────────────────────────────────────────────────────────────────
#
# Array runs produce a fan-out of per-task directories that get merged
# via ``scripts/aggregate_array_batch.py`` into a single ``array_<jobid>
# _aggregated/`` directory.  These helpers locate that aggregated dir
# (without relying on the ``latest`` symlink, which can point to a
# per-task dir if sync ordering happens to land that way) and surface
# the per-task metadata for paper-figure sanity checks.

import re as _re

_ARRAY_AGG_PATTERN = _re.compile(r"^array_(\d+)_aggregated$")
_ARRAY_TASK_PATTERN = _re.compile(r"^array_(\d+)_task_(\d+)$")


def find_aggregated_dirs(experiment_name: str,
                         results_root: Union[str, Path] = "results"):
    """List all ``array_<jobid>_aggregated/`` dirs under an experiment.

    Returns a list of ``(array_id_str, path)`` tuples sorted by
    array_id ascending.  Empty list if none exist.

    Parameters
    ----------
    experiment_name : str
        Bare experiment name, e.g. ``"sinr_cdf"``.
    results_root : str | Path, default ``"results"``
        Root directory containing ``exp_<name>/`` subfolders.
    """
    root = Path(results_root)
    if not root.is_absolute():
        root = _REPO_ROOT / root
    exp_dir = root / f"exp_{experiment_name}"
    if not exp_dir.is_dir():
        return []
    matches = []
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ARRAY_AGG_PATTERN.match(child.name)
        if m:
            matches.append((m.group(1), child))
    matches.sort(key=lambda t: int(t[0]))
    return matches


def find_per_task_dirs_for(experiment_name: str,
                           array_id: Union[str, int],
                           results_root: Union[str, Path] = "results"):
    """List ``array_<jobid>_task_<id>/`` dirs for one array job.

    Returns a list of ``(task_id_int, path)`` tuples sorted by
    task_id ascending.  Empty if the array job has no task dirs.
    """
    root = Path(results_root)
    if not root.is_absolute():
        root = _REPO_ROOT / root
    exp_dir = root / f"exp_{experiment_name}"
    array_id_str = str(array_id)
    if array_id_str.startswith("array_"):
        array_id_str = array_id_str[len("array_"):]
    matches = []
    if not exp_dir.is_dir():
        return matches
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ARRAY_TASK_PATTERN.match(child.name)
        if m and m.group(1) == array_id_str:
            matches.append((int(m.group(2)), child))
    matches.sort()
    return matches


def load_aggregated_array_result(experiment_name: str,
                                 array_id: Optional[Union[str, int]] = None,
                                 results_root: Union[str, Path] = "results"):
    """Load the aggregated :class:`ExperimentResult` from an array run.

    Independent of the ``latest`` symlink (which after rsync may not
    point at the aggregated dir).  Selects the array job by ID, or the
    most recent (highest job ID) if ``array_id`` is None.

    Parameters
    ----------
    experiment_name : str
        Bare experiment name, e.g. ``"sinr_cdf"`` or ``"gamma_sweep"``.
    array_id : str | int, optional
        SLURM array job ID.  When None, picks the highest-numbered
        ``array_<jobid>_aggregated/`` directory (typically the most
        recent run).
    results_root : str | Path, default ``"results"``

    Raises
    ------
    FileNotFoundError
        If no ``array_<jobid>_aggregated/`` directory exists for the
        experiment, or for the requested ``array_id``.
    """
    from cordis.experiments.result import ExperimentResult

    aggs = find_aggregated_dirs(experiment_name, results_root=results_root)
    if not aggs:
        raise FileNotFoundError(
            f"No array_<jobid>_aggregated/ directories found under "
            f"results/exp_{experiment_name}/.  Run "
            f"`python3 scripts/aggregate_array_batch.py "
            f"{experiment_name} <jobid>` first, or sync them from HPC3."
        )

    if array_id is None:
        # Pick the latest (highest ID).
        chosen_id, chosen_path = aggs[-1]
    else:
        target = str(array_id).lstrip("array_") or str(array_id)
        chosen = [(aid, p) for aid, p in aggs if aid == target]
        if not chosen:
            available = ", ".join(aid for aid, _ in aggs)
            raise FileNotFoundError(
                f"No array_{target}_aggregated/ under "
                f"results/exp_{experiment_name}/.  Available array IDs: "
                f"{available or '(none)'}."
            )
        chosen_id, chosen_path = chosen[0]

    print(f"Loaded aggregated result: {chosen_path}")
    return ExperimentResult.load(chosen_path), chosen_id


def per_task_summary(experiment_name: str,
                     array_id: Union[str, int],
                     results_root: Union[str, Path] = "results"):
    """Build a table of per-task metadata for an array job.

    Inspects each ``array_<jobid>_task_<id>/`` directory and reads its
    ``manifest.json`` + ``result.json`` sidecar to recover the
    per-task seed, n_trials, n_drops, and n_realizations.

    Returns
    -------
    list[dict]
        One dict per task with keys: ``task_id``, ``seed``,
        ``n_trials``, ``n_drops``, ``n_realizations``, ``path``.
        Returns empty list if no per-task dirs are found.

    Notes
    -----
    Seeds follow the recipe in ``_array_common.sh``:
    ``SEED = BASE_SEED + ARRAY_TASK_ID``.  Trial seeds within a task
    come from ``numpy.random.SeedSequence(SEED).spawn(n_trials)``, so
    different tasks produce disjoint trial streams.
    """
    import json as _json
    task_dirs = find_per_task_dirs_for(experiment_name, array_id,
                                       results_root=results_root)
    rows = []
    for task_id, path in task_dirs:
        # Manifest is small & always present after a successful run.
        manifest = {}
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = _json.load(f)
        # Sidecar (.json next to result.npz) carries seed + runner_cfg.
        sidecar = {}
        sidecar_path = path / "result.json"
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                sidecar = _json.load(f)
        runner_cfg = sidecar.get("runner_cfg") or {}
        per_algo = sidecar.get("per_algorithm_scalars") or {}
        # Trials: take max across algos (should all match).
        n_trials = max(
            (int(p.get("n_trials_total", 0)) for p in per_algo.values()),
            default=0,
        )
        rows.append({
            "task_id":         task_id,
            "seed":            runner_cfg.get("seed"),
            "n_trials":        n_trials,
            "n_drops":         runner_cfg.get("n_drops"),
            "n_realizations":  runner_cfg.get("n_realizations_per_drop"),
            "path":            path,
        })
    return rows


def print_per_task_summary(experiment_name: str,
                           array_id: Union[str, int],
                           results_root: Union[str, Path] = "results") -> None:
    """Pretty-print the per-task table from :func:`per_task_summary`.

    Includes column totals at the bottom so it's easy to verify that
    the array's total trials = sum(per-task trials).
    """
    rows = per_task_summary(experiment_name, array_id,
                            results_root=results_root)
    if not rows:
        print(f"No per-task directories found for "
              f"results/exp_{experiment_name}/array_{array_id}_task_*/.")
        return

    hdr = (f"  {'task':>4}  {'seed':>8}  {'trials':>7}  "
           f"{'drops':>6}  {'reals':>6}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"  Per-task summary for array_{array_id}:")
    print(hdr)
    print(sep)
    for r in rows:
        seed_s   = "?" if r["seed"]   is None else str(r["seed"])
        drops_s  = "?" if r["n_drops"]   is None else str(r["n_drops"])
        reals_s  = "?" if r["n_realizations"] is None else str(r["n_realizations"])
        print(f"  {r['task_id']:>4}  {seed_s:>8}  {r['n_trials']:>7}  "
              f"{drops_s:>6}  {reals_s:>6}")
    print(sep)
    # Aggregate row.
    total_trials = sum(r["n_trials"] for r in rows)
    # Seeds: report range (consecutive integers from BASE_SEED).
    valid_seeds = [r["seed"] for r in rows if r["seed"] is not None]
    seed_range = ""
    if valid_seeds:
        lo, hi = min(valid_seeds), max(valid_seeds)
        seed_range = f"{lo}-{hi}" if lo != hi else str(lo)
    print(f"  {'tot':>4}  {seed_range:>8}  {total_trials:>7}  "
          f"{'':>6}  {'':>6}")


def array_summary(result, experiment_name: str,
                  array_id: Union[str, int],
                  results_root: Union[str, Path] = "results") -> None:
    """One-shot summary for an aggregated array result.

    Prints headline counts (trials, drops, realizations, tasks) from
    the aggregated ``ExperimentResult`` + ``result.json`` sidecar,
    then the per-task table.  Use right after
    :func:`load_aggregated_array_result` for a self-documenting cell.
    """
    import json as _json
    print(f"Experiment:           {result.name}")
    print(f"Kind:                 {result.kind}")

    # Try to read the aggregated sidecar for trial count + runner_cfg.
    # For single-kind: result.json lives next to result.npz.
    # For sweep-kind: pick any per-value sidecar (they share runner_cfg).
    aggs = find_aggregated_dirs(experiment_name, results_root=results_root)
    target = str(array_id).lstrip("array_") or str(array_id)
    chosen = next((p for aid, p in aggs if aid == target), None)
    if chosen is None:
        print("(could not locate aggregated dir for runner_cfg metadata)")
        return

    sidecar_files = sorted(chosen.glob("result*.json"))
    if not sidecar_files:
        print(f"(no sidecars under {chosen})")
        return

    with open(sidecar_files[0]) as f:
        sidecar = _json.load(f)
    runner_cfg = sidecar.get("runner_cfg") or {}
    per_algo   = sidecar.get("per_algorithm_scalars") or {}
    n_trials_total = max(
        (int(p.get("n_trials_total", 0)) for p in per_algo.values()),
        default=0,
    )
    meta = (sidecar.get("metadata") or
            result.metadata if hasattr(result, "metadata") else {})
    n_tasks_merged = (meta.get("aggregated_from_n_tasks")
                      if isinstance(meta, dict) else None)

    print(f"Array job ID:         {target}")
    print(f"Total trials:         {n_trials_total}")
    print(f"n_drops:              {runner_cfg.get('n_drops', '?')}")
    print(f"n_realizations/drop:  {runner_cfg.get('n_realizations_per_drop', '?')}")
    if n_tasks_merged is not None:
        print(f"Tasks merged:         {n_tasks_merged}")
    if result.kind == "single":
        algos = list(result.sim_result.algorithm_results.keys())
        print(f"Algorithms ({len(algos)}):     {', '.join(algos)}")
    elif result.kind == "sweep":
        ax = result.sweep_axis
        print(f"Sweep axis:           {ax.name} = {ax.values}")
        first_sr = next(iter(result.sweep_results.values()))
        algos = list(first_sr.algorithm_results.keys())
        print(f"Algorithms ({len(algos)}):     {', '.join(algos)}")
    print()
    print_per_task_summary(experiment_name, target, results_root=results_root)


__all__ = [
    "setup_paper_style",
    "latest_run_dir",
    "load_latest_result",
    "load_run",
    "load_result",
    "summarize",
    # Stage 21 — array helpers
    "find_aggregated_dirs",
    "find_per_task_dirs_for",
    "load_aggregated_array_result",
    "per_task_summary",
    "print_per_task_summary",
    "array_summary",
]

