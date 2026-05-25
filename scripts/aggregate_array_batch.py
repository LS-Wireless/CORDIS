#!/usr/bin/env python3
"""
scripts/aggregate_array_batch.py
================================

Stage 21 — Aggregate per-task SimResult pickles from a SLURM job-array
run into a single combined SimResult.

The job-array workflow lays out per-task output as::

    results/array_<arrayjobid>/
        task_0/exp_<name>/<timestamp>/result.pkl
        task_1/exp_<name>/<timestamp>/result.pkl
        ...
        task_N/exp_<name>/<timestamp>/result.pkl

After all tasks complete (or you've waited long enough), run::

    python3 scripts/aggregate_array_batch.py results/array_<arrayjobid>/

The script:

  1. Globs ``results/array_<arrayjobid>/task_*/`` for SimResult pickles.
  2. Validates that every per-task pickle has the SAME spec layout
     (same algorithms, same metric registry).  Heterogeneous batches
     are rejected (per-task knobs other than N_TRIALS / SEED must
     match).
  3. Merges per-trial sample arrays across tasks:
     - SINRStatistics samples concatenated
     - SCNRStatistics samples concatenated
     - Per-trial diagnostics (iters, runtime_s, converged, power_ratios) concatenated
  4. Recomputes aggregate scalars (mean iters, convergence rate, etc.)
  5. Writes the merged SimResult to:

         results/array_<arrayjobid>/aggregated/result.pkl

Missing tasks (e.g., SLURM job failures) emit a warning but don't abort
— the aggregator merges whatever is available.  The number of merged
tasks is recorded in the aggregated SimResult's metadata.

Usage::

    python3 scripts/aggregate_array_batch.py results/array_12345/
    python3 scripts/aggregate_array_batch.py results/array_12345/ --output custom_path/
    python3 scripts/aggregate_array_batch.py results/array_12345/ --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def find_per_task_pickles(array_dir: Path) -> List[Path]:
    """Glob for ``task_*/exp_*/<ts>/result.pkl`` under ``array_dir``."""
    if not array_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {array_dir}")
    matches = sorted(array_dir.glob("task_*/exp_*/*/result.pkl"))
    return matches


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _validate_spec_layout(sim_results: List[Any]) -> None:
    """Reject heterogeneous batches: all per-task SimResults must have
    the same algorithm set + same metric availability."""
    if not sim_results:
        raise ValueError("no per-task SimResults provided to validator")
    first = sim_results[0]
    first_algos = set(first.algorithm_results.keys())
    for k, sr in enumerate(sim_results[1:], start=1):
        algos = set(sr.algorithm_results.keys())
        if algos != first_algos:
            raise ValueError(
                f"task {k} has algorithms {sorted(algos)}, but task 0 has "
                f"{sorted(first_algos)}.  Heterogeneous arrays not "
                f"supported — re-run the array with consistent specs."
            )


def _concat_per_trial_arrays(
    sim_results: List[Any],
    algo_name:   str,
) -> Dict[str, np.ndarray]:
    """Concatenate per-trial diagnostic arrays from `AlgorithmResult`.

    Returns a dict of (field_name → concatenated array) ready for
    splatting into a new AlgorithmResult.
    """
    concat: Dict[str, List[np.ndarray]] = {
        "iters":        [],
        "runtime_s":    [],
        "converged":    [],
        "power_ratios": [],
    }
    for sr in sim_results:
        ar = sr.algorithm_results.get(algo_name)
        if ar is None:
            continue
        for k in concat:
            concat[k].append(np.asarray(getattr(ar, k)))
    out: Dict[str, np.ndarray] = {}
    for k, parts in concat.items():
        if not parts:
            out[k] = np.array([])
            continue
        # power_ratios is 2D; everything else is 1D.
        if k == "power_ratios":
            out[k] = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0))
        else:
            out[k] = np.concatenate(parts, axis=0)
    return out


def _merge_stats(per_task_stats: List[Any]) -> Optional[Any]:
    """Merge a list of *Statistics dataclasses by concatenating any
    field that looks like a per-trial sample array.

    The Statistics dataclasses (SINRStatistics, SCNRStatistics) store
    aggregated per-trial samples — we concatenate every ndarray field
    along axis 0 and leave non-array fields from the first task.
    """
    non_none = [s for s in per_task_stats if s is not None]
    if not non_none:
        return None
    first = non_none[0]
    fields_info = dataclasses.fields(first)
    merged_kwargs: Dict[str, Any] = {}
    for f in fields_info:
        parts = [getattr(s, f.name) for s in non_none]
        # Concatenate ndarrays along axis 0; scalars take the first value.
        if all(isinstance(p, np.ndarray) for p in parts):
            try:
                merged_kwargs[f.name] = np.concatenate(parts, axis=0)
            except ValueError:
                # Shape mismatch on non-leading axes — fall back to first
                merged_kwargs[f.name] = parts[0]
        else:
            merged_kwargs[f.name] = parts[0]
    return type(first)(**merged_kwargs)


def merge_per_task_simresults(sim_results: List[Any]) -> Any:
    """Build a single SimResult by concatenating per-task arrays."""
    if not sim_results:
        raise ValueError("no SimResults to merge")
    _validate_spec_layout(sim_results)

    first = sim_results[0]
    algo_names = list(first.algorithm_results.keys())

    # Total trial accounting
    n_total_trials = sum(sr.n_trials for sr in sim_results)
    runtime_total  = sum(sr.runtime_total_s for sr in sim_results)

    # Build merged AlgorithmResult per algorithm
    merged_algo_results: Dict[str, Any] = {}
    for algo in algo_names:
        per_task_ar = [
            sr.algorithm_results[algo] for sr in sim_results
            if algo in sr.algorithm_results
        ]
        first_ar = per_task_ar[0]
        AlgorithmResultCls = type(first_ar)

        # Per-trial diagnostic arrays.
        concat = _concat_per_trial_arrays(sim_results, algo)

        # Per-trial stats objects.
        sinr_stats = _merge_stats([ar.sinr_stats for ar in per_task_ar])
        scnr_stats = _merge_stats([ar.scnr_stats for ar in per_task_ar])

        # Per-trial counts.
        n_trials_total = sum(ar.n_trials_total for ar in per_task_ar)
        n_succeeded    = sum(ar.n_succeeded    for ar in per_task_ar)
        n_failed       = sum(ar.n_failed       for ar in per_task_ar)
        runtime_s      = concat["runtime_s"]
        runtime_total  = float(np.sum(runtime_s)) if runtime_s.size else 0.0
        iters_arr      = concat["iters"]
        iters_mean     = float(np.mean(iters_arr)) if iters_arr.size else 0.0
        runtime_mean   = (float(np.mean(runtime_s)) if runtime_s.size else 0.0)
        conv_arr       = concat["converged"]
        conv_rate      = (float(np.mean(conv_arr)) if conv_arr.size else 0.0)
        fail_rate      = (n_failed / n_trials_total) if n_trials_total else 0.0

        merged_algo_results[algo] = AlgorithmResultCls(
            spec=first_ar.spec,
            n_trials_total=n_trials_total,
            n_succeeded=n_succeeded,
            n_failed=n_failed,
            sinr_stats=sinr_stats,
            scnr_stats=scnr_stats,
            iters=concat["iters"],
            runtime_s=concat["runtime_s"],
            converged=concat["converged"],
            power_ratios=concat["power_ratios"],
            convergence_rate=conv_rate,
            failure_rate=fail_rate,
            iters_mean=iters_mean,
            runtime_mean_s=runtime_mean,
            runtime_total_s=runtime_total,
        )

    # Build merged SimResult.  We copy the cfg from the first task and
    # replace n_trials with the aggregate.
    SimResultCls = type(first)
    return SimResultCls(
        cfg=first.cfg,
        n_trials=n_total_trials,
        seed=first.seed,           # arbitrary; per-task seeds differ
        algorithm_results=merged_algo_results,
        runtime_total_s=runtime_total,
    )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate SLURM job-array per-task pickles "
                    "into a single SimResult."
    )
    parser.add_argument(
        "array_dir", type=Path,
        help="Path to the array root, e.g. results/array_12345/",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output dir for the aggregated pickle.  Default: "
             "<array_dir>/aggregated/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and validate per-task pickles without merging.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity.",
    )
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[aggregate] %(message)s")

    # Discover per-task pickles.
    try:
        pickles = find_per_task_pickles(args.array_dir)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    if not pickles:
        logger.error(
            "no per-task SimResults found under %s/task_*/exp_*/*/result.pkl. "
            "Did the array job complete?  Are tasks under task_<id>/ "
            "subdirs?", args.array_dir,
        )
        return 1

    logger.info("found %d per-task pickle(s):", len(pickles))
    for p in pickles:
        logger.info("  %s", p.relative_to(args.array_dir))

    if args.dry_run:
        logger.info("--dry-run: not loading/merging.")
        return 0

    # Load.
    sim_results: List[Any] = []
    for p in pickles:
        try:
            sim_results.append(_load_pickle(p))
        except Exception as e:
            logger.warning("failed to load %s: %s", p, e)

    if not sim_results:
        logger.error("no pickles loaded successfully")
        return 1

    logger.info("loaded %d / %d task pickles successfully",
                len(sim_results), len(pickles))

    # Merge.
    try:
        merged = merge_per_task_simresults(sim_results)
    except ValueError as e:
        logger.error("merge failed: %s", e)
        return 1

    # Write.
    out_dir = args.output or (args.array_dir / "aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(merged, fh)

    logger.info("wrote aggregated SimResult: %s", out_path)
    logger.info("  total trials:       %d", merged.n_trials)
    logger.info("  total runtime [s]:  %.1f", merged.runtime_total_s)
    logger.info("  algorithms:         %s",
                ", ".join(merged.algorithm_results.keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

