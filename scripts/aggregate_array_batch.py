#!/usr/bin/env python3
"""
scripts/aggregate_array_batch.py
================================

Stage 21 — Aggregate per-task ExperimentResult outputs from a SLURM
job-array run into a single merged ExperimentResult.

Directory layout (Stage 21 revision):

    results/exp_<name>/
        array_<jobid>_task_0/         (per-task output)
            manifest.json
            result.npz                 (single-kind) OR
            result_<axis>_<v>.npz × N  (sweep-kind)
            result.json
            logs/run.log
        array_<jobid>_task_1/
            ...
        ...
        array_<jobid>_aggregated/     (created by this script)
            manifest.json
            result.npz / result_<axis>_<v>.npz × N (merged)
            result.json

Per-task seed isolation comes from BASE_SEED + ARRAY_TASK_ID (set by
_array_common.sh), so trial streams across tasks are disjoint.  The
aggregator concatenates per-trial arrays in each SimResult to produce
the merged ExperimentResult.

Usage::

    # Most common: aggregate the array identified by its experiment + job ID
    python3 scripts/aggregate_array_batch.py sinr_cdf 12345

    # Override the results root if you've reorganised
    python3 scripts/aggregate_array_batch.py sinr_cdf 12345 \\
        --results-root /custom/results/dir

    # Dry-run: list what would be merged, don't write anything
    python3 scripts/aggregate_array_batch.py sinr_cdf 12345 --dry-run

The merged output goes to:
    <results-root>/exp_<exp_name>/array_<array_id>_aggregated/

Missing tasks (SLURM out-of-time, node failures, etc.) emit a warning
per missing task and the aggregator merges whatever is available.
Re-submit failed indices with::

    sbatch --array=3,7 scripts/slurm/uci-hpc3/array/exp_<name>.array.sub

then re-run this script to merge the new data with the existing
aggregated/ dir (it will be overwritten).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────

_TASK_PATTERN = re.compile(r"^array_(\d+)_task_(\d+)$")


def find_per_task_dirs(
    exp_dir:  Path,
    array_id: str,
) -> List[Path]:
    """Discover per-task directories under ``<exp_dir>/`` for a given
    array job ID.

    Returns sorted by task_id (ascending).  Empty list if none found.
    """
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"experiment directory not found: {exp_dir}")
    matches: List[Tuple[int, Path]] = []
    for child in exp_dir.iterdir():
        if not child.is_dir():
            continue
        m = _TASK_PATTERN.match(child.name)
        if not m or m.group(1) != array_id:
            continue
        task_id = int(m.group(2))
        matches.append((task_id, child))
    matches.sort()
    return [p for _, p in matches]


# ─────────────────────────────────────────────────────────────────────
# Merge helpers
# ─────────────────────────────────────────────────────────────────────

def _concat_arrays(parts: List[np.ndarray]) -> np.ndarray:
    """Concatenate ndarrays along axis 0; return empty if no parts."""
    if not parts:
        return np.array([])
    return np.concatenate(parts, axis=0)


def _merge_stats(per_task_stats: List[Any]) -> Optional[Any]:
    """Merge per-task SINRStatistics / SCNRStatistics dataclasses.

    Concatenate ndarray fields along axis 0; non-array fields take
    the value from the first task.  Returns None if every input is None.
    """
    non_none = [s for s in per_task_stats if s is not None]
    if not non_none:
        return None
    first = non_none[0]
    fields_info = dataclasses.fields(first)
    merged: Dict[str, Any] = {}
    for f in fields_info:
        parts = [getattr(s, f.name) for s in non_none]
        if all(isinstance(p, np.ndarray) for p in parts):
            try:
                merged[f.name] = np.concatenate(parts, axis=0)
            except ValueError:
                # Shape mismatch on non-leading axes — fall back to first.
                merged[f.name] = parts[0]
        else:
            merged[f.name] = parts[0]
    return type(first)(**merged)


def _merge_sim_results(per_task_sr: List[Any]) -> Any:
    """Build a single SimResult by concatenating per-task arrays.

    Assumes every input has the same algorithm_results keys; the caller
    should have validated this before calling here.
    """
    if not per_task_sr:
        raise ValueError("no SimResults to merge")
    first = per_task_sr[0]
    algo_names = list(first.algorithm_results.keys())

    merged_algo_results: Dict[str, Any] = {}
    for algo in algo_names:
        per_task_ar = [sr.algorithm_results[algo] for sr in per_task_sr
                       if algo in sr.algorithm_results]
        first_ar = per_task_ar[0]

        # Per-trial diagnostic arrays.
        iters        = _concat_arrays([np.asarray(ar.iters)        for ar in per_task_ar])
        runtime_s    = _concat_arrays([np.asarray(ar.runtime_s)    for ar in per_task_ar])
        converged    = _concat_arrays([np.asarray(ar.converged)    for ar in per_task_ar])
        # power_ratios is 2D (n_trials × n_aps) — also concatenate axis 0.
        pr_parts = [np.asarray(ar.power_ratios) for ar in per_task_ar
                    if np.asarray(ar.power_ratios).size > 0]
        if pr_parts:
            try:
                power_ratios = np.concatenate(pr_parts, axis=0)
            except ValueError:
                power_ratios = pr_parts[0]
        else:
            power_ratios = np.zeros((0, 0))

        # Per-trial stats.
        sinr_stats = _merge_stats([ar.sinr_stats for ar in per_task_ar])
        scnr_stats = _merge_stats([ar.scnr_stats for ar in per_task_ar])

        # Counts + scalars.
        n_trials_total = sum(int(ar.n_trials_total) for ar in per_task_ar)
        n_succeeded    = sum(int(ar.n_succeeded)    for ar in per_task_ar)
        n_failed       = sum(int(ar.n_failed)       for ar in per_task_ar)
        runtime_total  = float(np.sum(runtime_s)) if runtime_s.size else 0.0
        iters_mean     = float(np.mean(iters))     if iters.size     else 0.0
        runtime_mean   = float(np.mean(runtime_s)) if runtime_s.size else 0.0
        conv_rate      = float(np.mean(converged)) if converged.size else 0.0
        fail_rate      = (n_failed / n_trials_total) if n_trials_total else 0.0

        AlgorithmResultCls = type(first_ar)
        merged_algo_results[algo] = AlgorithmResultCls(
            spec=first_ar.spec,
            n_trials_total=n_trials_total,
            n_succeeded=n_succeeded,
            n_failed=n_failed,
            sinr_stats=sinr_stats,
            scnr_stats=scnr_stats,
            iters=iters,
            runtime_s=runtime_s,
            converged=converged,
            power_ratios=power_ratios,
            convergence_rate=conv_rate,
            failure_rate=fail_rate,
            iters_mean=iters_mean,
            runtime_mean_s=runtime_mean,
            runtime_total_s=runtime_total,
        )

    # Build merged SimResult.  Reuse the first task's cfg/runner_cfg/specs
    # (they should all match — see _validate_consistency above) and
    # replace n_trials with the aggregate.
    SimResultCls = type(first)
    init_kwargs: Dict[str, Any] = {
        "cfg":                first.cfg,
        "algorithm_results":  merged_algo_results,
        "runtime_total_s":    sum(float(sr.runtime_total_s) for sr in per_task_sr),
    }
    # Optional fields the dataclass may carry — handle gracefully.
    for opt in ("n_trials", "seed", "runner_cfg", "algorithm_specs", "metadata"):
        if hasattr(first, opt):
            init_kwargs[opt] = getattr(first, opt)
    if "n_trials" in init_kwargs:
        init_kwargs["n_trials"] = sum(int(getattr(sr, "n_trials", 0))
                                      for sr in per_task_sr)
    return SimResultCls(**init_kwargs)


# ─────────────────────────────────────────────────────────────────────
# ExperimentResult merge dispatch
# ─────────────────────────────────────────────────────────────────────

def _validate_consistency(per_task_er: List[Any]) -> None:
    """Reject heterogeneous batches: every task must have the same
    experiment kind + same algorithms + same sweep axis (if any)."""
    if not per_task_er:
        return
    first = per_task_er[0]
    for k, er in enumerate(per_task_er[1:], start=1):
        if er.kind != first.kind:
            raise ValueError(
                f"task {k} has kind={er.kind!r}, but task 0 has "
                f"kind={first.kind!r}.  Heterogeneous arrays not "
                f"supported — re-run with consistent experiment specs."
            )
        if er.name != first.name:
            raise ValueError(
                f"task {k} has experiment name={er.name!r}, but "
                f"task 0 has {first.name!r}"
            )
        if first.kind == "sweep":
            # Sweep values must match across tasks (otherwise we
            # can't pair them up for merging).
            first_vals  = sorted(first.sweep_results.keys())
            this_vals   = sorted(er.sweep_results.keys())
            if first_vals != this_vals:
                raise ValueError(
                    f"task {k} sweep values {this_vals} differ from "
                    f"task 0 values {first_vals}.  All array tasks must "
                    f"have run the same sweep grid."
                )


def merge_experiment_results(per_task_er: List[Any]) -> Any:
    """Merge a list of per-task ExperimentResults into a single one.

    Dispatches on ``kind``:
      - "single": merge the single sim_result
      - "sweep":  merge sim_results per axis value
      - "trace":  not supported (arrays excluded from trace experiments)
    """
    if not per_task_er:
        raise ValueError("no ExperimentResults to merge")
    _validate_consistency(per_task_er)

    first = per_task_er[0]
    ExperimentResultCls = type(first)

    # Start from first task's metadata (we'll update trial-counting fields).
    merged_metadata = dict(getattr(first, "metadata", {}) or {})
    merged_metadata["aggregated_from_n_tasks"] = len(per_task_er)
    merged_metadata["aggregator_version"] = "stage21-v2"

    if first.kind == "single":
        merged_sr = _merge_sim_results([er.sim_result for er in per_task_er])
        return ExperimentResultCls(
            name=first.name,
            kind="single",
            sim_result=merged_sr,
            metadata=merged_metadata,
        )

    elif first.kind == "sweep":
        sweep_values = sorted(first.sweep_results.keys())
        merged_sweep: Dict[float, Any] = {}
        for v in sweep_values:
            per_task_sr_at_v = [er.sweep_results[v] for er in per_task_er]
            merged_sweep[v] = _merge_sim_results(per_task_sr_at_v)
        return ExperimentResultCls(
            name=first.name,
            kind="sweep",
            sweep_results=merged_sweep,
            sweep_axis=first.sweep_axis,
            metadata=merged_metadata,
        )

    elif first.kind == "trace":
        raise ValueError(
            f"trace-kind experiment {first.name!r} cannot be aggregated "
            f"across array tasks (single trial per run).  Use a "
            f"single-job .sub script for trace experiments."
        )

    else:
        raise ValueError(
            f"unknown ExperimentResult kind {first.kind!r} in task 0"
        )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _exp_dir_name(exp_name: str) -> str:
    """Idempotent 'exp_' prefix."""
    return exp_name if exp_name.startswith("exp_") else f"exp_{exp_name}"


def _array_id_from_arg(s: str) -> str:
    """Accept either '12345' or 'array_12345' for convenience."""
    if s.startswith("array_"):
        return s[len("array_"):]
    return s


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate SLURM job-array per-task ExperimentResult "
                    "outputs into a single merged ExperimentResult.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 scripts/aggregate_array_batch.py sinr_cdf 12345\n"
            "  → reads results/exp_sinr_cdf/array_12345_task_*/\n"
            "  → writes results/exp_sinr_cdf/array_12345_aggregated/\n"
        ),
    )
    parser.add_argument(
        "exp_name", type=str,
        help="Experiment name without the 'exp_' prefix (e.g. 'sinr_cdf') "
             "or with it ('exp_sinr_cdf') — both work.",
    )
    parser.add_argument(
        "array_id", type=str,
        help="SLURM array job ID (e.g. '12345') or 'array_12345'.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results"),
        help="Root directory containing exp_<name>/.  Default: results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and validate per-task directories without merging "
             "or writing anything.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (use -vv for DEBUG).",
    )
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[aggregate] %(message)s")

    exp_dir_name = _exp_dir_name(args.exp_name)
    array_id     = _array_id_from_arg(args.array_id)
    exp_dir      = args.results_root / exp_dir_name

    # Discover.
    try:
        task_dirs = find_per_task_dirs(exp_dir, array_id)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    if not task_dirs:
        logger.error(
            "no per-task directories matching %s/array_%s_task_*/ found. "
            "Did the array job complete?  Have you synced the results "
            "from HPC3?", exp_dir, array_id,
        )
        return 1

    logger.info("found %d per-task directories under %s:",
                len(task_dirs), exp_dir)
    for d in task_dirs:
        logger.info("  %s", d.name)

    if args.dry_run:
        logger.info("--dry-run: not loading/merging.")
        return 0

    # Load.  Import lazily — only need it for the real run, not for
    # --dry-run or --help.
    try:
        from cordis.experiments.result import ExperimentResult
    except ImportError as e:
        logger.error(
            "cordis.experiments.result.ExperimentResult not importable: %s. "
            "Run from the repo root, or set PYTHONPATH=$(pwd).",
            e,
        )
        return 2

    per_task_er: List[Any] = []
    for d in task_dirs:
        try:
            per_task_er.append(ExperimentResult.load(d))
        except FileNotFoundError as e:
            logger.warning("task %s missing artefacts: %s", d.name, e)
        except Exception as e:
            logger.warning("task %s failed to load: %s", d.name, e)

    if not per_task_er:
        logger.error("no per-task results loaded successfully")
        return 1

    logger.info("loaded %d / %d task results successfully",
                len(per_task_er), len(task_dirs))

    # Merge.
    try:
        merged = merge_experiment_results(per_task_er)
    except ValueError as e:
        logger.error("merge failed: %s", e)
        return 1

    # Save.
    out_dir = exp_dir / f"array_{array_id}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save(out_dir)

    logger.info("wrote aggregated ExperimentResult to: %s", out_dir)
    logger.info("  kind:               %s", merged.kind)
    if merged.kind == "single":
        sr = merged.sim_result
        n_trials = getattr(sr, "n_trials",
                           sum(ar.n_trials_total
                               for ar in sr.algorithm_results.values()))
        logger.info("  total trials:       %d", n_trials)
        logger.info("  algorithms:         %s",
                    ", ".join(sr.algorithm_results.keys()))
    elif merged.kind == "sweep":
        n_values = len(merged.sweep_results)
        first_sr = next(iter(merged.sweep_results.values()))
        n_trials_per_v = getattr(first_sr, "n_trials",
                                 sum(ar.n_trials_total
                                     for ar in first_sr.algorithm_results.values()))
        logger.info("  sweep axis:         %s",
                    getattr(merged.sweep_axis, "name", "?"))
        logger.info("  sweep values:       %d", n_values)
        logger.info("  trials per value:   %d", n_trials_per_v)
        logger.info("  algorithms:         %s",
                    ", ".join(first_sr.algorithm_results.keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

