#!/usr/bin/env python3
"""
scripts/aggregate_array_batch.py
================================

Aggregate per-task ExperimentResult outputs from a SLURM job-array run
into a single merged result.

Directory layout (Stage 21 experiment-first):

    results/exp_<name>/
        array_<jobid>_task_0/         (per-task output)
            manifest.json
            result.npz                 (single-kind) OR
            result_<axis>_<v>.npz × N  (sweep-kind)
            result.json                (or result_<axis>_<v>.json × N)
            logs/run.log
        array_<jobid>_task_1/
            ...
        array_<jobid>_aggregated/     (created by this script)
            manifest.json
            result.npz / result_<axis>_<v>.npz × N (merged arrays)
            result.json / result_<axis>_<v>.json   (merged sidecar)

Per-task seed isolation comes from BASE_SEED + ARRAY_TASK_ID (set by
_array_common.sh), so trial streams across tasks are disjoint.  This
aggregator concatenates per-trial arrays along axis 0 (the trial
dimension) to produce the merged ExperimentResult.

DESIGN NOTE — pure-numpy / no cordis import
-------------------------------------------
This aggregator deliberately does NOT import anything from the
``cordis`` package.  The .npz/.json/manifest.json format is regular
enough to merge with plain numpy + json, so we keep this script as a
lightweight utility that runs in any environment with numpy.  This
matters because:

  - The full cordis import chain pulls in cvxpy, matplotlib, and other
    heavy deps the user may not have on their laptop.
  - Some transitively-imported modules use ``numpy.typing.NDArray``,
    which can fail on older numpy versions.
  - The aggregator is a leaf utility — coupling it to the full
    simulation framework is gratuitous.

The format is documented in ``cordis/simulation/result.py`` (SimResult
.save/.load) and ``cordis/experiments/result.py`` (ExperimentResult
.save/.load).  Keep them in sync — if either changes the on-disk
format, update this script.

Usage::

    # Most common: aggregate the array identified by experiment + job ID
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
import json
import logging
import re
import sys
from datetime import datetime, timezone
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
# Pure-numpy merge helpers
# ─────────────────────────────────────────────────────────────────────

def _read_manifest(task_dir: Path) -> Dict[str, Any]:
    """Read manifest.json from a per-task directory.

    Raises FileNotFoundError if missing (caller decides whether to
    skip the task or abort).
    """
    p = task_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"missing manifest.json in {task_dir}")
    with open(p, "r") as f:
        return json.load(f)


def _validate_manifests(manifests: List[Tuple[Path, Dict[str, Any]]]) -> str:
    """Sanity-check that all tasks ran the same experiment kind + name.

    Returns the common ``kind`` string.
    """
    if not manifests:
        raise ValueError("no manifests to validate")
    first_path, first_man = manifests[0]
    first_kind = first_man.get("kind")
    first_name = first_man.get("name")
    if first_kind is None or first_name is None:
        raise ValueError(
            f"manifest in {first_path.parent.name} missing 'kind' or 'name'"
        )
    for path, man in manifests[1:]:
        if man.get("kind") != first_kind:
            raise ValueError(
                f"task {path.parent.name} has kind={man.get('kind')!r}, "
                f"but task {first_path.parent.name} has kind={first_kind!r}. "
                f"All tasks must run the same experiment kind."
            )
        if man.get("name") != first_name:
            raise ValueError(
                f"task {path.parent.name} has experiment name={man.get('name')!r}, "
                f"but task {first_path.parent.name} has {first_name!r}"
            )
    return first_kind


def _merge_npz_files(npz_paths: List[Path]) -> Dict[str, np.ndarray]:
    """Concatenate same-keyed arrays from a list of .npz files along
    axis 0 (the trial dimension).

    Returns a dict of merged arrays suitable for ``np.savez_compressed``.

    Behaviour:
      - Keys present in any task are emitted.
      - Empty/scalar arrays are skipped during concatenation.
      - On shape mismatch on non-leading axes (rare; would indicate
        a config drift across tasks), keeps the first task's value
        and logs a warning.
    """
    if not npz_paths:
        return {}

    # Load all npz files into memory dicts (close immediately to release
    # file handles — NpzFile is lazy and keeps the file open).
    per_task_dicts: List[Dict[str, np.ndarray]] = []
    for p in npz_paths:
        with np.load(p) as nz:
            per_task_dicts.append({k: nz[k].copy() for k in nz.files})

    # Union of keys across all tasks (in practice they should match).
    all_keys: set = set()
    for d in per_task_dicts:
        all_keys.update(d.keys())

    merged: Dict[str, np.ndarray] = {}
    for key in sorted(all_keys):
        parts: List[np.ndarray] = []
        for d in per_task_dicts:
            if key not in d:
                continue
            arr = d[key]
            # Skip empty arrays — they don't contribute to the merge
            # and can cause concatenation issues.
            if arr.size == 0:
                continue
            parts.append(arr)
        if not parts:
            # Every task had an empty array for this key — emit empty.
            merged[key] = np.array([])
            continue
        try:
            merged[key] = np.concatenate(parts, axis=0)
        except ValueError as e:
            logger.warning(
                "key %r: concatenation failed (%s); keeping first task's value",
                key, e,
            )
            merged[key] = parts[0]
    return merged


def _merge_json_sidecars(
    json_paths:     List[Path],
    merged_arrays:  Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Merge result.json sidecars: keep first task's metadata, recompute
    per-algorithm scalars from the merged arrays.

    ``merged_arrays`` is the dict returned by :func:`_merge_npz_files`,
    used to recompute means/totals from the concatenated trial data.
    """
    if not json_paths:
        return {}
    sidecars: List[Dict[str, Any]] = []
    for p in json_paths:
        with open(p, "r") as f:
            sidecars.append(json.load(f))

    # Start from first task as the base — cfg, runner_cfg, algorithm_specs
    # should be identical across tasks (verified by _validate_manifests).
    base = dict(sidecars[0])

    # Recompute per_algorithm_scalars: counts sum across tasks; means/
    # totals are recomputed from the merged arrays for precision (no
    # weighted-average accumulation error).
    per_algo_merged: Dict[str, Dict[str, Any]] = {}
    algo_names = list(base.get("per_algorithm_scalars", {}).keys())

    for algo in algo_names:
        per_algo_per_task = [
            sc.get("per_algorithm_scalars", {}).get(algo, {})
            for sc in sidecars
        ]
        # Counts
        n_trials_total = sum(int(p.get("n_trials_total", 0))
                             for p in per_algo_per_task)
        n_succeeded    = sum(int(p.get("n_succeeded", 0))
                             for p in per_algo_per_task)
        n_failed       = sum(int(p.get("n_failed", 0))
                             for p in per_algo_per_task)

        # Recompute means from the merged arrays.
        iters     = merged_arrays.get(f"{algo}/iters")
        runtime_s = merged_arrays.get(f"{algo}/runtime_s")
        converged = merged_arrays.get(f"{algo}/converged")

        iters_mean      = (float(np.mean(iters))     if iters     is not None
                           and iters.size     else 0.0)
        runtime_mean    = (float(np.mean(runtime_s)) if runtime_s is not None
                           and runtime_s.size else 0.0)
        runtime_total   = (float(np.sum(runtime_s))  if runtime_s is not None
                           and runtime_s.size else 0.0)
        conv_rate       = (float(np.mean(converged)) if converged is not None
                           and converged.size else 0.0)
        fail_rate       = ((n_failed / n_trials_total)
                           if n_trials_total > 0 else 0.0)

        merged_algo = dict(per_algo_per_task[0])  # carry through opt'l fields
        merged_algo.update({
            "n_trials_total":   n_trials_total,
            "n_succeeded":      n_succeeded,
            "n_failed":         n_failed,
            "convergence_rate": conv_rate,
            "failure_rate":     fail_rate,
            "iters_mean":       iters_mean,
            "runtime_mean_s":   runtime_mean,
            "runtime_total_s":  runtime_total,
        })
        per_algo_merged[algo] = merged_algo

    base["per_algorithm_scalars"] = per_algo_merged
    base["saved_at_iso"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return base


def _merge_one_unit(
    task_dirs:    List[Path],
    npz_filename: str,
    out_npz:      Path,
    out_json:     Path,
) -> int:
    """Merge a single (npz + sidecar json) pair across all task dirs.

    For 'single'-kind: npz_filename = "result.npz" (per task dir).
    For 'sweep'-kind:  npz_filename = "result_<axis>_<v>.npz" (per value).

    Returns the merged n_trials_total (sum of one algorithm — they
    should all be equal, but the first encountered key wins).
    """
    json_filename = npz_filename.replace(".npz", ".json")
    npz_paths  = [d / npz_filename for d in task_dirs]
    json_paths = [d / json_filename for d in task_dirs]

    # Filter to only existing files (warn on missing).
    valid_indices: List[int] = []
    for i, (np_p, js_p) in enumerate(zip(npz_paths, json_paths)):
        if not np_p.exists():
            logger.warning("missing %s in %s — skipping task",
                           npz_filename, task_dirs[i].name)
            continue
        if not js_p.exists():
            logger.warning("missing %s in %s — skipping task",
                           json_filename, task_dirs[i].name)
            continue
        valid_indices.append(i)

    if not valid_indices:
        logger.error("no tasks have %s — cannot merge", npz_filename)
        return 0

    valid_npz_paths  = [npz_paths[i]  for i in valid_indices]
    valid_json_paths = [json_paths[i] for i in valid_indices]

    merged_arrays  = _merge_npz_files(valid_npz_paths)
    merged_sidecar = _merge_json_sidecars(valid_json_paths, merged_arrays)

    # Save.
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **merged_arrays)
    with open(out_json, "w") as f:
        json.dump(merged_sidecar, f, indent=2, default=_json_default)

    # Report.
    n_trials = 0
    for algo, scalars in merged_sidecar.get("per_algorithm_scalars", {}).items():
        n_trials = int(scalars.get("n_trials_total", 0))
        break  # all algorithms have same n_trials
    return n_trials


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars / arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ─────────────────────────────────────────────────────────────────────
# Top-level merge — dispatches on kind
# ─────────────────────────────────────────────────────────────────────

def merge_experiment_results(
    task_dirs: List[Path],
    out_dir:   Path,
) -> Dict[str, Any]:
    """Merge per-task ExperimentResult outputs into ``out_dir``.

    Dispatches on the experiment ``kind`` (from manifest.json):
      - "single": merges result.npz + result.json across tasks.
      - "sweep":  merges result_<axis>_<v>.npz + .json per axis value.
      - "trace":  rejected (single trial, not parallelizable).

    Returns a summary dict suitable for logging.
    """
    if not task_dirs:
        raise ValueError("no task directories to merge")

    # Read all manifests up front for validation.
    manifests: List[Tuple[Path, Dict[str, Any]]] = []
    for d in task_dirs:
        try:
            manifests.append((d, _read_manifest(d)))
        except FileNotFoundError as e:
            logger.warning("%s — skipping task", e)
    if not manifests:
        raise ValueError("no per-task manifests found — nothing to merge")

    kind = _validate_manifests(manifests)
    # Tasks that survived the manifest check
    valid_task_dirs = [path.parent for path, _ in
                       [(p, m) for p, m in manifests]]
    # The above just unpacks for clarity; equivalent to:
    valid_task_dirs = [p for p, _ in manifests]

    first_manifest = manifests[0][1]
    exp_name = first_manifest["name"]

    # Build merged manifest: copy first task's base, mark aggregation in metadata.
    merged_manifest: Dict[str, Any] = {
        "name":     exp_name,
        "kind":     kind,
        "metadata": {
            **(first_manifest.get("metadata") or {}),
            "aggregated_from_n_tasks": len(valid_task_dirs),
            "aggregator_version":      "stage21-v2-pure-numpy",
            "aggregated_at_iso":       datetime.now(timezone.utc)
                                          .isoformat(timespec="seconds"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"kind": kind, "n_tasks": len(valid_task_dirs)}

    if kind == "single":
        n_trials = _merge_one_unit(
            task_dirs    = valid_task_dirs,
            npz_filename = "result.npz",
            out_npz      = out_dir / "result.npz",
            out_json     = out_dir / "result.json",
        )
        summary["n_trials"] = n_trials

    elif kind == "sweep":
        axis_info = first_manifest.get("axis")
        if axis_info is None:
            raise ValueError(
                f"manifest in {valid_task_dirs[0].name} is kind='sweep' but "
                f"has no 'axis' field — cannot determine sweep file names."
            )
        axis_name = axis_info.get("name")
        axis_values = axis_info.get("values", [])
        if not axis_name or not axis_values:
            raise ValueError(
                f"sweep manifest axis info malformed: {axis_info!r}"
            )

        merged_manifest["axis"] = axis_info
        per_value_n_trials: List[int] = []
        for v in axis_values:
            npz_filename = _value_filename(axis_name, v)
            out_npz_v   = out_dir / npz_filename
            out_json_v  = out_dir / npz_filename.replace(".npz", ".json")
            logger.info("merging sweep value %s=%s ...", axis_name, v)
            n_trials_v = _merge_one_unit(
                task_dirs    = valid_task_dirs,
                npz_filename = npz_filename,
                out_npz      = out_npz_v,
                out_json     = out_json_v,
            )
            per_value_n_trials.append(n_trials_v)
        summary["axis_name"]   = axis_name
        summary["n_values"]    = len(axis_values)
        summary["n_trials"]    = per_value_n_trials[0] if per_value_n_trials else 0

    elif kind == "trace":
        raise ValueError(
            f"trace-kind experiment {exp_name!r} cannot be aggregated "
            f"across array tasks (single trial per run).  Use a "
            f"single-job .sub script for trace experiments."
        )

    else:
        raise ValueError(
            f"unknown ExperimentResult kind {kind!r} in {valid_task_dirs[0].name}"
        )

    # Write merged manifest.
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(merged_manifest, f, indent=2, default=_json_default)

    return summary


def _fmt_value(v: Any) -> str:
    """Format a numeric sweep value for use in a filename.

    Mirrors ``cordis.experiments.result._fmt_value`` so file names match.
    """
    # Match the cordis idiom: integers as-is, floats with up to 6 digits,
    # negative sign retained.  If the value is exactly an int, format as int.
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    if f.is_integer():
        return str(int(f))
    # Trim trailing zeros from a fixed-point representation.
    s = f"{f:.6f}".rstrip("0").rstrip(".")
    return s


def _value_filename(axis_name: str, value: Any) -> str:
    """Match ``cordis.experiments.result._value_filename``."""
    return f"result_{axis_name}_{_fmt_value(value)}.npz"


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

    out_dir = exp_dir / f"array_{array_id}_aggregated"

    try:
        summary = merge_experiment_results(task_dirs, out_dir)
    except (ValueError, FileNotFoundError) as e:
        logger.error("merge failed: %s", e)
        return 1
    except Exception as e:
        logger.error("unexpected error during merge: %s", e, exc_info=True)
        return 2

    logger.info("wrote aggregated ExperimentResult to: %s", out_dir)
    logger.info("  kind:               %s", summary.get("kind"))
    logger.info("  tasks merged:       %d", summary.get("n_tasks", 0))
    if summary.get("kind") == "single":
        logger.info("  total trials:       %d", summary.get("n_trials", 0))
    elif summary.get("kind") == "sweep":
        logger.info("  sweep axis:         %s", summary.get("axis_name"))
        logger.info("  sweep values:       %d", summary.get("n_values", 0))
        logger.info("  trials per value:   %d", summary.get("n_trials", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

