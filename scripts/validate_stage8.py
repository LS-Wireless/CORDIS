#!/usr/bin/env python3
"""
scripts/validate_stage8.py
==========================

Umbrella validator: runs every CORDIS sub-validator in sequence and
reports an aggregate result.  The name is historical (it originally
covered Stage 8 sub-stages only) — it now drives every stage with a
matching ``validate_stage<tag>.py``:

* **8a** — paper-style plotting primitives (``cordis.plotting``)
* **8b** — experiment registry, runner, and result I/O
* **8c** — per-experiment runner / plot scripts (paper figures)
* **8d** — Makefile, SLURM wrappers, this umbrella validator
* **9**  — config-cleanup: n_trials decomposition, gamma_db consolidation,
           sigma_clt override semantics
* **10** — named spec-set selection (``--specs`` CLI flag across all experiments)
* **11** — UCI HPC3 site-specific SLURM scripts (``scripts/slurm/uci-hpc3/``)
* **12** — playground notebooks (one per result kind, under ``notebooks/``)
* **13** — HPC3-to-laptop sync script (``scripts/sync_results_from_hpc3.sh``)
* **14** — plotting fixes: ``plot_cdf``/``plot_admm_convergence`` default
  to ``legend_loc="best"``; notebooks use ``save_figure`` (not the
  non-existent ``save_paper_figure``)
* **15** — AST-based call-signature validation for playground notebooks;
  catches wrong kwargs / arity / metric names before they ship
* **16** — experiment-dispatch bug fixes: ``_set_field`` preserves int
  destination types, ``sweep_config_field`` re-validates per point,
  ``_exp_common.py`` filters ``n_drops``/``n_realizations`` by signature
* **17** — experiment polish: uniform ``spec_set="all_algorithms"``
  defaults across CDF/sweep experiments; per-iteration progress bar
  in ``solve_cordis_admm`` (used by ``convergence_trace``); fronthaul
  table now stores per-round real-scalar counts + iterations field
  (apples-to-apples bar chart across algorithms)
* **18** — notebook polish: ``load_result(name, exp_dir=None)`` helper
  for explicit-folder loading; CDF + sweep playgrounds parameterized
  via ``exp_name`` so switching between metric families / sweep params
  is one edit; ``best_iter`` field added to ADMMResult + safe getattr-
  guarded usage in trace notebook

Each sub-validator is a self-contained script under ``scripts/``.  This
runner invokes them via ``subprocess`` so each runs in its own process
state.  Aggregate exit code: 0 iff every sub-validator returns 0.

Usage::

    python3 scripts/validate_stage8.py                # run them all
    python3 scripts/validate_stage8.py --list         # list sub-validators
    python3 scripts/validate_stage8.py --only 9       # run only Stage 9
    python3 scripts/validate_stage8.py --stop-on-fail # halt after first failure
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Sub-validators, in execution order.  Tag (e.g. "8a") must match the
# suffix of validate_stage<tag>.py for --only filtering to work.
SUBVALIDATORS: List[Tuple[str, str]] = [
    ("8a", "validate_stage8a.py"),
    ("8b", "validate_stage8b.py"),
    ("8c", "validate_stage8c.py"),
    ("8d", "validate_stage8d.py"),
    ("9",  "validate_stage9.py"),
    ("10", "validate_stage10.py"),
    ("11", "validate_stage11.py"),
    ("12", "validate_stage12.py"),
    ("13", "validate_stage13.py"),
    ("14", "validate_stage14.py"),
    ("15", "validate_stage15.py"),
    ("16", "validate_stage16.py"),
    ("17", "validate_stage17.py"),
    ("18", "validate_stage18.py"),
    ("19", "validate_stage19.py"),
    ("20", "validate_stage20.py"),
]


def _subprocess_env() -> dict:
    """Env for sub-validators: REPO_ROOT on PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_stage8.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true",
                   help="List sub-validators and exit.")
    p.add_argument("--only", nargs="+", metavar="TAG", default=None,
                   help="Run only the given sub-validators (e.g. 8b 8d).")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Halt at the first sub-validator that fails.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress sub-validator stdout (only print summary).")
    return p.parse_args()


def _filter_subvalidators(only: List[str] | None
                          ) -> List[Tuple[str, str]]:
    if only is None:
        return SUBVALIDATORS
    only_set = set(only)
    known = {tag for tag, _ in SUBVALIDATORS}
    unknown = only_set - known
    if unknown:
        sys.exit(f"unknown --only tag(s): {sorted(unknown)} "
                 f"(known: {sorted(known)})")
    return [(t, f) for (t, f) in SUBVALIDATORS if t in only_set]


def _run_one(tag: str, fname: str, *, quiet: bool) -> Tuple[bool, float]:
    """Run one sub-validator, return (passed, duration_seconds)."""
    path = REPO_ROOT / "scripts" / fname
    if not path.exists():
        print(f"  ✗ {fname} not found at {path}")
        return False, 0.0

    t0 = time.monotonic()
    if quiet:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
    else:
        # Stream live so the user sees progress.
        proc = subprocess.run(
            [sys.executable, str(path)],
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
    dt = time.monotonic() - t0
    return (proc.returncode == 0), dt


def main() -> int:
    args = _parse_args()

    if args.list:
        print("Stage 8 sub-validators (execution order):")
        for tag, fname in SUBVALIDATORS:
            present = "✓" if (REPO_ROOT / "scripts" / fname).exists() else "✗ MISSING"
            print(f"  {tag}   scripts/{fname:<25s}  [{present}]")
        return 0

    targets = _filter_subvalidators(args.only)

    print("=" * 78)
    print(f"  Stage 8 umbrella validator — running {len(targets)} sub-validator(s)")
    print("=" * 78)

    results: List[Tuple[str, str, bool, float]] = []
    for tag, fname in targets:
        print(f"\n┌── {tag}: scripts/{fname} ────────────────────────")
        ok, dt = _run_one(tag, fname, quiet=args.quiet)
        results.append((tag, fname, ok, dt))
        status = "✓ PASSED" if ok else "✗ FAILED"
        print(f"└── {tag}: {status}   ({dt:.1f}s)")
        if not ok and args.stop_on_fail:
            print("\n--stop-on-fail set; halting.")
            break

    # ── Aggregate summary ──────────────────────────────────────────────
    passed = sum(1 for *_, ok, _ in results if ok)
    failed = sum(1 for *_, ok, _ in results if not ok)
    total_dt = sum(dt for *_, dt in results)

    print()
    print("=" * 78)
    if failed == 0:
        print(f"  ✓ ALL SUB-VALIDATORS PASSED   ({passed}/{len(results)})"
              f"   total wall time: {total_dt:.1f}s")
    else:
        print(f"  ✗ {failed} SUB-VALIDATOR(S) FAILED   "
              f"({passed} passed, {failed} failed)"
              f"   total wall time: {total_dt:.1f}s")
        failed_tags = [tag for tag, _, ok, _ in results if not ok]
        print(f"    failed: {', '.join(failed_tags)}")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

