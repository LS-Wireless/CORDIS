#!/usr/bin/env python3
"""
scripts/validate_stage21.py
============================

Stage 21 — HPC3 job-array support.

After running this, the user has:

  - Existing single-job ``scripts/slurm/uci-hpc3/exp_*.sub`` bumped to
    40 CPUs / 8 h / no ``--mem`` (full node on HPC3 standard partition,
    proportional ~192 GB RAM).
  - 9 new ``scripts/slurm/uci-hpc3/array/exp_*.array.sub`` for the
    sweeps + CDFs (the experiments that benefit from per-trial
    parallelism), each with 40 CPUs / 4 h / 10 tasks default.
  - ``scripts/slurm/uci-hpc3/array/_array_common.sh`` shared
    per-task setup (computes SEED, N_TRIALS, OUTPUT_ROOT, loads env).
  - ``scripts/aggregate_array_batch.py`` to merge per-task pickles.

Tests verify each piece at three tiers:
  Tier 1: source-grep on regenerated files (resource directives,
          template inclusions)
  Tier 2: signature + structure (aggregator CLI, _array_common.sh
          variable computations)
  Tier 3: end-to-end (aggregator merges synthetic per-task pickles)

Run from the repo root::

    python3 scripts/validate_stage21.py
"""
from __future__ import annotations

import inspect
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn):
        _TESTS.append((label, fn))
        return fn
    return deco


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1: Resource directives on regenerated single-job .sub scripts
# ═════════════════════════════════════════════════════════════════════════════

EXISTING_SUB_NAMES = [
    "exp_antennas_sweep.sub", "exp_clutter_cnr_sweep.sub",
    "exp_convergence_trace.sub", "exp_fronthaul_table.sub",
    "exp_gamma_sweep.sub", "exp_kappa_sweep.sub", "exp_n_ap_sweep.sub",
    "exp_n_ue_sweep.sub", "exp_scnr_cdf.sub", "exp_sinr_cdf.sub",
    "exp_snr_sweep.sub",
]


@_register("Test  1: all 11 existing exp_*.sub scripts use 40 CPUs, "
           "8h, and have no --mem directive")
def test_01_existing_sub_resource_directives():
    sub_dir = REPO_ROOT / "scripts" / "slurm" / "uci-hpc3"
    for name in EXISTING_SUB_NAMES:
        path = sub_dir / name
        assert path.exists(), f"missing single-job .sub script: {path}"
        text = path.read_text()
        # 40 CPUs
        m = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
        assert m, f"{name}: missing #SBATCH --cpus-per-task directive"
        assert int(m.group(1)) == 40, (
            f"{name}: cpus-per-task should be 40 (Stage 21); got {m.group(1)}"
        )
        # 8h time
        m = re.search(r"#SBATCH --time=(\d+):(\d+):(\d+)", text)
        assert m, f"{name}: missing #SBATCH --time directive"
        hours = int(m.group(1))
        assert hours == 8, (
            f"{name}: --time should be 8 hours (Stage 21); got {hours}h"
        )
        # No --mem
        assert "#SBATCH --mem" not in text, (
            f"{name}: --mem directive should be removed (Stage 21).  "
            f"With 40 CPUs on HPC3 standard partition (40c/192GB), "
            f"proportional RAM default is ~192 GB — specifying --mem "
            f"artificially caps it."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1: Array .sub scripts exist + have expected resource directives
# ═════════════════════════════════════════════════════════════════════════════

ARRAY_SUB_NAMES = [
    "exp_antennas_sweep.array.sub",
    "exp_clutter_cnr_sweep.array.sub",
    "exp_gamma_sweep.array.sub",
    "exp_kappa_sweep.array.sub",
    "exp_n_ap_sweep.array.sub",
    "exp_n_ue_sweep.array.sub",
    "exp_scnr_cdf.array.sub",
    "exp_sinr_cdf.array.sub",
    "exp_snr_sweep.array.sub",
]


@_register("Test  2: all 9 array .sub scripts exist in "
           "scripts/slurm/uci-hpc3/array/")
def test_02_array_sub_scripts_exist():
    array_dir = REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
    assert array_dir.is_dir(), (
        f"array sub-directory must exist: {array_dir}.  Run "
        f"scripts/regenerate_experiment_scripts.py to generate."
    )
    for name in ARRAY_SUB_NAMES:
        assert (array_dir / name).exists(), (
            f"missing array .sub script: {array_dir / name}"
        )


@_register("Test  3: array .sub scripts use 40 CPUs, 4h, and have a "
           "#SBATCH --array=0-N directive")
def test_03_array_sub_resource_directives():
    array_dir = REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
    for name in ARRAY_SUB_NAMES:
        text = (array_dir / name).read_text()
        # 40 CPUs
        m = re.search(r"#SBATCH --cpus-per-task=(\d+)", text)
        assert m and int(m.group(1)) == 40, (
            f"{name}: cpus-per-task should be 40; got "
            f"{m.group(1) if m else 'MISSING'}"
        )
        # 4h time
        m = re.search(r"#SBATCH --time=(\d+):", text)
        assert m and int(m.group(1)) == 4, (
            f"{name}: --time should be 4h (Stage 21 array default); "
            f"got {m.group(1) if m else 'MISSING'}h"
        )
        # --array=0-N
        m = re.search(r"#SBATCH --array=(\d+)-(\d+)", text)
        assert m, f"{name}: missing #SBATCH --array directive"
        # No --mem
        assert "#SBATCH --mem" not in text, (
            f"{name}: --mem directive should not be present"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1: _array_common.sh + per-task variable derivation
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  4: _array_common.sh exists and defines SEED, N_TRIALS, "
           "OUTPUT_ROOT, N_WORKERS, TAG export variables")
def test_04_array_common_exists_and_defines_per_task_vars():
    common = (REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
              / "_array_common.sh")
    assert common.exists(), f"missing: {common}"
    text = common.read_text()
    for var in ("SEED", "N_TRIALS", "OUTPUT_ROOT", "N_WORKERS", "TAG"):
        # Must be exported (export VAR=... or be referenced with export)
        assert re.search(rf"^export {var}=", text, re.M), (
            f"_array_common.sh must `export {var}=...`  so the value is "
            f"visible to the python runner spawned by scripts/exp_*.sh"
        )


@_register("Test  5: _array_common.sh derives SEED = BASE_SEED + "
           "ARRAY_TASK_ID so per-task seeds are distinct")
def test_05_array_common_seed_derivation():
    common = (REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
              / "_array_common.sh")
    text = common.read_text()
    # Look for the seed-offset pattern.
    assert re.search(r"SEED=\$\(\(\s*BASE_SEED\s*\+\s*ARRAY_TASK_ID\s*\)\)", text), (
        "_array_common.sh must derive SEED as $((BASE_SEED + "
        "ARRAY_TASK_ID)) so each task uses an independent "
        "numpy.random.SeedSequence stream."
    )


@_register("Test  6: _array_common.sh derives N_TRIALS = "
           "ceil(N_TRIALS_TOTAL / N_ARRAY_TASKS)")
def test_06_array_common_trial_split():
    common = (REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
              / "_array_common.sh")
    text = common.read_text()
    # Ceiling division: (N_TRIALS_TOTAL + N_ARRAY_TASKS - 1) / N_ARRAY_TASKS
    pattern = (r"N_TRIALS=\$\(\(\s*\(\s*N_TRIALS_TOTAL\s*\+\s*"
               r"N_ARRAY_TASKS\s*-\s*1\s*\)\s*/\s*N_ARRAY_TASKS\s*\)\)")
    assert re.search(pattern, text), (
        "_array_common.sh must compute per-task trials via ceiling "
        "division so the final task picks up any remainder when "
        "N_TRIALS_TOTAL isn't divisible by N_ARRAY_TASKS"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 2: Aggregator script
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  7: scripts/aggregate_array_batch.py exists and exposes "
           "find_per_task_dirs + merge_experiment_results")
def test_07_aggregator_signature():
    agg_path = REPO_ROOT / "scripts" / "aggregate_array_batch.py"
    assert agg_path.exists(), f"missing aggregator: {agg_path}"
    # Import the module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aggregate_array_batch", agg_path,
    )
    if spec is None or spec.loader is None:
        raise _SkipTest(f"could not import {agg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "find_per_task_dirs"), (
        "aggregator must expose find_per_task_dirs(exp_dir, array_id) -> [Path]"
    )
    assert hasattr(mod, "merge_experiment_results"), (
        "aggregator must expose merge_experiment_results(task_dirs, out_dir)"
    )
    # The aggregator should be pure-numpy — no `cordis` import.
    # This protects against future regressions where someone adds a
    # `from cordis...` line and re-introduces the heavy import chain.
    src = agg_path.read_text()
    assert "import cordis" not in src and "from cordis" not in src, (
        "aggregate_array_batch.py must NOT import cordis — it's a "
        "leaf utility that should work in any env with numpy + json. "
        "Found a 'cordis' import; remove it."
    )
    sig = inspect.signature(mod.find_per_task_dirs)
    params = list(sig.parameters.keys())
    for required in ("exp_dir", "array_id"):
        assert required in params, (
            f"find_per_task_dirs should take {required!r}; got {params}"
        )


@_register("Test  8: aggregator CLI --help runs without error")
def test_08_aggregator_help():
    agg_path = REPO_ROOT / "scripts" / "aggregate_array_batch.py"
    result = subprocess.run(
        [sys.executable, str(agg_path), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"aggregator --help failed: stderr={result.stderr[:500]}"
    )
    # Stage 21 v2: two positional args (exp_name + array_id)
    assert "exp_name" in result.stdout and "array_id" in result.stdout, (
        f"aggregator --help should mention exp_name + array_id positionals; "
        f"got: {result.stdout[:500]}"
    )


@_register("Test  9: aggregator gracefully handles non-existent "
           "results-root (returns non-zero exit, doesn't crash)")
def test_09_aggregator_missing_dir():
    agg_path = REPO_ROOT / "scripts" / "aggregate_array_batch.py"
    result = subprocess.run(
        [sys.executable, str(agg_path),
         "sinr_cdf", "99999",
         "--results-root", "/nonexistent/path/xyz"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, (
        "aggregator should fail (non-zero exit) on missing results-root; "
        "got 0"
    )


@_register("Test 10: aggregator --dry-run discovers per-task dirs under "
           "the new results/exp_<name>/array_<id>_task_*/ layout")
def test_10_aggregator_dryrun_discovery():
    """Create a fake layout matching the experiment-first array workflow,
    invoke the aggregator with --dry-run, expect zero exit + per-task
    dir names in output."""
    agg_path = REPO_ROOT / "scripts" / "aggregate_array_batch.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        results_root = Path(tmpdir) / "results"
        exp_dir = results_root / "exp_sinr_cdf"
        for task_id in range(3):
            (exp_dir / f"array_99999_task_{task_id}").mkdir(parents=True)

        result = subprocess.run(
            [sys.executable, str(agg_path),
             "sinr_cdf", "99999",
             "--results-root", str(results_root),
             "--dry-run"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"--dry-run should succeed on fake layout; "
            f"stderr={result.stderr[:500]}"
        )
        # Output should mention the 3 task dirs we created.
        combined = result.stdout + result.stderr
        for task_id in range(3):
            assert f"array_99999_task_{task_id}" in combined, (
                f"expected array_99999_task_{task_id} in --dry-run output; "
                f"got: {combined[:500]}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Tier 1: regenerator script knows about array variants
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test 11: regenerate_experiment_scripts.py defines "
           "ARRAY_ENABLED_EXPERIMENTS and the right experiments")
def test_11_regenerator_array_set():
    """Source-grep the regenerator to ensure the array set has the
    right 9 experiments (excludes convergence_trace + fronthaul_table)."""
    text = (REPO_ROOT / "scripts" / "regenerate_experiment_scripts.py").read_text()
    m = re.search(
        r"ARRAY_ENABLED_EXPERIMENTS\s*=\s*\{([^}]+)\}",
        text, re.S,
    )
    assert m, "regenerator must define ARRAY_ENABLED_EXPERIMENTS"
    body = m.group(1)
    must_include = {"sinr_cdf", "scnr_cdf", "gamma_sweep", "kappa_sweep",
                    "snr_sweep", "n_ap_sweep", "n_ue_sweep",
                    "antennas_sweep", "clutter_cnr_sweep"}
    must_exclude = {"convergence_trace", "fronthaul_table"}
    for name in must_include:
        assert f'"{name}"' in body, (
            f"ARRAY_ENABLED_EXPERIMENTS should include {name!r}; got body: {body}"
        )
    for name in must_exclude:
        assert f'"{name}"' not in body, (
            f"ARRAY_ENABLED_EXPERIMENTS should NOT include {name!r} — it "
            f"doesn't benefit from per-trial array parallelism"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 21 v2 — run-id plumbing for experiment-first directory layout
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test 12: _exp_common.py adds --run-id flag and passes it as "
           "timestamp to experiment_dir()")
def test_12_run_id_plumbing():
    src = (REPO_ROOT / "scripts" / "_exp_common.py").read_text()
    assert re.search(r'add_argument\(\s*"--run-id"', src), (
        "_exp_common.py must add --run-id CLI flag (Stage 21 v2): used by "
        "SLURM array tasks to give each task a predictable leaf-dir name "
        "in place of the auto-timestamp."
    )
    # Must be passed as `timestamp=args.run_id` to experiment_dir
    assert re.search(
        r"experiment_dir.*\n.*timestamp\s*=\s*args\.run_id",
        src, re.S,
    ), (
        "experiment_dir() must be called with timestamp=args.run_id so "
        "the runner honours the custom run-id when set"
    )


@_register("Test 13: _array_common.sh exports RUN_ID = "
           "array_<jobid>_task_<id> and OUTPUT_ROOT (not nested under array_<id>)")
def test_13_array_common_new_layout():
    common = (REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
              / "_array_common.sh")
    text = common.read_text()
    # RUN_ID composition
    assert re.search(
        r'export RUN_ID="array_\$\{ARRAY_JOB_ID\}_task_\$\{ARRAY_TASK_ID\}"',
        text,
    ), (
        "_array_common.sh must derive RUN_ID as "
        '"array_${ARRAY_JOB_ID}_task_${ARRAY_TASK_ID}" (Stage 21 v2 - '
        "experiment-first layout)"
    )
    # OUTPUT_ROOT defaults to 'results' (no array_<id> nesting)
    assert "export OUTPUT_ROOT=" in text
    # Should NOT include the old 'results/array_${ARRAY_JOB_ID}/task_' pattern
    assert "results/array_${ARRAY_JOB_ID}/task_" not in text, (
        "_array_common.sh must NOT use the old "
        "'results/array_<jobid>/task_<id>' nesting — Stage 21 v2 puts "
        "the experiment dir first.  Use RUN_ID + default OUTPUT_ROOT='results' "
        "and let the runner create exp_<name>/<run_id>/ automatically."
    )


@_register("Test 14: every exp_*.sh wrapper plumbs RUN_ID env var "
           "through to the --run-id CLI flag")
def test_14_exp_sh_run_id_plumbing():
    """Spot-check 3 representative wrappers (CDF + sweep + scnr) for
    the RUN_ID → --run-id pattern."""
    representative = ["exp_sinr_cdf.sh", "exp_gamma_sweep.sh",
                      "exp_scnr_cdf.sh"]
    for fname in representative:
        path = REPO_ROOT / "scripts" / fname
        assert path.exists(), f"missing wrapper: {path}"
        text = path.read_text()
        assert re.search(r'RUN_ID="\$\{RUN_ID:-\}"', text), (
            f"{fname} must read RUN_ID env var (Stage 21 v2)"
        )
        assert "--run-id" in text, (
            f"{fname} must pass --run-id flag to scripts/exp_*.py"
        )


@_register("Test 15: sync_results_from_hpc3.sh accepts --array-id "
           "filter for syncing one array's tasks + aggregated result")
def test_15_sync_array_id_flag():
    sync = (REPO_ROOT / "scripts" / "sync_results_from_hpc3.sh").read_text()
    assert "--array-id" in sync, (
        "sync_results_from_hpc3.sh must accept --array-id flag for "
        "Stage 21 v2 (lets you pull one array's results without "
        "the rest of the experiment's history)"
    )
    # The flag should require an EXPERIMENT scope (otherwise rsync filtering
    # gets ugly across multiple exp_*/ trees).
    assert re.search(r"--array-id requires.*EXPERIMENT", sync), (
        "sync script must require --array-id be paired with an "
        "EXPERIMENT positional argument"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 78)
    print(f"  Stage 21 validator — {len(_TESTS)} tests")
    print("=" * 78)

    passed = skipped = failed = 0
    failures: List[Tuple[str, str]] = []

    for label, fn in _TESTS:
        print(f"\n── {label} ──")
        try:
            fn()
            print("  ✓ PASSED")
            passed += 1
        except _SkipTest as e:
            print(f"  ⏭  SKIPPED  ({e})")
            skipped += 1
        except AssertionError as e:
            print("  ✗ FAILED")
            for line in str(e).splitlines():
                print(f"      {line}")
            failed += 1
            failures.append((label, str(e)))
        except Exception:
            print("  ✗ FAILED (unexpected exception)")
            tb = traceback.format_exc()
            print("      " + tb.replace("\n", "\n      ").rstrip())
            failed += 1
            failures.append((label, "unexpected exception"))

    print()
    print("=" * 78)
    if failed == 0:
        print(f"  ✓ ALL TESTS PASSED   "
              f"({passed}/{len(_TESTS)} passed, {skipped} skipped)")
    else:
        print(f"  ✗ {failed} FAILURE(S)   "
              f"({passed} passed, {skipped} skipped, {failed} failed)")
        print("    failed: " + ", ".join(lbl for lbl, _ in failures))
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

