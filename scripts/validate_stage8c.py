#!/usr/bin/env python3
"""
scripts/validate_stage8c.py
===========================

Comprehensive validation for the Stage 8c per-experiment scripts.

Three layers of testing:

* **Structural** — every expected file exists, in the right place,
  with the right shebang and (for ``.sh``) executable permissions.
* **Syntactic** — every shell script passes ``bash -n``; every
  Python script compiles; every Python script's ``--help`` renders
  without error (catches missing imports and broken argparse setup).
* **Functional** — runs a tiny end-to-end pipeline for two
  representative experiments (one CDF, one sweep) IF the full
  cordis install is available; skips with a clear reason if not.

Run::

    python3 scripts/validate_stage8c.py

Exit code is 0 iff every test passes (skips do not fail the run).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the repo root importable so `from cordis...` works when this
# script is invoked directly (Python only puts the script's directory
# on sys.path, not the cwd).  Matches validate_stage8b.py.
sys.path.insert(0, str(REPO_ROOT))

# The twelve experiments Stage 8c covers, in registry order.
EXPERIMENTS = [
    "sinr_cdf", "scnr_cdf",
    "gamma_sweep", "kappa_sweep", "clutter_cnr_sweep",
    "snr_sweep", "csi_sweep",
    "n_ue_sweep", "n_ap_sweep", "antennas_sweep",
    "convergence_trace", "fronthaul_table",
]


# ─────────────────────────────────────────────────────────────────────
# Test registration scaffolding
# ─────────────────────────────────────────────────────────────────────

class _SkipTest(Exception):
    """Raised by tests that decide they cannot meaningfully run."""


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def decorator(fn):
        _TESTS.append((label, fn))
        return fn
    return decorator


# ─────────────────────────────────────────────────────────────────────
# STRUCTURAL TESTS — files exist and are well-formed
# ─────────────────────────────────────────────────────────────────────

@_register("Test  1: shared infrastructure files exist")
def test_01_shared_infra_exists():
    for rel in [
        "scripts/_exp_common.py",
        "scripts/_plot_common.py",
        "configs/recipes/_defaults.sh",
        "scripts/regenerate_experiment_scripts.py",
    ]:
        path = REPO_ROOT / rel
        assert path.exists(), f"missing: {rel}"
        assert path.stat().st_size > 0, f"empty: {rel}"


@_register("Test  2: every experiment has all four files")
def test_02_per_experiment_files_exist():
    missing = []
    for name in EXPERIMENTS:
        for rel in [
            f"configs/recipes/exp_{name}.sh",
            f"scripts/exp_{name}.py",
            f"scripts/exp_{name}.sh",
            f"scripts/plot_{name}.py",
        ]:
            if not (REPO_ROOT / rel).exists():
                missing.append(rel)
    assert not missing, f"missing files:\n  " + "\n  ".join(missing)


@_register("Test  3: every shell script has a bash shebang and +x perm")
def test_03_shell_scripts_executable():
    bad_shebang, not_exec = [], []
    # Only check files this stage owns — leave pre-existing scripts alone.
    sh_files = (
        [REPO_ROOT / "configs/recipes" / f"exp_{n}.sh" for n in EXPERIMENTS] +
        [REPO_ROOT / "scripts"         / f"exp_{n}.sh" for n in EXPERIMENTS] +
        [REPO_ROOT / "configs/recipes" / "_defaults.sh"]
    )
    for path in sh_files:
        if not path.exists():
            continue
        first = path.read_text(encoding="utf-8").splitlines()[0]
        if not (first.startswith("#!/") and "bash" in first):
            bad_shebang.append(str(path.relative_to(REPO_ROOT)))
        # _-prefixed files (e.g. _defaults.sh) are sourced, not run.
        if path.name.startswith("_"):
            continue
        if not os.access(path, os.X_OK):
            not_exec.append(str(path.relative_to(REPO_ROOT)))
    assert not bad_shebang, f"missing bash shebang:\n  " + "\n  ".join(bad_shebang)
    assert not not_exec, f"missing +x:\n  " + "\n  ".join(not_exec)


@_register("Test  4: every Python script has a python3 shebang")
def test_04_python_scripts_shebang():
    bad = []
    py_files = (
        [REPO_ROOT / f"scripts/exp_{n}.py"  for n in EXPERIMENTS] +
        [REPO_ROOT / f"scripts/plot_{n}.py" for n in EXPERIMENTS]
    )
    for path in py_files:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        if not (first.startswith("#!/") and "python" in first):
            bad.append(str(path.relative_to(REPO_ROOT)))
    assert not bad, f"missing python shebang:\n  " + "\n  ".join(bad)


# ─────────────────────────────────────────────────────────────────────
# SYNTACTIC TESTS — bash -n, py_compile, --help
# ─────────────────────────────────────────────────────────────────────

def _subprocess_env() -> dict:
    """Env dict that makes ``cordis`` importable by subprocess Python.

    In production, ``cordis`` is installed via setup.py/pyproject.toml
    and importable from any cwd.  In test setups it may live only in
    the working tree, so prepend REPO_ROOT to PYTHONPATH explicitly.
    """
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(REPO_ROOT) + (os.pathsep + pp if pp else ""))
    return env

@_register("Test  5: all shell scripts pass bash -n")
def test_05_shell_syntax():
    fails = []
    sh_files = (
        list((REPO_ROOT / "configs/recipes").glob("*.sh")) +
        list((REPO_ROOT / "scripts").glob("*.sh"))
    )
    for path in sh_files:
        proc = subprocess.run(["bash", "-n", str(path)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            fails.append(f"{path.relative_to(REPO_ROOT)}: {proc.stderr.strip()}")
    assert not fails, "shell syntax errors:\n  " + "\n  ".join(fails)


@_register("Test  6: all Python scripts pass py_compile")
def test_06_python_syntax():
    import py_compile
    fails = []
    py_files = (
        [REPO_ROOT / "scripts/_exp_common.py",
         REPO_ROOT / "scripts/_plot_common.py"] +
        [REPO_ROOT / f"scripts/exp_{n}.py"  for n in EXPERIMENTS] +
        [REPO_ROOT / f"scripts/plot_{n}.py" for n in EXPERIMENTS]
    )
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            fails.append(f"{path.relative_to(REPO_ROOT)}: {e}")
    assert not fails, "python syntax errors:\n  " + "\n  ".join(fails)


@_register("Test  7: every exp_<name>.py renders --help")
def test_07_exp_help_renders():
    fails = []
    for name in EXPERIMENTS:
        path = REPO_ROOT / f"scripts/exp_{name}.py"
        proc = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, timeout=15,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            fails.append(f"exp_{name}.py: rc={proc.returncode}\n"
                         f"  stderr: {proc.stderr.strip()[:200]}")
        elif "usage:" not in proc.stdout.lower():
            fails.append(f"exp_{name}.py: --help did not print usage")
    assert not fails, "exp_<name>.py --help failures:\n  " + "\n  ".join(fails)


@_register("Test  8: every plot_<name>.py renders --help")
def test_08_plot_help_renders():
    # The plot wrappers import from cordis.plotting at module level.
    # Check importability IN A SUBPROCESS (matching the env that tests
    # the plot scripts use) so we skip when the real cordis stack is
    # only partially available.
    probe = subprocess.run(
        [sys.executable, "-c", "import cordis.plotting"],
        capture_output=True, text=True, timeout=15,
        env=_subprocess_env(),
        cwd=str(REPO_ROOT),
    )
    if probe.returncode != 0:
        raise _SkipTest(
            "cordis.plotting not importable in subprocess: "
            + probe.stderr.strip().splitlines()[-1][:120]
        )

    fails = []
    for name in EXPERIMENTS:
        path = REPO_ROOT / f"scripts/plot_{name}.py"
        proc = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, timeout=15,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            fails.append(f"plot_{name}.py: rc={proc.returncode}\n"
                         f"  stderr: {proc.stderr.strip()[:200]}")
        elif "usage:" not in proc.stdout.lower():
            fails.append(f"plot_{name}.py: --help did not print usage")
    assert not fails, "plot_<name>.py --help failures:\n  " + "\n  ".join(fails)


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC TESTS — code matches what the registry actually provides
# ─────────────────────────────────────────────────────────────────────

@_register("Test  9: EXPERIMENTS set matches the registry exactly")
def test_09_experiments_match_registry():
    try:
        from cordis.experiments import REGISTRY
    except ImportError as e:
        raise _SkipTest(f"cordis.experiments not importable: {e}")
    have_in_validate  = set(EXPERIMENTS)
    have_in_registry  = set(REGISTRY.keys())
    missing_in_validate = have_in_registry - have_in_validate
    extra_in_validate   = have_in_validate - have_in_registry
    assert not missing_in_validate, \
        f"registry has experiments missing from validate_stage8c.py: " \
        f"{sorted(missing_in_validate)}"
    assert not extra_in_validate, \
        f"validate_stage8c.py references unregistered experiments: " \
        f"{sorted(extra_in_validate)}"


@_register("Test 10: every recipe sources _defaults.sh")
def test_10_recipes_source_defaults():
    fails = []
    for name in EXPERIMENTS:
        recipe = REPO_ROOT / "configs/recipes" / f"exp_{name}.sh"
        txt = recipe.read_text(encoding="utf-8")
        if "_defaults.sh" not in txt:
            fails.append(f"exp_{name}.sh does not source _defaults.sh")
    assert not fails, "\n  ".join(fails)


@_register("Test 11: every recipe sets NAME=exp_<name>")
def test_11_recipes_set_name():
    fails = []
    for name in EXPERIMENTS:
        recipe = REPO_ROOT / "configs/recipes" / f"exp_{name}.sh"
        txt = recipe.read_text(encoding="utf-8")
        if f'NAME="exp_{name}"' not in txt:
            fails.append(f"exp_{name}.sh missing NAME=\"exp_{name}\"")
    assert not fails, "\n  ".join(fails)


@_register("Test 12: every exp_<name>.py calls registry's run_<name>")
def test_12_exp_py_calls_registry():
    fails = []
    for name in EXPERIMENTS:
        path = REPO_ROOT / "scripts" / f"exp_{name}.py"
        txt  = path.read_text(encoding="utf-8")
        # Every wrapper should dispatch via run_experiment(<name>, ...)
        expected = f'run_experiment("{name}"'
        if expected not in txt:
            fails.append(f"exp_{name}.py does not call {expected}...)")
    assert not fails, "\n  ".join(fails)


@_register("Test 13: every plot_<name>.py loads result with matching name")
def test_13_plot_py_loads_result():
    fails = []
    for name in EXPERIMENTS:
        path = REPO_ROOT / "scripts" / f"plot_{name}.py"
        txt  = path.read_text(encoding="utf-8")
        expected = f'load_result("{name}"'
        if expected not in txt:
            fails.append(f"plot_{name}.py does not call {expected}, ...)")
    assert not fails, "\n  ".join(fails)


@_register("Test 14: _defaults.sh defines SET_ARGS array")
def test_14_defaults_has_set_args():
    txt = (REPO_ROOT / "configs/recipes/_defaults.sh").read_text(encoding="utf-8")
    assert "SET_ARGS=(" in txt, "_defaults.sh missing SET_ARGS array"
    # All 16 sensing fields should be present (1:1 with PARAM_REGISTRY).
    for key in [
        "sensing.clutter_center_strategy",
        "sensing.clutter_offset_az_deg",
        "sensing.clutter_cnr_db",
        "sensing.clutter_as_deg",
    ]:
        assert key in txt, f"_defaults.sh missing {key}"


# ─────────────────────────────────────────────────────────────────────
# FUNCTIONAL TESTS — run a tiny pipeline end-to-end (skip if no install)
# ─────────────────────────────────────────────────────────────────────

def _have_full_install() -> bool:
    """True iff we can import the full cordis stack."""
    try:
        from cordis.utils.config import load_config           # noqa: F401
        from cordis.simulation.runner import RunnerConfig     # noqa: F401
        from cordis.experiments import REGISTRY               # noqa: F401
        return True
    except ImportError:
        return False


def _smoke_config(base_cfg_path: Path, out_dir: Path) -> Path:
    """Build a config tuned for fast smoke-testing.

    Loads the repo's default config, overrides only the knobs that
    blow up wall-clock time for a pipeline smoke test (ADMM iteration
    cap, mostly), and writes the modified config to ``out_dir``.

    This decouples Tests 15/16 from any future bump to algorithmic
    defaults — the smoke test exists to verify the runner+plot
    pipeline produces output, not to test algorithm convergence, so
    capping ``algorithm.admm.n_max`` to a small value is exactly right.

    Stage 22a raised the default ``admm.n_max`` from 50 → 200, which
    quadrupled the wall-clock of the end-to-end tests and pushed the
    sweep test (2 trials × 2 γ × 9 algorithms ≈ 36 ADMM solves) past
    its 300-second timeout.  This helper restores headroom by capping
    iters at 10 — still enough for the pipeline to exercise every
    code path without measuring algorithmic quality.
    """
    cfg = json.loads(base_cfg_path.read_text())
    cfg.setdefault("algorithm", {}).setdefault("admm", {})["n_max"] = 10
    smoke_path = out_dir / "smoke_config.json"
    smoke_path.write_text(json.dumps(cfg, indent=2))
    return smoke_path


@_register("Test 15: end-to-end — exp_sinr_cdf.py + plot_sinr_cdf.py")
def test_15_end_to_end_sinr_cdf():
    if not _have_full_install():
        raise _SkipTest("full cordis install not available")
    cfg_path = REPO_ROOT / "configs/default.json"
    if not cfg_path.exists():
        raise _SkipTest(f"no config at {cfg_path}")

    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td) / "results"
        fig_root = Path(td) / "figures"
        smoke_cfg = _smoke_config(cfg_path, Path(td))

        # Run a tiny experiment.
        proc = subprocess.run(
            [sys.executable, "scripts/exp_sinr_cdf.py",
             "--base-config", str(smoke_cfg),
             "--n-drops", "2", "--n-realizations", "1",
             "--n-workers", "1",
             "--output-root", str(out_root)],
            capture_output=True, text=True, timeout=300,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"exp_sinr_cdf.py failed (rc={proc.returncode}):\n"
                f"  stderr: {proc.stderr.strip()[-500:]}"
            )

        # Verify exactly one timestamped result directory was created.
        # The runner also drops a `latest` symlink — exclude it.
        exp_root = out_root / "exp_sinr_cdf"
        assert exp_root.exists(), "no exp_sinr_cdf/ directory created"
        runs = sorted(p for p in exp_root.iterdir()
                      if p.is_dir() and not p.is_symlink())
        assert len(runs) == 1, (
            f"expected 1 timestamped run dir, got {len(runs)}: "
            f"{[p.name for p in runs]}"
        )
        run_dir = runs[0]
        assert (run_dir / "manifest.json").exists(), "manifest.json missing"
        assert (run_dir / "logs/run.log").exists(),  "logs/run.log missing"

        # Plot.
        proc = subprocess.run(
            [sys.executable, "scripts/plot_sinr_cdf.py",
             "--exp-dir", str(run_dir),
             "--figures-root", str(fig_root),
             "--formats", "pdf"],
            capture_output=True, text=True, timeout=60,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"plot_sinr_cdf.py failed (rc={proc.returncode}):\n"
                f"  stderr: {proc.stderr.strip()[-500:]}"
            )
        pdfs = list(fig_root.rglob("*.pdf"))
        assert pdfs, "no PDF figures produced"


@_register("Test 16: end-to-end — exp_gamma_sweep.py + plot_gamma_sweep.py")
def test_16_end_to_end_gamma_sweep():
    if not _have_full_install():
        raise _SkipTest("full cordis install not available")
    cfg_path = REPO_ROOT / "configs/default.json"
    if not cfg_path.exists():
        raise _SkipTest(f"no config at {cfg_path}")

    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td) / "results"
        fig_root = Path(td) / "figures"
        smoke_cfg = _smoke_config(cfg_path, Path(td))

        proc = subprocess.run(
            [sys.executable, "scripts/exp_gamma_sweep.py",
             "--base-config", str(smoke_cfg),
             "--n-drops", "2", "--n-realizations", "1",
             "--n-workers", "1",
             "--gamma-values", "0,6",
             "--output-root", str(out_root)],
            capture_output=True, text=True, timeout=300,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"exp_gamma_sweep.py failed (rc={proc.returncode}):\n"
                f"  stderr: {proc.stderr.strip()[-500:]}"
            )

        exp_root = out_root / "exp_gamma_sweep"
        assert exp_root.exists(), "no exp_gamma_sweep/ directory created"
        runs = sorted(p for p in exp_root.iterdir()
                      if p.is_dir() and not p.is_symlink())
        assert len(runs) == 1, (
            f"expected 1 timestamped run dir, got {len(runs)}: "
            f"{[p.name for p in runs]}"
        )
        run_dir = runs[0]

        proc = subprocess.run(
            [sys.executable, "scripts/plot_gamma_sweep.py",
             "--exp-dir", str(run_dir),
             "--figures-root", str(fig_root),
             "--formats", "pdf"],
            capture_output=True, text=True, timeout=60,
            env=_subprocess_env(),
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"plot_gamma_sweep.py failed (rc={proc.returncode}):\n"
                f"  stderr: {proc.stderr.strip()[-500:]}"
            )
        pdfs = list(fig_root.rglob("*.pdf"))
        assert pdfs, "no PDF figures produced"


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    bar = "=" * 78
    print(bar)
    print(f"  Stage 8c validation — {len(_TESTS)} tests")
    print(bar)

    n_pass, n_skip, n_fail = 0, 0, 0
    failures = []
    for label, fn in _TESTS:
        print(f"\n── {label} ──")
        try:
            fn()
            print("  ✓ PASSED")
            n_pass += 1
        except _SkipTest as e:
            print(f"  ⏭  SKIPPED  ({e})")
            n_skip += 1
        except AssertionError as e:
            print(f"  ✗ FAILED\n      {str(e)[:1500]}")
            failures.append(label)
            n_fail += 1
        except Exception as e:
            print(f"  ✗ ERRORED  ({type(e).__name__})")
            print("      " + "\n      ".join(
                traceback.format_exc().splitlines()[-6:]))
            failures.append(label)
            n_fail += 1

    print("\n" + "=" * 78)
    if n_fail == 0:
        print(f"  ✓ ALL TESTS PASSED   ({n_pass}/{len(_TESTS)} passed, "
              f"{n_skip} skipped)")
    else:
        print(f"  ✗ {n_fail} FAILURE(S)   ({n_pass} passed, "
              f"{n_skip} skipped, {n_fail} failed)")
        print("    failed: " + ", ".join(failures))
    print("=" * 78)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

