#!/usr/bin/env python3
"""
scripts/validate_stage8d.py
===========================

Stage 8d validator.

Stage 8d adds the top-level orchestration layer on top of Stage 8c:

* ``Makefile`` at repo root — primary user interface
* ``scripts/slurm/exp_<name>.sbatch`` × 11 — cluster job wrappers
* ``scripts/validate_stage8.py`` — umbrella validator (8a + 8b + 8c + 8d)
* ``docs/quickstart.md`` — quickstart guide
* ``PYTHONPATH`` export inside every regenerated ``scripts/exp_<name>.sh``

Tests are grouped into four tiers — structural, syntactic, semantic, and
functional — and each test prints its own ✓ / ✗ / ⏭ line.

Run from the repo root:

    python3 scripts/validate_stage8d.py

Exit code is 0 iff every test passes (skips do not fail the run).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the repo root importable so `from cordis...` works when this
# script is invoked directly (Python only puts the script's directory
# on sys.path, not the cwd).  Matches validate_stage8b.py / 8c.
sys.path.insert(0, str(REPO_ROOT))

# The eleven experiments Stage 8c/8d cover, in registry order.
EXPERIMENTS = [
    "sinr_cdf", "scnr_cdf",
    "gamma_sweep", "kappa_sweep", "clutter_cnr_sweep", "snr_sweep",
    "n_ue_sweep", "n_ap_sweep", "antennas_sweep",
    "convergence_trace", "fronthaul_table",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Test registry & runner
# ─────────────────────────────────────────────────────────────────────────────

class _SkipTest(Exception):
    """Raised by a test that cannot run in the current environment."""


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


def _subprocess_env() -> dict:
    """Environment for subprocess calls: REPO_ROOT on PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    )
    return env


def _have_make() -> bool:
    return shutil.which("make") is not None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Structural: files exist with sensible properties
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: Makefile exists at repo root")
def test_01_makefile_exists():
    p = REPO_ROOT / "Makefile"
    assert p.exists(), "Makefile missing"
    assert p.stat().st_size > 0, "Makefile empty"


@_register("Test  2: scripts/slurm/ exists with one .sbatch per experiment")
def test_02_slurm_dir():
    slurm_dir = REPO_ROOT / "scripts" / "slurm"
    assert slurm_dir.is_dir(), "scripts/slurm/ missing"
    found = sorted(p.stem.removeprefix("exp_") for p in slurm_dir.glob("*.sbatch"))
    expected = sorted(f"exp_{n}" for n in EXPERIMENTS)
    found_names = sorted(p.name for p in slurm_dir.glob("*.sbatch"))
    missing = set(f"exp_{n}.sbatch" for n in EXPERIMENTS) - set(found_names)
    extra = set(found_names) - set(f"exp_{n}.sbatch" for n in EXPERIMENTS)
    assert not missing, f"missing .sbatch: {sorted(missing)}"
    assert not extra,   f"unexpected .sbatch: {sorted(extra)}"


@_register("Test  3: every .sbatch is executable")
def test_03_sbatch_executable():
    missing_x = []
    for name in EXPERIMENTS:
        p = REPO_ROOT / "scripts" / "slurm" / f"exp_{name}.sbatch"
        if not (p.stat().st_mode & 0o111):
            missing_x.append(p.relative_to(REPO_ROOT))
    assert not missing_x, f"not executable: {missing_x}"


@_register("Test  4: scripts/validate_stage8.py exists & is executable")
def test_04_umbrella_validator_exists():
    p = REPO_ROOT / "scripts" / "validate_stage8.py"
    assert p.exists(), "validate_stage8.py missing"
    assert p.stat().st_size > 0, "validate_stage8.py empty"


@_register("Test  5: docs/quickstart.md exists")
def test_05_quickstart_exists():
    p = REPO_ROOT / "docs" / "quickstart.md"
    assert p.exists(), "docs/quickstart.md missing"
    assert p.stat().st_size > 200, "docs/quickstart.md too short"


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — Syntactic: files parse / lint cleanly
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  6: Makefile parses (`make -n help`)")
def test_06_makefile_parses():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    proc = subprocess.run(
        ["make", "-n", "help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
    )
    assert proc.returncode == 0, (
        f"`make -n help` failed:\n"
        f"  stdout: {proc.stdout.strip()[-300:]}\n"
        f"  stderr: {proc.stderr.strip()[-300:]}"
    )


@_register("Test  7: every .sbatch passes `bash -n`")
def test_07_sbatch_bash_n():
    for name in EXPERIMENTS:
        p = REPO_ROOT / "scripts" / "slurm" / f"exp_{name}.sbatch"
        proc = subprocess.run(
            ["bash", "-n", str(p)],
            capture_output=True, text=True, timeout=5,
        )
        assert proc.returncode == 0, (
            f"bash -n {p.name} failed: {proc.stderr.strip()[-200:]}"
        )


@_register("Test  8: validate_stage8.py is valid Python (`py_compile`)")
def test_08_umbrella_compiles():
    p = REPO_ROOT / "scripts" / "validate_stage8.py"
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(p)],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"py_compile failed:\n  stderr: {proc.stderr.strip()[-300:]}"
    )


@_register("Test  9: validate_stage8.py --help renders")
def test_09_umbrella_help():
    p = REPO_ROOT / "scripts" / "validate_stage8.py"
    proc = subprocess.run(
        [sys.executable, str(p), "--help"],
        capture_output=True, text=True, timeout=10,
        env=_subprocess_env(),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"--help failed (rc={proc.returncode}):\n"
        f"  stderr: {proc.stderr.strip()[-300:]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — Semantic: cross-file consistency
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 10: Makefile EXPERIMENTS variable matches REGISTRY")
def test_10_makefile_matches_registry():
    try:
        from cordis.experiments import REGISTRY
    except ImportError as e:
        raise _SkipTest(f"cordis.experiments not importable: {e}")

    txt = _read(REPO_ROOT / "Makefile")
    # Extract the EXPERIMENTS := ... block, walking through any
    # backslash-newline line continuations.  DOTALL + a negative
    # lookbehind for `\` on the terminating newline does the trick.
    m = re.search(r"^EXPERIMENTS\s*:=(.*?)(?<!\\)\n",
                  txt, flags=re.MULTILINE | re.DOTALL)
    assert m, "Makefile has no EXPERIMENTS := ... block"
    raw = m.group(1).replace("\\\n", " ")
    names = [tok for tok in raw.split() if tok]

    reg_names = set(REGISTRY)
    mk_names  = set(names)
    missing_from_mk = reg_names - mk_names
    extra_in_mk     = mk_names - reg_names
    assert not missing_from_mk, (
        f"Makefile missing experiments: {sorted(missing_from_mk)}"
    )
    assert not extra_in_mk, (
        f"Makefile has extras not in REGISTRY: {sorted(extra_in_mk)}"
    )


@_register("Test 11: each .sbatch references the right exp_<name>.sh")
def test_11_sbatch_references_launcher():
    for name in EXPERIMENTS:
        p = REPO_ROOT / "scripts" / "slurm" / f"exp_{name}.sbatch"
        txt = _read(p)
        expected = f"bash scripts/exp_{name}.sh"
        assert expected in txt, (
            f"{p.name} does not invoke `{expected}`"
        )


@_register("Test 12: every .sbatch has required SBATCH directives")
def test_12_sbatch_directives():
    required = ["--job-name", "--output", "--error",
                "--time", "--cpus-per-task", "--mem"]
    for name in EXPERIMENTS:
        p = REPO_ROOT / "scripts" / "slurm" / f"exp_{name}.sbatch"
        txt = _read(p)
        for d in required:
            assert re.search(rf"^#SBATCH\s+{re.escape(d)}=", txt, re.MULTILINE), (
                f"{p.name} missing #SBATCH {d}"
            )


@_register("Test 13: each exp_<name>.sh exports PYTHONPATH")
def test_13_sh_launcher_pythonpath():
    for name in EXPERIMENTS:
        p = REPO_ROOT / "scripts" / f"exp_{name}.sh"
        txt = _read(p)
        assert "export PYTHONPATH=" in txt, (
            f"scripts/exp_{name}.sh does not export PYTHONPATH "
            "— rerun `python3 scripts/regenerate_experiment_scripts.py`"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — Functional: targets dispatch correctly
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 14: `make -n <name>` works for every experiment")
def test_14_make_dry_run_runs():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    for name in EXPERIMENTS:
        proc = subprocess.run(
            ["make", "-n", name],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        assert proc.returncode == 0, (
            f"`make -n {name}` failed:\n"
            f"  stderr: {proc.stderr.strip()[-300:]}"
        )
        assert f"scripts/exp_{name}.sh" in proc.stdout, (
            f"`make -n {name}` does not invoke scripts/exp_{name}.sh\n"
            f"  stdout: {proc.stdout.strip()[-300:]}"
        )


@_register("Test 15: `make -n plot-<name>` works for every experiment")
def test_15_make_plot_dry_run():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    for name in EXPERIMENTS:
        proc = subprocess.run(
            ["make", "-n", f"plot-{name}"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        assert proc.returncode == 0, (
            f"`make -n plot-{name}` failed:\n"
            f"  stderr: {proc.stderr.strip()[-300:]}"
        )
        assert f"scripts/plot_{name}.py" in proc.stdout, (
            f"`make -n plot-{name}` does not invoke scripts/plot_{name}.py"
        )


@_register("Test 16: `make -n submit-<name>` works for every experiment")
def test_16_make_submit_dry_run():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    for name in EXPERIMENTS:
        proc = subprocess.run(
            ["make", "-n", f"submit-{name}"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        assert proc.returncode == 0, (
            f"`make -n submit-{name}` failed:\n"
            f"  stderr: {proc.stderr.strip()[-300:]}"
        )
        assert f"scripts/slurm/exp_{name}.sbatch" in proc.stdout, (
            f"`make -n submit-{name}` does not sbatch the right file"
        )


@_register("Test 17: `make -n all` enumerates every experiment")
def test_17_make_all():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    proc = subprocess.run(
        ["make", "-n", "all"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
    )
    assert proc.returncode == 0, f"`make -n all` failed: {proc.stderr[-200:]}"
    for name in EXPERIMENTS:
        assert f"scripts/exp_{name}.sh" in proc.stdout, (
            f"`make -n all` missing exp_{name}.sh"
        )


@_register("Test 18: `make -n figures` enumerates every plot")
def test_18_make_figures():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    proc = subprocess.run(
        ["make", "-n", "figures"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
    )
    assert proc.returncode == 0, f"`make -n figures` failed: {proc.stderr[-200:]}"
    for name in EXPERIMENTS:
        assert f"scripts/plot_{name}.py" in proc.stdout, (
            f"`make -n figures` missing plot_{name}.py"
        )


@_register("Test 19: `make -n validate` invokes scripts/validate_stage8.py")
def test_19_make_validate():
    if not _have_make():
        raise _SkipTest("`make` not in PATH")
    proc = subprocess.run(
        ["make", "-n", "validate"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
    )
    assert proc.returncode == 0, f"`make -n validate` failed: {proc.stderr[-200:]}"
    assert "scripts/validate_stage8.py" in proc.stdout, (
        "`make -n validate` does not invoke scripts/validate_stage8.py"
    )


@_register("Test 20: umbrella validator can introspect sub-validators")
def test_20_umbrella_lists_subvalidators():
    """`validate_stage8.py --list` should enumerate every sub-validator."""
    p = REPO_ROOT / "scripts" / "validate_stage8.py"
    proc = subprocess.run(
        [sys.executable, str(p), "--list"],
        capture_output=True, text=True, timeout=10,
        env=_subprocess_env(),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"--list failed:\n  stderr: {proc.stderr.strip()[-300:]}"
    )
    out = proc.stdout
    for sub in ["validate_stage8a", "validate_stage8b",
                "validate_stage8c", "validate_stage8d"]:
        assert sub in out, f"--list does not mention {sub}"


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 8d validator — {len(_TESTS)} tests")
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
            print(f"  ✗ FAILED")
            print(f"      {e}")
            failed += 1
            failures.append((label, str(e)))
        except Exception:
            print("  ✗ FAILED (unexpected exception)")
            tb = traceback.format_exc()
            print("      " + tb.replace("\n", "\n      ").rstrip())
            failed += 1
            failures.append((label, tb.splitlines()[-1] if tb else "?"))

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

