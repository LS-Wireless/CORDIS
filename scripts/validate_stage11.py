#!/usr/bin/env python3
"""
scripts/validate_stage11.py
============================

Stage 11 validator.

Stage 11 introduces UCI HPC3 site-specific SLURM submission scripts
under ``scripts/slurm/uci-hpc3/``.  The generic ``.sbatch`` templates
under ``scripts/slurm/`` remain in place as portable starting points
for other clusters.

Tests verify:

* **directory structure** — ``scripts/slurm/uci-hpc3/`` exists,
  contains exactly one ``_config.sh`` plus 11 ``.sub`` files
  (one per registered experiment).
* **_config.sh contract** — exports the six ``CORDIS_*`` env vars
  with default values, each parameterised via ``${VAR:-default}``
  so user / site overrides work.
* **.sub contract** — every .sub file sources ``_config.sh``,
  loads modules, activates the venv, exports ``N_WORKERS``, and
  invokes the corresponding ``scripts/exp_<name>.sh`` launcher.
* **bash syntax** — every generated file passes ``bash -n``.
* **per-experiment resource consistency** — each .sub uses the same
  ``--time``, ``--mem``, ``--cpus-per-task`` as the generic .sbatch
  for the same experiment (drift-detection).

Run from the repo root::

    python3 scripts/validate_stage11.py

Exit code is 0 iff every test passes.
"""
from __future__ import annotations

import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

UCI_DIR = REPO_ROOT / "scripts" / "slurm" / "uci-hpc3"
GENERIC_DIR = REPO_ROOT / "scripts" / "slurm"

# Experiments are the same set used by validate_stage8c/8d.  Keep in sync.
EXPERIMENTS = [
    "sinr_cdf", "scnr_cdf",
    "gamma_sweep", "kappa_sweep", "clutter_cnr_sweep",
    "snr_sweep", "n_ue_sweep", "n_ap_sweep", "antennas_sweep",
    "convergence_trace", "fronthaul_table",
]

EXPECTED_CONFIG_VARS = [
    "CORDIS_ACCOUNT",
    "CORDIS_PARTITION",
    "CORDIS_MAIL_USER",
    "CORDIS_MAIL_TYPE",
    "CORDIS_REPO_DIR",
    "CORDIS_PYTHON_MODULE",
    "CORDIS_VENV",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Test registry & runner
# ─────────────────────────────────────────────────────────────────────────────

class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — directory structure
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: scripts/slurm/uci-hpc3/ directory exists")
def test_01_dir_exists():
    assert UCI_DIR.is_dir(), (
        f"expected directory {UCI_DIR} to exist; run "
        f"`python3 scripts/regenerate_experiment_scripts.py` first."
    )


@_register("Test  2: _config.sh exists")
def test_02_config_exists():
    p = UCI_DIR / "_config.sh"
    assert p.is_file(), f"missing {p}"


@_register("Test  3: exactly one .sub per registered experiment (11 total)")
def test_03_sub_count():
    sub_files = sorted(p.name for p in UCI_DIR.glob("exp_*.sub"))
    expected = sorted(f"exp_{n}.sub" for n in EXPERIMENTS)
    missing = set(expected) - set(sub_files)
    extra = set(sub_files) - set(expected)
    assert not missing and not extra, (
        f"sub-file mismatch — missing: {sorted(missing)}, "
        f"extra: {sorted(extra)}"
    )


@_register("Test  4: generic .sbatch counterparts also still present")
def test_04_generic_sbatch_present():
    # Stage 11 must NOT remove the portable templates.
    sbatch_files = sorted(p.name for p in GENERIC_DIR.glob("exp_*.sbatch"))
    expected = sorted(f"exp_{n}.sbatch" for n in EXPERIMENTS)
    missing = set(expected) - set(sbatch_files)
    assert not missing, (
        f"generic .sbatch templates missing: {sorted(missing)} — "
        f"Stage 11 should not delete them"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — _config.sh contract
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  5: _config.sh exports all 7 CORDIS_* env vars")
def test_05_config_vars():
    txt = (UCI_DIR / "_config.sh").read_text()
    for var in EXPECTED_CONFIG_VARS:
        # Must be `export VAR=...` (not just `VAR=...`) so child .sub
        # scripts can see them after `source _config.sh`.
        pat = rf'\bexport\s+{re.escape(var)}\s*=\s*"\$\{{{re.escape(var)}:-'
        assert re.search(pat, txt), (
            f"_config.sh: expected `export {var}=\"${{{var}:-default}}\"`; "
            f"not found.  Each variable must be exported AND parameterised "
            f"with the ${{VAR:-default}} idiom so submit-time overrides work."
        )


@_register("Test  6: _config.sh is bash-syntactically valid")
def test_06_config_bash_syntax():
    proc = subprocess.run(
        ["bash", "-n", str(UCI_DIR / "_config.sh")],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"`bash -n _config.sh` failed:\n{proc.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — .sub file contract
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  7: every .sub sources _config.sh via SLURM_SUBMIT_DIR")
def test_07_sub_sources_config():
    """sbatch COPIES the script to a spool dir before running, so
    BASH_SOURCE[0] alone resolves to the spool path — _config.sh
    won't be there.  The correct pattern is to use SLURM_SUBMIT_DIR
    (with BASH_SOURCE fallback for direct invocation)."""
    for name in EXPERIMENTS:
        p = UCI_DIR / f"exp_{name}.sub"
        txt = p.read_text()
        assert 'source "$SCRIPT_DIR/_config.sh"' in txt, (
            f"{p.name}: should `source` _config.sh"
        )
        # The KEY assertion: SLURM_SUBMIT_DIR must be consulted
        # before falling back to BASH_SOURCE.
        assert "SLURM_SUBMIT_DIR" in txt, (
            f"{p.name}: must reference SLURM_SUBMIT_DIR when locating "
            f"_config.sh.  Pure BASH_SOURCE-based path resolution "
            f"FAILS under sbatch because SLURM copies the script to a "
            f"spool directory (e.g. /export/spool/slurm/slurmd.spool/"
            f"jobNNNNN/) before executing it, so dirname BASH_SOURCE[0] "
            f"points to that spool dir and _config.sh isn't there."
        )


@_register("Test  8: every .sub loads modules + activates venv")
def test_08_sub_modules_and_venv():
    for name in EXPERIMENTS:
        p = UCI_DIR / f"exp_{name}.sub"
        txt = p.read_text()
        # module purge + load python module
        assert "module purge" in txt, f"{p.name}: missing `module purge`"
        assert 'module load "$CORDIS_PYTHON_MODULE"' in txt, (
            f"{p.name}: should load $CORDIS_PYTHON_MODULE"
        )
        # venv activation
        assert 'source "$CORDIS_VENV/bin/activate"' in txt, (
            f"{p.name}: should activate venv via $CORDIS_VENV"
        )


@_register("Test  9: every .sub exports N_WORKERS from SLURM_CPUS_PER_TASK")
def test_09_sub_n_workers():
    for name in EXPERIMENTS:
        p = UCI_DIR / f"exp_{name}.sub"
        txt = p.read_text()
        assert 'export N_WORKERS="${SLURM_CPUS_PER_TASK:-1}"' in txt, (
            f"{p.name}: must export N_WORKERS so the launcher honours "
            f"the SLURM allocation"
        )


@_register("Test 10: every .sub invokes its launcher")
def test_10_sub_invokes_launcher():
    for name in EXPERIMENTS:
        p = UCI_DIR / f"exp_{name}.sub"
        txt = p.read_text()
        expected = f"bash scripts/exp_{name}.sh"
        assert expected in txt, (
            f"{p.name}: should invoke `{expected}` (the launcher)"
        )


@_register("Test 11: every .sub has the cordis-<name> job-name + correct paths")
def test_11_sub_directives():
    for name in EXPERIMENTS:
        p = UCI_DIR / f"exp_{name}.sub"
        txt = p.read_text()
        # Standard SLURM directives
        for directive in (
            f"#SBATCH --job-name=cordis-{name}",
            f"#SBATCH --output=logs/slurm/{name}-%j.out",
            f"#SBATCH --error=logs/slurm/{name}-%j.err",
            "#SBATCH --partition=standard",
            "#SBATCH --account=swindle_lab",
        ):
            assert directive in txt, (
                f"{p.name}: missing directive `{directive}`"
            )


@_register("Test 12: every .sub is bash-syntactically valid")
def test_12_sub_bash_syntax():
    for name in EXPERIMENTS:
        p = UCI_DIR / f"exp_{name}.sub"
        proc = subprocess.run(
            ["bash", "-n", str(p)],
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode == 0, (
            f"`bash -n {p.name}` failed:\n{proc.stderr}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — resource consistency vs generic .sbatch
# ─────────────────────────────────────────────────────────────────────────────

_RES_RE = {
    "time":  re.compile(r"^#SBATCH\s+--time=(\S+)\s*$",          re.M),
    "cpus":  re.compile(r"^#SBATCH\s+--cpus-per-task=(\S+)\s*$", re.M),
    # Stage 21: --mem dropped from all .sub/.sbatch (HPC3 40-CPU
    # allocation gives ~192 GB proportional default).  No longer in
    # the drift check — but if it reappears in only one of the pair
    # the test still catches it because the dict keys must match.
}


def _extract_resources(text: str) -> dict:
    out = {}
    for key, rgx in _RES_RE.items():
        m = rgx.search(text)
        if m is None:
            raise AssertionError(f"could not find #SBATCH --{key}= line")
        out[key] = m.group(1)
    return out


@_register("Test 13: per-experiment resources match between .sub and .sbatch")
def test_13_resource_consistency():
    """A drift-detection guard.  Both files come from the same
    per-experiment metadata in regenerate_experiment_scripts.py, so
    the three resource lines must match exactly.  If they drift,
    someone edited a generated file by hand (the templates regenerate
    cleanly each time)."""
    mismatches = []
    for name in EXPERIMENTS:
        sub = _extract_resources((UCI_DIR / f"exp_{name}.sub").read_text())
        sb  = _extract_resources((GENERIC_DIR / f"exp_{name}.sbatch").read_text())
        if sub != sb:
            mismatches.append((name, sub, sb))
    assert not mismatches, (
        "resource drift between .sub and .sbatch — re-run "
        "`python3 scripts/regenerate_experiment_scripts.py` to fix:\n"
        + "\n".join(f"  {n}: .sub={sub} vs .sbatch={sb}"
                    for n, sub, sb in mismatches)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 11 validator — {len(_TESTS)} tests")
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
            print(f"      {e}")
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

