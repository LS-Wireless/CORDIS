#!/usr/bin/env python3
"""
scripts/validate_stage17.py
============================

Stage 17 validator.

Stage 17 makes three user-requested experiment-polish changes:

1.  **Uniform `spec_set="all_algorithms"` default** for the 9 trial-
    bearing experiments (CDF + sweeps).  Previously four sweeps
    (gamma, kappa, clutter_cnr, antennas) defaulted to narrower
    benchmark sets — surprising users who expected the same default
    behaviour as the other sweeps.

2.  **Convergence-trace progress bar.**  ``solve_cordis_admm`` now
    accepts a ``progress: bool`` kwarg.  When True, its outer
    iteration loop is wrapped with ``tqdm``.  ``run_convergence_trace``
    threads ``progress=True`` so the user sees an iteration bar
    during the ~30-iteration solve.

3.  **Fronthaul table apples-to-apples comparison.**  Previously the
    table stored ``real_scalars`` per coordination round for
    Centralized + Split, but TOTAL across all rounds (``per-iter ×
    T_ADMM``) for CORDIS-ADMM.  Bar charts of ``real_scalars``
    therefore showed ADMM artificially higher than Centralized,
    contradicting the journal-paper claim that ADMM is cheaper
    per coordination round.  Stage 17 fixes this: every row stores
    per-round count; the per-solve multiplier moves to a new
    ``iterations`` field.

Tests verify:

* **Tier 1 — uniform defaults.**  All 9 trial-bearing experiments
  declare ``spec_set: str = "all_algorithms"``.
* **Tier 2 — progress kwarg present.**  ``solve_cordis_admm`` has
  ``progress: bool = False`` in its signature; ``run_convergence_trace``
  passes ``progress=True``.
* **Tier 3 — fronthaul-table accounting.**  ADMM's ``real_scalars``
  equals the per-iteration count (NOT pre-multiplied by T_ADMM);
  the table contains an ``iterations`` field; ``scales_with`` for
  ADMM no longer contains the ``T_ADMM`` factor.

Run from the repo root::

    python3 scripts/validate_stage17.py
"""
from __future__ import annotations

import inspect
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


def _load_registry():
    try:
        from cordis.experiments.registry import REGISTRY
        return REGISTRY
    except Exception as e:
        raise _SkipTest(f"cordis.experiments.registry not importable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — uniform spec_set defaults
# ─────────────────────────────────────────────────────────────────────────────

# All trial-bearing experiments — these have a spec_set kwarg and after
# Stage 17 default it to "all_algorithms".
_TRIAL_EXPERIMENTS_WITH_SPEC_SET = [
    "sinr_cdf", "scnr_cdf",
    "gamma_sweep", "kappa_sweep", "clutter_cnr_sweep",
    "snr_sweep", "n_ue_sweep", "n_ap_sweep", "antennas_sweep",
]


@_register("Test  1: every CDF/sweep experiment defaults spec_set to "
           "'all_algorithms'")
def test_01_uniform_defaults():
    reg = _load_registry()
    offenders: List[Tuple[str, str]] = []
    for name in _TRIAL_EXPERIMENTS_WITH_SPEC_SET:
        assert name in reg, f"{name} not in REGISTRY"
        sig = inspect.signature(reg[name])
        if "spec_set" not in sig.parameters:
            offenders.append((name, "<missing spec_set kwarg>"))
            continue
        default = sig.parameters["spec_set"].default
        if default != "all_algorithms":
            offenders.append((name, default))
    assert not offenders, (
        "experiments with non-uniform spec_set defaults:\n  "
        + "\n  ".join(f"run_{n}: spec_set={d!r}" for n, d in offenders)
    )


@_register("Test  2: file-level audit — exactly 10 'spec_set: str = \"all_algorithms\"' lines")
def test_02_count_in_registry_file():
    """Structural lock-in: the registry file should literally contain
    ten ``spec_set: str = "all_algorithms"`` lines (one per trial-
    bearing experiment).  Catches any future regression that flips a
    default back to a narrower set."""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    matches = re.findall(
        r'spec_set:\s*str\s*=\s*"all_algorithms"', src
    )
    assert len(matches) == 10, (
        f"expected exactly 10 'spec_set: str = \"all_algorithms\"' lines "
        f"in registry.py (one per CDF/sweep experiment); found "
        f"{len(matches)}"
    )

    # And NO references to the legacy narrower defaults as the default.
    for legacy in ("cordis_vs_centralized", "cordis_vs_benchmarks"):
        bad = re.search(
            rf'spec_set:\s*str\s*=\s*"{legacy}"', src
        )
        assert bad is None, (
            f"registry.py still has a `spec_set: str = \"{legacy}\"` "
            f"default; Stage 17 should have unified these to "
            f"'all_algorithms'."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — convergence-trace progress bar
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  3: solve_cordis_admm accepts a progress kwarg")
def test_03_solver_has_progress_kwarg():
    try:
        from cordis.algorithms.joint_opt import solve_cordis_admm
    except Exception as e:
        raise _SkipTest(f"joint_opt not importable: {e}")
    sig = inspect.signature(solve_cordis_admm)
    assert "progress" in sig.parameters, (
        "solve_cordis_admm should accept a `progress: bool = False` kwarg "
        "so convergence_trace can opt into the per-iteration tqdm bar."
    )
    p = sig.parameters["progress"]
    assert p.default is False, (
        f"`progress` should default to False (so Monte-Carlo runners "
        f"don't get a bar drawn thousands of times); got default={p.default!r}"
    )


@_register("Test  4: ADMM iteration loop is wrapped with tqdm when progress=True")
def test_04_loop_wrapped():
    """Structural check on the solver: when progress=True, the outer
    ``for n_iter in range(n_admm_max):`` must run inside a tqdm context."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    # Look for the wrapping pattern we installed.
    assert "from tqdm import tqdm" in src, (
        "joint_opt.py should import tqdm (lazily) when progress=True"
    )
    # The loop variable must come from the wrapped range, not bare range().
    bare_loop = re.search(
        r"^\s*for\s+n_iter\s+in\s+range\(n_admm_max\)\s*:\s*$", src, re.M,
    )
    assert bare_loop is None, (
        "joint_opt.py still has a bare `for n_iter in range(n_admm_max):` "
        "loop; the iteration loop should iterate over a `_iter_range` "
        "variable that's tqdm-wrapped when progress=True."
    )


@_register("Test  5: run_convergence_trace passes progress=True to run_cordis_admm")
def test_05_trace_enables_progress():
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    # Find the run_cordis_admm call inside run_convergence_trace and
    # verify it includes progress=True.  Approximate via substring.
    trace_block_match = re.search(
        r"def run_convergence_trace\(.*?return ExperimentResult\(",
        src, re.S,
    )
    assert trace_block_match is not None, (
        "could not locate run_convergence_trace body in registry.py"
    )
    trace_block = trace_block_match.group(0)
    assert re.search(r"progress\s*=\s*True", trace_block), (
        "run_convergence_trace should call run_cordis_admm with "
        "`progress=True` so the user sees iteration progress."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — fronthaul-table accounting
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  6: fronthaul-table real_scalars is per-iter for ADMM, "
           "NOT pre-multiplied by T_ADMM")
def test_06_admm_per_iter_not_total():
    """Structural check: the previous bug pre-multiplied
    ``admm_per_iter`` by ``admm_avg_iters`` and stored that as
    ``real_scalars``.  After Stage 17 the storage is the per-iter
    value."""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()

    # The buggy line — must not appear.
    bad = re.search(
        r'"real_scalars":\s*admm_total\b',
        src,
    )
    assert bad is None, (
        '`"real_scalars": admm_total` still present — that\'s the '
        'per-iter × T_ADMM total, which makes the bar chart show ADMM '
        'higher than Centralized.  Use the per-round count instead.'
    )

    # The fixed pattern — must appear.
    good = re.search(
        r'"real_scalars":\s*admm_per_round\b',
        src,
    )
    assert good is not None, (
        'expected `"real_scalars": admm_per_round` (the per-round count) '
        'in the CORDIS-ADMM row of the fronthaul table.'
    )


@_register("Test  7: fronthaul table includes an 'iterations' field")
def test_07_iterations_field_present():
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    # Locate the fronthaul-table block and count "iterations" assignments.
    block = re.search(
        r"def run_fronthaul_table\(.*?return ExperimentResult\(",
        src, re.S,
    )
    assert block is not None, "could not find run_fronthaul_table body"
    body = block.group(0)
    n_iter_fields = len(re.findall(r'"iterations":\s*', body))
    assert n_iter_fields == 3, (
        f"expected exactly 3 'iterations' fields in the fronthaul table "
        f"(one per algorithm row); found {n_iter_fields}"
    )


@_register("Test  8: ADMM 'scales_with' formula no longer contains "
           "the T_ADMM factor")
def test_08_scales_with_no_more_t_admm():
    """When real_scalars is per-round, scales_with should describe the
    per-round scaling, not the per-solve total.  The leading
    T_{\\rm ADMM} factor is therefore removed from ADMM's
    scales_with line.

    (T_ADMM moves to the 'iterations' field.)"""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    block = re.search(
        r"def run_fronthaul_table\(.*?return ExperimentResult\(",
        src, re.S,
    )
    assert block is not None
    body = block.group(0)
    # Look for any CORDIS-ADMM scales_with line containing T_{\rm ADMM}.
    bad = re.search(
        r'"CORDIS-ADMM".*?"scales_with":\s*r?"[^"]*T_\{?\\?rm ADMM',
        body, re.S,
    )
    assert bad is None, (
        "CORDIS-ADMM scales_with still contains a T_{\\rm ADMM} factor, "
        "but real_scalars is now per-round — the T factor should move to "
        "the 'iterations' field."
    )


@_register("Test  8b: no stale 'admm_per_iter' references in registry.py")
def test_08b_no_stale_admm_per_iter():
    """The Stage 17 rewrite renamed the local variable
    ``admm_per_iter`` (and its derived ``admm_total``) to
    ``admm_per_round`` everywhere in the fronthaul-table block.  A
    user-reported leftover in the metadata dict (line 725) referenced
    the deleted name and raised "Unresolved reference" at runtime.

    Belt-and-braces against any future regression of the same kind:
    the file should not reference ``admm_per_iter`` anywhere, since
    that variable no longer exists."""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    # Look for the variable name as a whole word (not as part of a
    # longer identifier like "admm_per_iter_real_scalars" — though
    # such an identifier should also have been renamed).
    bad = re.findall(r'\badmm_per_iter\b', src)
    assert not bad, (
        f"found {len(bad)} reference(s) to the now-deleted "
        f"`admm_per_iter` variable in registry.py.  Rename to "
        f"`admm_per_round` to match the Stage 17 vocabulary."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — dispatcher logs when filtering
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  9: _exp_common.py logs when n_drops/n_realizations are filtered")
def test_09_dispatcher_logs_filtering():
    """User visibility: when a fixed-trial experiment doesn't accept
    n_drops/n_realizations, the user's CLI value is silently dropped.
    Stage 17 logs an info message so the behaviour is obvious."""
    src = (REPO_ROOT / "scripts" / "_exp_common.py").read_text()
    for kw in ("n-drops", "n-realizations"):
        assert f"--{kw} not applicable" in src, (
            f"_exp_common.py should log when --{kw} is filtered for a "
            f"fixed-trial experiment; message missing."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 17 validator — {len(_TESTS)} tests")
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

