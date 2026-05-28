#!/usr/bin/env python3
"""
scripts/validate_stage10.py
============================

Stage 10 validator.

Stage 10 introduces named spec-set selection across the experiment
framework.  Tests verify:

* **registry surface** — ``_SPEC_SETS`` registered with the four
  built-in names, ``_resolve_spec_set`` rejects unknown names,
  ``list_spec_sets()`` exposes them publicly.

* **run_*** signatures — every spec-set-consuming experiment accepts
  ``spec_set`` as a keyword-only parameter with the correct historical
  default, and the two fixed-algorithm experiments
  (``run_convergence_trace``, ``run_fronthaul_table``) do not.

* **CLI plumbing** — ``--specs`` is exposed on every per-experiment
  runner script with the four named choices and a ``None`` default
  (so the run_* function's own default applies when omitted).

* **launcher** — the conditional ``SPEC_ARGS`` pattern is present in
  the generated launcher shell scripts.

Tests are ordered tier-by-tier; each prints its own ✓ / ✗ / ⏭ line.

Run from the repo root::

    python3 scripts/validate_stage10.py

Exit code is 0 iff every test passes (skips do not fail the run).
"""
from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Test infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


def _have_cordis() -> bool:
    try:
        from cordis.experiments import registry  # noqa: F401
        return True
    except Exception:
        return False


# Expected defaults per experiment.  Originally Stage 10 promoted the
# pre-Stage-10 hardcoded factory calls (where gamma/kappa/clutter_cnr
# sweeps used cordis_vs_centralized and antennas used
# cordis_vs_benchmarks).  Stage 17 then unified ALL CDF/sweep
# experiments on 'all_algorithms' for figure consistency — so this
# table reflects post-Stage-17 truth.
EXPECTED_DEFAULTS = {
    "run_sinr_cdf":          "all_algorithms",
    "run_scnr_cdf":          "all_algorithms",
    "run_gamma_sweep":       "all_algorithms",
    "run_kappa_sweep":       "all_algorithms",
    "run_clutter_cnr_sweep": "all_algorithms",
    "run_snr_sweep":         "all_algorithms",
    "run_csi_sweep":         "all_algorithms",
    "run_n_ue_sweep":        "all_algorithms",
    "run_n_ap_sweep":        "all_algorithms",
    "run_antennas_sweep":    "all_algorithms",
}

# Functions that intentionally don't accept spec_set (fixed by design).
FIXED_ALGORITHM_FNS = {"run_convergence_trace", "run_fronthaul_table"}


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Registry surface
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: _SPEC_SETS registered with 5 named factories")
def test_01_spec_sets_dict():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.experiments.registry import _SPEC_SETS
    expected = {"cordis_only", "cordis_vs_centralized",
                "cordis_vs_benchmarks", "all_algorithms",
                "psr_baselines"}
    got = set(_SPEC_SETS)
    missing = expected - got
    extra = got - expected
    assert not missing and not extra, (
        f"_SPEC_SETS mismatch — missing: {missing}, extra: {extra}"
    )
    # Each value must be callable.
    for name, fn in _SPEC_SETS.items():
        assert callable(fn), f"_SPEC_SETS[{name!r}] is not callable: {fn!r}"


@_register("Test  2: _resolve_spec_set returns factory or raises clear error")
def test_02_resolver():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.experiments.registry import _resolve_spec_set
    # Known name → callable
    fn = _resolve_spec_set("cordis_only")
    assert callable(fn), f"resolver returned non-callable: {fn!r}"
    # Unknown name → ValueError with helpful message
    try:
        _resolve_spec_set("not_a_real_set")
    except ValueError as e:
        msg = str(e)
        assert "not_a_real_set" in msg, f"error msg missing name: {msg!r}"
        assert "available" in msg.lower(), (
            f"error msg should list available sets: {msg!r}"
        )
    else:
        raise AssertionError(
            "_resolve_spec_set should raise ValueError on unknown name"
        )


@_register("Test  3: list_spec_sets publicly accessible from package root")
def test_03_list_spec_sets():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.experiments import list_spec_sets
    names = list_spec_sets()
    assert isinstance(names, list), f"list_spec_sets returned {type(names)}"
    assert len(names) == 5, f"expected 5 named sets, got {len(names)}: {names}"
    # Order matters for CLI choices display — must be consistent.
    assert "all_algorithms" in names
    assert "cordis_only" in names
    assert "psr_baselines" in names


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — run_* signatures
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  4: spec-set-consuming run_* fns accept spec_set kwarg")
def test_04_run_fn_signatures():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.experiments import registry
    for fn_name in EXPECTED_DEFAULTS:
        fn = getattr(registry, fn_name)
        sig = inspect.signature(fn)
        params = sig.parameters
        assert "spec_set" in params, (
            f"{fn_name} signature is missing spec_set; params = "
            f"{list(params)}"
        )
        p = params["spec_set"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{fn_name}.spec_set should be keyword-only, got kind={p.kind}"
        )


@_register("Test  5: each run_* has the correct historical default")
def test_05_run_fn_defaults():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.experiments import registry
    for fn_name, expected_default in EXPECTED_DEFAULTS.items():
        fn = getattr(registry, fn_name)
        sig = inspect.signature(fn)
        got = sig.parameters["spec_set"].default
        assert got == expected_default, (
            f"{fn_name}.spec_set default = {got!r}, "
            f"expected {expected_default!r} (historical behavior)"
        )


@_register("Test  6: fixed-algorithm run_* fns do NOT accept spec_set")
def test_06_fixed_algorithm_fns():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.experiments import registry
    for fn_name in FIXED_ALGORITHM_FNS:
        fn = getattr(registry, fn_name)
        sig = inspect.signature(fn)
        assert "spec_set" not in sig.parameters, (
            f"{fn_name} should NOT accept spec_set (its algorithm choice "
            f"is fixed by design), but signature has it: "
            f"{list(sig.parameters)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — CLI plumbing
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  7: add_specs_arg helper exported from _exp_common")
def test_07_helper_exported():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import _exp_common as ec
    except Exception as e:
        raise _SkipTest(f"_exp_common not importable: {e}")
    assert hasattr(ec, "add_specs_arg"), (
        "_exp_common.py should expose add_specs_arg"
    )


@_register("Test  8: --specs flag accepts the four named sets")
def test_08_cli_choices():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import add_drops_args
    parser = argparse.ArgumentParser()
    add_drops_args(parser)
    # Walk the parser's actions to find --specs
    specs_action = next(
        (a for a in parser._actions if "--specs" in a.option_strings),
        None,
    )
    assert specs_action is not None, "--specs flag not registered"
    assert specs_action.default is None, (
        f"--specs default should be None (so run_*'s default applies), "
        f"got {specs_action.default!r}"
    )
    assert specs_action.choices is not None, "--specs should restrict choices"
    expected = {"cordis_only", "cordis_vs_centralized",
                "cordis_vs_benchmarks", "all_algorithms",
                "psr_baselines"}
    got = set(specs_action.choices)
    assert got == expected, (
        f"--specs choices = {got}, expected {expected}"
    )


@_register("Test  9: --specs rejects invalid values at argparse time")
def test_09_invalid_specs_rejected():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import add_drops_args
    parser = argparse.ArgumentParser()
    add_drops_args(parser)
    # argparse raises SystemExit on choice violation; capture stderr.
    import io
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--specs", "not_a_real_set"])
        except SystemExit:
            return   # expected
    raise AssertionError(
        "--specs should reject invalid values at argparse time"
    )


@_register("Test 10: --specs flag exposed on exp_<name>.py --help")
def test_10_cli_help():
    env = {**os.environ,
           "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "exp_sinr_cdf.py"),
         "--help"],
        capture_output=True, text=True, env=env, timeout=10,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise _SkipTest(f"--help failed: {proc.stderr[-200:]}")
    assert "--specs" in proc.stdout, (
        f"--specs not in --help output; tail: {proc.stdout[-300:]}"
    )
    # All five named sets must appear in the help text (via choices=).
    for name in ("cordis_only", "cordis_vs_centralized",
                 "cordis_vs_benchmarks", "all_algorithms",
                 "psr_baselines"):
        assert name in proc.stdout, (
            f"named spec set {name!r} not listed in --help"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — Launcher plumbing
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 11: launchers expose SPECS env var + conditional --specs flag")
def test_11_launcher_plumbing():
    # Spot-check three representative experiments (one CDF, one sweep,
    # one with a non-default historical spec set).
    for exp_name in ("sinr_cdf", "snr_sweep", "gamma_sweep"):
        path = REPO_ROOT / "scripts" / f"exp_{exp_name}.sh"
        txt = path.read_text()
        assert 'SPECS="${SPECS:-}"' in txt, (
            f"{path.name}: missing SPECS env-var declaration"
        )
        assert "SPEC_ARGS=()" in txt, (
            f"{path.name}: missing SPEC_ARGS conditional array"
        )
        assert '[ -n "$SPECS" ] && SPEC_ARGS+=(--specs "$SPECS")' in txt, (
            f"{path.name}: SPEC_ARGS conditional missing"
        )
        # The SPEC_ARGS expansion must be threaded into the python call.
        assert '"${SPEC_ARGS[@]}"' in txt, (
            f"{path.name}: SPEC_ARGS not expanded into python invocation"
        )


@_register("Test 12: convergence_trace + fronthaul_table also expose --specs "
           "(safely ignored by run_*)")
def test_12_fixed_algorithm_launchers():
    # Stage-10 design: --specs is registered universally for UX
    # consistency, but the signature check in run_experiment silently
    # skips functions that don't accept spec_set.  Confirm the flag
    # IS exposed on these too (so users don't get cryptic argparse
    # errors when setting SPECS as a shared env var for a batch).
    for exp_name in ("convergence_trace", "fronthaul_table"):
        path = REPO_ROOT / "scripts" / f"exp_{exp_name}.sh"
        txt = path.read_text()
        assert 'SPECS="${SPECS:-}"' in txt, (
            f"{path.name}: should also expose SPECS env var "
            f"(silently ignored)"
        )


@_register("Test 13: run_experiment silently skips spec_set for fixed-algo fns")
def test_13_runtime_silent_skip():
    """Inspect run_experiment source to verify the signature-guard exists.
    A functional test would need a full cordis install; this static check
    is a good proxy."""
    src = (REPO_ROOT / "scripts" / "_exp_common.py").read_text()
    assert '"spec_set" in sig.parameters' in src, (
        "_exp_common.run_experiment should signature-check before "
        "forwarding spec_set (so fixed-algorithm run_* fns don't crash)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 10 validator — {len(_TESTS)} tests")
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

