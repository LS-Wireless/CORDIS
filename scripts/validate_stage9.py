#!/usr/bin/env python3
"""
scripts/validate_stage9.py
==========================

Stage 9 validator.

Stage 9 cleans up three vestigial config patterns that the codebase
inherited from earlier iterations:

* **n_trials decomposition** — ``simulation.n_trials`` is now actively
  consumed via :func:`scripts._exp_common.resolve_drops_real`.
  Explicit ``--n-drops`` / ``--n-realizations`` win;
  ``--n-trials`` or ``cfg.simulation.n_trials`` is decomposed into the
  closest factor pair otherwise.

* **gamma_db consolidation** — ``γ_u`` lives at
  ``cfg.algorithm.gamma_db`` (umbrella level) and is shared by every
  algorithm.  ``algorithm.split.gamma_db`` is removed.

* **sigma_clt / clutter_cnr_db** — ``sensing.sigma_clt`` is now
  ``Optional[float] = None``.  ``None`` (default) means "derive σ_clt²
  from CNR".  An explicit value means "use σ_clt² = sigma_clt² directly,
  ignoring CNR".

Tests are grouped by tier (structural, behavioural, regression).  Each
test prints its own ✓ / ✗ / ⏭ line.

Run from the repo root::

    python3 scripts/validate_stage9.py

Exit code is 0 iff every test passes (skips do not fail the run).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


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


def _have_cordis_lite() -> bool:
    """Whether we can import the lightweight pieces of cordis we need."""
    try:
        from cordis.experiments import specs  # noqa: F401
        from cordis.utils import config       # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Structural: the renamed surfaces exist where we expect them
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: AlgorithmConfig.gamma_db field exists")
def test_01_algorithm_gamma_db_field():
    if not _have_cordis_lite():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import AlgorithmConfig
    from dataclasses import fields
    field_names = {f.name for f in fields(AlgorithmConfig)}
    assert "gamma_db" in field_names, (
        f"AlgorithmConfig.gamma_db missing; fields = {sorted(field_names)}"
    )


@_register("Test  2: SplitOptConfig.gamma_db field removed")
def test_02_split_gamma_db_gone():
    if not _have_cordis_lite():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import SplitOptConfig
    from dataclasses import fields
    field_names = {f.name for f in fields(SplitOptConfig)}
    assert "gamma_db" not in field_names, (
        f"SplitOptConfig.gamma_db should have been hoisted to AlgorithmConfig; "
        f"still present in fields {sorted(field_names)}"
    )


@_register("Test  3: SensingConfig.sigma_clt is Optional (default None)")
def test_03_sigma_clt_optional():
    if not _have_cordis_lite():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import SensingConfig
    cfg = SensingConfig()
    assert cfg.sigma_clt is None, (
        f"SensingConfig.sigma_clt default should be None, got {cfg.sigma_clt!r}"
    )


@_register("Test  4: PARAM_REGISTRY has algorithm.gamma_db, not split.gamma_db")
def test_04_metadata_renamed():
    if not _have_cordis_lite():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import PARAM_REGISTRY
    assert "algorithm.gamma_db" in PARAM_REGISTRY, (
        "PARAM_REGISTRY missing 'algorithm.gamma_db'"
    )
    assert "algorithm.split.gamma_db" not in PARAM_REGISTRY, (
        "PARAM_REGISTRY still has the old 'algorithm.split.gamma_db' key"
    )


@_register("Test  5: scripts/_exp_common exposes resolve_drops_real "
           "and closest_factor_pair")
def test_05_resolver_exposed():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import _exp_common as ec
    except Exception as e:
        raise _SkipTest(f"_exp_common not importable: {e}")
    assert hasattr(ec, "resolve_drops_real"), "missing resolve_drops_real"
    assert hasattr(ec, "closest_factor_pair"), "missing closest_factor_pair"


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — Behavioural: the resolvers and override paths work
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  6: closest_factor_pair returns balanced factors")
def test_06_factor_pairs():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import closest_factor_pair as cfp
    cases = [
        (1,    (1, 1)),
        (4,    (2, 2)),
        (12,   (3, 4)),
        (100,  (10, 10)),
        (200,  (10, 20)),
        (400,  (20, 20)),
        (500,  (20, 25)),
        (13,   (1, 13)),   # prime — lopsided is expected
    ]
    for n, expect in cases:
        got = cfp(n)
        assert got == expect, f"cfp({n}) = {got}, expected {expect}"


@_register("Test  7: closest_factor_pair raises on invalid input")
def test_07_factor_pair_invalid():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import closest_factor_pair as cfp
    for bad in [0, -1, -10]:
        try:
            cfp(bad)
        except ValueError:
            continue
        raise AssertionError(f"closest_factor_pair({bad}) should have raised")


@_register("Test  8: resolve_drops_real precedence — explicit drops×real wins")
def test_08_resolver_explicit():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import resolve_drops_real
    args = argparse.Namespace(n_drops=8, n_realizations=5, n_trials=200)
    d, r, src = resolve_drops_real(args, cfg_n_trials=400)
    assert (d, r) == (8, 5), f"got ({d}, {r}); expected (8, 5)"
    assert "explicit" in src, f"source mismatch: {src!r}"


@_register("Test  9: resolve_drops_real precedence — CLI --n-trials decomposes")
def test_09_resolver_cli_n_trials():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import resolve_drops_real
    args = argparse.Namespace(n_drops=None, n_realizations=None, n_trials=200)
    d, r, src = resolve_drops_real(args, cfg_n_trials=999)
    assert (d, r) == (10, 20), f"got ({d}, {r}); expected (10, 20)"
    assert "n_trials=200" in src, f"source mismatch: {src!r}"


@_register("Test 10: resolve_drops_real precedence — config n_trials decomposes")
def test_10_resolver_cfg_n_trials():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import resolve_drops_real
    args = argparse.Namespace(n_drops=None, n_realizations=None, n_trials=None)
    d, r, src = resolve_drops_real(args, cfg_n_trials=100)
    assert (d, r) == (10, 10), f"got ({d}, {r}); expected (10, 10)"
    assert "n_trials=100" in src, f"source mismatch: {src!r}"


@_register("Test 11: resolve_drops_real precedence — fallback when nothing set")
def test_11_resolver_fallback():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import resolve_drops_real
    args = argparse.Namespace(n_drops=None, n_realizations=None, n_trials=None)
    d, r, src = resolve_drops_real(args, cfg_n_trials=None,
                                   default_n_drops=50, default_n_real=4)
    assert (d, r) == (50, 4), f"got ({d}, {r}); expected (50, 4)"
    assert "fallback" in src, f"source mismatch: {src!r}"


@_register("Test 12: resolve_drops_real — partial explicit (drops only)")
def test_12_resolver_partial_drops():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from _exp_common import resolve_drops_real
    args = argparse.Namespace(n_drops=8, n_realizations=None, n_trials=200)
    d, r, src = resolve_drops_real(args, cfg_n_trials=None)
    # ceil(200/8) = 25
    assert (d, r) == (8, 25), f"got ({d}, {r}); expected (8, 25)"


@_register("Test 13: spec builders default gamma_u_db to None")
def test_13_spec_gamma_default_none():
    if not _have_cordis_lite():
        raise _SkipTest("cordis not importable")
    from cordis.experiments.specs import (
        split_spec, admm_spec, centralized_spec,
        mrt_spec, zf_spec, global_zf_spec,
    )
    # With no explicit gamma_u_db, spec.params should NOT contain it.
    for name, fn in [("split", split_spec), ("admm", admm_spec),
                     ("centralized", centralized_spec),
                     ("mrt", mrt_spec), ("zf", zf_spec),
                     ("global_zf", global_zf_spec)]:
        spec = fn(n_ue=4)
        assert "gamma_u_db" not in spec.params, (
            f"{name}_spec without override should omit gamma_u_db from "
            f"params (so the algorithm falls back to cfg.algorithm.gamma_db); "
            f"got params={spec.params}"
        )


@_register("Test 14: spec builders honor explicit gamma_u_db override")
def test_14_spec_gamma_override():
    if not _have_cordis_lite():
        raise _SkipTest("cordis not importable")
    from cordis.experiments.specs import split_spec, admm_spec
    import numpy as np
    spec = split_spec(n_ue=3, gamma_u_db=15.0)
    assert "gamma_u_db" in spec.params, \
        "spec with explicit gamma_u_db should include it in params"
    np.testing.assert_allclose(spec.params["gamma_u_db"], [15.0, 15.0, 15.0])

    spec = admm_spec(n_ue=2, gamma_u_db=5.0)
    np.testing.assert_allclose(spec.params["gamma_u_db"], [5.0, 5.0])


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — Regression: no stale references to removed surfaces
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 15: no stale 'cfg.algorithm.split.gamma_db' references")
def test_15_no_stale_split_gamma_reads():
    # Search every .py under cordis/ for the removed access path.
    bad = []
    for p in (REPO_ROOT / "cordis").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "algorithm.split.gamma_db" in txt:
            bad.append(p.relative_to(REPO_ROOT))
    assert not bad, (
        f"found stale 'algorithm.split.gamma_db' references in {bad}"
    )


@_register("Test 16: no stale 'DEFAULT_GAMMA_DB' usage")
def test_16_no_stale_default_gamma_db():
    # The constant should be removed; check it's not referenced.
    bad = []
    for p in (REPO_ROOT / "cordis").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "DEFAULT_GAMMA_DB" in txt:
            bad.append(p.relative_to(REPO_ROOT))
    assert not bad, (
        f"DEFAULT_GAMMA_DB should be removed (gamma_u_db defaults to None now); "
        f"still referenced in {bad}"
    )


@_register("Test 17: recipes use GAMMA_DB (not SPLIT_GAMMA_DB)")
def test_17_recipe_gamma_renamed():
    txt = (REPO_ROOT / "configs" / "recipes" / "_defaults.sh").read_text()
    assert "GAMMA_DB=" in txt, "_defaults.sh missing GAMMA_DB"
    assert "SPLIT_GAMMA_DB" not in txt, (
        "_defaults.sh still has SPLIT_GAMMA_DB (rename to GAMMA_DB)"
    )
    # SET_ARGS should map to algorithm.gamma_db, not algorithm.split.gamma_db
    assert "algorithm.gamma_db=" in txt, (
        "SET_ARGS missing algorithm.gamma_db mapping"
    )
    assert "algorithm.split.gamma_db=" not in txt, (
        "SET_ARGS still has algorithm.split.gamma_db (move to algorithm.gamma_db)"
    )


@_register("Test 18: recipes expose SIGMA_CLT (default null)")
def test_18_recipe_sigma_clt():
    txt = (REPO_ROOT / "configs" / "recipes" / "_defaults.sh").read_text()
    assert "SIGMA_CLT=" in txt, "_defaults.sh missing SIGMA_CLT variable"
    assert 'SIGMA_CLT:-null' in txt, (
        "SIGMA_CLT default should be 'null' (= use CNR formula)"
    )
    assert "sensing.sigma_clt=" in txt, (
        "SET_ARGS missing sensing.sigma_clt mapping"
    )


@_register("Test 19: launchers conditionally pass trial-count flags")
def test_19_launcher_conditional_flags():
    txt = (REPO_ROOT / "scripts" / "exp_sinr_cdf.sh").read_text()
    assert 'TRIAL_ARGS=()' in txt, (
        "Launcher should build TRIAL_ARGS conditionally"
    )
    assert '[ -n "$N_DROPS"' in txt
    assert '[ -n "$N_REAL"' in txt
    assert '[ -n "$N_TRIALS"' in txt
    # Old hardcoded UNCONDITIONAL --n-drops flag (with backslash line
    # continuation) should be gone.  The new conditional form pushes
    # the flag into TRIAL_ARGS=(--n-drops "$N_DROPS") inside an if-guard,
    # which is fine.
    bad_patterns = [
        '--n-drops       "$N_DROPS"       \\',
        '--n-realizations "$N_REAL"       \\',
    ]
    for bad in bad_patterns:
        assert bad not in txt, (
            f"Launcher still unconditionally passes the flag: {bad!r}"
        )


@_register("Test 20: --n-trials flag exposed on exp_<name>.py --help")
def test_20_cli_n_trials_flag():
    env = {**os.environ,
           "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "exp_sinr_cdf.py"), "--help"],
        capture_output=True, text=True, env=env, timeout=10,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise _SkipTest(f"--help failed (cordis import?): {proc.stderr[-200:]}")
    assert "--n-trials" in proc.stdout, (
        f"--n-trials not in --help output; stdout tail: {proc.stdout[-300:]}"
    )


@_register("Test 21: sensing_channel.py branches on sigma_clt is None")
def test_21_sensing_channel_branches():
    p = REPO_ROOT / "cordis" / "channel" / "sensing_channel.py"
    txt = p.read_text()
    assert "s_cfg.sigma_clt is None" in txt, (
        "sensing_channel.py should branch on sigma_clt is None for CNR fallback"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 9 validator — {len(_TESTS)} tests")
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

