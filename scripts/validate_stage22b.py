#!/usr/bin/env python3
"""
scripts/validate_stage22b.py
============================

Stage 22b umbrella validator.  Covers:

  * Adaptive ρ scheme (Boyd-Parikh-Chu §3.4.1, conservative variant)
  * Patience-based early stopping
  * Their config plumbing through ADMMConfig + default.json
  * Stability safeguards (bounded growth, warmup, instability detector)

Run as a standalone script or via the Stage 8 umbrella validator.
Returns non-zero exit code on any failure.

    python3 scripts/validate_stage22b.py
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
#  Test registration
# ─────────────────────────────────────────────────────────────────────────────

class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn):
        _TESTS.append((label, fn))
        return fn
    return deco


def _have_cordis() -> bool:
    try:
        import cordis  # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — Config plumbing
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: ADMMConfig has all 9 stage-22b fields with safe defaults")
def test_01_admm_config_fields():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import ADMMConfig
    cfg = ADMMConfig()
    expected = {
        # adaptive ρ
        "adaptive_rho":          True,
        "rho_mu_balance":        10.0,
        "rho_tau":               1.5,
        "rho_max_factor":        3.0,
        "rho_min_factor":        0.5,
        "rho_adapt_warmup":      5,
        "rho_adapt_interval":    3,
        # early-stop
        "early_stop_patience":   15,
        "early_stop_min_iters":  30,
    }
    for name, expected_val in expected.items():
        assert hasattr(cfg, name), f"ADMMConfig missing field: {name}"
        got = getattr(cfg, name)
        assert got == expected_val, (
            f"ADMMConfig.{name} = {got!r}, expected {expected_val!r}"
        )


@_register("Test  2: rho_max_factor ≤ 3.0 in defaults (SCA stability headroom)")
def test_02_rho_max_factor_safe():
    """The empirical instability threshold is ~5× of input rho_admm.
    The default cap must leave meaningful headroom (≤ 3.0 means the
    adaptive scheme can never push to within 60 % of instability)."""
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import ADMMConfig
    cfg = ADMMConfig()
    assert cfg.rho_max_factor <= 3.0, (
        f"rho_max_factor={cfg.rho_max_factor} too aggressive — "
        f"empirical instability at 5×, default must be ≤3 for safety"
    )
    assert cfg.rho_tau < 2.0, (
        f"rho_tau={cfg.rho_tau} too aggressive — Boyd's 2.0 destabilises "
        f"this SCA-based algorithm; use ≤1.75"
    )


@_register("Test  3: default.json mirrors ADMMConfig defaults")
def test_03_default_json_mirrors_dataclass():
    """The JSON default.json should expose the same defaults as the
    Python dataclass, so create_config.py audits and SET_ARGS overrides
    work consistently."""
    p = REPO_ROOT / "configs" / "default.json"
    if not p.exists():
        raise _SkipTest("configs/default.json not found")
    with open(p) as f:
        d = json.load(f)
    admm = d.get("algorithm", {}).get("admm", {})
    expected_keys = {
        "adaptive_rho", "rho_mu_balance", "rho_tau",
        "rho_max_factor", "rho_min_factor",
        "rho_adapt_warmup", "rho_adapt_interval",
        "early_stop_patience", "early_stop_min_iters",
    }
    missing = expected_keys - set(admm.keys())
    assert not missing, (
        f"default.json algorithm.admm missing stage-22b keys: "
        f"{sorted(missing)}"
    )


@_register("Test  4: PARAM_REGISTRY documents all 9 new fields")
def test_04_param_registry():
    if not _have_cordis():
        raise _SkipTest("cordis not importable")
    from cordis.utils.config import PARAM_REGISTRY
    expected = [
        "algorithm.admm.adaptive_rho",
        "algorithm.admm.rho_mu_balance",
        "algorithm.admm.rho_tau",
        "algorithm.admm.rho_max_factor",
        "algorithm.admm.rho_min_factor",
        "algorithm.admm.rho_adapt_warmup",
        "algorithm.admm.rho_adapt_interval",
        "algorithm.admm.early_stop_patience",
        "algorithm.admm.early_stop_min_iters",
    ]
    for key in expected:
        assert key in PARAM_REGISTRY, (
            f"PARAM_REGISTRY missing entry for {key}"
        )
        entry = PARAM_REGISTRY[key]
        assert "help" in entry and entry["help"], f"{key}: empty help"
        assert "range" in entry and entry["range"], f"{key}: empty range"


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — Algorithm behaviour (source-level audits)
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  5: solve_cordis_admm reads new fields via getattr from cfg.algorithm.admm")
def test_05_solve_reads_cfg():
    """The new knobs are algorithm meta-knobs — they live in cfg and
    don't need to thread through spec.params.  Verify they're read."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    for knob in ("adaptive_rho", "rho_tau", "rho_mu_balance",
                 "rho_max_factor", "rho_min_factor",
                 "rho_adapt_warmup", "rho_adapt_interval",
                 "early_stop_patience", "early_stop_min_iters"):
        # Look for getattr(cfg.algorithm.admm, "knob_name", ...)
        pattern = rf'getattr\s*\(\s*cfg\.algorithm\.admm\s*,\s*"{knob}"'
        assert re.search(pattern, src), (
            f"solve_cordis_admm doesn't read cfg.algorithm.admm.{knob} "
            f"via getattr"
        )


@_register("Test  6: ADMMResult has rho_history + early_stopped fields")
def test_06_admm_result_fields():
    # Try the live-import path first (best — exercises dataclass defaults);
    # fall back to source inspection if cordis can't be imported (e.g. in
    # a stripped-down test mirror without cvxpy).
    try:
        from cordis.algorithms.joint_opt import ADMMResult
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(ADMMResult)}
        assert "rho_history" in field_names, (
            "ADMMResult.rho_history field missing"
        )
        assert "early_stopped" in field_names, (
            "ADMMResult.early_stopped field missing"
        )
        r = ADMMResult(W_tx={})
        assert r.rho_history == [], f"rho_history default = {r.rho_history!r}"
        assert r.early_stopped is False, f"early_stopped default = {r.early_stopped!r}"
        return
    except (ImportError, ModuleNotFoundError):
        pass  # fall through to source-level check

    # Source-level fallback
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    assert re.search(
        r"rho_history\s*:\s*List\[float\]\s*=\s*field\s*\(\s*default_factory=list\s*\)",
        src,
    ), "ADMMResult.rho_history field declaration not found in source"
    assert re.search(
        r"early_stopped\s*:\s*bool\s*=\s*False", src
    ), "ADMMResult.early_stopped field declaration not found in source"


@_register("Test  7: source contains ν rescaling block on ρ change")
def test_07_nu_rescaling_present():
    """When ρ changes by factor τ in scaled-form ADMM, the dual variable
    ν must be rescaled by 1/τ (so λ = ρ·ν stays consistent).  Verify
    the code does this — without it, dual residual would jump on every
    ρ update and the convergence behaviour would be wrong."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    # Look for the nu_rescale block — should be inside the adaptive ρ section
    assert "nu_rescale" in src, (
        "Adaptive-ρ block missing the ν rescaling step.  Without it, "
        "the dual variable's effective Lagrange multiplier λ=ρ·ν would "
        "jump on every ρ change, violating scaled-form ADMM theory."
    )
    # The rescaling factor should be old/new (so ν shrinks when ρ grows)
    assert re.search(
        r"nu_rescale\s*=\s*rho_factor_current\s*/\s*new_factor", src
    ), (
        "ν rescaling formula is wrong — must be (rho_old / rho_new) so "
        "ν shrinks when ρ grows (since u = λ/ρ in scaled form)"
    )


@_register("Test  8: source contains instability detector (r_pri_growing)")
def test_08_instability_detector():
    """The adaptive ρ scheme must NOT raise ρ when r_pri is growing —
    that's a symptom of SCA trust-region violation.  Empirically, ρ≥5×
    destabilises this algorithm; the detector blocks growth past the
    safe regime."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    assert "r_pri_growing" in src, (
        "Adaptive-ρ block missing the r_pri_growing instability detector"
    )
    # And it should be used to gate the increase branch
    # (search for the pattern: if r_pri_growing ... and the increase branch
    # in an elif so it's blocked when r_pri_growing is True)
    pattern = re.search(
        r"if\s+r_pri_growing\s*:.*?elif\s+r_pri_rel\s*>\s*rho_mu_balance",
        src, re.S,
    )
    assert pattern is not None, (
        "Adaptive-ρ control flow doesn't gate ρ-increase on "
        "r_pri_growing.  Expected:  if r_pri_growing: <decrease>  "
        "elif r_pri_rel > μ·r_dual_rel: <increase>"
    )


@_register("Test  9: patience-stop only fires under residual_norm criterion")
def test_09_patience_only_residual_norm():
    """The legacy 'min_sinr' criterion is not designed for patience-stop
    (the best-iter changes too often).  Verify the source gates the
    early-stop logic on best_iter_criterion == 'residual_norm'."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    # Find the early-stop block — it should test the criterion FIRST
    pattern = re.search(
        r'best_iter_criterion\s+in\s+\(\s*"residual_norm",\s*'
        r'"feasible_then_residual"\s*\)\s*\n\s*'
        r'and\s+early_stop_patience\s*>\s*0',
        src,
    )
    assert pattern is not None, (
        "Patience-stop block must gate on best_iter_criterion in "
        "('residual_norm', 'feasible_then_residual') AND early_stop_patience>0"
    )
    # And the early_stopped flag must be set on bail-out
    assert "early_stopped = True" in src, (
        "Patience-stop must set early_stopped=True so the post-loop "
        "ADMMResult records why the loop ended"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    width = 78
    print("=" * width)
    print(f"  Stage 22b validator — {len(_TESTS)} tests")
    print("=" * width)

    passed = skipped = failed = 0
    failed_tests: List[str] = []

    for label, fn in _TESTS:
        print(f"\n── {label} ──")
        try:
            fn()
        except _SkipTest as e:
            print(f"  ↷ SKIPPED  ({e})")
            skipped += 1
            continue
        except AssertionError as e:
            print("  ✗ FAILED")
            print(f"      {e}")
            failed += 1
            failed_tests.append(label)
            continue
        except Exception:
            print("  ✗ FAILED (unexpected exception)")
            traceback.print_exc()
            failed += 1
            failed_tests.append(label)
            continue
        print("  ✓ PASSED")
        passed += 1

    print()
    print("=" * width)
    if failed:
        print(f"  ✗ {failed} FAILURE(S)   "
              f"({passed} passed, {skipped} skipped, {failed} failed)")
        for t in failed_tests:
            print(f"    failed: {t}")
    else:
        print(f"  ✓ ALL TESTS PASSED   ({passed}/{passed + skipped} "
              f"passed, {skipped} skipped)")
    print("=" * width)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

