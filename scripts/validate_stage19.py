#!/usr/bin/env python3
"""
scripts/validate_stage19.py
============================

Stage 19 validator.

Stage 19 fixes a user-reported config-plumbing bug in CORDIS-ADMM:
``cfg.algorithm.admm.kappa`` and ``cfg.algorithm.admm.rho`` (and
``cfg.algorithm.admm.n_max``) had **no effect on the algorithm**.
Changing them in the JSON config produced identical results.

The bug.  ``admm_spec()`` has function-level defaults
``DEFAULT_KAPPA = 1.0`` and ``DEFAULT_RHO_ADMM = 1.0`` that get
written into ``spec.params``.  ``_dispatch_cordis_admm`` forwards
``spec.params["kappa"]`` and ``["rho_admm"]`` to
``run_cordis_admm()``.  But every ``run_*`` registry function called
``factory(**spec_kwargs)`` without first reading
``cfg.algorithm.admm.*`` into ``spec_kwargs`` — so the
function-level defaults always won, regardless of cfg.  Result:
silent silent override of user config.

(``cfg.algorithm.admm.kappa`` did work for Centralized — see
``solve_centralized``, which reads cfg directly — making the asymmetry
particularly confusing.  ``cfg.algorithm.admm.n_max`` partially
worked via an indirect path for ``run_convergence_trace`` but not
through admm_spec.)

The fix.  Stage 19 adds ``_admm_kwargs_from_cfg(cfg)`` to
``registry.py`` and calls it in every ``run_*`` function before
spec construction:

.. code-block:: python

    spec_kwargs.setdefault("n_ue", _get_n_ue(cfg))
    for k, v in _admm_kwargs_from_cfg(cfg).items():
        spec_kwargs.setdefault(k, v)
    specs = factory(**spec_kwargs)

``setdefault`` is critical: explicit per-call kwargs (e.g. the
swept value in ``run_kappa_sweep``) still win, but anything not
explicitly set picks up the cfg value.

Tests verify:

* **Tier 1** — helper present, signature, behaviour, override semantics
* **Tier 2** — every run_* call site forwards the cfg dict
* **Tier 3** — end-to-end: a synthetic cfg with non-default κ, ρ_admm,
  n_admm_max actually produces an ``admm_spec`` whose ``params`` carry
  those values (the path the user was originally testing manually)

Run from the repo root::

    python3 scripts/validate_stage19.py
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


def _load_registry_module():
    try:
        import cordis.experiments.registry as reg
        return reg
    except ModuleNotFoundError as e:
        raise _SkipTest(f"cordis.experiments.registry not importable: {e}")


def _load_admm_spec():
    try:
        from cordis.experiments.specs import admm_spec
        return admm_spec
    except ModuleNotFoundError as e:
        raise _SkipTest(f"cordis.experiments.specs not importable: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — helper present & correct behaviour
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: _admm_kwargs_from_cfg helper exists in registry.py")
def test_01_helper_exists():
    reg = _load_registry_module()
    assert hasattr(reg, "_admm_kwargs_from_cfg"), (
        "cordis/experiments/registry.py should define "
        "`_admm_kwargs_from_cfg(cfg)` — the canonical bridge between "
        "cfg.algorithm.admm.* and admm_spec()."
    )
    sig = inspect.signature(reg._admm_kwargs_from_cfg)
    assert len(sig.parameters) == 1, (
        f"helper should take cfg as its only positional arg; "
        f"got signature {sig}"
    )


@_register("Test  2: helper extracts kappa, rho_admm, n_admm_max from cfg")
def test_02_helper_extracts_all_three():
    reg = _load_registry_module()
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            admm=SimpleNamespace(kappa=2.5, rho=0.5, n_max=300),
        ),
    )
    kw = reg._admm_kwargs_from_cfg(cfg)
    expected = {"kappa": 2.5, "rho_admm": 0.5, "n_admm_max": 300}
    assert kw == expected, (
        f"_admm_kwargs_from_cfg should extract all three knobs the user "
        f"can set in cfg.algorithm.admm.  Got: {kw}\nExpected: {expected}"
    )
    # Type promises.
    assert isinstance(kw["kappa"], float)
    assert isinstance(kw["rho_admm"], float)
    assert isinstance(kw["n_admm_max"], int)


@_register("Test  3: setdefault semantics preserve explicit overrides")
def test_03_setdefault_preserves_overrides():
    """``run_kappa_sweep`` passes ``kappa=k`` explicitly per sweep point.
    The cfg-forwarding must not clobber that — setdefault must NOT
    overwrite a key that's already in spec_kwargs."""
    reg = _load_registry_module()
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            admm=SimpleNamespace(kappa=1.0, rho=1.0, n_max=50),
        ),
    )
    spec_kwargs = {"n_ue": 4, "kappa": 99.0}  # explicit override
    for k, v in reg._admm_kwargs_from_cfg(cfg).items():
        spec_kwargs.setdefault(k, v)
    assert spec_kwargs["kappa"] == 99.0, (
        "explicit kappa=99.0 in spec_kwargs should win over cfg.kappa=1.0"
    )
    assert spec_kwargs["rho_admm"] == 1.0, (
        "rho_admm should be filled in from cfg (not set explicitly)"
    )


@_register("Test  4: helper degrades gracefully on a cfg missing the "
           "admm namespace")
def test_04_helper_no_admm_namespace():
    reg = _load_registry_module()
    from types import SimpleNamespace
    cfg = SimpleNamespace()  # no .algorithm at all
    kw = reg._admm_kwargs_from_cfg(cfg)
    assert kw == {}, (
        f"helper should return an empty dict when cfg has no "
        f"algorithm.admm namespace; got {kw}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — every run_* call site forwards the cfg dict
# ─────────────────────────────────────────────────────────────────────────────

# The list of run_* functions that build specs and need the forwarding.
_RUN_FNS_NEEDING_FORWARD = [
    "run_sinr_cdf",
    "run_scnr_cdf",
    "run_gamma_sweep",
    "run_kappa_sweep",
    "run_clutter_cnr_sweep",
    "run_snr_sweep",
    "run_n_ue_sweep",
    "run_n_ap_sweep",
    "run_antennas_sweep",
    "run_convergence_trace",
    "run_fronthaul_table",
]


@_register("Test  5: every run_* function calls _admm_kwargs_from_cfg "
           "before building its specs")
def test_05_every_run_fn_forwards():
    """Structural check on the registry source.  Each run_* function's
    body must contain a reference to ``_admm_kwargs_from_cfg(cfg)`` — or
    explicitly opt out via a comment, which none currently do."""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    # Extract each function's body.
    function_blocks = {}
    pattern = re.compile(
        r"^def\s+(run_\w+)\s*\(.*?(?=^def\s+\w|\Z)",
        re.M | re.S,
    )
    for m in pattern.finditer(src):
        function_blocks[m.group(1)] = m.group(0)

    missing = []
    for fn_name in _RUN_FNS_NEEDING_FORWARD:
        body = function_blocks.get(fn_name, "")
        if not body:
            missing.append(f"{fn_name} (not found in registry.py)")
            continue
        if "_admm_kwargs_from_cfg(cfg)" not in body:
            missing.append(f"{fn_name} (no cfg-forwarding)")
    assert not missing, (
        "run_* functions missing the cfg-forwarding pattern:\n  "
        + "\n  ".join(missing)
        + "\n\nEach such function silently ignores cfg.algorithm.admm.kappa "
          "and cfg.algorithm.admm.rho, falling back to admm_spec's hardcoded "
          "defaults.  Add the standard pattern:\n\n"
          "    for k, v in _admm_kwargs_from_cfg(cfg).items():\n"
          "        spec_kwargs.setdefault(k, v)"
    )


@_register("Test  6: no run_* function calls factory()/admm_spec() with "
           "ONLY n_ue (the pre-Stage-19 bug pattern)")
def test_06_no_naked_factory_calls():
    """The pre-Stage-19 bug pattern was a naked ``factory(n_ue=...)`` or
    ``admm_spec(n_ue=...)`` call with no preceding cfg forwarding.  The
    fix turns these into dict-builders that merge cfg defaults first.
    Catch any regression that reintroduces a naked call."""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    # Patterns that were the bug.  All call sites should now be either:
    #   factory(**spec_kwargs)  (after cfg-forwarding into spec_kwargs)
    #   factory(**_kw)          (after building _kw with cfg forwarding)
    bad_patterns = [
        r"specs\s*=\s*factory\(n_ue\s*=[^)]*\)\s*$",
        r"specs\s*=\s*\[\s*admm_spec\(n_ue\s*=[^)]*\)\s*\]\s*$",
        r"spec_obj\s*=\s*admm_spec\(n_ue\s*=[^,)]*\)\s*$",
    ]
    offenders = []
    for line_num, line in enumerate(src.splitlines(), 1):
        for pat in bad_patterns:
            if re.search(pat, line):
                offenders.append((line_num, line.strip()))
    assert not offenders, (
        "found naked factory/admm_spec call(s) that bypass cfg-forwarding:"
        + "\n  ".join(f"\n  line {ln}: {txt}" for ln, txt in offenders)
        + "\n\nWrap with the _kw-builder pattern (see other run_* "
          "functions for the template)."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — end-to-end: cfg → spec.params
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  7: end-to-end — admm_spec() with cfg-forwarded kwargs "
           "carries non-default kappa/rho_admm/n_admm_max into spec.params")
def test_07_end_to_end_spec_params():
    """The user's original test: set cfg.algorithm.admm.kappa=2.5 and
    verify the algorithm actually uses 2.5, not the hardcoded default.

    Lock-in for the prime symptom.  Pure construction; no algorithm
    execution needed (we just check that the kappa value lands in
    spec.params where the dispatcher will read it from)."""
    reg = _load_registry_module()
    admm_spec = _load_admm_spec()
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            admm=SimpleNamespace(kappa=2.5, rho=0.5, n_max=300),
        ),
    )
    # Simulate what e.g. run_sinr_cdf now does internally.
    spec_kwargs = {"n_ue": 4}
    for k, v in reg._admm_kwargs_from_cfg(cfg).items():
        spec_kwargs.setdefault(k, v)
    spec = admm_spec(**spec_kwargs)
    p = spec.params
    assert p["kappa"]      == 2.5,  f"kappa: got {p['kappa']!r}, expected 2.5"
    assert p["rho_admm"]   == 0.5,  f"rho_admm: got {p['rho_admm']!r}, expected 0.5"
    assert p["n_admm_max"] == 300,  f"n_admm_max: got {p['n_admm_max']!r}, expected 300"


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 19 validator — {len(_TESTS)} tests")
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

