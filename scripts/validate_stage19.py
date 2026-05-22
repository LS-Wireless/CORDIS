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


@_register("Test  2: helper extracts all six admm-tunable knobs from cfg "
           "(kappa, rho_admm, n_admm_max, eps_pri, eps_dual, xi_slack)")
def test_02_helper_extracts_all_three():
    reg = _load_registry_module()
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            admm=SimpleNamespace(
                kappa=2.5, rho=0.5, n_max=300,
                eps_pri=5e-4, eps_dual=7e-4, xi_slack=2e3,
            ),
        ),
    )
    kw = reg._admm_kwargs_from_cfg(cfg)
    expected = {
        "kappa": 2.5, "rho_admm": 0.5, "n_admm_max": 300,
        "eps_pri": 5e-4, "eps_dual": 7e-4, "xi_slack": 2e3,
    }
    assert kw == expected, (
        f"_admm_kwargs_from_cfg should extract all six knobs the user "
        f"can set in cfg.algorithm.admm.  Got: {kw}\nExpected: {expected}"
    )
    # Type promises.
    for k in ("kappa", "rho_admm", "eps_pri", "eps_dual", "xi_slack"):
        assert isinstance(kw[k], float), f"{k} should be float, got {type(kw[k])}"
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
        # Accept any cfg-like variable name (cfg, cfg_v for kappa_sweep's
        # overridden cfg, etc.) — what matters is that the helper is
        # invoked with something cfg-shaped.
        if not re.search(r"_admm_kwargs_from_cfg\(\s*cfg\w*\s*\)", body):
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
           "carries non-default kappa/rho/n_max/eps_pri/eps_dual/xi_slack "
           "into spec.params")
def test_07_end_to_end_spec_params():
    """The user's original test: set cfg.algorithm.admm.kappa=2.5 and
    verify the algorithm actually uses 2.5, not the hardcoded default.
    Extended to cover all six fields after Stage 19b audit (eps_pri,
    eps_dual, xi_slack were caught by the same audit pattern)."""
    reg = _load_registry_module()
    admm_spec = _load_admm_spec()
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(
            admm=SimpleNamespace(
                kappa=2.5, rho=0.5, n_max=300,
                eps_pri=5e-4, eps_dual=7e-4, xi_slack=2e3,
            ),
        ),
    )
    # Simulate what e.g. run_sinr_cdf now does internally.
    spec_kwargs = {"n_ue": 4}
    for k, v in reg._admm_kwargs_from_cfg(cfg).items():
        spec_kwargs.setdefault(k, v)
    spec = admm_spec(**spec_kwargs)
    p = spec.params
    checks = (
        ("kappa",       2.5),
        ("rho_admm",    0.5),
        ("n_admm_max",  300),
        ("eps_pri",     5e-4),
        ("eps_dual",    7e-4),
        ("xi_slack",    2e3),
    )
    for key, expected in checks:
        assert p[key] == expected, (
            f"{key}: got {p[key]!r}, expected {expected!r}"
        )


@_register("Test  8: dispatcher (_dispatch_cordis_admm) declares "
           "eps_pri/eps_dual/xi_slack in its fwd_keys list (so spec.params "
           "actually reach the algorithm)")
def test_08_dispatcher_fwd_keys():
    """The full chain is cfg → spec.params → fwd_keys filter →
    solve_cordis_admm kwargs.  Stage 19 fixed the cfg → spec.params step;
    this test locks in that fwd_keys includes the new fields, so the
    spec.params hop actually reaches the algorithm.

    Source-grep is sufficient because the fwd_keys tuple is a literal in
    cordis/simulation/scenario.py."""
    src = (REPO_ROOT / "cordis" / "simulation" / "scenario.py").read_text()
    # Find the _dispatch_cordis_admm function body.
    m = re.search(
        r"def\s+_dispatch_cordis_admm\s*\(.*?(?=^def\s+\w)",
        src, re.M | re.S,
    )
    assert m, "could not locate _dispatch_cordis_admm in scenario.py"
    body = m.group(0)
    # Extract the fwd_keys tuple text.
    fk = re.search(r"fwd_keys\s*=\s*\(\s*([^)]+)\)", body, re.S)
    assert fk, "could not locate fwd_keys tuple in _dispatch_cordis_admm"
    fwd_keys_text = fk.group(1)
    required = ("kappa", "rho_admm", "n_admm_max",
                "eps_pri", "eps_dual", "xi_slack")
    missing = [k for k in required if f'"{k}"' not in fwd_keys_text]
    assert not missing, (
        f"_dispatch_cordis_admm.fwd_keys is missing: {missing}.  Even if "
        f"spec.params carries these (Stage 19), they won't reach "
        f"solve_cordis_admm without the dispatcher forwarding them.  "
        f"Current fwd_keys text:\n{fwd_keys_text}"
    )


@_register("Test  9: run_kappa_sweep updates BOTH cfg.algorithm.admm.kappa "
           "AND cfg.algorithm.split.kappa per sweep point")
def test_09_kappa_sweep_updates_split_too():
    """User-reported: a κ-sweep figure showed Split as a flat horizontal
    line because only admm.kappa was being overridden per sweep point —
    Split (which reads cfg.algorithm.split.kappa directly inside
    solve_cordis_split) kept its dataclass default 1.0 throughout.

    Source-grep the body of run_kappa_sweep for both override paths."""
    src = (REPO_ROOT / "cordis" / "experiments" / "registry.py").read_text()
    m = re.search(
        r"def\s+run_kappa_sweep\s*\(.*?(?=^def\s+\w)",
        src, re.M | re.S,
    )
    assert m, "could not locate run_kappa_sweep"
    body = m.group(0)
    assert '"algorithm.admm.kappa"' in body, (
        "run_kappa_sweep must override cfg.algorithm.admm.kappa "
        "(read directly by solve_centralized)"
    )
    assert '"algorithm.split.kappa"' in body, (
        "run_kappa_sweep must override cfg.algorithm.split.kappa "
        "(read directly by solve_cordis_split — without this, Split "
        "appears as a flat line in κ-sweep figures because it keeps "
        "the dataclass default 1.0)"
    )


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

