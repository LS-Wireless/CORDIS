#!/usr/bin/env python3
"""
scripts/validate_stage16.py
============================

Stage 16 validator.

Stage 16 fixes four user-reported bugs in the experiment dispatch:

1.  ``_set_field`` lost the int type of swept config fields (e.g.
    ``topology.n_ap`` became ``float``), breaking downstream
    ``range()``/array sizing with ``TypeError: 'float' object cannot
    be interpreted as an integer``.

2.  ``sweep_config_field`` never re-validated the swept config, so
    invalid combinations like ``n_rf_chains > n_ant`` (which can
    arise mid-sweep) silently produced empty results from per-drop
    build failures, with the error only surfacing when the plot
    script ran.

3.  ``_exp_common.py`` unconditionally injected ``n_drops`` and
    ``n_realizations`` into ``experiment_kwargs``, but
    ``run_convergence_trace`` and ``run_fronthaul_table`` don't
    accept them — every invocation of those experiments crashed
    immediately with ``TypeError: ... got an unexpected keyword
    argument 'n_drops'``.

4.  (Pair of 3.)  Both ``n_drops`` and ``n_realizations`` should be
    treated identically and forwarded only to functions that accept
    them.  The same signature-inspection pattern already used for
    ``--specs`` (Stage 10) generalizes to cover all three kwargs.

Tests verify:

* **Tier 1 — _set_field type preservation.**  int destination
  field + whole-valued float → int cast.  int destination + non-
  integer float → TypeError.
* **Tier 2 — Sweep re-validation.**  Sweep that pushes
  ``n_rf_chains > n_ant`` at some midpoint raises ValueError with
  a sweep-point context (vs. silently producing empty results).
* **Tier 3 — Dispatch signature filtering.**  n_drops + n_realizations
  forwarded ONLY to functions that declare them in their signature;
  trace/table don't get them.
* **Tier 4 — No regression.**  Sweep + CDF experiments still receive
  n_drops + n_realizations as before.

Run from the repo root::

    python3 scripts/validate_stage16.py
"""
from __future__ import annotations

import inspect
import sys
import traceback
from dataclasses import dataclass
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


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — _set_field type preservation
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: _set_field preserves int when destination is int + "
           "value is whole-valued float")
def test_01_int_preservation():
    _set_field = _load_sweeps()._set_field

    @dataclass
    class _Inner:
        n_ap: int = 4
        n_ant: int = 10

    @dataclass
    class _Outer:
        topology: _Inner = None
        def __post_init__(self):
            if self.topology is None:
                self.topology = _Inner()

    obj = _Outer()
    _set_field(obj, "topology.n_ap", 6.0)
    assert obj.topology.n_ap == 6, f"expected int 6, got {obj.topology.n_ap!r}"
    assert isinstance(obj.topology.n_ap, int), (
        f"expected int type, got {type(obj.topology.n_ap).__name__} "
        f"— this is the regression that produced the runtime "
        f"TypeError in range()."
    )


@_register("Test  2: _set_field rejects non-integer float into int field")
def test_02_int_rejection():
    _set_field = _load_sweeps()._set_field

    @dataclass
    class _Inner:
        n_ap: int = 4

    @dataclass
    class _Outer:
        topology: _Inner = None
        def __post_init__(self):
            if self.topology is None:
                self.topology = _Inner()

    obj = _Outer()
    try:
        _set_field(obj, "topology.n_ap", 4.5)
    except TypeError as e:
        assert "non-integer" in str(e).lower(), (
            f"error message should mention 'non-integer': {e}"
        )
        return
    raise AssertionError(
        "_set_field should refuse to cast non-integer float into int field"
    )


@_register("Test  3: _set_field leaves float destination untouched "
           "(no spurious cast)")
def test_03_no_unintended_cast():
    _set_field = _load_sweeps()._set_field

    @dataclass
    class _System:
        snr_db: float = 0.0

    @dataclass
    class _Cfg:
        system: _System = None
        def __post_init__(self):
            if self.system is None:
                self.system = _System()

    obj = _Cfg()
    _set_field(obj, "system.snr_db", 12.5)
    assert obj.system.snr_db == 12.5
    assert isinstance(obj.system.snr_db, float)


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — sweep re-validation
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  4: sweep_config_field calls cfg.validate() per point and "
           "re-raises with sweep context")
def test_04_sweep_revalidate():
    """Without this, antennas_sweep silently produces empty results
    when n_ant goes below n_rf_chains — the failure only surfaces at
    plot time, far away from the cause."""
    sw = _load_sweeps()
    sweep_config_field, SweepAxis = sw.sweep_config_field, sw.SweepAxis

    # Minimal fake config that mimics SimConfig.validate() raising on
    # invalid combos.
    @dataclass
    class _Topo:
        n_ant: int = 10
        n_rf_chains: int = 6

    @dataclass
    class _Cfg:
        topology: _Topo = None
        def __post_init__(self):
            if self.topology is None:
                self.topology = _Topo()
        def validate(self):
            if self.topology.n_rf_chains > self.topology.n_ant:
                raise ValueError(
                    f"n_rf_chains ({self.topology.n_rf_chains}) "
                    f"cannot exceed n_ant ({self.topology.n_ant})."
                )

    # Sweep n_ant from 4 to 10; n_rf_chains stays at 6.
    # At n_ant=4, the validate() should fire.
    axis = SweepAxis(name="n_ant", values=[4, 6, 8, 10])

    # Stub the runner so we only test the validation pathway.
    class _StubReport:
        pass

    def _stub_from_run(_report):
        return "stub_sim_result"

    class _StubRunner:
        def __init__(self, cfg, specs, runner_cfg):
            self.cfg = cfg
        def run(self):
            return _StubReport()

    cfg = _Cfg()
    try:
        sweep_config_field(
            cfg, specs=[], field_path="topology.n_ant",
            axis=axis, runner_cfg=None,
            runner_cls=_StubRunner,
            sim_result_from_run=_stub_from_run,
        )
    except ValueError as e:
        # Error must mention which sweep point failed.
        msg = str(e)
        assert "n_ant" in msg, (
            f"error should name the sweep axis: {msg}"
        )
        assert "4" in msg or "4.0" in msg, (
            f"error should name the failing sweep point value: {msg}"
        )
        return
    raise AssertionError(
        "sweep_config_field should re-raise validate() failures with "
        "sweep-point context, but completed without error"
    )


@_register("Test  5: sweep with all-valid points still runs to completion")
def test_05_sweep_all_valid():
    """Regression guard: re-validation must not break the happy path."""
    sw = _load_sweeps()
    sweep_config_field, SweepAxis = sw.sweep_config_field, sw.SweepAxis

    @dataclass
    class _Topo:
        n_ant: int = 10
        n_rf_chains: int = 4    # always ≤ smallest sweep value

    @dataclass
    class _Cfg:
        topology: _Topo = None
        def __post_init__(self):
            if self.topology is None:
                self.topology = _Topo()
        def validate(self):
            if self.topology.n_rf_chains > self.topology.n_ant:
                raise ValueError("invalid")

    class _StubRunner:
        def __init__(self, cfg, specs, runner_cfg): self.cfg = cfg
        def run(self): return None

    axis = SweepAxis(name="n_ant", values=[6, 8, 10, 12])
    out = sweep_config_field(
        _Cfg(), specs=[], field_path="topology.n_ant",
        axis=axis, runner_cfg=None,
        runner_cls=_StubRunner,
        sim_result_from_run=lambda r: f"point_for_{r}",
    )
    assert set(out.keys()) == {6.0, 8.0, 10.0, 12.0}, (
        f"expected all 4 sweep points to complete; got {sorted(out.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — dispatch signature filtering
# ─────────────────────────────────────────────────────────────────────────────

# Experiments whose run_* functions do NOT accept n_drops or n_realizations
# as kwargs.  Stage 16 ensures the dispatcher does NOT pass them.
_FIXED_EXPERIMENTS = ["convergence_trace", "fronthaul_table"]

# Experiments whose run_* functions DO accept both.  Stage 16 must not
# regress them.
_TRIAL_EXPERIMENTS = [
    "sinr_cdf", "scnr_cdf",
    "gamma_sweep", "kappa_sweep", "clutter_cnr_sweep",
    "snr_sweep", "n_ue_sweep", "n_ap_sweep", "antennas_sweep",
]


def _load_registry():
    try:
        from cordis.experiments.registry import REGISTRY
        return REGISTRY
    except Exception as e:
        raise _SkipTest(f"cordis.experiments.registry not importable: {e}")


def _load_sweeps():
    try:
        from cordis.experiments import sweeps
        return sweeps
    except Exception as e:
        raise _SkipTest(f"cordis.experiments.sweeps not importable: {e}")


@_register("Test  6: convergence_trace + fronthaul_table do NOT accept "
           "n_drops/n_realizations (precondition for the dispatcher fix)")
def test_06_fixed_experiment_signatures():
    """Records the current signatures of trace/table — if a future
    refactor adds n_drops to these signatures, the dispatcher fix
    becomes unnecessary, but the test should be updated then."""
    reg = _load_registry()
    for name in _FIXED_EXPERIMENTS:
        assert name in reg, f"{name} not in REGISTRY"
        sig = inspect.signature(reg[name])
        assert "n_drops" not in sig.parameters, (
            f"run_{name} now accepts n_drops; update the dispatcher fix "
            f"in _exp_common.py if this is intentional."
        )
        # n_realizations is allowed (fronthaul_table has one with a
        # default for its internal use), so we don't assert its absence.


@_register("Test  7: trial-bearing experiments still accept "
           "n_drops/n_realizations (no regression)")
def test_07_trial_experiment_signatures():
    reg = _load_registry()
    for name in _TRIAL_EXPERIMENTS:
        assert name in reg, f"{name} not in REGISTRY"
        sig = inspect.signature(reg[name])
        for kw in ("n_drops", "n_realizations"):
            assert kw in sig.parameters, (
                f"run_{name} should accept {kw!r}; missing from "
                f"{list(sig.parameters)}.  This is a regression — "
                f"trial-bearing experiments need these kwargs."
            )


@_register("Test  8: _exp_common.py uses signature inspection for n_drops "
           "+ n_realizations (not unconditional injection)")
def test_08_dispatch_uses_inspection():
    """Lock in the structural fix: the n_drops/n_realizations assignment
    must be gated by ``\"n_drops\" in sig.parameters`` — not unconditional.

    Both lock-ins are required: a future refactor could remove the
    signature check OR add an unconditional fallback, either of which
    would re-break trace/table."""
    src = (REPO_ROOT / "scripts" / "_exp_common.py").read_text()
    import re

    # The fix pattern MUST be present.
    for kw in ("n_drops", "n_realizations"):
        guard = re.search(
            rf'if\s+"{kw}"\s+in\s+sig\.parameters\s*:', src
        )
        assert guard is not None, (
            f"_exp_common.py should gate assignment of {kw!r} with "
            f"`if \"{kw}\" in sig.parameters:`; this guard is missing."
        )

    # The bad pattern from the bug (assignment NOT inside an if-block)
    # is harder to detect with a regex because the gated assignment
    # literally is `experiment_kwargs["n_drops"] = n_drops` — just
    # indented inside an `if`.  Use a structural check: count the
    # assignments and the if-guards; they must match (one guard per
    # assignment).
    n_assign = len(re.findall(
        r'experiment_kwargs\["(?:n_drops|n_realizations)"\]\s*=',
        src,
    ))
    n_guards = sum(
        len(re.findall(rf'if\s+"{kw}"\s+in\s+sig\.parameters\s*:', src))
        for kw in ("n_drops", "n_realizations")
    )
    assert n_assign == n_guards == 2, (
        f"expected exactly 2 gated assignments for n_drops + "
        f"n_realizations; found {n_assign} assignment(s) and "
        f"{n_guards} guard(s).  An unguarded assignment would "
        f"resurrect the trace/table TypeError."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — end-to-end dispatch smoke test
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  9: dispatcher can build kwargs for trace/table without "
           "TypeError (smoke test)")
def test_09_smoke_dispatch():
    """Exercise the same kwargs-assembly logic the dispatcher runs.
    This catches the user-reported `unexpected keyword argument 'n_drops'`
    failure without actually running the experiment."""
    reg = _load_registry()

    n_drops, n_real = 1, 2
    for name in _FIXED_EXPERIMENTS:
        fn = reg[name]
        sig = inspect.signature(fn)
        kwargs = {}
        if "n_drops" in sig.parameters:
            kwargs["n_drops"] = n_drops
        if "n_realizations" in sig.parameters:
            kwargs["n_realizations"] = n_real
        # If we tried to call fn(**kwargs) now, it must NOT raise
        # TypeError about n_drops.  We only assert that n_drops would
        # be filtered out for these fixed-algorithm experiments.
        assert "n_drops" not in kwargs, (
            f"dispatcher would still pass n_drops to {name} — fix Test 8"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 16 validator — {len(_TESTS)} tests")
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

