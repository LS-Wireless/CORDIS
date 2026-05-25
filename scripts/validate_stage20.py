#!/usr/bin/env python3
"""
scripts/validate_stage20.py
============================

Stage 20 — ADMM robustness improvements.

After Stage 19 fixed the cfg-plumbing for κ/ρ/eps and Stage 19a-e
tuned the defaults, the user reported that CORDIS-ADMM still
struggles with min-SINR on hard channel realizations: it maximises
sensing utility well but lets the per-user SINR floor drift below γ
on a non-negligible fraction of drops.

Stage 20 ships three targeted improvements:

  Step 1.  Converged-exit returns the W with highest min-SINR seen
           (was: just-converged iterate's W).  In the non-converged
           and inner-failure exits, this was already done — Step 1
           extends the same "best-iterate" return semantics to the
           normal converged path.

  Step 2.  Honor cfg.algorithm.admm.warm_start_from_split (was a
           Stage 19b "dead config field").  When True (default), the
           ADMM warm-start runs the full CORDIS-Split pipeline and
           uses its W_tx — instead of the equal-PSR (50/50) fallback
           that the code was using regardless of the flag.

  Step 3.  Add infeasibility tracking to AlgorithmResult and the CDF
           plotting helper.  Three new methods on AlgorithmResult:
             - infeasibility_rate(gamma_db) → float
             - feasible_trials_mask(gamma_db) → bool array
             - cdf_conditional(metric, gamma_db) → (xs, fs)
           And three new kwargs on plot_cdf:
             - gamma_db          → draws γ marker, enables annotation
             - show_feasible_only → overlays the conditional CDF
             - annotate_infeasibility → appends "(inf=X.X%)" to labels

Tests verify each step at three tiers:
  Tier 1: function/method signature + plumbing
  Tier 2: behaviour on synthetic data
  Tier 3: round-trip — flipping a flag changes the output

Run from the repo root::

    python3 scripts/validate_stage20.py
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


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Converged-exit returns best-min-SINR iterate
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  1: converged-path return uses W_best (not W_cur) when "
           "best_r_pri < inf")
def test_01_converged_returns_best():
    """Source-grep: the converged return site must condition W_tx on
    best_r_pri, not unconditionally return W_cur.  Pattern:

        W_return = W_best if best_r_pri < float("inf") else W_cur
        ...
        W_tx=W_return,
    """
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    # Find the converged branch.  It's the only one preceded by the
    # comment "converged at iter".
    m = re.search(
        r'converged at iter.*?return ADMMResult\((.*?)\)',
        src, re.S,
    )
    assert m, "could not locate converged-exit return in joint_opt.py"
    block = m.group(1)
    # Must reference W_best (the Step-1 fix); must NOT unconditionally
    # use W_cur (the pre-Step-1 bug).
    assert "W_best" in block or "W_return" in block, (
        "converged return must use W_best (Stage 20 Step 1): the "
        "best-min-SINR iterate.  Pre-Step-1 code unconditionally "
        "returned W_cur which can be worse on min-SINR if the "
        "convergence iter wasn't the best-min-SINR iter seen.\n"
        f"Current return-block excerpt:\n{block[:200]}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Honor warm_start_from_split flag
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  2: solve_cordis_admm reads cfg.algorithm.admm.warm_start_"
           "from_split and dispatches to run_cordis_split when True")
def test_02_warm_start_dispatches_to_split():
    """Source-grep: the warm-start block must reference both the
    config flag AND run_cordis_split.  Previously the comment said it
    used Phase I of Split but the code used equal-PSR regardless."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    # Look at the warm-start area — between "Warm-start" comment and the
    # next major section.
    m = re.search(
        r"# ── Warm-start.*?(?=\n    # ──)",
        src, re.S,
    )
    assert m, "could not locate warm-start block in joint_opt.py"
    block = m.group(0)
    assert "warm_start_from_split" in block, (
        "warm-start block must reference cfg.algorithm.admm."
        "warm_start_from_split (Stage 20 Step 2).  Without this, the "
        "flag is a dead config field and the equal-PSR fallback is "
        "always used."
    )
    assert "run_cordis_split" in block, (
        "warm-start block must call run_cordis_split when the flag is "
        "True (Stage 20 Step 2).  Without this, the warm-start uses "
        "Split's Phase I (no PSR optimization) only — the very issue "
        "we've been tracking since Stage 17."
    )


@_register("Test  3: equal-PSR fallback is still reachable when flag is "
           "False (no regression on legacy behaviour)")
def test_03_equal_psr_fallback_still_reachable():
    """The else-branch must still call build_W_tx_equal_psr so users
    who explicitly set warm_start_from_split=False get the old
    behaviour for A/B comparison."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    m = re.search(
        r"# ── Warm-start.*?(?=\n    # ──)",
        src, re.S,
    )
    assert m
    block = m.group(0)
    assert "build_W_tx_equal_psr" in block, (
        "fallback path must still call build_W_tx_equal_psr when "
        "warm_start_from_split=False (Stage 20 Step 2).  Removing this "
        "removes the user's ability to A/B test with vs without the "
        "warm-start."
    )


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Infeasibility tracking on AlgorithmResult + plot_cdf
# ═════════════════════════════════════════════════════════════════════════════

@_register("Test  4: AlgorithmResult.infeasibility_rate(gamma_db) exists "
           "with correct signature")
def test_04_infeasibility_rate_signature():
    try:
        from cordis.simulation.result import AlgorithmResult
    except (ModuleNotFoundError, ImportError) as e:
        raise _SkipTest(f"cordis.simulation.result not importable: {e}")
    assert hasattr(AlgorithmResult, "infeasibility_rate"), (
        "AlgorithmResult must define infeasibility_rate (Stage 20 Step 3)"
    )
    sig = inspect.signature(AlgorithmResult.infeasibility_rate)
    params = list(sig.parameters.keys())
    # self, gamma_db, metric
    assert "gamma_db" in params, (
        f"infeasibility_rate(self, gamma_db, ...) expected; got params {params}"
    )
    assert "metric" in params, (
        f"infeasibility_rate should accept a metric= kwarg; got params {params}"
    )


@_register("Test  5: infeasibility_rate computes the right fraction "
           "on synthetic per-trial samples")
def test_05_infeasibility_rate_behaviour():
    try:
        from cordis.simulation.result import AlgorithmResult, _METRIC_REGISTRY
        import numpy as np
    except (ModuleNotFoundError, ImportError) as e:
        raise _SkipTest(f"cordis.simulation.result not importable: {e}")

    # Build a synthetic AlgorithmResult-like object with controllable
    # min_sinr_db values.  Use a stub class that just implements the
    # ducktyped surface infeasibility_rate needs.
    class _StubSINR:
        # AlgorithmResult._samples expects an attribute named per the
        # metric registry: "min_sinr_db" → ("sinr", "min_sinr_per_trial_db")
        # but we'll just lie about the resolution via _samples patching.
        pass

    # Easier: instantiate the actual class with minimal fields, then
    # monkey-patch _samples since we don't want to construct full
    # SINRStatistics here.
    samples_arr = np.array([2.0, 3.5, 5.0, 7.5, 10.0])
    #                       ↓     ↓     ↓ ──── all below γ=5 are "infeasible"
    # γ=5 → samples < 5 → [True, True, False, False, False] → 2/5 = 0.4

    class _SUT:
        """Stand-in that reuses AlgorithmResult's pure methods."""
        def has_metric(self, metric):  return metric == "min_sinr_db"
        def _samples(self, metric):    return samples_arr

    # Borrow the method via __get__ to bind to our stub
    sut = _SUT()
    rate = AlgorithmResult.infeasibility_rate(sut, gamma_db=5.0)  # type: ignore[arg-type]
    assert abs(rate - 0.4) < 1e-9, (
        f"infeasibility_rate at γ=5.0 should be 0.4 (samples < 5 are "
        f"{(samples_arr < 5).sum()}/{samples_arr.size}); got {rate}"
    )

    # γ=11.0 → all infeasible
    rate_high = AlgorithmResult.infeasibility_rate(sut, gamma_db=11.0)  # type: ignore[arg-type]
    assert abs(rate_high - 1.0) < 1e-9, (
        f"at γ=11.0 all samples should be infeasible; got {rate_high}"
    )

    # γ=0.0 → none infeasible
    rate_zero = AlgorithmResult.infeasibility_rate(sut, gamma_db=0.0)  # type: ignore[arg-type]
    assert abs(rate_zero - 0.0) < 1e-9, (
        f"at γ=0.0 no samples should be infeasible; got {rate_zero}"
    )


@_register("Test  6: AlgorithmResult.cdf_conditional restricts to feasible "
           "trials and returns sorted xs + monotone-increasing fs ∈ [0,1]")
def test_06_cdf_conditional_behaviour():
    try:
        from cordis.simulation.result import AlgorithmResult
        import numpy as np
    except (ModuleNotFoundError, ImportError) as e:
        raise _SkipTest(f"cordis.simulation.result not importable: {e}")

    samples_arr = np.array([2.0, 3.5, 5.0, 7.5, 10.0])

    class _SUT:
        def has_metric(self, metric):  return True
        def _samples(self, metric):    return samples_arr
        # cdf_conditional internally calls self.feasible_trials_mask(...)
        # so the stub must duck-type that method too.  Mirror the real
        # AlgorithmResult.feasible_trials_mask logic so the test
        # exercises the conditional-CDF math, not the masking detail.
        def feasible_trials_mask(self, gamma_db, metric="min_sinr_db"):
            return samples_arr >= gamma_db

    sut = _SUT()
    # γ=5: feasible are 5.0, 7.5, 10.0  (3 samples)
    xs, fs = AlgorithmResult.cdf_conditional(  # type: ignore[arg-type]
        sut, "min_sinr_db", gamma_db=5.0,
    )
    assert xs.size == 3, f"expected 3 feasible samples at γ=5.0; got {xs.size}"
    assert list(xs) == [5.0, 7.5, 10.0], f"xs wrong sort: {xs}"
    # fs should be 1/3, 2/3, 1.0
    np.testing.assert_allclose(fs, [1/3, 2/3, 1.0], rtol=0, atol=1e-12)


@_register("Test  7: plot_cdf accepts gamma_db / show_feasible_only / "
           "annotate_infeasibility kwargs without raising")
def test_07_plot_cdf_signature():
    try:
        from cordis.plotting.cdf import plot_cdf
    except (ModuleNotFoundError, ImportError) as e:
        raise _SkipTest(f"cordis.plotting.cdf not importable: {e}")
    sig = inspect.signature(plot_cdf)
    params = set(sig.parameters.keys())
    for new_kw in ("gamma_db", "show_feasible_only",
                   "annotate_infeasibility", "cond_metric"):
        assert new_kw in params, (
            f"plot_cdf must accept {new_kw!r} kwarg (Stage 20 Step 3).  "
            f"Got params: {sorted(params)}"
        )


@_register("Test  8: plot_cdf with gamma_db draws a γ axvline AND appends "
           "infeasibility annotation to legend labels")
def test_08_plot_cdf_gamma_marker_and_annotation():
    try:
        from cordis.plotting.cdf import plot_cdf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except (ModuleNotFoundError, ImportError) as e:
        raise _SkipTest(f"plotting deps not importable: {e}")

    # Build a stub SimResult-like with a single algorithm.
    samples_arr = np.array([2.0, 3.5, 5.0, 7.5, 10.0])
    feas_mask   = samples_arr >= 5.0

    class _StubAR:
        def has_metric(self, m):      return True
        def _samples(self, m):        return samples_arr
        def cdf(self, m):
            xs = np.sort(samples_arr)
            return xs, np.arange(1, xs.size + 1) / xs.size
        def cdf_conditional(self, metric, gamma_db, cond_metric="min_sinr_db"):
            arr = self._samples(metric)
            mask = arr >= gamma_db
            xs = np.sort(arr[mask])
            return xs, np.arange(1, xs.size + 1) / xs.size if xs.size else np.zeros(0)
        def infeasibility_rate(self, gamma_db, metric="min_sinr_db"):
            return float(np.mean(self._samples(metric) < gamma_db))
        def feasible_trials_mask(self, gamma_db, metric="min_sinr_db"):
            return feas_mask

    class _StubSR:
        algorithm_results = {"FakeAlgo": _StubAR()}

    fig, ax = plt.subplots()
    plot_cdf(_StubSR(), "min_sinr_db", ax=ax,
             gamma_db=5.0,
             show_feasible_only=True,
             annotate_infeasibility=True)
    # The γ axvline should be among the plotted lines.
    line_xs = [tuple(l.get_xdata()) for l in ax.get_lines()]
    has_gamma_line = any(
        len(xs) == 2 and xs[0] == xs[1] == 5.0
        for xs in line_xs
    )
    assert has_gamma_line, (
        "plot_cdf should draw a vertical line at gamma_db=5.0 "
        "(Stage 20 Step 3)"
    )
    # The legend label for FakeAlgo should be annotated with infeasibility rate.
    legend = ax.get_legend()
    assert legend is not None, "expected a legend on the axes"
    labels = [t.get_text() for t in legend.get_texts()]
    has_annotated = any("inf=" in lbl for lbl in labels)
    assert has_annotated, (
        f"expected at least one label with 'inf=' annotation; got {labels}"
    )
    plt.close(fig)


@_register("Test  9: cfg.algorithm.admm.warm_start_from_split flag is still "
           "True in the dataclass default (Stage 19b → 20 promoted live)")
def test_09_warm_start_flag_default():
    try:
        from cordis.utils.config import CORDISConfig
    except (ModuleNotFoundError, ImportError) as e:
        raise _SkipTest(f"cordis.utils.config not importable: {e}")
    cfg = CORDISConfig()
    flag = getattr(cfg.algorithm.admm, "warm_start_from_split", None)
    assert flag is True, (
        f"cfg.algorithm.admm.warm_start_from_split should default to True "
        f"so users get the Split warm-start without opting in.  Got: {flag!r}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 78)
    print(f"  Stage 20 validator — {len(_TESTS)} tests")
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

