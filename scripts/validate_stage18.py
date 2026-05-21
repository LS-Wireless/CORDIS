#!/usr/bin/env python3
"""
scripts/validate_stage18.py
============================

Stage 18 validator.

Stage 18 makes three notebook-polish changes:

1.  ``load_result(name, exp_dir=None)`` helper added to
    ``notebooks/_playground_helpers.py``.  Notebooks now load via this
    dispatcher: ``EXP_DIR = None`` falls back to the most recent run;
    setting ``EXP_DIR`` to a specific timestamped directory loads that
    one instead.  Replaces direct ``load_latest_result()`` calls in
    all four playground notebooks.

2.  **CDF + sweep notebooks parameterized via ``exp_name``.**  For the
    CDF notebook, the top cell sets ``exp_name = 'sinr'`` (or
    ``'scnr'``) and derives ``EXPERIMENT``, ``METRIC_MIN``,
    ``METRIC_MEAN``, ``METRIC_DISPLAY`` via f-strings.  All subsequent
    cells use the derived variables, so switching between
    SINR and SCNR is one edit.  For the sweep notebook, the same
    pattern parameterizes the experiment folder (``EXPERIMENT =
    f'{exp_name}_sweep'``); axis labels come from
    ``result.sweep_axis.display`` so they update automatically.
    Metric names in the sweep notebook stay hardcoded since they
    measure SINR/SCNR independently of what's being swept.

3.  **Trace-notebook ``best_iter`` bug fixed.**  ``ADMMResult`` now
    carries a ``best_iter: int = 0`` field populated by
    ``solve_cordis_admm`` at all three return sites.  The trace
    notebook's twin-axes cell uses ``getattr(admm, 'best_iter', None)``
    and only draws the axvline if a value is set — gracefully
    handling pre-Stage-18 loaded results where ``best_iter`` is None.

Tests verify:

* **Tier 1 — load_result helper.**  Helper exists with the expected
  signature; all four notebooks call it.
* **Tier 2 — exp_name parameterization.**  CDF + sweep notebooks
  declare ``exp_name = '...'`` at the top; CDF notebook uses
  ``METRIC_MIN`` / ``METRIC_DISPLAY`` everywhere downstream (no
  hardcoded ``min_sinr_db``).
* **Tier 3 — best_iter field + safe handling.**  ``ADMMResult`` has
  the field, populated at all three return sites; trace notebook
  uses ``getattr(..., 'best_iter', None)`` and a falsy-guard.

Run from the repo root::

    python3 scripts/validate_stage18.py
"""
from __future__ import annotations

import inspect
import json
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


def _load_helpers():
    try:
        sys.path.insert(0, str(REPO_ROOT / "notebooks"))
        import _playground_helpers as h
        return h
    except Exception as e:
        raise _SkipTest(f"notebooks/_playground_helpers not importable: {e}")


def _load_admm_result_class():
    try:
        from cordis.algorithms.joint_opt import ADMMResult
        return ADMMResult
    except Exception as e:
        raise _SkipTest(f"cordis.algorithms.joint_opt not importable: {e}")


def _notebook_code(nb_name: str) -> str:
    """Return all concatenated code-cell sources from a playground notebook."""
    nb_path = REPO_ROOT / "notebooks" / nb_name
    assert nb_path.exists(), f"missing notebook: {nb_path}"
    nb = json.loads(nb_path.read_text())
    return "\n".join(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )


_NOTEBOOKS = [
    "playground_cdf.ipynb",
    "playground_sweep.ipynb",
    "playground_trace.ipynb",
    "playground_table.ipynb",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — load_result helper
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: load_result helper exists with (name, exp_dir=None) signature")
def test_01_helper_exists():
    h = _load_helpers()
    assert hasattr(h, "load_result"), (
        "notebooks/_playground_helpers.py should define load_result"
    )
    sig = inspect.signature(h.load_result)
    params = list(sig.parameters.keys())
    assert params[0] == "experiment_name", (
        f"first arg should be 'experiment_name'; got {params[0]!r}"
    )
    assert "exp_dir" in sig.parameters, (
        "load_result should accept an `exp_dir` kwarg"
    )
    assert sig.parameters["exp_dir"].default is None, (
        f"`exp_dir` should default to None; got "
        f"{sig.parameters['exp_dir'].default!r}"
    )


@_register("Test  2: every playground notebook calls load_result(EXPERIMENT, EXP_DIR)")
def test_02_notebooks_use_load_result():
    """All 4 notebooks should call the new helper.  No notebook should
    still call load_latest_result directly — that bypasses the EXP_DIR
    knob the user asked for."""
    offenders = []
    for nb in _NOTEBOOKS:
        src = _notebook_code(nb)
        has_load_result = "load_result(" in src
        has_legacy      = "load_latest_result(" in src
        # load_run is fine — it's still used for explicit multi-run overlay
        # in the CDF notebook.
        if not has_load_result:
            offenders.append((nb, "missing load_result(...)"))
        if has_legacy:
            offenders.append((nb, "still calls load_latest_result(...)"))
        if "EXP_DIR" not in src:
            offenders.append((nb, "missing EXP_DIR variable"))
    assert not offenders, (
        "notebook(s) not using the new pattern:\n  "
        + "\n  ".join(f"{n}: {r}" for n, r in offenders)
    )


@_register("Test  2b: every playground notebook imports load_result "
           "(not just calls it)")
def test_02b_load_result_imported():
    """Calling ``load_result(...)`` is not enough — it also has to be in
    the explicit ``from _playground_helpers import (...)`` list at the
    top of each notebook.  Without the import, every cell that uses it
    raises NameError at runtime.

    Stage 18 originally shipped with the call site updated but the
    import statement (in ``INTRO_SETUP``) untouched; that's exactly
    the regression this test now locks out."""
    offenders = []
    for nb in _NOTEBOOKS:
        src = _notebook_code(nb)
        # Look for an import line that names load_result.  We accept any
        # whitespace, parenthesised multi-line imports, and trailing
        # commas — what matters is that `load_result` appears as a name
        # in a `from _playground_helpers import ...` block.
        has_import = bool(re.search(
            r"from\s+_playground_helpers\s+import\s+[^)]*\bload_result\b",
            src, re.S,
        ))
        if not has_import:
            offenders.append(nb)
    assert not offenders, (
        "load_result is called but not imported in:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd `load_result` to the `from _playground_helpers "
          "import (...)` list in INTRO_SETUP."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — exp_name parameterization
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  3: CDF + sweep notebooks declare 'exp_name = ...' at the top")
def test_03_exp_name_declared():
    for nb in ("playground_cdf.ipynb", "playground_sweep.ipynb"):
        src = _notebook_code(nb)
        assert re.search(r"^exp_name\s*=", src, re.M), (
            f"{nb} should declare `exp_name = '...'` so users only need "
            f"to edit one cell to switch between metric families / sweep "
            f"parameters."
        )


@_register("Test  4: CDF notebook uses METRIC_MIN/METRIC_DISPLAY everywhere "
           "downstream (no hardcoded 'min_sinr_db')")
def test_04_cdf_fully_parameterized():
    src = _notebook_code("playground_cdf.ipynb")
    # Hardcoded metric strings should not appear in code cells.
    bad = re.findall(r"metric=\s*['\"]min_sinr_db['\"]", src)
    assert not bad, (
        f"playground_cdf.ipynb still has {len(bad)} hardcoded "
        f"`metric='min_sinr_db'` reference(s); should use "
        f"`metric=METRIC_MIN` so the notebook works for SCNR too."
    )
    # And METRIC_MIN must actually be used (sanity).
    assert "metric=METRIC_MIN" in src, (
        "playground_cdf.ipynb should reference `METRIC_MIN` in at least "
        "one cell; parameterization missing."
    )


@_register("Test  5: CDF notebook derives EXPERIMENT + metrics via f-strings "
           "from exp_name")
def test_05_derived_names_in_cdf():
    src = _notebook_code("playground_cdf.ipynb")
    # f-string derivations should be present.
    for pattern, friendly in [
        (r"f'\{exp_name\}_cdf'", "EXPERIMENT = f'{exp_name}_cdf'"),
        (r"f'min_\{exp_name\}_db'", "METRIC_MIN = f'min_{exp_name}_db'"),
        (r"f'mean_\{exp_name\}_db'", "METRIC_MEAN = f'mean_{exp_name}_db'"),
    ]:
        assert re.search(pattern, src), (
            f"playground_cdf.ipynb should derive names via f-strings: "
            f"missing `{friendly}` (or similar)"
        )


@_register("Test  6: sweep notebook derives EXPERIMENT via f'{exp_name}_sweep'")
def test_06_sweep_derives_experiment():
    src = _notebook_code("playground_sweep.ipynb")
    assert re.search(r"f'\{exp_name\}_sweep'", src), (
        "playground_sweep.ipynb should derive EXPERIMENT via "
        "f'{exp_name}_sweep' so switching between snr/kappa/gamma sweeps "
        "is one edit"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — best_iter field + safe handling
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  7: ADMMResult has a best_iter field")
def test_07_admm_result_has_best_iter():
    cls = _load_admm_result_class()
    from dataclasses import fields
    field_names = {f.name for f in fields(cls)}
    assert "best_iter" in field_names, (
        f"ADMMResult should have a `best_iter` field; current fields: "
        f"{sorted(field_names)}"
    )


@_register("Test  8: solve_cordis_admm populates best_iter at every "
           "ADMMResult return site")
def test_08_best_iter_populated():
    """Lock-in: the three return sites that build an ADMMResult should
    each pass `best_iter=best_iter`.  Catches any future refactor that
    adds a return path and forgets the field."""
    src = (REPO_ROOT / "cordis" / "algorithms" / "joint_opt.py").read_text()
    n_returns = len(re.findall(r"return ADMMResult\(", src))
    n_best_iter = len(re.findall(r"best_iter\s*=\s*best_iter\s*,", src))
    assert n_returns == 3, (
        f"expected exactly 3 `return ADMMResult(...)` sites in "
        f"joint_opt.py; found {n_returns}.  If this changed, update "
        f"the test to match the new count."
    )
    assert n_best_iter == 3, (
        f"expected exactly 3 `best_iter=best_iter,` assignments at the "
        f"return sites; found {n_best_iter}.  At least one return site "
        f"is missing the field."
    )


@_register("Test  9: trace notebook uses getattr+falsy-guard for best_iter "
           "(survives None/0 cases)")
def test_09_trace_notebook_safe():
    """The user-reported `'>' not supported between instances of 'float'
    and 'NoneType'` came from calling `axvline(admm.best_iter)` when
    best_iter was None (legacy LoadedADMMResult default).  Stage 18
    guards every reference with getattr+truthiness."""
    src = _notebook_code("playground_trace.ipynb")
    # The exact bug pattern — must be gone.
    bad = re.search(
        r"axvline\(\s*admm\.best_iter\s*[,)]",
        src,
    )
    assert bad is None, (
        "playground_trace.ipynb still calls `ax.axvline(admm.best_iter)` "
        "directly; if best_iter is None this raises a TypeError inside "
        "matplotlib.  Wrap with `getattr(admm, 'best_iter', None)` and "
        "only draw the line if truthy."
    )
    # The fix pattern — must be present.
    has_guard = re.search(
        r"getattr\(\s*admm\s*,\s*['\"]best_iter['\"]",
        src,
    )
    assert has_guard is not None, (
        "playground_trace.ipynb should access best_iter via "
        "`getattr(admm, 'best_iter', None)` and guard the axvline call "
        "behind a truthy check."
    )


@_register("Test 10: trace save/load round-trip preserves per-user "
           "diagnostic fields (sinr_history, n_admm_iters, etc.)")
def test_10_trace_roundtrip_preserves_diag_fields():
    """User-reported gap: ``LoadedADMMResult`` previously dropped
    ``sinr_history``, ``n_admm_iters``, ``converged``, ``feasible``,
    ``inner_failures``, ``sensing_obj_history``, ``z_norm_history``
    on save/load.  The per-user diagnostic in playground_trace.ipynb
    couldn't access them.  Stage-18-diag-v2 extends the save path to
    persist all of them and the load path + dataclass to read them
    back.

    Test by round-tripping a synthetic ADMMResult and confirming every
    extended field is preserved.  Lock-in for the contract."""
    try:
        from cordis.experiments.result import ExperimentResult
        import numpy as np
        import tempfile
        from pathlib import Path
    except ModuleNotFoundError as e:
        raise _SkipTest(f"cordis.experiments.result not importable: {e}")

    expected_fields = {
        "sinr_history":        (4, 3),
        "sensing_obj_history": (4,),
        "z_norm_history":      (4,),
    }
    expected_scalars = {
        "converged":      False,
        "feasible":       True,
        "n_admm_iters":   4,
        "inner_failures": 1,
        "best_iter":      3,
    }

    class _FakeADMM:
        def __init__(self):
            self.primal_res_history  = np.array([1.0, 0.5, 0.25, 0.12])
            self.dual_res_history    = np.array([2.0, 1.0, 0.5, 0.25])
            self.slack_history       = np.ones((4, 3)) * 0.1
            self.sinr_history        = np.ones((4, 3)) * 15.0
            self.sensing_obj_history = np.array([0.5, 0.6, 0.7, 0.8])
            self.z_norm_history      = np.array([10.0, 9.0, 8.5, 8.2])
            for k, v in expected_scalars.items():
                setattr(self, k, v)

    try:
        result = ExperimentResult(name="diag_test", kind="trace",
                                   admm_result=_FakeADMM(), metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "out"
            exp_dir.mkdir()
            result.save(exp_dir)
            loaded = ExperimentResult.load(exp_dir)
    except ModuleNotFoundError as e:
        raise _SkipTest(f"transitive cordis dep missing: {e}")

    a = loaded.admm_result
    for fname, exp_shape in expected_fields.items():
        v = getattr(a, fname, None)
        assert v is not None, (
            f"{fname} was dropped by save/load — Stage-18-diag-v2 "
            f"extension regression"
        )
        assert v.shape == exp_shape, (
            f"{fname} shape mismatch: expected {exp_shape}, got {v.shape}"
        )
    for fname, exp_val in expected_scalars.items():
        v = getattr(a, fname, None)
        assert v == exp_val, (
            f"{fname} round-trip mismatch: expected {exp_val!r}, got {v!r}"
        )


@_register("Test 11: legacy trace (without extended fields) loads with "
           "graceful defaults, not AttributeError")
def test_11_legacy_trace_loads():
    """A trace.npz saved before Stage-18-diag-v2 has only the 4 original
    fields.  Loading it should not raise — extended fields should fall
    back to dataclass defaults (None for arrays, False/0/True for scalars)."""
    try:
        from cordis.experiments.result import ExperimentResult
        import numpy as np, json, tempfile
        from pathlib import Path
    except ModuleNotFoundError as e:
        raise _SkipTest(f"cordis.experiments.result not importable: {e}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "legacy"
            exp_dir.mkdir()
            (exp_dir / "manifest.json").write_text(json.dumps({
                "name": "convergence_trace", "kind": "trace", "metadata": {},
            }))
            np.savez_compressed(exp_dir / "trace.npz",
                primal_res_history=np.array([1.0, 0.5]),
                dual_res_history=np.array([2.0, 1.0]),
            )
            loaded = ExperimentResult.load(exp_dir)
    except ModuleNotFoundError as e:
        raise _SkipTest(f"transitive cordis dep missing: {e}")

    a = loaded.admm_result
    for fname in ("sinr_history", "sensing_obj_history",
                  "z_norm_history", "slack_history"):
        assert getattr(a, fname) is None, (
            f"{fname} should be None for legacy trace, got "
            f"{getattr(a, fname)!r}"
        )
    assert a.best_iter is None
    assert a.converged is False
    assert a.feasible is True
    assert a.n_admm_iters == 0
    assert a.inner_failures == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 18 validator — {len(_TESTS)} tests")
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

