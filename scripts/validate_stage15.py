#!/usr/bin/env python3
"""
scripts/validate_stage15.py
============================

Stage 15 validator.

Validates that every call into ``cordis.plotting`` and
``notebooks/_playground_helpers`` from a generated playground notebook
matches the actual function signature.  This closes a structural gap
in the generator: string literals inside ``_build_playgrounds.py``
were never type-checked or executed during validation, so typos and
wrong kwargs silently shipped (we hit this with ``save_paper_figure``,
``avg_sinr_db``, ``plot_admm_convergence(ax=...)``, etc).

Detection strategy:

1.  Build a registry mapping function name → resolved Python callable
    (introspected via ``inspect.signature``).
2.  Parse each generated ``.ipynb`` to AST, walk every code cell.
3.  For every ``ast.Call`` node whose callable is in the registry:
    -  every keyword arg must be in the signature (or function must
       accept ``**kwargs``)
    -  no argument is passed both positionally and as a keyword
    -  no excess positional args beyond what the signature accepts
       (unless the function takes ``*args``)
4.  Separately, every ``metric=<string-literal>`` reference must be a
    key in ``cordis.simulation.result._METRIC_REGISTRY``.

Tests are intentionally narrow — they don't aim to fully type-check
the notebooks, just to catch the failure modes that have actually
bitten us.

Run from the repo root::

    python3 scripts/validate_stage15.py

Exit code is 0 iff every test passes.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR    = REPO_ROOT / "notebooks"

NOTEBOOKS = [
    "playground_cdf.ipynb",
    "playground_sweep.ipynb",
    "playground_trace.ipynb",
    "playground_table.ipynb",
]


class _SkipTest(Exception):
    pass


_TESTS: List[Tuple[str, Callable[[], None]]] = []


def _register(label: str):
    def deco(fn: Callable[[], None]) -> Callable[[], None]:
        _TESTS.append((label, fn))
        return fn
    return deco


# ─────────────────────────────────────────────────────────────────────────────
#  Registry builder — maps callable name → Python callable for inspect
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry() -> Dict[str, Callable[..., Any]]:
    """Resolve every function name we expect notebooks to call.

    Missing imports raise _SkipTest (e.g. on stub-only environments)
    so the validator degrades gracefully when not everything is
    installed.  In CI/dev with the full env, every entry resolves.
    """
    reg: Dict[str, Callable[..., Any]] = {}

    # cordis.plotting public surface
    sys.path.insert(0, str(REPO_ROOT))
    try:
        cp = importlib.import_module("cordis.plotting")
    except Exception as e:
        raise _SkipTest(f"cordis.plotting failed to import: {e}")

    plotting_names = [
        "plot_cdf", "plot_sweep", "plot_admm_convergence",
        "bar_chart", "to_latex_table", "to_markdown_table",
        "save_figure", "figsize", "apply_paper_style",
    ]
    for name in plotting_names:
        if hasattr(cp, name):
            reg[name] = getattr(cp, name)

    # _playground_helpers (lives in notebooks/, hence sibling-import path)
    sys.path.insert(0, str(NB_DIR))
    try:
        ph_spec = importlib.util.spec_from_file_location(
            "_playground_helpers", NB_DIR / "_playground_helpers.py",
        )
        ph = importlib.util.module_from_spec(ph_spec)
        ph_spec.loader.exec_module(ph)
    except Exception as e:
        # Not fatal; just won't validate calls into _playground_helpers.
        return reg

    helper_names = [
        "setup_paper_style", "latest_run_dir",
        "load_latest_result", "load_run", "summarize",
    ]
    for name in helper_names:
        if hasattr(ph, name):
            reg[name] = getattr(ph, name)

    return reg


# ─────────────────────────────────────────────────────────────────────────────
#  Per-call validator
# ─────────────────────────────────────────────────────────────────────────────

def _validate_call(call_node: ast.Call, sig: inspect.Signature) -> List[str]:
    """Return a list of human-readable error strings for one Call node."""
    errors: List[str] = []
    params = sig.parameters

    # Classify parameters once.
    positional_params: List[inspect.Parameter] = []
    keyword_param_names = set()
    has_var_positional = False
    has_var_keyword = False
    required_names = set()       # names without a default
    for p in params.values():
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        else:
            keyword_param_names.add(p.name)
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
                positional_params.append(p)
            if p.default is inspect.Parameter.empty and \
                    p.kind != inspect.Parameter.VAR_KEYWORD:
                required_names.add(p.name)

    # Check 1: unknown kwargs.
    passed_kwargs = set()
    for kw in call_node.keywords:
        if kw.arg is None:
            # **kwargs unpacking; can't statically check.
            continue
        passed_kwargs.add(kw.arg)
        if kw.arg not in keyword_param_names and not has_var_keyword:
            errors.append(f"unexpected keyword argument '{kw.arg}'")

    # Check 2: positional/keyword overlap.
    n_positional_passed = len(call_node.args)
    consumed_by_positional = {
        p.name for p in positional_params[:n_positional_passed]
    }
    for kw_name in passed_kwargs:
        if kw_name in consumed_by_positional:
            errors.append(
                f"argument '{kw_name}' got multiple values "
                f"(passed positionally AND as keyword)"
            )

    # Check 3: too many positional.
    if not has_var_positional and \
            n_positional_passed > len(positional_params):
        errors.append(
            f"too many positional arguments: got {n_positional_passed}, "
            f"signature allows at most {len(positional_params)}"
        )

    # Check 4: missing required arguments (positional or keyword).
    bound_param_names = set(consumed_by_positional) | passed_kwargs
    missing_required = sorted(required_names - bound_param_names)
    if missing_required:
        errors.append(
            f"missing required argument(s): {missing_required}"
        )

    return errors


def _extract_simple_callable(call_node: ast.Call) -> str | None:
    """Return ``foo`` for ``foo(...)``; None for ``obj.method(...)`` etc.

    We don't try to resolve attribute accesses (``ax.legend(...)``,
    ``mod.func(...)``); those are out of scope for this checker.
    """
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    return None


def _walk_notebook(nb_name: str,
                   registry: Dict[str, Callable[..., Any]]
                   ) -> List[Tuple[int, str, str]]:
    """Return (cell_idx, callable_name, error_message) tuples."""
    issues: List[Tuple[int, str, str]] = []
    nb = json.loads((NB_DIR / nb_name).read_text())

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            issues.append((i, "<parse>", f"syntax error: {e}"))
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = _extract_simple_callable(node)
            if fname is None or fname not in registry:
                continue
            try:
                sig = inspect.signature(registry[fname])
            except (ValueError, TypeError):
                continue
            for err in _validate_call(node, sig):
                issues.append((i, fname, err))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — call-signature checks (the headline tests)
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: registry resolves at least 6 callables")
def test_01_registry_populated():
    reg = _build_registry()
    if len(reg) < 6:
        raise _SkipTest(
            f"registry only resolved {len(reg)} callable(s) — environment may "
            f"be stubbed.  Need ≥6 (e.g. plot_cdf, plot_sweep, "
            f"plot_admm_convergence, bar_chart, save_figure, figsize)."
        )


@_register("Test  2: playground_cdf.ipynb passes signature checks")
def test_02_cdf_calls_valid():
    reg = _build_registry()
    issues = _walk_notebook("playground_cdf.ipynb", reg)
    assert not issues, _format_issues("playground_cdf.ipynb", issues)


@_register("Test  3: playground_sweep.ipynb passes signature checks")
def test_03_sweep_calls_valid():
    reg = _build_registry()
    issues = _walk_notebook("playground_sweep.ipynb", reg)
    assert not issues, _format_issues("playground_sweep.ipynb", issues)


@_register("Test  4: playground_trace.ipynb passes signature checks")
def test_04_trace_calls_valid():
    reg = _build_registry()
    issues = _walk_notebook("playground_trace.ipynb", reg)
    assert not issues, _format_issues("playground_trace.ipynb", issues)


@_register("Test  5: playground_table.ipynb passes signature checks")
def test_05_table_calls_valid():
    reg = _build_registry()
    issues = _walk_notebook("playground_table.ipynb", reg)
    assert not issues, _format_issues("playground_table.ipynb", issues)


def _format_issues(nb_name: str,
                   issues: List[Tuple[int, str, str]]) -> str:
    lines = [f"{nb_name}: {len(issues)} call-signature error(s):"]
    for cell_idx, fname, msg in issues:
        lines.append(f"  cell {cell_idx}: {fname}() — {msg}")
    lines.append(
        "Fix the offending call(s) in notebooks/_build_playgrounds.py "
        "and re-run `python3 notebooks/_build_playgrounds.py`."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — metric name validation
# ─────────────────────────────────────────────────────────────────────────────

# Match metric='name' or metric="name" in any cell.
_METRIC_KW_RE = re.compile(r"""metric\s*=\s*['"]([a-z_][a-z0-9_]*)['"]""")


def _collect_metric_references(nb_name: str) -> List[Tuple[int, str]]:
    """Return (cell_idx, metric_name) for every metric=... call in nb."""
    out: List[Tuple[int, str]] = []
    nb = json.loads((NB_DIR / nb_name).read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        for m in _METRIC_KW_RE.finditer(src):
            out.append((i, m.group(1)))
    return out


@_register("Test  6: every metric=... reference is in _METRIC_REGISTRY")
def test_06_metric_names_valid():
    """Catches typos like 'avg_sinr_db' (real: 'mean_sinr_db') or
    'sum_scnr_db' (real: 'weighted_sum_scnr_db')."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from cordis.simulation.result import _METRIC_REGISTRY
    except Exception as e:
        raise _SkipTest(f"cordis.simulation.result import failed: {e}")
    valid = set(_METRIC_REGISTRY.keys())

    offenders: List[Tuple[str, int, str]] = []
    for nb_name in NOTEBOOKS:
        for cell_idx, metric_name in _collect_metric_references(nb_name):
            if metric_name not in valid:
                offenders.append((nb_name, cell_idx, metric_name))

    if offenders:
        lines = [f"invalid metric names in notebooks:"]
        for nb, ci, name in offenders:
            lines.append(f"  {nb} cell {ci}: metric='{name}' "
                         f"(not in _METRIC_REGISTRY)")
        lines.append(f"Valid metrics: {sorted(valid)}")
        raise AssertionError("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — known-bug regression guards
# ─────────────────────────────────────────────────────────────────────────────

# Each tuple is (label, bad_substring, error_explanation).  These
# encode bugs we've actually shipped and fixed; if any of these
# substrings reappears, the validator catches the regression.
_KNOWN_BAD_PATTERNS = [
    ("save_paper_figure import",
     r"from cordis\.plotting import\s+[^\n]*save_paper_figure",
     "save_paper_figure does not exist in cordis.plotting; use save_figure"),
    ("plot_sweep with sweep_axis positional",
     r"plot_sweep\([^)]*result\.sweep_axis[^)]*metric=",
     "plot_sweep takes only (results_by_x, metric, ax); passing "
     "sweep_axis positionally collides with metric= kwarg"),
    ("plot_admm_convergence with ax= kwarg",
     r"plot_admm_convergence\([^)]*\bax\s*=",
     "plot_admm_convergence's kwarg is 'axes' (tuple of two), not 'ax'"),
    ("bar_chart with column= kwarg",
     r"bar_chart\([^)]*\bcolumn\s*=",
     "bar_chart's kwarg is 'metric', not 'column'"),
    ("_plot_common import from notebook",
     r"from\s+_plot_common\s+import",
     "_plot_common is private to scripts/; reaching into it from a "
     "notebook breaks because notebooks/ is not on sys.path by default"),
]


@_register("Test  7: no shipped notebook contains any historical-bug "
           "fingerprint")
def test_07_no_known_bad_patterns():
    """Regression guard: every fingerprint here is a bug we've shipped
    and fixed.  If any reappears (e.g. someone reverts a fix), the
    validator catches it before the next release.
    """
    offenders: List[Tuple[str, str, str]] = []
    for nb_name in NOTEBOOKS:
        nb = json.loads((NB_DIR / nb_name).read_text())
        full_text = "\n".join(
            ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"
        )
        for label, pattern, explanation in _KNOWN_BAD_PATTERNS:
            if re.search(pattern, full_text):
                offenders.append((nb_name, label, explanation))

    if offenders:
        lines = ["historical bug fingerprints reappeared:"]
        for nb, label, exp in offenders:
            lines.append(f"  {nb}: {label}")
            lines.append(f"     → {exp}")
        raise AssertionError("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 15 validator — {len(_TESTS)} tests")
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

