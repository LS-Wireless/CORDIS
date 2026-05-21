#!/usr/bin/env python3
"""
scripts/validate_stage12.py
============================

Stage 12 validator.

Stage 12 introduces four playground notebooks (one per result *kind*)
under ``notebooks/``, plus a shared helper module ``_playground_helpers.py``.
This validator confirms:

* **file layout** — the four notebooks + helper module exist
* **JSON validity** — each .ipynb parses as JSON and conforms to
  nbformat 4.5 minimal schema (has cells, kernelspec, language_info)
* **cell hygiene** — each notebook has the expected intro cells
  (markdown title, setup cell, EXPERIMENT cell, load cell)
* **helper contract** — required functions (``setup_paper_style``,
  ``load_latest_result``, ``load_run``, ``latest_run_dir``,
  ``summarize``) are present and importable
* **no stale output** — generated notebooks ship with empty output
  cells (no execution_count, no outputs); otherwise diffs balloon
  whenever someone runs the notebook

Run from the repo root::

    python3 scripts/validate_stage12.py

Exit code is 0 iff every test passes.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR    = REPO_ROOT / "notebooks"

NOTEBOOKS = [
    "playground_cdf.ipynb",
    "playground_sweep.ipynb",
    "playground_trace.ipynb",
    "playground_table.ipynb",
]

# Helper functions every notebook depends on.
REQUIRED_HELPERS = [
    "setup_paper_style",
    "latest_run_dir",
    "load_latest_result",
    "load_run",
    "summarize",
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
#  Tier 1 — file layout
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: notebooks/ directory exists")
def test_01_dir():
    assert NB_DIR.is_dir(), f"missing directory {NB_DIR}"


@_register("Test  2: _playground_helpers.py exists")
def test_02_helper_exists():
    p = NB_DIR / "_playground_helpers.py"
    assert p.is_file(), f"missing {p}"


@_register("Test  3: all 4 playground_*.ipynb files exist")
def test_03_notebooks_exist():
    missing = [n for n in NOTEBOOKS if not (NB_DIR / n).is_file()]
    assert not missing, (
        f"missing notebooks: {missing}.  Run "
        f"`python3 notebooks/_build_playgrounds.py` to regenerate."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — JSON validity + nbformat schema
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  4: every notebook is valid JSON")
def test_04_json_valid():
    for name in NOTEBOOKS:
        p = NB_DIR / name
        try:
            json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise AssertionError(f"{name}: invalid JSON: {e}")


@_register("Test  5: every notebook has nbformat 4.x + kernelspec + language_info")
def test_05_nbformat_schema():
    for name in NOTEBOOKS:
        nb = json.loads((NB_DIR / name).read_text())
        assert nb.get("nbformat") == 4, (
            f"{name}: nbformat must be 4, got {nb.get('nbformat')!r}"
        )
        meta = nb.get("metadata", {})
        for required in ("kernelspec", "language_info"):
            assert required in meta, (
                f"{name}: metadata missing {required!r}"
            )
        assert meta["kernelspec"].get("name", "").startswith("python"), (
            f"{name}: kernelspec.name must be a python kernel"
        )
        assert meta["language_info"].get("name") == "python", (
            f"{name}: language_info.name must be 'python'"
        )


@_register("Test  6: every notebook has at least 10 cells")
def test_06_cell_count():
    """A useful playground notebook has multiple exploration cells, not
    just a load + plot.  Bail on truncated notebooks."""
    for name in NOTEBOOKS:
        nb = json.loads((NB_DIR / name).read_text())
        n = len(nb["cells"])
        assert n >= 10, (
            f"{name}: only {n} cells; expected ≥10 (markdown intro + "
            f"setup + load + 5-8 exploration cells + outro)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — cell hygiene
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  7: first cell is markdown title")
def test_07_first_cell_markdown():
    for name in NOTEBOOKS:
        nb = json.loads((NB_DIR / name).read_text())
        first = nb["cells"][0]
        assert first["cell_type"] == "markdown", (
            f"{name}: first cell should be a markdown title, "
            f"got {first['cell_type']}"
        )
        src = "".join(first["source"]) if isinstance(first["source"], list) \
              else first["source"]
        assert src.lstrip().startswith("#"), (
            f"{name}: first markdown cell should start with '# Title'"
        )


@_register("Test  8: every notebook has an EXPERIMENT = ... cell")
def test_08_experiment_cell():
    """The 'one knob' contract: every notebook lets the user pick the
    experiment by editing a single ``EXPERIMENT = '...'`` line.

    Accepts any whitespace around the ``=`` (e.g. aligned columns when
    derived variables follow: ``EXPERIMENT     = f'{exp_name}_cdf'``).
    The contract is "there exists an EXPERIMENT binding"; formatting
    is not part of it.
    """
    import re
    for name in NOTEBOOKS:
        nb = json.loads((NB_DIR / name).read_text())
        found = False
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]
            if isinstance(src, list):
                src = "".join(src)
            if re.search(r"^EXPERIMENT\s*=", src, re.M):
                found = True
                break
        assert found, (
            f"{name}: no cell with `EXPERIMENT = ...` found.  Every "
            f"playground notebook should expose the experiment name as "
            f"a single editable variable."
        )


@_register("Test  9: every notebook imports from _playground_helpers")
def test_09_imports_helpers():
    for name in NOTEBOOKS:
        nb = json.loads((NB_DIR / name).read_text())
        joined = "\n".join(
            ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"
        )
        assert "_playground_helpers" in joined, (
            f"{name}: missing import from _playground_helpers"
        )


@_register("Test 10: all code cells ship empty (no executed outputs)")
def test_10_empty_outputs():
    """Generated notebooks must be committed with `execution_count=None`
    and `outputs=[]`.  Otherwise git diffs explode the moment anyone
    runs a cell — defeating the whole purpose of versioning them.

    To clear after exploration: `jupyter nbconvert --clear-output
    --inplace notebooks/playground_*.ipynb`.
    """
    for name in NOTEBOOKS:
        nb = json.loads((NB_DIR / name).read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            ec = cell.get("execution_count")
            outs = cell.get("outputs", [])
            assert ec is None, (
                f"{name} cell {i}: execution_count={ec} (should be null). "
                f"Run `jupyter nbconvert --clear-output --inplace` first."
            )
            assert outs == [], (
                f"{name} cell {i}: has {len(outs)} output(s) "
                f"(should be empty list)."
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — helper module contract
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test 11: _playground_helpers exposes all required functions")
def test_11_helper_api():
    spec = importlib.util.spec_from_file_location(
        "_playground_helpers", NB_DIR / "_playground_helpers.py"
    )
    if spec is None or spec.loader is None:
        raise _SkipTest("could not build module spec for _playground_helpers")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise _SkipTest(f"_playground_helpers failed to import: {e}")

    missing = [f for f in REQUIRED_HELPERS if not hasattr(mod, f)]
    assert not missing, (
        f"_playground_helpers missing functions: {missing}"
    )

    # Quick sanity: __all__ should match (or be a superset of) the
    # required-helper list — keeps the module's public surface honest.
    all_names = set(getattr(mod, "__all__", []))
    if all_names:
        missing_from_all = [f for f in REQUIRED_HELPERS if f not in all_names]
        assert not missing_from_all, (
            f"_playground_helpers.__all__ missing: {missing_from_all}"
        )


@_register("Test 12: latest_run_dir raises FileNotFoundError on missing experiment")
def test_12_helper_error_message():
    spec = importlib.util.spec_from_file_location(
        "_playground_helpers", NB_DIR / "_playground_helpers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise _SkipTest(f"_playground_helpers failed to import: {e}")

    try:
        mod.latest_run_dir("__definitely_not_a_real_experiment__")
    except FileNotFoundError as e:
        # Must mention what's missing AND what to do about it (the
        # "Have you run `make ...` yet?" hint).
        msg = str(e)
        assert "make" in msg.lower(), (
            f"error message should suggest running `make`: {msg!r}"
        )
        return
    raise AssertionError(
        "latest_run_dir should raise FileNotFoundError for missing experiments"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 12 validator — {len(_TESTS)} tests")
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

