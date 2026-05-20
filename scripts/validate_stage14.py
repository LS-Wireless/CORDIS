#!/usr/bin/env python3
"""
scripts/validate_stage14.py
============================

Stage 14 validator.

Stage 14 fixes two paper cuts in the plotting pipeline:

1. **Legend defaults** — ``plot_cdf`` defaulted to ``legend_loc="lower right"``,
   which crowds the curves in high-SNR CDF plots where most realizations
   cluster near the right edge.  Changed to ``"best"`` (matches ``plot_sweep``).
   ``plot_admm_convergence`` previously hardcoded ``"upper right"`` with no
   override; now accepts a ``legend_loc`` kwarg defaulting to ``"best"``.

2. **Wrong save function name in playground notebooks** — Stage 12's
   ``GENERIC_SAVE_CELL`` invoked ``cordis.plotting.save_paper_figure``,
   which does not exist (the actual function is ``save_figure``).
   ``save_paper_figure`` lives in ``scripts/_plot_common.py`` as an
   argparse-aware wrapper; notebooks don't have argparse args so they
   should call ``save_figure`` directly.

Tests verify:

* **plot_cdf default** — signature default for ``legend_loc`` is ``"best"``
* **plot_admm_convergence accepts legend_loc** — kwarg present + threaded
  through to both ``ax_res.legend`` and ``ax_slack.legend`` calls
* **plot_sweep unchanged** — already defaulted to ``"best"``; verify
  no regression
* **playground notebooks use save_figure** — none of the four notebooks
  reference the non-existent ``save_paper_figure``
* **_build_playgrounds.py source-of-truth** — GENERIC_SAVE_CELL contains
  the correct symbol so future regenerations stay clean

Run from the repo root::

    python3 scripts/validate_stage14.py
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


def _nb_code_text(name: str) -> str:
    """Return concatenated source of all code cells in a notebook."""
    nb = json.loads((REPO_ROOT / "notebooks" / name).read_text())
    parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        parts.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1 — plot_cdf legend_loc default
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  1: plot_cdf signature defaults legend_loc to 'best'")
def test_01_plot_cdf_default():
    src = (REPO_ROOT / "cordis" / "plotting" / "cdf.py").read_text()
    # Find the legend_loc default in the signature.
    m = re.search(r'legend_loc\s*:\s*str\s*=\s*"([^"]+)"', src)
    assert m is not None, (
        "could not find `legend_loc: str = \"...\"` in plot_cdf signature"
    )
    assert m.group(1) == "best", (
        f"plot_cdf default legend_loc is {m.group(1)!r}; should be 'best' "
        f"to avoid legend-vs-curve overlap on high-SNR CDFs.  "
        f"(\"lower right\" was the historical default, chosen because CDFs "
        f"always pass through (xmin, 0) — but curves cluster near the right "
        f"edge in high-SNR experiments and crowd that corner.)"
    )


@_register("Test  2: plot_cdf docstring documents the new default")
def test_02_plot_cdf_doc():
    src = (REPO_ROOT / "cordis" / "plotting" / "cdf.py").read_text()
    # Look for a mention of 'best' near 'legend_loc' in the docstring.
    # Approximate: find the legend_loc docstring entry.
    m = re.search(r'legend_loc\s*:\s*str\s*\n\s+([^\n]+(?:\n\s+[^\n]+)*)',
                  src)
    assert m is not None, "missing legend_loc entry in plot_cdf docstring"
    doc = m.group(1)
    assert '"best"' in doc or "'best'" in doc, (
        f"plot_cdf docstring should explain that 'best' is the new default; "
        f"got: {doc[:150]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2 — plot_admm_convergence accepts legend_loc
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  3: plot_admm_convergence has legend_loc kwarg with "
           "default 'best'")
def test_03_admm_signature():
    src = (REPO_ROOT / "cordis" / "plotting" / "convergence.py").read_text()
    # Match across multiple lines: `legend_loc: str = "best",`
    m = re.search(r'legend_loc\s*:\s*str\s*=\s*"([^"]+)"', src)
    assert m is not None, (
        "plot_admm_convergence is missing a `legend_loc` parameter.  Add "
        "`legend_loc: str = \"best\"` to the keyword-only section of the "
        "signature so users can override the legend location."
    )
    assert m.group(1) == "best", (
        f"legend_loc default in plot_admm_convergence is {m.group(1)!r}; "
        f"should be 'best' (the old hardcoded 'upper right' worked for some "
        f"traces but crowded the convergence curve in others)."
    )


@_register("Test  4: plot_admm_convergence threads legend_loc into "
           "BOTH .legend() calls")
def test_04_admm_threading():
    """The function plots two panels (residuals + slack), each with its
    own .legend() call.  Both must honor the kwarg, otherwise the
    parameter only half-works."""
    src = (REPO_ROOT / "cordis" / "plotting" / "convergence.py").read_text()
    legend_calls = re.findall(r"\.legend\(loc=([^)]+)\)", src)
    assert len(legend_calls) >= 2, (
        f"expected ≥2 .legend(loc=...) calls in plot_admm_convergence "
        f"(residuals + slack panels); found {len(legend_calls)}"
    )
    # Each one should be `loc=legend_loc`, not a hardcoded string.
    bad = [c for c in legend_calls if '"' in c or "'" in c]
    assert not bad, (
        f"plot_admm_convergence has hardcoded legend locations: {bad}.  "
        f"Both .legend() calls must use loc=legend_loc."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 3 — plot_sweep unchanged (regression guard)
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  5: plot_sweep still defaults legend_loc to 'best' "
           "(no regression)")
def test_05_plot_sweep_default():
    src = (REPO_ROOT / "cordis" / "plotting" / "sweep.py").read_text()
    m = re.search(r'legend_loc\s*:\s*str\s*=\s*"([^"]+)"', src)
    assert m is not None, "plot_sweep is missing its legend_loc kwarg"
    assert m.group(1) == "best", (
        f"plot_sweep legend_loc default regressed to {m.group(1)!r}; "
        f"should still be 'best'."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 4 — playground notebooks: no save_paper_figure
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  6: no playground notebook imports save_paper_figure "
           "from cordis.plotting")
def test_06_no_bogus_save():
    """save_paper_figure does not exist in cordis.plotting.  The
    function lives in scripts/_plot_common.py as an argparse wrapper.
    Notebooks should call cordis.plotting.save_figure directly."""
    offenders = []
    for name in NOTEBOOKS:
        text = _nb_code_text(name)
        if "save_paper_figure" in text:
            offenders.append(name)
    assert not offenders, (
        f"notebooks reference non-existent save_paper_figure: {offenders}.  "
        f"Regenerate with `python3 notebooks/_build_playgrounds.py` after "
        f"the _build_playgrounds.py fix."
    )


@_register("Test  7: every playground notebook uses cordis.plotting.save_figure")
def test_07_uses_save_figure():
    missing = []
    for name in NOTEBOOKS:
        text = _nb_code_text(name)
        # Look for `from cordis.plotting import save_figure` or `save_figure(`.
        if "save_figure" not in text:
            missing.append(name)
    assert not missing, (
        f"notebooks missing save_figure usage: {missing}"
    )


@_register("Test  8: cordis.plotting actually exports save_figure")
def test_08_save_figure_exists():
    """Sanity check the symbol we're now telling notebooks to import."""
    init = (REPO_ROOT / "cordis" / "plotting" / "__init__.py").read_text()
    assert "save_figure" in init, (
        "cordis/plotting/__init__.py should export save_figure"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 5 — _build_playgrounds source-of-truth is fixed
# ─────────────────────────────────────────────────────────────────────────────

@_register("Test  9: _build_playgrounds.py GENERIC_SAVE_CELL uses save_figure")
def test_09_build_script_fixed():
    """Even if the notebooks are correct today, the generator must be
    correct or `make regenerate-scripts` will silently put the bug
    back the next time someone runs it."""
    src = (REPO_ROOT / "notebooks" / "_build_playgrounds.py").read_text()
    assert "save_paper_figure" not in src, (
        "_build_playgrounds.py still references save_paper_figure.  Fix "
        "GENERIC_SAVE_CELL to import + call save_figure instead."
    )
    assert "save_figure" in src, (
        "_build_playgrounds.py should reference save_figure in its "
        "GENERIC_SAVE_CELL"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 78)
    print(f"  Stage 14 validator — {len(_TESTS)} tests")
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

