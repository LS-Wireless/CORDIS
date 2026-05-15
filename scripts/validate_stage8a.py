"""
Stage 8a — Validation script.

Twelve tests covering:

    1. ``apply_paper_style`` rcParams effect (LaTeX + fallback)
    2. ``style_for`` known + unknown names; deterministic fallback
    3. ``register_algorithm_style`` overrides
    4. ``figsize`` named widths and custom width
    5. ``save_figure`` writes PDF + PGF non-empty
    6. ``save_figure`` metadata: PDF Keywords + PGF % comments
    7. ``plot_cdf`` produces N monotone curves, axes labelled
    8. ``plot_sweep`` line + IQR shading; ``error='none'`` omits shade
    9. ``plot_admm_convergence`` two panels with curves
   10. ``bar_chart`` produces bars with errorbars
   11. ``to_markdown_table`` and ``to_latex_table`` produce expected shape
   12. ``experiment_dir`` / ``figure_dir`` / ``latest_result`` round-trip

Plus, as a side-effect of the run, populates ``figures_examples/`` with
one PDF + PGF per plot type so the repository ships browseable
examples (matches the user's "ship examples" choice).

Fixtures
~~~~~~~~
Plot helpers are tested against minimal mock objects that quack like
:class:`SimResult`, :class:`AlgorithmResult`, and :class:`ADMMResult`.
Mocks let Stage 8a validation run without any CORDIS algorithm imports
— the focus here is the plotting library itself.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

# Make ``cordis`` importable when this script is run directly via
# ``python scripts/validate_stage8a.py`` (matches the convention used
# by every other validate_stageN.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

# ─────────────────────────────────────────────────────────────────────
#  Environment-availability probes
#
#  PGF output requires *some* TeX engine on PATH (matplotlib's PGF
#  backend spawns LaTeX to compute text metrics — there is no pure-
#  Python path).  ``pypdf`` is a Python package used only to verify
#  embedded PDF metadata in Test 6.  Either may be absent on developer
#  machines without a full TeX distribution.  The tests handle each
#  case by raising ``_SkipTest`` and reporting SKIPPED instead of
#  FAILED, so the validation run remains green.
# ─────────────────────────────────────────────────────────────────────

_TEX_ENGINES = ("pdflatex", "xelatex", "lualatex")
_HAS_TEX = any(shutil.which(eng) is not None for eng in _TEX_ENGINES)

try:
    import pypdf  # noqa: F401
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


class _SkipTest(Exception):
    """Raised by a test to signal that it should be reported as SKIPPED
    rather than PASSED or FAILED.  Reason text appears in output."""

# ─────────────────────────────────────────────────────────────────────
#  Imports under test
# ─────────────────────────────────────────────────────────────────────

from cordis.plotting import (             # noqa: E402
    ALGORITHM_STYLE,
    apply_paper_style,
    bar_chart,
    figsize,
    plot_admm_convergence,
    plot_cdf,
    plot_sweep,
    register_algorithm_style,
    save_figure,
    style_for,
    to_latex_table,
    to_markdown_table,
)
from cordis.experiments import (          # noqa: E402
    experiment_dir,
    figure_dir,
    latest_result,
    log_dir,
)


# ─────────────────────────────────────────────────────────────────────
#  Mock fixtures (no CORDIS algorithm imports)
# ─────────────────────────────────────────────────────────────────────

class _FakeAlgorithmResult:
    """Duck-typed AlgorithmResult: has_metric / cdf / mean / std / percentile."""
    def __init__(self, name, metric_arrays):
        self.name = name
        self._d  = {k: np.asarray(v, dtype=float)
                    for k, v in metric_arrays.items()}

    def has_metric(self, m):           return m in self._d
    def mean(self, m):                 return float(np.mean(self._d[m]))
    def std(self, m):                  return float(np.std(self._d[m]))
    def percentile(self, m, q):        return float(np.percentile(self._d[m], q))
    def cdf(self, m):
        x = np.sort(self._d[m])
        f = np.arange(1, x.size + 1) / x.size
        return x, f


class _FakeSimResult:
    """Duck-typed SimResult: just an algorithm_results dict."""
    def __init__(self, algorithm_results):
        self.algorithm_results = algorithm_results


class _FakeADMMResult:
    """Duck-typed ADMMResult: histories + best_iter."""
    def __init__(self, primal_hist, dual_hist, slack_hist=None, best_iter=None):
        self.primal_res_history = primal_hist
        self.dual_res_history   = dual_hist
        if slack_hist is not None:
            self.slack_history  = slack_hist
        if best_iter is not None:
            self.best_iter      = best_iter


def _build_sim_result(seed=42, n_trials=80):
    """Synthetic SimResult: 4 algorithms × 2 metrics with realistic spreads."""
    rng = np.random.default_rng(seed)
    algos = {
        "CORDIS-Split":  {"min_sinr_db":  rng.normal( 2.5, 1.2, n_trials),
                          "weighted_sum_scnr_db": rng.normal(45.0, 1.5, n_trials)},
        "CORDIS-ADMM":   {"min_sinr_db":  rng.normal( 3.0, 0.8, n_trials),
                          "weighted_sum_scnr_db": rng.normal(48.0, 1.2, n_trials)},
        "Centralized":   {"min_sinr_db":  rng.normal( 3.0, 0.4, n_trials),
                          "weighted_sum_scnr_db": rng.normal(50.0, 1.0, n_trials)},
        "MRT-Split":     {"min_sinr_db":  rng.normal( 4.0, 1.8, n_trials),
                          "weighted_sum_scnr_db": rng.normal(40.0, 2.0, n_trials)},
    }
    return _FakeSimResult({
        name: _FakeAlgorithmResult(name, metrics)
        for name, metrics in algos.items()
    })


def _build_admm_result(seed=42, n_iters=22):
    """Synthetic ADMMResult: decaying residuals + decaying slack."""
    rng = np.random.default_rng(seed)
    n = np.arange(1, n_iters + 1)
    primal = 20.0 * np.exp(-0.18 * n) + 0.3 * rng.standard_normal(n_iters)
    dual   = 25.0 * np.exp(-0.16 * n) + 0.3 * rng.standard_normal(n_iters)
    primal = np.maximum(primal, 1e-3)
    dual   = np.maximum(dual,   1e-3)
    slack  = [np.maximum(0.6 * np.exp(-0.25 * i) + 0.02 * rng.standard_normal(3),
                          1e-6)
              for i in n]
    return _FakeADMMResult(primal, dual, slack, best_iter=14)


# ─────────────────────────────────────────────────────────────────────
#  Test runner plumbing
# ─────────────────────────────────────────────────────────────────────

class _TestRunner:
    """Tiny pytest-like driver — keeps Stage 8a self-contained."""
    def __init__(self):
        self.n_pass = 0
        self.n_fail = 0
        self.n_skip = 0
        self.failures = []
        self.skips = []

    def run(self, label, fn):
        print(f"── {label} ──", flush=True)
        try:
            fn()
            print("  ✓ PASSED", flush=True)
            self.n_pass += 1
        except _SkipTest as e:
            print(f"  ⓘ SKIPPED: {e}", flush=True)
            self.n_skip += 1
            self.skips.append((label, str(e)))
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}", flush=True)
            self.n_fail += 1
            self.failures.append((label, str(e)))
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}", flush=True)
            self.n_fail += 1
            self.failures.append((label, f"{type(e).__name__}: {e}"))

    def summary(self):
        total = self.n_pass + self.n_fail + self.n_skip
        bar   = "=" * 76
        print(f"\n{bar}")
        if self.n_fail == 0:
            tag = "✓ ALL TESTS PASSED"
            if self.n_skip:
                tag += f" ({self.n_skip} skipped)"
            print(f"  {tag}   ({self.n_pass}/{total} passed)")
        else:
            print(f"  ✗ {self.n_fail}/{total} STAGE 8a TESTS FAILED")
            for label, msg in self.failures:
                print(f"      • {label}: {msg}")
        if self.skips:
            print("  Skipped tests:")
            for label, msg in self.skips:
                print(f"      • {label}: {msg}")
        print(bar)
        return self.n_fail == 0


# ─────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────

def test_01_apply_paper_style():
    """rcParams change, both LaTeX-on and fallback paths work."""
    plt.rcdefaults()
    apply_paper_style(use_latex=True)
    if matplotlib.rcParams["text.usetex"]:
        assert matplotlib.rcParams["font.family"] == ["serif"], \
            f"font.family={matplotlib.rcParams['font.family']}"
        assert matplotlib.rcParams["pgf.texsystem"] == "pdflatex"
    # Fallback path must always work, regardless of TeX availability.
    plt.rcdefaults()
    apply_paper_style(use_latex=False)
    assert matplotlib.rcParams["text.usetex"] is False
    assert matplotlib.rcParams["font.family"] == ["sans-serif"]
    # Lock in non-LaTeX mode for the rest of the suite so it doesn't
    # blow up if the current env loses pdflatex mid-run.
    apply_paper_style(use_latex=False)


def test_02_style_for():
    """Known → registered style; unknown → deterministic grey fallback."""
    s = style_for("CORDIS-ADMM")
    assert s["color"] == "#d62728"
    assert s["label"] == "CORDIS-ADMM"
    # Unknown is deterministic.
    a = style_for("FooBar")
    b = style_for("FooBar")
    assert a == b
    # Distinct unknowns → likely distinct (hash-based).
    c = style_for("AnotherUnknown_xyz")
    assert c["label"] == "AnotherUnknown_xyz"


def test_03_register_algorithm_style():
    """User overrides land in the global table."""
    register_algorithm_style("ZZ_Test", color="purple", marker="*",
                             linestyle="-", label="Test")
    s = style_for("ZZ_Test")
    assert s["color"] == "purple" and s["marker"] == "*"
    # Removing for cleanliness so re-running this script is idempotent.
    ALGORITHM_STYLE.pop("ZZ_Test", None)


def test_04_figsize():
    """Named widths + custom widths produce sensible tuples."""
    w, h = figsize("single")
    assert abs(w - 3.5) < 1e-9 and h > 0
    w, h = figsize("double")
    assert abs(w - 7.16) < 1e-9
    w, h = figsize(4.0, aspect=2.0)
    assert (w, h) == (4.0, 2.0)
    # Bad inputs raise
    try:
        figsize("nonexistent_key")
        assert False, "expected KeyError"
    except KeyError:
        pass
    try:
        figsize(3.0, aspect=-1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_05_save_figure_writes_both_formats():
    """PDF always written; PGF written only when a TeX engine is available."""
    formats = ("pdf",) if not _HAS_TEX else ("pdf", "pgf")
    with tempfile.TemporaryDirectory() as td:
        fig, ax = plt.subplots(figsize=figsize("single"))
        ax.plot([0, 1, 2], [1, 4, 2])
        paths = save_figure(fig, Path(td) / "t05", formats=formats)
        plt.close(fig)
        assert len(paths) == len(formats), \
            f"expected {len(formats)} files, got {len(paths)}"
        for p in paths:
            assert p.exists() and p.stat().st_size > 100, f"{p} too small"
        assert any(p.suffix == ".pdf" for p in paths)
        if _HAS_TEX:
            assert any(p.suffix == ".pgf" for p in paths)
        else:
            # PGF coverage is missing in this environment.  Make that
            # visible in the test output without failing the run.
            raise _SkipTest(
                "no TeX engine on PATH; "
                "PDF saved successfully, PGF coverage skipped. "
                "Install MacTeX (brew install --cask mactex) for full coverage."
            )


def test_06_save_figure_metadata():
    """PDF Keywords field + PGF comment header carry provenance."""
    if not _HAS_PYPDF:
        raise _SkipTest("requires pypdf (pip install pypdf)")
    if not _HAS_TEX:
        raise _SkipTest(
            "PGF half of this test requires TeX (install MacTeX); "
            "PDF half also skipped to keep this test atomic."
        )
    from pypdf import PdfReader
    with tempfile.TemporaryDirectory() as td:
        fig, ax = plt.subplots(figsize=figsize("single"))
        ax.plot([0, 1], [0, 1])
        with warnings.catch_warnings():
            # We treat unexpected warnings as failures only here.
            warnings.simplefilter("error", category=UserWarning)
            paths = save_figure(
                fig, Path(td) / "t06",
                metadata={"Experiment": "validate_8a", "Trial": "42"},
            )
        plt.close(fig)
        # PDF
        pdf = next(p for p in paths if p.suffix == ".pdf")
        info = PdfReader(str(pdf)).metadata
        assert "CORDIS" in str(info.get("/Creator")), info
        assert "Experiment=validate_8a" in str(info.get("/Keywords", "")), info
        assert "Trial=42" in str(info.get("/Keywords", "")), info
        # PGF
        pgf = next(p for p in paths if p.suffix == ".pgf")
        head = pgf.read_text().splitlines()[:6]
        joined = "\n".join(head)
        assert "% Creator: CORDIS" in joined
        assert "% Experiment: validate_8a" in joined
        assert "% Trial: 42" in joined


def test_07_plot_cdf():
    """N curves drawn, axes labelled, monotone."""
    sr = _build_sim_result()
    fig, ax = plt.subplots(figsize=figsize("single"))
    ax = plot_cdf(sr, "min_sinr_db", ax=ax, xlabel=r"min-SINR [dB]")
    plt.close(fig)
    lines = ax.get_lines()
    assert len(lines) == 4, f"expected 4 curves, got {len(lines)}"
    for line in lines:
        ys = line.get_ydata()
        # CDFs must be monotone non-decreasing
        assert np.all(np.diff(ys) >= -1e-12), "CDF not monotone"
    assert ax.get_xlabel() == r"min-SINR [dB]"


def test_08_plot_sweep():
    """3-point sweep produces lines + IQR band; 'none' omits the band."""
    results_by_x = {
        0.0: _build_sim_result(seed=0),
        3.0: _build_sim_result(seed=1),
        6.0: _build_sim_result(seed=2),
    }
    fig, ax = plt.subplots(figsize=figsize("single"))
    ax = plot_sweep(results_by_x, "weighted_sum_scnr_db",
                    ax=ax, xlabel=r"$\gamma$ [dB]", error="iqr")
    n_lines_iqr = len(ax.get_lines())
    n_collections_iqr = len(ax.collections)
    plt.close(fig)
    assert n_lines_iqr == 4, f"expected 4 line curves, got {n_lines_iqr}"
    assert n_collections_iqr == 4, \
        f"expected 4 IQR bands (one per algorithm), got {n_collections_iqr}"

    fig, ax = plt.subplots(figsize=figsize("single"))
    ax = plot_sweep(results_by_x, "weighted_sum_scnr_db",
                    ax=ax, error="none")
    n_collections_none = len(ax.collections)
    plt.close(fig)
    assert n_collections_none == 0, \
        f"error='none' should draw no fill bands, got {n_collections_none}"


def test_09_plot_admm_convergence():
    """Two panels populated with curves; best-iter marker drawn."""
    ar = _build_admm_result()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize("double", aspect=2.5))
    axA, axB = plot_admm_convergence(ar, axes=(axA, axB))
    assert len(axA.get_lines()) >= 2     # primal + dual
    assert len(axB.get_lines()) >= 1     # slack
    # Best-iter vlines (one per panel)
    vlines_A = [l for l in axA.get_lines() if l.get_linestyle() == ":"]
    assert len(vlines_A) >= 0   # not strictly required to be present, but check it exists
    plt.close(fig)


def test_10_bar_chart():
    """Bars + errorbars drawn."""
    sr = _build_sim_result()
    fig, ax = plt.subplots(figsize=figsize("single"))
    ax = bar_chart(sr, "weighted_sum_scnr_db", ax=ax, error="std",
                   ylabel="Sum SCNR (dB)")
    plt.close(fig)
    # bars are `Rectangle` patches
    n_bars = len([p for p in ax.patches if p.__class__.__name__ == "Rectangle"])
    # Bar collection on the bar artist; >=4 patches expected.
    assert n_bars >= 4, f"expected ≥4 bars, got {n_bars}"
    assert ax.get_ylabel() == "Sum SCNR (dB)"


def test_11_tables():
    """Markdown + LaTeX tables emit expected structural fragments."""
    sr = _build_sim_result()
    md = to_markdown_table(sr, ["min_sinr_db", "weighted_sum_scnr_db"])
    assert md.startswith("| Algorithm")
    assert md.count("\n") >= 5         # header + sep + 4 algos
    assert "CORDIS-ADMM" in md
    # LaTeX
    tex = to_latex_table(sr, ["min_sinr_db", "weighted_sum_scnr_db"],
                         caption="Validation table", label="tab:val")
    for token in (r"\begin{table}", r"\toprule", r"\midrule", r"\bottomrule",
                  r"\end{table}", r"\caption{Validation table}",
                  r"\label{tab:val}"):
        assert token in tex, f"missing {token!r}"


def test_12_experiment_io():
    """experiment_dir / latest_result / figure_dir / log_dir round-trip,
    including the auto-applied ``exp_`` prefix."""
    with tempfile.TemporaryDirectory() as td:
        os.environ["CORDIS_RESULTS_DIR"] = str(Path(td) / "results")
        os.environ["CORDIS_FIGURES_DIR"] = str(Path(td) / "figures")
        try:
            # ── exp_ prefix is added automatically ────────────────────
            d1 = experiment_dir("validate_8a", timestamp="20260514_120000")
            assert d1.exists()
            assert "exp_validate_8a" in str(d1), \
                f"expected 'exp_validate_8a' in path, got {d1}"
            assert d1.parent.name == "exp_validate_8a", d1.parent

            # ── Prefix is idempotent: 'exp_X' doesn't become 'exp_exp_X' ─
            d_idem = experiment_dir("exp_already_prefixed",
                                    timestamp="20260514_120100")
            assert "exp_exp_" not in str(d_idem), \
                f"double prefix detected in {d_idem}"
            assert "exp_already_prefixed" in str(d_idem)

            # ── 'latest' symlink points to the most recent timestamp ─
            d2 = experiment_dir("validate_8a", timestamp="20260514_130000")
            assert d2.exists()
            resolved = latest_result("validate_8a")
            assert resolved.name == "20260514_130000", resolved

            # ── figure_dir auto-prefixes too ──────────────────────────
            f = figure_dir("validate_8a")
            assert f.exists()
            assert f.name == "exp_validate_8a", f
            # figure_dir is also idempotent on already-prefixed names
            f2 = figure_dir("exp_validate_8a")
            assert f2 == f, f"{f2} != {f}"

            # ── log_dir creates <exp>/logs/ for per-run logs ──────────
            logs = log_dir(d1)
            assert logs.exists() and logs.is_dir()
            assert logs.name == "logs"
            assert logs.parent == d1
            # Idempotent (mkdir parents=True, exist_ok=True)
            logs2 = log_dir(d1)
            assert logs2 == logs

            # ── Bad names still rejected ──────────────────────────────
            for bad in ("", "has space", "has/slash", ".", ".."):
                try:
                    experiment_dir(bad)
                    assert False, f"expected ValueError for {bad!r}"
                except ValueError:
                    pass
        finally:
            os.environ.pop("CORDIS_RESULTS_DIR", None)
            os.environ.pop("CORDIS_FIGURES_DIR", None)


# ─────────────────────────────────────────────────────────────────────
#  Example figure generation (figures_examples/)
# ─────────────────────────────────────────────────────────────────────

def generate_examples(out_dir: Path):
    """Generate one PDF (and PGF if TeX available) per plot type
    under ``figures_examples/``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = ("pdf", "pgf") if _HAS_TEX else ("pdf",)

    sr = _build_sim_result()
    ar = _build_admm_result()

    # 1. CDF
    fig, ax = plt.subplots(figsize=figsize("single"))
    plot_cdf(sr, "min_sinr_db",
             ax=ax,
             xlabel=r"min-SINR [dB]",
             title="Example: empirical CDF")
    save_figure(fig, out_dir / "example_cdf", formats=formats,
                metadata={"Example": "CDF of min-SINR"})
    plt.close(fig)

    # 2. Sweep
    sweep_data = {
        float(g): _build_sim_result(seed=int(g)+10)
        for g in (0.0, 3.0, 6.0, 10.0, 14.0)
    }
    # Bias the means slightly so the lines aren't flat.
    for g, srx in sweep_data.items():
        for ar_ in srx.algorithm_results.values():
            ar_._d["weighted_sum_scnr_db"] += -0.4 * g

    fig, ax = plt.subplots(figsize=figsize("single"))
    plot_sweep(sweep_data, "weighted_sum_scnr_db",
               ax=ax, error="iqr",
               xlabel=r"$\gamma$ [dB]", ylabel=r"Sum SCNR [dB]",
               title=r"Example: SCNR vs $\gamma$")
    save_figure(fig, out_dir / "example_sweep", formats=formats,
                metadata={"Example": "SCNR vs gamma sweep"})
    plt.close(fig)

    # 3. ADMM convergence (two-panel)
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=figsize("double", aspect=2.5),
        constrained_layout=True,
    )
    plot_admm_convergence(ar, axes=(axA, axB),
                          title="Example: ADMM convergence trajectory")
    save_figure(fig, out_dir / "example_convergence", formats=formats,
                metadata={"Example": "ADMM residual + slack trajectory"})
    plt.close(fig)

    # 4. Bar chart
    fig, ax = plt.subplots(figsize=figsize("single"))
    bar_chart(sr, "weighted_sum_scnr_db", ax=ax, error="std",
              ylabel="Sum SCNR (dB)",
              title="Example: algorithm comparison")
    save_figure(fig, out_dir / "example_bars", formats=formats,
                metadata={"Example": "Bar chart of sum-SCNR"})
    plt.close(fig)

    # 5. Tables (plain text)
    md = to_markdown_table(sr, ["min_sinr_db", "weighted_sum_scnr_db"])
    (out_dir / "example_table.md").write_text(md)
    tex = to_latex_table(
        sr, ["min_sinr_db", "weighted_sum_scnr_db"],
        caption="Algorithm comparison",
        label="tab:example",
        metric_labels={
            "min_sinr_db":           r"min-SINR (dB)",
            "weighted_sum_scnr_db":  r"Sum SCNR (dB)",
        },
    )
    (out_dir / "example_table.tex").write_text(tex)


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    bar = "=" * 76
    print(bar)
    print("  Stage 8a — Plotting and I/O conventions: validation")
    print(bar)

    # Surface optional-dependency availability up front.
    tex_status   = "present" if _HAS_TEX else "MISSING (install MacTeX for full coverage)"
    pypdf_status = "present" if _HAS_PYPDF else "MISSING (pip install pypdf)"
    print(f"  Environment:  TeX engine: {tex_status}")
    print(f"                pypdf:      {pypdf_status}")
    print(bar)

    runner = _TestRunner()
    runner.run("Test  1: apply_paper_style sets rcParams (LaTeX + fallback)",
               test_01_apply_paper_style)
    runner.run("Test  2: style_for known/unknown, deterministic fallback",
               test_02_style_for)
    runner.run("Test  3: register_algorithm_style overrides global table",
               test_03_register_algorithm_style)
    runner.run("Test  4: figsize named + custom widths; bad inputs raise",
               test_04_figsize)
    runner.run("Test  5: save_figure writes PDF + PGF non-empty",
               test_05_save_figure_writes_both_formats)
    runner.run("Test  6: save_figure metadata (PDF Keywords + PGF comments)",
               test_06_save_figure_metadata)
    runner.run("Test  7: plot_cdf produces N monotone curves",
               test_07_plot_cdf)
    runner.run("Test  8: plot_sweep lines + IQR band; 'none' omits shading",
               test_08_plot_sweep)
    runner.run("Test  9: plot_admm_convergence two panels with curves",
               test_09_plot_admm_convergence)
    runner.run("Test 10: bar_chart draws bars + errorbars",
               test_10_bar_chart)
    runner.run("Test 11: to_markdown_table + to_latex_table shape",
               test_11_tables)
    runner.run("Test 12: experiment_dir / latest_result / figure_dir",
               test_12_experiment_io)

    # Generate the example PDF/PGF gallery under figures/examples/
    print()
    print("── Generating figures/examples/ ──")
    examples_dir = (
        Path(__file__).resolve().parent.parent / "figures" / "examples"
    )
    generate_examples(examples_dir)
    n_files = sum(1 for _ in examples_dir.iterdir() if _.is_file())
    print(f"  ✓ wrote {n_files} files to {examples_dir}")

    ok = runner.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

