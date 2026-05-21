#!/usr/bin/env python3
"""
Build the four playground notebooks from in-script cell definitions.

This script is a build-time tool used to produce
``notebooks/playground_<kind>.ipynb``.  It is kept under
``notebooks/`` so the notebooks live next to their source-of-truth.

Run::

    python3 notebooks/_build_playgrounds.py

Re-run after editing this script to regenerate all four notebooks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent       # notebooks/


# ─────────────────────────────────────────────────────────────────────
#  Notebook-cell builders
# ─────────────────────────────────────────────────────────────────────

def code(*lines: str) -> dict:
    """Build a code cell from a list of source lines (no trailing \\n needed)."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "\n".join(lines),
    }


def md(*lines: str) -> dict:
    """Build a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
    }


def notebook(cells: List[dict]) -> dict:
    """Wrap cells in a minimal Jupyter nbformat 4.5 document."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {
                "name":      "python",
                "version":   "3.x",
                "mimetype":  "text/x-python",
                "file_extension":   ".py",
            },
        },
        "nbformat":       4,
        "nbformat_minor": 5,
    }


# ─────────────────────────────────────────────────────────────────────
#  Shared cell snippets — used across all four notebooks
# ─────────────────────────────────────────────────────────────────────

INTRO_SETUP = code(
    "# Stage 12: shared setup — make `cordis` importable when this notebook",
    "# is launched from notebooks/, then apply the IEEE paper rcParams.",
    "import sys, logging",
    "from pathlib import Path",
    "from _playground_helpers import (",
    "    setup_paper_style, load_latest_result, load_run, load_result, summarize,",
    ")",
    "",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "",
    "# Set use_latex=False if pdflatex isn't on PATH (e.g. on a compute node).",
    "setup_paper_style(use_latex=True)",
    "",
    "logging.basicConfig(level=logging.WARNING, format='%(levelname)-7s %(message)s')",
)


GENERIC_FIGSIZE_CELL = code(
    "# Figsize variants — `figsize` returns (w, h) in inches for matplotlib.",
    "# width: 'single' (one column), 'double' (two-column), 'third' (3-up panel).",
    "# aspect: w/h ratio.  Tweak both to fit your paper layout.",
    "from cordis.plotting import figsize",
    "",
    "for width in ('single', 'double', 'third'):",
    "    w, h = figsize(width=width, aspect=3/2)",
    "    print(f'{width:>6}: ({w:.2f}, {h:.2f}) inches')",
    "",
    "# Example: tight three-up panel for a paper sub-figure",
    "# fig, axes = plt.subplots(1, 3, figsize=figsize(width='double', aspect=3.5/1.5))",
)


GENERIC_SAVE_CELL = code(
    "# Save with provenance metadata (Git SHA, creation date, etc. — embedded",
    "# into the PDF's metadata, prepended as comments in the .pgf).",
    "from cordis.plotting import save_figure",
    "",
    "# Adjust EXPERIMENT and metric labels to match the figure above.",
    "out = save_figure(",
    "    fig,",
    "    base_path=f'../figures/playground/{EXPERIMENT}_demo',",
    "    formats=('pdf', 'png'),       # add 'pgf' on systems with LaTeX",
    "    metadata={'Experiment': EXPERIMENT, 'Notebook': 'playground'},",
    ")",
    "for p in out:",
    "    print('wrote', p)",
)


# ─────────────────────────────────────────────────────────────────────
#  CDF notebook (sinr_cdf, scnr_cdf)
# ─────────────────────────────────────────────────────────────────────

def build_cdf_notebook() -> dict:
    cells = [
        md(
            "# CDF Playground",
            "",
            "Interactive exploration of empirical-CDF figures (`sinr_cdf`, `scnr_cdf`).",
            "",
            "**Workflow.**  Run a CDF experiment first — e.g. `make sinr_cdf` — to",
            "populate `results/exp_sinr_cdf/<ts>/`.  Set `exp_name` below to the",
            "metric family (`'sinr'` or `'scnr'`) and iterate on the cells.",
            "Switching between SINR and SCNR is one edit; all subsequent cells",
            "derive their experiment folder, metric names, and titles from",
            "`exp_name`.",
            "",
            "**Compatible experiments:** `sinr_cdf`, `scnr_cdf`",
            "(both use `kind == 'single'` and the `plot_cdf` function).",
        ),
        INTRO_SETUP,
        code(
            "# ── The one knob: which metric family to load ──",
            "exp_name = 'sinr'              # 'sinr' or 'scnr'",
            "",
            "# Optional: load a specific run directory instead of the latest.",
            "# Set to a path like 'results/exp_sinr_cdf/20260520_113500'",
            "# (relative to the repo root) to re-render an older comparison.",
            "EXP_DIR = None",
            "",
            "# Derived names — change `exp_name` above and these update automatically.",
            "EXPERIMENT     = f'{exp_name}_cdf'        # 'sinr_cdf' / 'scnr_cdf'",
            "METRIC_MIN     = f'min_{exp_name}_db'     # 'min_sinr_db' / 'min_scnr_db'",
            "METRIC_MEAN    = f'mean_{exp_name}_db'    # 'mean_sinr_db' / 'mean_scnr_db'",
            "METRIC_DISPLAY = exp_name.upper()         # 'SINR' / 'SCNR'  (for titles)",
            "",
            "result = load_result(EXPERIMENT, EXP_DIR)",
            "assert result.kind == 'single', (",
            "    f'This notebook is for CDF (kind=\"single\") experiments; '",
            "    f'got kind={result.kind}.  Use playground_sweep / _trace / _table.'",
            ")",
            "summarize(result)",
        ),
        md("## 1. Quick CDF — all algorithms"),
        code(
            "from cordis.plotting import plot_cdf, figsize",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_cdf(result.sim_result, metric=METRIC_MIN, ax=ax)",
            "ax.set_title(f'{EXPERIMENT}: min-{METRIC_DISPLAY} CDF')",
            "ax.set_xlabel(f'min-{METRIC_DISPLAY} [dB]')",
            "plt.show()",
        ),
        md(
            "## 2. Filter algorithms via `only=`",
            "",
            "`plot_cdf` accepts an `only=` list of algorithm display names — anything",
            "not in the list is silently skipped.  Colors/markers stay consistent with",
            "`ALGORITHM_STYLE` even when you drop algorithms.",
        ),
        code(
            "# Pick just the algorithms you want — easy A/B comparison.",
            "PICK = ['CORDIS-Split', 'CORDIS-ADMM', 'Centralized']",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_cdf(result.sim_result, metric=METRIC_MIN, ax=ax, only=PICK)",
            "ax.set_title(f'{EXPERIMENT}: CORDIS vs. Centralized')",
            "ax.set_xlabel(f'min-{METRIC_DISPLAY} [dB]')",
            "plt.show()",
        ),
        md(
            "## 3. Alternate metrics — min vs. mean",
            "",
            "The SimResult typically carries several metrics per run.  For the",
            "currently-selected family (SINR or SCNR), the canonical pair is",
            "`min_<family>_db` and `mean_<family>_db`.  Comparing the two side-",
            "by-side shows how algorithm rankings shift between the worst-user",
            "view and the average view.",
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=figsize(width='double', aspect=3/2))",
            "plot_cdf(result.sim_result, metric=METRIC_MIN, ax=axes[0], only=PICK)",
            "axes[0].set_title(f'Worst-user {METRIC_DISPLAY}')",
            "axes[0].set_xlabel(f'min-{METRIC_DISPLAY} [dB]')",
            "plot_cdf(result.sim_result, metric=METRIC_MEAN, ax=axes[1], only=PICK)",
            "axes[1].set_title(f'Average-user {METRIC_DISPLAY}')",
            "axes[1].set_xlabel(f'mean-{METRIC_DISPLAY} [dB]')",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## 4. Per-algorithm style override",
            "",
            "`ALGORITHM_STYLE` is a dict; monkey-patch entries for one-off tweaks",
            "without committing changes to `cordis/plotting/style.py`.",
        ),
        code(
            "from cordis.plotting.style import ALGORITHM_STYLE",
            "import copy",
            "",
            "# Snapshot original so cell is re-runnable.",
            "_orig = copy.deepcopy(ALGORITHM_STYLE)",
            "",
            "# Bump CORDIS-Split to a heavier line, switch its color.",
            "if 'CORDIS-Split' in ALGORITHM_STYLE:",
            "    ALGORITHM_STYLE['CORDIS-Split']['linewidth'] = 2.5",
            "    ALGORITHM_STYLE['CORDIS-Split']['color']     = '#d62728'",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_cdf(result.sim_result, metric=METRIC_MIN, ax=ax, only=PICK)",
            "ax.set_title('Style override demo (red, thicker Split)')",
            "plt.show()",
            "",
            "# Restore the original style so subsequent cells aren't affected.",
            "ALGORITHM_STYLE.clear(); ALGORITHM_STYLE.update(_orig)",
        ),
        md(
            "## 5. Tail behavior — log x-axis",
            "",
            "Useful when the interesting story is in the outage region",
            "(e.g. 5th-percentile SINR).",
        ),
        code(
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_cdf(result.sim_result, metric=METRIC_MIN, ax=ax, only=PICK)",
            "# Mark the 5%/50%/95% lines",
            "for q in (0.05, 0.5, 0.95):",
            "    ax.axhline(q, color='lightgray', linestyle=':', linewidth=0.8, zorder=0)",
            "ax.set_ylim(0, 1)",
            "plt.show()",
        ),
        md("## 6. Figsize variants"),
        GENERIC_FIGSIZE_CELL,
        md("## 7. Save with provenance metadata"),
        GENERIC_SAVE_CELL,
        md(
            "## 8. Multi-run overlay",
            "",
            "Compare two runs of the same experiment — e.g. before vs. after",
            "tuning a parameter — by loading both and overlaying their curves.",
            "",
            "Set `RUN_A` / `RUN_B` to two specific run paths to enable this cell;",
            "remove them or set to `None` to skip.",
        ),
        code(
            "RUN_A = None    # e.g. 'results/exp_sinr_cdf/2026-05-18_12-00-00'",
            "RUN_B = None    # e.g. 'results/exp_sinr_cdf/2026-05-19_15-30-00'",
            "",
            "if RUN_A and RUN_B:",
            "    res_a = load_run(RUN_A)",
            "    res_b = load_run(RUN_B)",
            "    fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "    # plot_cdf doesn't accept linestyle/label_suffix kwargs, so we",
            "    # post-process the Line2D objects to differentiate the two runs.",
            "    n_before_a = len(ax.lines)",
            "    plot_cdf(res_a.sim_result, metric=METRIC_MIN, ax=ax, only=PICK)",
            "    n_before_b = len(ax.lines)",
            "    for ln in ax.lines[n_before_a:n_before_b]:",
            "        ln.set_linestyle('--')",
            "        ln.set_label(ln.get_label() + ' (A)')",
            "    plot_cdf(res_b.sim_result, metric=METRIC_MIN, ax=ax, only=PICK)",
            "    for ln in ax.lines[n_before_b:]:",
            "        ln.set_label(ln.get_label() + ' (B)')",
            "    ax.legend(loc='best')        # refresh legend after relabeling",
            "    plt.show()",
            "else:",
            "    print('Set RUN_A and RUN_B above to compare two runs.')",
        ),
    ]
    return notebook(cells)


# ─────────────────────────────────────────────────────────────────────
#  Sweep notebook (snr_sweep, gamma_sweep, kappa_sweep, ...)
# ─────────────────────────────────────────────────────────────────────

def build_sweep_notebook() -> dict:
    cells = [
        md(
            "# Sweep Playground",
            "",
            "Interactive exploration of sweep figures — single-variable parameter",
            "sweeps that produce a metric-vs-axis curve per algorithm.",
            "",
            "**Compatible experiments:** `snr_sweep`, `gamma_sweep`, `kappa_sweep`,",
            "`clutter_cnr_sweep`, `n_ue_sweep`, `n_ap_sweep`, `antennas_sweep`",
            "(all use `kind == 'sweep'` and the `plot_sweep` function).",
            "",
            "**The one knob.**  Set `exp_name` (e.g. `'snr'`, `'kappa'`, `'gamma'`)",
            "and the experiment folder is derived as `f'{exp_name}_sweep'`.",
            "Switching between, say, `snr_sweep` and `kappa_sweep` results is one",
            "edit — axis labels come from `result.sweep_axis.display` so they update",
            "automatically.",
        ),
        INTRO_SETUP,
        code(
            "# ── The one knob: which sweep parameter ──",
            "exp_name = 'snr'              # snr / gamma / kappa / clutter_cnr",
            "                              # / n_ue / n_ap / antennas",
            "",
            "# Optional: load a specific run directory instead of the latest.",
            "# Set to a path like 'results/exp_snr_sweep/20260520_113500'",
            "# (relative to the repo root) to re-render an older comparison.",
            "EXP_DIR = None",
            "",
            "EXPERIMENT = f'{exp_name}_sweep'",
            "",
            "result = load_result(EXPERIMENT, EXP_DIR)",
            "assert result.kind == 'sweep', (",
            "    f'This notebook is for sweep experiments; got kind={result.kind}.'",
            ")",
            "summarize(result)",
        ),
        md(
            "## 1. Quick sweep plot — one metric, all algorithms",
            "",
            "`plot_sweep` takes the keyed-by-x results dict (its x-axis comes",
            "from the dict's keys, not from `sweep_axis`).  The `sweep_axis`",
            "metadata is used for human-readable axis labels — set them yourself",
            "via `ax.set_xlabel(result.sweep_axis.display)` etc.",
        ),
        code(
            "from cordis.plotting import plot_sweep, figsize",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_sweep(result.sweep_results, metric='min_sinr_db', ax=ax)",
            "ax.set_xlabel(result.sweep_axis.display)",
            "ax.set_ylabel('min-SINR [dB]')",
            "ax.set_title(f'{EXPERIMENT}: min-SINR vs. {result.sweep_axis.display}')",
            "plt.show()",
        ),
        md(
            "## 2. Side-by-side: SINR + SCNR",
            "",
            "Most paper figures pair a communication metric with a sensing metric.",
            "Build the two-panel layout inline — no helper needed.",
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=figsize(width='double', aspect=2.5/1.5))",
            "plot_sweep(result.sweep_results, metric='min_sinr_db', ax=axes[0])",
            "axes[0].set_xlabel(result.sweep_axis.display)",
            "axes[0].set_ylabel('min-SINR [dB]')",
            "axes[0].set_title('min-SINR [dB]')",
            "plot_sweep(result.sweep_results, metric='mean_scnr_db', ax=axes[1])",
            "axes[1].set_xlabel(result.sweep_axis.display)",
            "axes[1].set_ylabel('mean-SCNR [dB]')",
            "axes[1].set_title('mean-SCNR [dB]')",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## 3. Algorithm filter — focus on CORDIS vs. Centralized",
            "",
            "`plot_sweep` honors the same `only=` keyword as `plot_cdf`.",
        ),
        code(
            "PICK = ['CORDIS-Split', 'CORDIS-ADMM', 'Centralized']",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_sweep(result.sweep_results, metric='min_sinr_db', ax=ax, only=PICK)",
            "ax.set_xlabel(result.sweep_axis.display)",
            "ax.set_ylabel('min-SINR [dB]')",
            "plt.show()",
        ),
        md(
            "## 4. Log-y for SCNR sweeps",
            "",
            "Sensing metrics often span orders of magnitude across the sweep range.",
        ),
        code(
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_sweep(result.sweep_results, metric='mean_scnr_db', ax=ax, only=PICK)",
            "ax.set_yscale('log')",
            "ax.set_xlabel(result.sweep_axis.display)",
            "ax.set_ylabel('mean SCNR (log scale)')",
            "plt.show()",
        ),
        md(
            "## 5. Annotate the operating point",
            "",
            "If your paper highlights a specific axis value (e.g. SNR=10 dB), draw",
            "a vertical line + label at that point.",
        ),
        code(
            "x_marker = 10.0                # adjust to your sweep's axis units",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "plot_sweep(result.sweep_results, metric='min_sinr_db', ax=ax, only=PICK)",
            "ax.set_xlabel(result.sweep_axis.display)",
            "if min(result.sweep_axis.values) <= x_marker <= max(result.sweep_axis.values):",
            "    ax.axvline(x_marker, color='gray', linestyle=':', linewidth=1)",
            "    ax.annotate(f'  operating point ({x_marker})',",
            "                xy=(x_marker, ax.get_ylim()[1]),",
            "                xytext=(5, -10), textcoords='offset points',",
            "                fontsize=8, color='gray')",
            "plt.show()",
        ),
        md("## 6. Figsize variants"),
        GENERIC_FIGSIZE_CELL,
        md("## 7. Save with provenance metadata"),
        GENERIC_SAVE_CELL,
    ]
    return notebook(cells)


# ─────────────────────────────────────────────────────────────────────
#  Trace notebook (convergence_trace)
# ─────────────────────────────────────────────────────────────────────

def build_trace_notebook() -> dict:
    cells = [
        md(
            "# ADMM Convergence Trace Playground",
            "",
            "Interactive exploration of ADMM convergence diagnostics —",
            "primal/dual residuals and slack history across iterations.",
            "",
            "**Compatible experiments:** `convergence_trace`",
            "(uses `kind == 'trace'` and the `plot_admm_convergence` function).",
        ),
        INTRO_SETUP,
        code(
            "EXPERIMENT = 'convergence_trace'",
            "",
            "# Optional: load a specific run directory instead of the latest.",
            "# Set to a path like 'results/exp_convergence_trace/20260520_113500'",
            "# (relative to the repo root) to re-render an older trace.",
            "EXP_DIR = None",
            "",
            "result = load_result(EXPERIMENT, EXP_DIR)",
            "assert result.kind == 'trace'",
            "summarize(result)",
            "",
            "# `best_iter` may be 0 (no iterations run) or None (loaded from a",
            "# pre-Stage-18 result that didn't store it).  Cells below handle",
            "# both gracefully.",
            "best_iter = getattr(result.admm_result, 'best_iter', None)",
            "print(f'Best iter: {best_iter}')",
            "print(f'Total iters: {len(result.admm_result.primal_res_history)}')",
        ),
        md("## 1. Standard convergence plot"),
        code(
            "from cordis.plotting import plot_admm_convergence, figsize",
            "",
            "# plot_admm_convergence produces a TWO-panel figure (residuals + slack).",
            "# Pre-create both axes and pass them as the `axes=(ax_top, ax_bot)` tuple,",
            "# or call without `axes` to let the function build its own figure.",
            "fig, (ax_res, ax_slack) = plt.subplots(",
            "    1, 2, figsize=figsize(width='double', aspect=2.5/1.5),",
            ")",
            "plot_admm_convergence(result.admm_result, axes=(ax_res, ax_slack))",
            "fig.suptitle('ADMM convergence (primal/dual residuals + slack history)')",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## 2. Primal / dual on twin axes",
            "",
            "When primal and dual residuals have very different scales, a single",
            "log-y axis still works but a twin-axes layout makes both visible.",
            "If `best_iter` is set (Stage 18+ trace results), a vertical guide",
            "marks it.",
        ),
        code(
            "admm = result.admm_result",
            "iters = np.arange(1, len(admm.primal_res_history) + 1)",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "ln1 = ax.semilogy(iters, admm.primal_res_history, label='primal', color='C0')",
            "ax.set_xlabel('ADMM iteration')",
            "ax.set_ylabel('primal residual', color='C0')",
            "ax.tick_params(axis='y', labelcolor='C0')",
            "",
            "ax2 = ax.twinx()",
            "ln2 = ax2.semilogy(iters, admm.dual_res_history, label='dual', color='C3', linestyle='--')",
            "ax2.set_ylabel('dual residual', color='C3')",
            "ax2.tick_params(axis='y', labelcolor='C3')",
            "",
            "# Guard against best_iter being None (legacy loaded result) or 0",
            "# (solver returned before recording any iteration).",
            "_bi = getattr(admm, 'best_iter', None)",
            "if _bi:",
            "    ax.axvline(_bi, color='gray', linestyle=':', linewidth=0.8)",
            "    ax.set_title(f'Twin-axes residuals (best iter = {_bi})')",
            "else:",
            "    ax.set_title('Twin-axes residuals (best_iter unavailable)')",
            "plt.show()",
        ),
        md(
            "## 3. Slack history overlay",
            "",
            "When ALM/MoM is used, the slack penalty rises as it discovers QoS",
            "violations.  Overlaying its history on top of residuals shows when",
            "the algorithm 'gives up' enforcing a tight constraint.",
        ),
        code(
            "if hasattr(admm, 'slack_history') and admm.slack_history is not None \\",
            "        and len(admm.slack_history):",
            "    fig, axes = plt.subplots(2, 1, figsize=figsize(width='single', aspect=3/2.5),",
            "                              sharex=True)",
            "    axes[0].semilogy(iters, admm.primal_res_history, label='primal', color='C0')",
            "    axes[0].semilogy(iters, admm.dual_res_history,   label='dual',   color='C3')",
            "    axes[0].legend()",
            "    axes[0].set_ylabel('residual')",
            "    axes[1].plot(iters, admm.slack_history, color='C2')",
            "    axes[1].set_ylabel('slack')",
            "    axes[1].set_xlabel('ADMM iteration')",
            "    plt.tight_layout()",
            "    plt.show()",
            "else:",
            "    print('No slack_history in this trace.')",
        ),
        md(
            "## 4. Convergence-rate fit (rough)",
            "",
            "Fit a geometric rate to the primal residual tail (last ~30% of iters)",
            "to quantify how fast ADMM is converging.",
        ),
        code(
            "tail_frac = 0.3",
            "n         = len(admm.primal_res_history)",
            "tail_start = max(1, int(n * (1 - tail_frac)))",
            "tail_iters = iters[tail_start:]",
            "tail_res   = admm.primal_res_history[tail_start:]",
            "",
            "# log y = a + b * iter  →  y = exp(a) * exp(b)^iter",
            "if np.all(tail_res > 0):",
            "    coeffs = np.polyfit(tail_iters, np.log(tail_res), 1)",
            "    rate   = np.exp(coeffs[0])     # geometric ratio per iter",
            "    print(f'Tail geometric rate: r ≈ {rate:.4f}  (per iter)')",
            "    print(f'                     1/(1-r) ≈ {1/(1-rate):.1f}  iters to settle')",
            "else:",
            "    print('Tail contains zeros/negatives; skipping rate fit.')",
        ),
        md("## 5. Figsize variants"),
        GENERIC_FIGSIZE_CELL,
        md("## 6. Save with provenance metadata"),
        GENERIC_SAVE_CELL,
    ]
    return notebook(cells)


# ─────────────────────────────────────────────────────────────────────
#  Table notebook (fronthaul_table)
# ─────────────────────────────────────────────────────────────────────

def build_table_notebook() -> dict:
    cells = [
        md(
            "# Table Playground",
            "",
            "Interactive exploration of tabular results — render as Markdown,",
            "LaTeX (booktabs), or as a bar chart for visual comparison.",
            "",
            "**Compatible experiments:** `fronthaul_table`",
            "(uses `kind == 'table'`).",
            "",
            "Note: `result.table_data` is a `Dict[algorithm_name, Dict[field, value]]`",
            "structure — NOT a `SimResult`.  The `cordis.plotting.bar_chart` and",
            "`to_*_table` functions expect a `SimResult`, so this notebook builds",
            "tables and charts from `result.table_data` directly (matching what",
            "`scripts/plot_fronthaul_table.py` does).",
        ),
        INTRO_SETUP,
        code(
            "EXPERIMENT = 'fronthaul_table'",
            "",
            "# Optional: load a specific run directory instead of the latest.",
            "# Set to a path like 'results/exp_fronthaul_table/20260520_113500'",
            "# (relative to the repo root) to re-render an older table.",
            "EXP_DIR = None",
            "",
            "result = load_result(EXPERIMENT, EXP_DIR)",
            "assert result.kind == 'table'",
            "summarize(result)",
            "print()",
            "print('Algorithms in table:', list(result.table_data.keys()))",
            "first_algo = next(iter(result.table_data))",
            "print(f'Fields per algorithm: {list(result.table_data[first_algo].keys())}')",
        ),
        md("## 1. Markdown render (good for previews + GitHub READMEs)"),
        code(
            "# Mirrors what scripts/plot_fronthaul_table.py emits.",
            "# real_scalars is PER COORDINATION ROUND (apples-to-apples across",
            "# algorithms).  iterations holds the per-solve multiplier",
            "# (1 for Centralized/Split, T_ADMM for ADMM).",
            "lines = [",
            "    '| Algorithm | Data shared | Size | Real scalars / round | Iterations | Total / solve | Scalable |',",
            "    '|---|---|---|---|---|---|---|',",
            "]",
            "for algo, row in result.table_data.items():",
            "    rs = row.get('real_scalars')",
            "    it = row.get('iterations')",
            "    rs_str = '—' if rs is None else (",
            "        f'{rs:.1f}' if isinstance(rs, float) else f'{rs:d}'",
            "    )",
            "    it_str = '—' if it is None else (",
            "        f'{it:.1f}' if isinstance(it, float) and not float(it).is_integer()",
            "        else f'{int(it):d}'",
            "    )",
            "    tot_str = '—' if (rs is None or it is None) else (",
            "        f'{float(rs) * float(it):.0f}'",
            "    )",
            "    lines.append(",
            "        f\"| {algo} | {row['data_to_share']} | {row['size']} | \"",
            "        f\"{rs_str} | {it_str} | {tot_str} | \"",
            "        f\"{'✓' if row['scalable'] else '✗'} |\"",
            "    )",
            "print('\\n'.join(lines))",
        ),
        md(
            "## 2. LaTeX render for the paper",
            "",
            "Uses `booktabs` style.  Copy-paste the output into your paper's `.tex`",
            "(or use `\\input{...}` from a saved `.tex` file).",
        ),
        code(
            "tex_lines = [",
            "    r'\\begin{table}[t]',",
            "    r'\\centering',",
            "    r'\\caption{Per-AP fronthaul coordination overhead.}',",
            "    r'\\label{tab:fronthaul}',",
            "    r'\\begin{tabular}{lcccc}',",
            "    r'\\toprule',",
            "    r'Algorithm & Data shared & Size & Real / iter & Scalable \\\\',",
            "    r'\\midrule',",
            "]",
            "for algo, row in result.table_data.items():",
            "    rs = row.get('real_scalars')",
            "    rs_str = '--' if rs is None else (",
            "        f'{rs:.1f}' if isinstance(rs, float) else f'{rs:d}'",
            "    )",
            "    tex_lines.append(",
            "        f\"{algo} & {row['data_to_share']} & {row['size']} & \"",
            "        f\"{rs_str} & \"",
            "        + (r'$\\checkmark$' if row['scalable'] else r'$\\times$')",
            "        + r' \\\\'",
            "    )",
            "tex_lines += [r'\\bottomrule', r'\\end{tabular}', r'\\end{table}']",
            "tex = '\\n'.join(tex_lines)",
            "print(tex)",
            "",
            "# Save it for \\input{} from your paper.",
            "out = Path('../figures/playground/fronthaul_table.tex')",
            "out.parent.mkdir(parents=True, exist_ok=True)",
            "out.write_text(tex)",
            "print(f'wrote {out}')",
        ),
        md(
            "## 3. Bar chart — per-round fronthaul cost",
            "",
            "All algorithms reported per coordination round (apples-to-apples).",
            "Centralized + Split run 1 round per coherence block; CORDIS-ADMM",
            "runs T_ADMM rounds — shown in the `iterations` column above.",
            "",
            "When numbers span orders of magnitude (e.g. fronthaul scalars from",
            "a few to a few hundred), a log-y bar chart often communicates the",
            "gap better than the numeric table alone.  Built from",
            "`result.table_data` directly with raw matplotlib (since",
            "`cordis.plotting.bar_chart` expects a SimResult).",
        ),
        code(
            "from cordis.plotting import figsize",
            "",
            "# Pull the per-round count for plotting, skip rows with None.",
            "algs   = []",
            "values = []",
            "for algo, row in result.table_data.items():",
            "    rs = row.get('real_scalars')",
            "    if rs is None:",
            "        continue",
            "    algs.append(algo)",
            "    values.append(float(rs))",
            "",
            "fig, ax = plt.subplots(figsize=figsize(width='single', aspect=3/2))",
            "bars = ax.bar(range(len(algs)), values, color='C0', edgecolor='black')",
            "ax.set_yscale('log')",
            "ax.set_xticks(range(len(algs)))",
            "ax.set_xticklabels(algs, rotation=20, ha='right')",
            "ax.set_ylabel('real scalars per AP per coordination round')",
            "ax.set_title('Per-round fronthaul cost (log scale)')",
            "ax.grid(True, axis='y', alpha=0.3)",
            "",
            "# Annotate each bar with the count for clarity.",
            "for rect, v in zip(bars, values):",
            "    ax.annotate(f'{v:.0f}', xy=(rect.get_x() + rect.get_width()/2,",
            "                                rect.get_height()),",
            "                xytext=(0, 3), textcoords='offset points',",
            "                ha='center', va='bottom', fontsize=8)",
            "",
            "plt.tight_layout()",
            "plt.show()",
        ),
        md(
            "## 4. Highlight CORDIS rows in the LaTeX output",
            "",
            "For paper figures it's nice to bold-face the proposed algorithms",
            "so the reader's eye lands on them.  This is a post-process on the",
            "LaTeX text from cell 2.",
        ),
        code(
            "tex_with_highlight = tex",
            "for alg in ('CORDIS-Split', 'CORDIS-ADMM'):",
            "    tex_with_highlight = tex_with_highlight.replace(",
            "        f'{alg}', f'\\\\textbf{{{alg}}}'",
            "    )",
            "print(tex_with_highlight)",
        ),
        md("## 5. Figsize variants"),
        GENERIC_FIGSIZE_CELL,
        md("## 6. Save the bar chart with provenance"),
        GENERIC_SAVE_CELL,
    ]
    return notebook(cells)


# ─────────────────────────────────────────────────────────────────────
#  Build all four
# ─────────────────────────────────────────────────────────────────────

NOTEBOOKS = {
    "playground_cdf.ipynb":    build_cdf_notebook,
    "playground_sweep.ipynb":  build_sweep_notebook,
    "playground_trace.ipynb":  build_trace_notebook,
    "playground_table.ipynb":  build_table_notebook,
}


def main() -> None:
    for filename, builder in NOTEBOOKS.items():
        nb = builder()
        path = HERE / filename
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        n_cells = len(nb["cells"])
        print(f"  wrote {path.name}  ({n_cells} cells)")
    print(f"\nTotal: {len(NOTEBOOKS)} notebooks regenerated.")


if __name__ == "__main__":
    main()

