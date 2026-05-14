"""
Stage 8a — Comparison helpers: bar chart + tables.

Aggregate per-algorithm statistics into three outputs:

* :func:`bar_chart`         — matplotlib bar plot with optional error bars
* :func:`to_markdown_table` — GitHub-flavoured markdown
* :func:`to_latex_table`    — booktabs-style LaTeX table

Tables share the same column logic so a single source of truth drives
both renderings of any given metric set.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from cordis.plotting.style import figsize, style_for

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Shared aggregation helper
# ─────────────────────────────────────────────────────────────────────

def _collect_rows(
    sim_result: Any,
    metrics: Sequence[str],
    aggregate: Literal["mean", "median"],
    only: Optional[Sequence[str]],
    precision: int,
) -> Tuple[List[str], List[List[str]], List[List[Optional[float]]]]:
    """Aggregate metrics for all (algorithm, metric) pairs.

    Returns
    -------
    names : list[str]
        Algorithm names in iteration order.
    rows_str : list[list[str]]
        ``rows_str[i][j]`` = formatted ``metrics[j]`` for ``names[i]``,
        or ``"—"`` if missing.
    rows_num : list[list[float|None]]
        Same shape, numeric values (``None`` if missing).
    """
    if aggregate not in {"mean", "median"}:
        raise ValueError(f"aggregate must be 'mean' or 'median', got {aggregate!r}")

    only_set = set(only) if only else None
    names: List[str] = []
    rows_str: List[List[str]] = []
    rows_num: List[List[Optional[float]]] = []

    for name, ar in sim_result.algorithm_results.items():
        if only_set is not None and name not in only_set:
            continue
        row_s: List[str] = []
        row_n: List[Optional[float]] = []
        for m in metrics:
            if not ar.has_metric(m):
                row_s.append("—")
                row_n.append(None)
                continue
            try:
                v = ar.mean(m) if aggregate == "mean" else ar.percentile(m, 50)
            except (KeyError, ValueError) as e:
                logger.warning("aggregate failed for %s.%s: %s", name, m, e)
                row_s.append("—")
                row_n.append(None)
                continue
            row_s.append(f"{v:.{precision}f}")
            row_n.append(float(v))
        names.append(name)
        rows_str.append(row_s)
        rows_num.append(row_n)
    return names, rows_str, rows_num


# ─────────────────────────────────────────────────────────────────────
#  Bar chart
# ─────────────────────────────────────────────────────────────────────

def bar_chart(
    sim_result: Any,
    metric: str,
    ax: Optional[Axes] = None,
    *,
    aggregate: Literal["mean", "median"] = "mean",
    error: Literal["std", "iqr", "none"] = "std",
    only: Optional[Sequence[str]] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> Axes:
    """
    Bar chart of ``aggregate(metric)`` per algorithm.

    Parameters
    ----------
    sim_result : SimResult
    metric : str
    ax : Axes, optional
    aggregate : 'mean' | 'median'
    error : 'std' | 'iqr' | 'none'
        Error-bar style.  'iqr' uses half the IQR.
    only : sequence of algorithm names, optional
    ylabel, title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if error not in {"std", "iqr", "none"}:
        raise ValueError(f"error must be 'std', 'iqr', or 'none', got {error!r}")
    if aggregate not in {"mean", "median"}:
        raise ValueError(f"aggregate must be 'mean' or 'median', got {aggregate!r}")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize("single"))

    only_set = set(only) if only else None
    names, values, errs, colors = [], [], [], []

    for name, ar in sim_result.algorithm_results.items():
        if only_set is not None and name not in only_set:
            continue
        if not ar.has_metric(metric):
            continue
        try:
            v = ar.mean(metric) if aggregate == "mean" else ar.percentile(metric, 50)
            if error == "std":
                e = ar.std(metric)
            elif error == "iqr":
                e = (ar.percentile(metric, 75) - ar.percentile(metric, 25)) / 2.0
            else:
                e = 0.0
        except (KeyError, ValueError) as ex:
            logger.warning("bar_chart: failed for %s.%s: %s", name, metric, ex)
            continue
        names.append(name)
        values.append(v)
        errs.append(e)
        colors.append(style_for(name).get("color", "gray"))

    if not names:
        logger.warning("bar_chart: no algorithm had metric %r", metric)
        return ax

    positions = np.arange(len(names))
    yerr = errs if error != "none" else None
    ax.bar(positions, values, yerr=yerr,
           color=colors, edgecolor="black", linewidth=0.5,
           capsize=2.0, error_kw=dict(linewidth=0.7))
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────
#  Markdown table
# ─────────────────────────────────────────────────────────────────────

def to_markdown_table(
    sim_result: Any,
    metrics: Sequence[str],
    *,
    aggregate: Literal["mean", "median"] = "mean",
    only: Optional[Sequence[str]] = None,
    precision: int = 3,
) -> str:
    """
    Build a GitHub-flavoured markdown table of aggregate metrics.

    Columns: algorithm name + one column per metric.  Missing values
    appear as ``"—"``.  Column widths are computed for clean alignment.

    Returns
    -------
    str — multi-line markdown.
    """
    names, rows_str, _ = _collect_rows(
        sim_result, metrics, aggregate, only, precision,
    )
    if not names:
        return ""

    headers = ["Algorithm"] + list(metrics)
    n_cols = len(headers)
    all_rows = [headers] + [[n] + r for n, r in zip(names, rows_str)]
    widths = [max(len(str(row[i])) for row in all_rows) for i in range(n_cols)]

    def _fmt(row):
        return "| " + " | ".join(f"{row[i]:<{widths[i]}}" for i in range(n_cols)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    return "\n".join([_fmt(headers), sep] + [_fmt([n] + r)
                                              for n, r in zip(names, rows_str)])


# ─────────────────────────────────────────────────────────────────────
#  LaTeX table
# ─────────────────────────────────────────────────────────────────────

def to_latex_table(
    sim_result: Any,
    metrics: Sequence[str],
    *,
    aggregate: Literal["mean", "median"] = "mean",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    only: Optional[Sequence[str]] = None,
    precision: int = 3,
    metric_labels: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a booktabs-style LaTeX table of aggregate metrics.

    Parameters
    ----------
    metric_labels : dict, optional
        Map raw metric name → display label, e.g.
        ``{'min_sinr_db': r'min-SINR (dB)'}``.  Defaults to the raw
        name with ``_`` escaped.

    Notes
    -----
    Output uses ``\\begin{table}[h] \\centering \\caption ... \\label
    ... \\begin{tabular}{lr...r} \\toprule ... \\midrule ... \\bottomrule
    \\end{tabular} \\end{table}``.  Requires the ``booktabs`` package
    in the LaTeX preamble.
    """
    names, rows_str, _ = _collect_rows(
        sim_result, metrics, aggregate, only, precision,
    )
    if not names:
        return ""

    metric_labels = metric_labels or {}
    headers = ["Algorithm"] + [
        metric_labels.get(m, m).replace("_", r"\_")
        for m in metrics
    ]
    # "—" is non-ASCII; LaTeX with utf8 inputenc handles it, but the
    # safest portable rendering is "---" (em-dash via three hyphens).
    rows_str_tex = [[cell if cell != "—" else "---" for cell in r]
                    for r in rows_str]

    col_spec = "l" + "r" * len(metrics)
    lines = [r"\begin{table}[h]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for name, row in zip(names, rows_str_tex):
        lines.append(" & ".join([name] + row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

