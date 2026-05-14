"""
Stage 8a — Parameter sweep plotting helper.

Given a ``{x_value: SimResult}`` mapping, plot ``mean(metric)`` vs ``x``
for each algorithm, with optional IQR or std-band shading.

Example
-------
.. code-block:: python

    results_by_gamma = {0.0: sim0, 3.0: sim3, 6.0: sim6, 10.0: sim10}
    ax = plot_sweep(results_by_gamma, "weighted_sum_scnr_db",
                    xlabel=r"$\\gamma$ [dB]",
                    ylabel="Sum SCNR [dB]",
                    error="iqr")
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from cordis.plotting.style import figsize, style_for

logger = logging.getLogger(__name__)


def plot_sweep(
    results_by_x: Mapping[float, Any],     # {x_value: SimResult}
    metric: str,
    ax: Optional[Axes] = None,
    *,
    error: Literal["iqr", "std", "none"] = "iqr",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend_loc: str = "best",
    only: Optional[Sequence[str]] = None,
    log_x: bool = False,
    grid: bool = True,
) -> Axes:
    """
    Plot ``mean(metric)`` vs ``x`` for each algorithm across the sweep.

    Parameters
    ----------
    results_by_x : Mapping[float, SimResult]
        One :class:`SimResult` per swept value of x.
    metric : str
        Metric name from the result registry.
    ax : Axes, optional
    error : {'iqr', 'std', 'none'}
        Error-band style.

        * ``'iqr'`` — fill between 25th and 75th percentile
        * ``'std'`` — fill between ``mean ± std``
        * ``'none'`` — no band

    xlabel, ylabel, title : str, optional
    legend_loc : str
    only : sequence of algorithm names, optional
        Restrict to a subset of algorithms.
    log_x : bool
        Log-scale the x-axis.
    grid : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    if error not in {"iqr", "std", "none"}:
        raise ValueError(
            f"error must be 'iqr', 'std', or 'none'; got {error!r}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize("single"))

    xs_sorted = sorted(results_by_x.keys())

    # Algorithms that appear at least once across the sweep.
    all_names = set()
    for sr in results_by_x.values():
        all_names.update(sr.algorithm_results.keys())
    if only:
        all_names &= set(only)

    for name in sorted(all_names):
        means: list = []
        lo: list = []
        hi: list = []
        xs_used: list = []
        for x in xs_sorted:
            sr = results_by_x[x]
            if name not in sr.algorithm_results:
                continue
            ar = sr.algorithm_results[name]
            if not ar.has_metric(metric):
                continue
            try:
                m = ar.mean(metric)
            except (KeyError, ValueError) as e:
                logger.warning("mean failed for %s at x=%s: %s", name, x, e)
                continue
            xs_used.append(x)
            means.append(m)
            if error == "iqr":
                lo.append(ar.percentile(metric, 25))
                hi.append(ar.percentile(metric, 75))
            elif error == "std":
                s = ar.std(metric)
                lo.append(m - s)
                hi.append(m + s)
        if not xs_used:
            continue
        line, = ax.plot(xs_used, means, **style_for(name))
        if error != "none":
            ax.fill_between(xs_used, lo, hi,
                            color=line.get_color(), alpha=0.15,
                            linewidth=0)

    ax.set_xlabel(xlabel if xlabel is not None else "x")
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    if title:
        ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    if grid:
        ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)
    return ax

