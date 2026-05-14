"""
Stage 8a — Empirical CDF plotting helper.

Plots one CDF curve per algorithm in a :class:`SimResult`.  Designed
to be the workhorse for SINR-CDF / SCNR-CDF figures in the paper.

Example
-------
.. code-block:: python

    from cordis.plotting import apply_paper_style, plot_cdf, save_figure
    apply_paper_style()
    ax = plot_cdf(sim_result, "min_sinr_db",
                  xlabel=r"min-SINR [dB]")
    save_figure(ax.figure, "figures/sinr_cdf")
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from cordis.plotting.style import figsize, style_for

logger = logging.getLogger(__name__)


def plot_cdf(
    sim_result: Any,                       # SimResult (duck-typed)
    metric: str,
    ax: Optional[Axes] = None,
    *,
    xlabel: Optional[str] = None,
    ylabel: str = "Empirical CDF",
    title: Optional[str] = None,
    legend_loc: str = "lower right",
    only: Optional[Sequence[str]] = None,
    grid: bool = True,
) -> Axes:
    """
    Plot empirical CDF of ``metric`` for each algorithm in ``sim_result``.

    Parameters
    ----------
    sim_result
        A :class:`SimResult` (or any object with ``.algorithm_results``
        dict whose values implement ``.has_metric(name)`` and
        ``.cdf(name) -> (xs, fs)``).
    metric : str
        Name of metric registered in ``result.py`` (e.g.
        ``"min_sinr_db"``, ``"weighted_sum_scnr_db"``).
    ax : matplotlib Axes, optional
        Plot into this axis; creates a new figure if None.
    xlabel, ylabel, title : str, optional
        Axis labels.  ``xlabel`` defaults to the metric name.
    legend_loc : str
        Legend location (matplotlib convention).
    only : sequence of algorithm names, optional
        Restrict the plot to a subset of algorithms.
    grid : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize("single"))

    only_set = set(only) if only else None
    n_plotted = 0

    for name, ar in sim_result.algorithm_results.items():
        if only_set is not None and name not in only_set:
            continue
        if not ar.has_metric(metric):
            logger.debug("Skipping %s: no metric %r", name, metric)
            continue
        try:
            xs, fs = ar.cdf(metric)
        except (KeyError, ValueError) as e:
            logger.warning("CDF failed for %s on %r: %s", name, metric, e)
            continue
        ax.plot(xs, fs, **style_for(name))
        n_plotted += 1

    if n_plotted == 0:
        logger.warning("plot_cdf: no algorithm provided metric %r", metric)

    ax.set_xlabel(xlabel if xlabel is not None else metric)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    if n_plotted > 0:
        ax.legend(loc=legend_loc)
    return ax

