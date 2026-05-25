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
    legend_loc: str = "best",
    only: Optional[Sequence[str]] = None,
    grid: bool = True,
    gamma_db: Optional[float] = None,
    show_feasible_only: bool = False,
    annotate_infeasibility: bool = True,
    cond_metric: str = "min_sinr_db",
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
        Legend location (matplotlib convention).  Default ``"best"``
        lets matplotlib pick the corner with least overlap, which
        works well when curves cluster near the right edge (high-SNR
        experiments).  Override with ``"lower right"`` when you want
        a fixed corner — exploiting the fact that any CDF passes
        through (xmin, 0), so lower-right is mathematically empty.
    only : sequence of algorithm names, optional
        Restrict the plot to a subset of algorithms.
    grid : bool
    gamma_db : float, optional
        Stage 20: if provided, marks γ with a vertical dashed line and
        enables the infeasibility-tracking features described below.
    show_feasible_only : bool, default False
        Stage 20: if True (and ``gamma_db`` is set), draw a SECOND CDF
        curve per algorithm using only trials where ``cond_metric ≥
        gamma_db`` (a "feasible-trials only" CDF).  Dashed style so it
        doesn't visually compete with the all-trials curve.  Useful for
        CORDIS-ADMM whose unconditional CDF has a long left tail from
        infeasible drops.
    annotate_infeasibility : bool, default True
        Stage 20: if True (and ``gamma_db`` is set), append the
        infeasibility rate per algorithm to the legend label, formatted
        as ``"  (inf=X.X%)"``.  Lets the reader see the trade-off
        between mean performance and constraint-satisfaction rate at a
        glance, without filtering trials silently.
    cond_metric : str, default "min_sinr_db"
        Stage 20: per-trial metric used to classify a trial as
        feasible (``cond_metric ≥ gamma_db``).  Almost always
        ``"min_sinr_db"``.

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

        # Build legend label, optionally annotated with infeasibility rate.
        style = dict(style_for(name))
        if (gamma_db is not None and annotate_infeasibility
                and ar.has_metric(cond_metric)):
            inf_rate = ar.infeasibility_rate(gamma_db, metric=cond_metric)
            base_label = style.get("label", name)
            style["label"] = f"{base_label}  (inf={100.0*inf_rate:.1f}%)"
        ax.plot(xs, fs, **style)

        # Optional second curve: feasible-trials-only conditional CDF.
        if (gamma_db is not None and show_feasible_only
                and ar.has_metric(cond_metric)):
            try:
                xs_f, fs_f = ar.cdf_conditional(
                    metric, gamma_db, cond_metric=cond_metric,
                )
                if xs_f.size > 0:
                    # Inherit color, drop the label (avoid legend bloat),
                    # use dashed linestyle to mark "conditional".
                    cond_style = dict(style)
                    cond_style.pop("label", None)
                    cond_style["linestyle"] = "--"
                    cond_style["alpha"] = 0.7
                    ax.plot(xs_f, fs_f, **cond_style)
            except (KeyError, ValueError) as e:
                logger.warning("conditional CDF failed for %s: %s", name, e)
        n_plotted += 1

    # γ marker (after curves so it stays on top).
    if gamma_db is not None:
        ax.axvline(float(gamma_db), color="k", linestyle=":",
                   linewidth=0.8, alpha=0.6,
                   label=rf"$\gamma$ = {float(gamma_db):.1f} dB")

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

