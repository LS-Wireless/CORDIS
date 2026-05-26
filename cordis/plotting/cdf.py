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
        Stage 20: if True (and ``gamma_db`` is set), draw the CDF
        using only trials where ``cond_metric >= gamma_db`` (a
        "feasible-trials only" CDF) instead of the full CDF.  Useful
        for CORDIS-ADMM whose unconditional CDF has a long left tail
        from infeasible drops; the feasible-only curve isolates the
        algorithm's behavior on the trials it actually satisfied.

        If ``gamma_db`` is None this flag has no effect and a warning
        is logged.

        If every trial is infeasible at the given threshold (empty
        conditional CDF), the unconditional CDF is drawn as a fallback
        with a warning, so the user always sees a curve.
    annotate_infeasibility : bool, default True
        Stage 20: if True (and ``gamma_db`` is set), append the
        infeasibility rate per algorithm to the legend label, formatted
        as ``"  (inf=X.X%)"``.  Lets the reader see the trade-off
        between mean performance and constraint-satisfaction rate at a
        glance, without filtering trials silently.  When matplotlib is
        configured with ``text.usetex=True``, the ``%`` is escaped to
        ``\%`` automatically (otherwise LaTeX treats it as a comment
        and eats the rest of the label).
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

    # User asked for show_feasible_only but no threshold — no-op silently
    # before, now warn so the user notices.
    if show_feasible_only and gamma_db is None:
        logger.warning(
            "plot_cdf: show_feasible_only=True has no effect without "
            "gamma_db.  Either pass gamma_db=<value> to enable the "
            "feasibility filter, or set show_feasible_only=False to "
            "silence this warning."
        )

    # Detect LaTeX rendering — '%' is the LaTeX comment character and
    # eats everything to end-of-line, including ')'.  We must escape it
    # to '\%' inside labels when text.usetex is on.
    using_latex = bool(plt.rcParams.get("text.usetex", False))
    pct_token = r"\%" if using_latex else "%"

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
            style["label"] = (f"{base_label}  "
                              f"(inf={100.0*inf_rate:.1f}{pct_token})")

        # Decide which curves to draw:
        #   - show_feasible_only=False (default) → unconditional only
        #   - show_feasible_only=True without gamma_db → unconditional
        #     only (the warning above tells the user why)
        #   - show_feasible_only=True with gamma_db → feasible-only ONLY
        #     (the previous behavior drew BOTH curves overlapping, which
        #     defeats the purpose of the filter)
        draw_feasible = (gamma_db is not None and show_feasible_only
                         and ar.has_metric(cond_metric))

        if not draw_feasible:
            ax.plot(xs, fs, **style)
        else:
            try:
                xs_f, fs_f = ar.cdf_conditional(
                    metric, gamma_db, cond_metric=cond_metric,
                )
                if xs_f.size > 0:
                    # Solid line, same style — this IS the curve now,
                    # so it gets the (possibly-annotated) label.
                    ax.plot(xs_f, fs_f, **style)
                else:
                    # Empty conditional CDF (every trial infeasible) —
                    # fall back to unconditional so the user sees
                    # something rather than nothing.
                    logger.warning(
                        "plot_cdf: %s has zero feasible trials at "
                        "gamma_db=%s — falling back to unconditional CDF.",
                        name, gamma_db,
                    )
                    ax.plot(xs, fs, **style)
            except (KeyError, ValueError) as e:
                logger.warning("conditional CDF failed for %s: %s", name, e)
                ax.plot(xs, fs, **style)
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

