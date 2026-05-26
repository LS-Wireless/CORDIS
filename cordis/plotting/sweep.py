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
    gamma_db: Optional[float] = None,
    show_feasible_only: bool = False,
    annotate_infeasibility: bool = True,
    cond_metric: str = "min_sinr_db",
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
    gamma_db : float, optional
        Stage 21 v3: if provided, draws a horizontal dashed line at
        ``y = gamma_db`` (the SINR floor the algorithms target) and
        enables the infeasibility-tracking features described below.
        Most useful when ``metric == "min_sinr_db"`` since γ is then on
        the same scale as the plot; for other metrics the line is still
        drawn but its interpretation is up to the reader.
    show_feasible_only : bool, default False
        Stage 21 v3: if True (and ``gamma_db`` is set), recompute the
        per-axis-value central tendency + error band using ONLY trials
        where ``cond_metric >= gamma_db`` (the "feasible" trials).  At
        axis values where every trial is infeasible the point is
        skipped, leaving a visible gap in the curve and a warning in
        the log.  Useful for showing CORDIS-ADMM's behavior on the
        trials where it actually solved the problem.

        Requires ``gamma_db`` to be set; warns and falls back to the
        full mean if you set ``show_feasible_only=True`` without it.
    annotate_infeasibility : bool, default True
        Stage 21 v3: if True (and ``gamma_db`` is set), append a
        per-algorithm infeasibility summary to its legend label as
        ``"  (max inf=X.X%)"``, where X.X is the WORST infeasibility
        rate observed across all sweep axis values.  The "max" framing
        surfaces the worst-case constraint-satisfaction rate, which is
        typically what you want from a sweep — at the operating points
        where the algorithm struggles most.  Auto-escapes ``%`` to
        ``\\%`` when ``text.usetex=True``.
    cond_metric : str, default "min_sinr_db"
        Stage 21 v3: per-trial metric used to classify a trial as
        feasible (``cond_metric >= gamma_db``).  Almost always
        ``"min_sinr_db"``.

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

    # NumPy needed for the conditional/feasible-only branch.
    import numpy as _np

    # No-op warning, matching plot_cdf's behavior.
    if show_feasible_only and gamma_db is None:
        logger.warning(
            "plot_sweep: show_feasible_only=True has no effect without "
            "gamma_db.  Either pass gamma_db=<value> to enable the "
            "feasibility filter, or set show_feasible_only=False to "
            "silence this warning."
        )

    # LaTeX-aware '%' escape — '%' is the LaTeX comment character and
    # would eat the rest of the legend label.
    using_latex = bool(plt.rcParams.get("text.usetex", False))
    pct_token = r"\%" if using_latex else "%"

    xs_sorted = sorted(results_by_x.keys())

    # Algorithms that appear at least once across the sweep.
    all_names = set()
    for sr in results_by_x.values():
        all_names.update(sr.algorithm_results.keys())
    if only:
        all_names &= set(only)

    def _feasible_stats(ar, x_val):
        """Compute (mean, p25, p75, std) over feasible trials only.

        Uses the public ``feasible_trials_mask`` + the (semi-public)
        ``_samples`` accessor — same path that ``cdf_conditional``
        uses internally.  Returns ``(None, None, None, None)`` when
        zero feasible trials exist at this axis value, so the caller
        can skip the point (leaving a gap in the curve).
        """
        if not ar.has_metric(cond_metric):
            return None, None, None, None
        try:
            arr  = ar._samples(metric)
            mask = ar.feasible_trials_mask(gamma_db, metric=cond_metric)
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(
                "plot_sweep: feasibility filter failed for %s at x=%s: %s",
                ar.spec.name if hasattr(ar, "spec") else "?", x_val, e,
            )
            return None, None, None, None
        if mask.size != arr.size or not _np.any(mask):
            return None, None, None, None
        filt = arr[mask]
        return (float(_np.mean(filt)),
                float(_np.percentile(filt, 25)),
                float(_np.percentile(filt, 75)),
                float(_np.std(filt)))

    for name in sorted(all_names):
        means: list = []
        lo: list = []
        hi: list = []
        xs_used: list = []

        # For the legend annotation: track the worst (max) infeasibility
        # rate across all axis values where this algorithm has data.
        worst_inf_rate = 0.0
        saw_any_cond_metric = False

        for x in xs_sorted:
            sr = results_by_x[x]
            if name not in sr.algorithm_results:
                continue
            ar = sr.algorithm_results[name]
            if not ar.has_metric(metric):
                continue

            # Track worst-case infeasibility for the legend annotation
            # (independent of whether we filter by feasibility below).
            if (gamma_db is not None and annotate_infeasibility
                    and ar.has_metric(cond_metric)):
                inf_rate = ar.infeasibility_rate(gamma_db, metric=cond_metric)
                worst_inf_rate = max(worst_inf_rate, float(inf_rate))
                saw_any_cond_metric = True

            # Compute the central tendency + error band — either over
            # all trials (default) or over feasible trials only.
            if show_feasible_only and gamma_db is not None:
                m, p25, p75, s = _feasible_stats(ar, x)
                if m is None:
                    logger.warning(
                        "plot_sweep: %s has 0 feasible trials at x=%s, "
                        "gamma_db=%s — skipping point.",
                        name, x, gamma_db,
                    )
                    continue
            else:
                try:
                    m = ar.mean(metric)
                except (KeyError, ValueError) as e:
                    logger.warning("mean failed for %s at x=%s: %s",
                                   name, x, e)
                    continue
                if error == "iqr":
                    p25 = ar.percentile(metric, 25)
                    p75 = ar.percentile(metric, 75)
                    s   = None
                elif error == "std":
                    s   = ar.std(metric)
                    p25 = p75 = None
                else:
                    p25 = p75 = s = None

            xs_used.append(x)
            means.append(m)
            if error == "iqr":
                lo.append(p25); hi.append(p75)
            elif error == "std":
                lo.append(m - s); hi.append(m + s)

        if not xs_used:
            continue

        # Build label (optionally annotated with worst-case infeasibility).
        style = dict(style_for(name))
        if (gamma_db is not None and annotate_infeasibility
                and saw_any_cond_metric):
            base_label = style.get("label", name)
            style["label"] = (f"{base_label}  "
                              f"(max inf={100.0*worst_inf_rate:.1f}{pct_token})")

        line, = ax.plot(xs_used, means, **style)
        if error != "none":
            ax.fill_between(xs_used, lo, hi,
                            color=line.get_color(), alpha=0.15,
                            linewidth=0)

    # γ marker (horizontal line at the SINR floor).
    if gamma_db is not None:
        ax.axhline(float(gamma_db), color="k", linestyle=":",
                   linewidth=0.8, alpha=0.6,
                   label=rf"$\gamma$ = {float(gamma_db):.1f} dB")

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

