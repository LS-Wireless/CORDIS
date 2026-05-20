"""
Stage 8a — ADMM convergence trajectory plot.

Two-panel layout:

* Left:  primal residual r_pri and dual residual r_dual vs iteration.
* Right: SOC slack ``max_u ε_u`` vs iteration (if the ADMMResult
  carries a ``slack_history`` field; otherwise the panel shows a
  "not available" annotation so the figure remains well-formed).

Takes an :class:`ADMMResult` directly rather than a :class:`SimResult`
because these are per-trial trajectories, not aggregate statistics.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from cordis.plotting.style import figsize

logger = logging.getLogger(__name__)

# Curve colors chosen so each trace reads distinctly even in greyscale.
_PRIMAL_COLOR = "#d62728"   # red
_DUAL_COLOR   = "#1f77b4"   # blue
_SLACK_COLOR  = "#2ca02c"   # green


def plot_admm_convergence(
    admm_result: Any,                       # ADMMResult (duck-typed)
    axes: Optional[Tuple[Axes, Axes]] = None,
    *,
    log_y: bool = True,
    title: Optional[str] = None,
    show_best_iter: bool = True,
    legend_loc: str = "best",
) -> Tuple[Axes, Axes]:
    """
    Plot ADMM primal/dual residuals and SOC slack vs iteration.

    Parameters
    ----------
    admm_result
        :class:`ADMMResult` (or any object exposing
        ``primal_res_history``, ``dual_res_history``, optionally
        ``slack_history`` (list of arrays of length ``n_ue``), and
        optionally ``best_iter`` (int)).
    axes : tuple of two Axes, optional
        Two-panel axes to plot into.  Creates a new figure if None.
    log_y : bool
        Use log scale on both panels' y-axes.
    title : str, optional
        Suptitle for the figure.
    show_best_iter : bool
        If the result has ``best_iter`` set, mark it with a vertical
        dotted line on both panels.
    legend_loc : str
        Legend location (matplotlib convention).  Default ``"best"``
        lets matplotlib auto-pick the corner with least overlap.
        Override (e.g. ``"upper right"``) for series of figures where
        a consistent corner across panels matters more than minimal
        overlap.

    Returns
    -------
    (ax_residuals, ax_slack)
    """
    if axes is None:
        fig, axes = plt.subplots(
            1, 2, figsize=figsize("double", aspect=2.5),
            constrained_layout=True,
        )
    ax_res, ax_slack = axes

    primal = np.asarray(admm_result.primal_res_history, dtype=float)
    dual   = np.asarray(admm_result.dual_res_history,   dtype=float)
    if primal.size == 0:
        logger.warning("Empty primal_res_history; producing empty plot")
        return ax_res, ax_slack

    iters = np.arange(1, primal.size + 1)

    ax_res.plot(iters, primal, color=_PRIMAL_COLOR, marker="o",
                markersize=2.5, linewidth=1.0, label=r"$r_{\rm pri}$")
    ax_res.plot(iters, dual,   color=_DUAL_COLOR,   marker="s",
                markersize=2.5, linewidth=1.0, label=r"$r_{\rm dual}$")
    if log_y:
        ax_res.set_yscale("log")
    ax_res.set_xlabel("Iteration")
    ax_res.set_ylabel("Residual")
    ax_res.grid(True, alpha=0.3)
    ax_res.legend(loc=legend_loc)

    # Slack panel — only if slack_history is present.
    slack_hist = getattr(admm_result, "slack_history", None)
    if slack_hist is not None and len(slack_hist) > 0:
        slack = np.asarray(slack_hist, dtype=float)
        if slack.ndim > 1:
            slack = slack.max(axis=1)
        slack_plot = np.maximum(slack, 1e-30) if log_y else slack
        ax_slack.plot(iters[:slack_plot.size], slack_plot,
                      color=_SLACK_COLOR, marker="^",
                      markersize=2.5, linewidth=1.0,
                      label=r"$\max_u \varepsilon_u$")
        if log_y:
            ax_slack.set_yscale("log")
        ax_slack.set_xlabel("Iteration")
        ax_slack.set_ylabel("SOC slack")
        ax_slack.grid(True, alpha=0.3)
        ax_slack.legend(loc=legend_loc)
    else:
        ax_slack.text(0.5, 0.5, "slack_history not available",
                      ha="center", va="center",
                      transform=ax_slack.transAxes,
                      fontsize=7, alpha=0.6)
        ax_slack.set_xlabel("Iteration")
        ax_slack.set_xticks([])
        ax_slack.set_yticks([])

    # Optional best-iter marker.
    if show_best_iter and hasattr(admm_result, "best_iter"):
        bi = getattr(admm_result, "best_iter", None)
        if bi is not None and bi > 0:
            for a in (ax_res, ax_slack):
                a.axvline(bi, color="0.3", linestyle=":",
                          linewidth=0.8, alpha=0.7)

    if title:
        ax_res.figure.suptitle(title)

    return ax_res, ax_slack

