"""
CORDIS plotting helpers — Stage 8a.

Modules:

* :mod:`cordis.plotting.style`       — matplotlib rcParams + per-algorithm style
* :mod:`cordis.plotting._output`     — :func:`save_figure` (PDF + PGF + metadata)
* :mod:`cordis.plotting.cdf`         — :func:`plot_cdf`
* :mod:`cordis.plotting.sweep`       — :func:`plot_sweep`
* :mod:`cordis.plotting.convergence` — :func:`plot_admm_convergence`
* :mod:`cordis.plotting.comparison`  — :func:`bar_chart`,
                                       :func:`to_markdown_table`,
                                       :func:`to_latex_table`

Typical usage
-------------
.. code-block:: python

    from cordis.plotting import (
        apply_paper_style, figsize, style_for, save_figure,
        plot_cdf, plot_sweep, plot_admm_convergence,
        bar_chart, to_markdown_table, to_latex_table,
    )
    import matplotlib.pyplot as plt

    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize("single"))
    plot_cdf(sim_result, "min_sinr_db",
             ax=ax, xlabel=r"min-SINR [dB]")
    save_figure(fig, "figures/sinr_cdf")
"""
from cordis.plotting.style import (
    ALGORITHM_STYLE,
    GOLDEN_RATIO,
    WIDTHS,
    apply_paper_style,
    figsize,
    register_algorithm_style,
    style_for,
)
from cordis.plotting._output import save_figure
from cordis.plotting.cdf import plot_cdf
from cordis.plotting.sweep import plot_sweep
from cordis.plotting.convergence import plot_admm_convergence
from cordis.plotting.comparison import (
    bar_chart,
    to_markdown_table,
    to_latex_table,
)

__all__ = [
    # style
    "apply_paper_style",
    "figsize",
    "WIDTHS",
    "GOLDEN_RATIO",
    "style_for",
    "register_algorithm_style",
    "ALGORITHM_STYLE",
    # output
    "save_figure",
    # plot helpers
    "plot_cdf",
    "plot_sweep",
    "plot_admm_convergence",
    "bar_chart",
    "to_markdown_table",
    "to_latex_table",
]

