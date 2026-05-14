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
* :mod:`cordis.plotting.topology`    — :func:`plot_topology_2d`,
                                       :func:`plot_topology_3d`,
                                       :func:`plot_lsf_heatmap`,
                                       :func:`plot_los_matrix`,
                                       :func:`plot_pathloss_vs_distance`,
                                       :func:`plot_topology_summary`

(``cordis.plotting.topology`` was moved from ``cordis.visualization``
in Stage 8a so all matplotlib output lives in one module.)

Typical usage
-------------
.. code-block:: python

    from cordis.plotting import (
        apply_paper_style, figsize, style_for, save_figure,
        plot_cdf, plot_sweep, plot_admm_convergence,
        bar_chart, to_markdown_table, to_latex_table,
        plot_topology_2d, plot_topology_summary,
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
from cordis.plotting.topology import (
    plot_topology_2d,
    plot_topology_3d,
    plot_lsf_heatmap,
    plot_los_matrix,
    plot_pathloss_vs_distance,
    plot_topology_summary,
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
    # data plots
    "plot_cdf",
    "plot_sweep",
    "plot_admm_convergence",
    "bar_chart",
    "to_markdown_table",
    "to_latex_table",
    # topology / system-model plots
    "plot_topology_2d",
    "plot_topology_3d",
    "plot_lsf_heatmap",
    "plot_los_matrix",
    "plot_pathloss_vs_distance",
    "plot_topology_summary",
]

