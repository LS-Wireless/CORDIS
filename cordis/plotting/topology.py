"""
cordis/plotting/topology.py
============================
Topology visualization tools for the CORDIS simulation framework.

(Moved from ``cordis/visualization/topology_viz.py`` in Stage 8a so all
matplotlib-based figure rendering lives under one module; the network
"system model" figure is paper content and deserves the same provenance
pipeline as the data plots.)

Provides clean, publication-quality plots of the network layout and
large-scale channel statistics, all callable with a single function.

Available plots
---------------
plot_topology_2d(topo, ...)
    2-D bird's-eye view of APs, UEs, and targets with optional coverage
    circles and AP-to-UE link lines.

plot_topology_3d(topo, ...)
    Interactive 3-D scatter plot of the same entities.

plot_lsf_heatmap(topo, lsf, ...)
    Heatmap of large-scale fading β_{au} [dB] for every AP-UE pair.

plot_los_matrix(topo, lsf, ...)
    Binary LoS/NLoS state matrix for every AP-UE pair.

plot_pathloss_vs_distance(topo, lsf, ...)
    Scatter plot of path loss [dB] vs. 2-D distance for LoS and NLoS links.

plot_topology_summary(topo, lsf, ...)
    2×2 dashboard combining the four most useful plots in one figure.

Save behaviour
--------------
All functions accept an optional ``save_path`` argument.  Format is
inferred from the path's extension:

* ``.pdf`` / no extension → routed through
  :func:`cordis.plotting.save_figure` so the file embeds CORDIS
  provenance (Creator, GitSHA, CreationDate, Subject="topology").
  Also produces a sidecar ``.pgf`` when LaTeX is available.
* ``.png`` / ``.svg`` / ``.eps`` / etc. → legacy ``fig.savefig`` path
  (no metadata; these formats don't carry a Keywords field).

The custom topology palette is intentionally distinct from the
algorithm-curve palette used by ``plot_cdf`` / ``plot_sweep`` — system
diagrams and statistical plots have different visual conventions.  For
fonts to match the rest of the paper, call ``apply_paper_style()``
before invoking these functions.

Usage example
-------------
    from cordis.utils.config import load_config
    from cordis.utils.io_utils import make_rng
    from cordis.channel.topology import generate_topology
    from cordis.channel.pathloss import compute_large_scale_fading
    from cordis.plotting import apply_paper_style
    from cordis.plotting.topology import plot_topology_2d, plot_topology_summary

    cfg  = load_config("configs/default.json")
    rng  = make_rng(42)
    topo = generate_topology(cfg, rng)
    lsf  = compute_large_scale_fading(topo, cfg, rng)

    apply_paper_style()                # optional: paper-style fonts
    plot_topology_2d(topo, title="Trial 1")
    plot_topology_summary(topo, lsf,
                          save_path="figures/topology/trial1")  # PDF + PGF
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from cordis.channel.topology import NetworkTopology
from cordis.plotting._output import save_figure
from cordis.utils.logger import get_logger

logger = get_logger(__name__)

# ── Colour palette (consistent across all plots) ───────────────────────────────
_C = {
    "tx_ap":    "#2563EB",   # blue  — transmit AP
    "rx_ap":    "#DC2626",   # red   — receive (sensing) AP
    "ue":       "#16A34A",   # green — UE
    "target":   "#D97706",   # amber — target
    "link_los": "#93C5FD",   # light blue — LoS link
    "link_nlos":"#CBD5E1",   # slate      — NLoS link
    "bg":       "#F8FAFC",   # near-white background
    "grid":     "#E2E8F0",   # light grid lines
}

_MARKER = {
    "tx_ap": "^",    # upward triangle
    "rx_ap": "v",    # downward triangle
    "ue":    "o",    # circle
    "target":"*",    # star
}

_SIZE = {
    "tx_ap": 120,
    "rx_ap": 120,
    "ue":    80,
    "target":200,
}


# =============================================================================
# Internal helpers
# =============================================================================

def _save_or_show(
    fig: plt.Figure,
    save_path: Optional[Union[str, Path]],
    dpi: int = 150,
) -> None:
    """
    Save figure to disk if ``save_path`` is given, then show.

    Format is inferred from the path's extension:

    * ``.pdf`` or no extension → routed through
      :func:`cordis.plotting.save_figure` for provenance metadata
      (Creator, GitSHA, CreationDate, Subject="topology"); also
      produces a ``.pgf`` sidecar when LaTeX is available.
    * Any other extension (``.png``, ``.svg``, ``.eps``, …) → plain
      ``fig.savefig`` since those formats don't carry a Keywords
      field that metadata can be embedded in.
    """
    if save_path is not None:
        p = Path(save_path)
        ext = p.suffix.lower().lstrip(".")
        if ext in {"pdf", "pgf", ""}:
            # PDF / PGF — use the new pipeline so the figure carries
            # CORDIS provenance metadata.  Default to PDF when caller
            # didn't pick an extension.
            formats = (ext,) if ext else ("pdf",)
            try:
                paths = save_figure(
                    fig,
                    p.with_suffix(""),                  # save_figure adds the ext
                    formats=formats,
                    metadata={"Subject": "topology"},
                )
                for sp in paths:
                    logger.info("Figure saved → %s", sp.resolve())
            except Exception as e:
                # PGF requires LaTeX; if it isn't available, fall back
                # to PDF only rather than failing the whole save.
                if "pgf" in formats and len(formats) > 1:
                    logger.warning(
                        "PGF save failed (%s); retrying PDF only.", e
                    )
                    paths = save_figure(
                        fig, p.with_suffix(""),
                        formats=("pdf",),
                        metadata={"Subject": "topology"},
                    )
                    for sp in paths:
                        logger.info("Figure saved → %s", sp.resolve())
                else:
                    raise
        else:
            # Legacy path for PNG / SVG / EPS / etc.
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            logger.info("Figure saved → %s", p.resolve())
    plt.show()


def _ap_colour(ap) -> str:
    return _C["rx_ap"] if ap.is_receive else _C["tx_ap"]


def _ap_marker(ap) -> str:
    return _MARKER["rx_ap"] if ap.is_receive else _MARKER["tx_ap"]


def _ap_label(ap) -> str:
    mode = "Rx" if ap.is_receive else "Tx"
    return f"AP{ap.idx}({mode})"


def _legend_handles() -> list:
    """Return a fixed set of legend handles for entity types."""
    handles = [
        mpatches.Patch(color=_C["tx_ap"],  label="Transmit AP"),
        mpatches.Patch(color=_C["rx_ap"],  label="Receive AP (sensing)"),
        mpatches.Patch(color=_C["ue"],     label="UE"),
        mpatches.Patch(color=_C["target"], label="Sensing target"),
    ]
    return handles


# =============================================================================
# plot_topology_2d
# =============================================================================

def plot_topology_2d(
    topo: NetworkTopology,
    lsf=None,
    title: str = "Network Topology",
    show_labels: bool = True,
    show_links: bool = False,
    show_coverage_circle: bool = True,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    2-D bird's-eye view of the network topology.

    Parameters
    ----------
    topo : NetworkTopology
    lsf : LargeScaleFading or None
        If supplied, LoS links are drawn in blue and NLoS in grey.
    title : str
    show_labels : bool
        Annotate each entity with its index.
    show_links : bool
        Draw lines from every TX AP to every UE.
    show_coverage_circle : bool
        Draw a dashed circle indicating the UE placement boundary.
    figsize : tuple
    save_path : str or Path or None
        If given, saves the figure to this path.
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["bg"])
    ax.set_facecolor(_C["bg"])
    ax.grid(True, color=_C["grid"], linewidth=0.6, zorder=0)
    ax.set_aspect("equal")

    # ── AP-UE link lines ──────────────────────────────────────────────────
    if show_links:
        for ap in topo.tx_aps:
            for ue in topo.ues:
                if lsf is not None:
                    is_los = bool(lsf.los_state[ap.idx, ue.idx])
                    lc = _C["link_los"] if is_los else _C["link_nlos"]
                    lw = 0.8 if is_los else 0.4
                else:
                    lc, lw = _C["link_nlos"], 0.4
                ax.plot(
                    [ap.pos[0], ue.pos[0]],
                    [ap.pos[1], ue.pos[1]],
                    color=lc, linewidth=lw, zorder=1, alpha=0.7,
                )

    # ── Coverage boundary circle ──────────────────────────────────────────
    if show_coverage_circle:
        ue_radii = np.linalg.norm(topo.ue_positions[:, :2], axis=1)
        r_max = float(ue_radii.max()) * 1.05 if len(ue_radii) else 1000.0
        circle = plt.Circle(
            (0, 0), r_max, fill=False,
            linestyle="--", linewidth=0.8, color="#94A3B8", zorder=1,
        )
        ax.add_patch(circle)

    # ── Plot entities ─────────────────────────────────────────────────────
    for ap in topo.aps:
        color  = _ap_colour(ap)
        marker = _ap_marker(ap)
        ax.scatter(
            ap.pos[0], ap.pos[1],
            c=color, marker=marker, s=_SIZE["tx_ap"],
            zorder=4, edgecolors="white", linewidths=0.8,
        )
        if show_labels:
            ax.annotate(
                _ap_label(ap),
                xy=(ap.pos[0], ap.pos[1]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=7, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=5,
            )

    for ue in topo.ues:
        ax.scatter(
            ue.pos[0], ue.pos[1],
            c=_C["ue"], marker=_MARKER["ue"], s=_SIZE["ue"],
            zorder=3, edgecolors="white", linewidths=0.8,
        )
        if show_labels:
            ax.annotate(
                f"UE{ue.idx}",
                xy=(ue.pos[0], ue.pos[1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color=_C["ue"],
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=5,
            )

    for tg in topo.targets:
        ax.scatter(
            tg.pos[0], tg.pos[1],
            c=_C["target"], marker=_MARKER["target"], s=_SIZE["target"],
            zorder=3, edgecolors="white", linewidths=0.8,
        )
        if show_labels:
            ax.annotate(
                f"T{tg.idx}",
                xy=(tg.pos[0], tg.pos[1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color=_C["target"], fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=5,
            )

    # ── Legend & labels ───────────────────────────────────────────────────
    ax.legend(
        handles=_legend_handles(),
        loc="upper right", fontsize=8, framealpha=0.9,
        edgecolor=_C["grid"],
    )
    ax.set_xlabel("x  [m]", fontsize=9)
    ax.set_ylabel("y  [m]", fontsize=9)
    ax.set_title(
        f"{title}\n"
        f"{topo.n_ap} APs (Tx={topo.n_tx}, Rx={topo.n_rx})  |  "
        f"{topo.n_ue} UEs  |  {topo.n_targets} Targets",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# =============================================================================
# plot_topology_3d
# =============================================================================

def plot_topology_3d(
    topo: NetworkTopology,
    title: str = "Network Topology (3-D)",
    show_labels: bool = True,
    figsize: Tuple[float, float] = (9, 7),
    elev: float = 25.0,
    azim: float = -60.0,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Interactive 3-D scatter plot of the network topology.

    Parameters
    ----------
    topo : NetworkTopology
    title : str
    show_labels : bool
    figsize : tuple
    elev, azim : float
        Initial elevation and azimuth angles for the 3-D view.
    save_path : str or Path or None
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor=_C["bg"])
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_C["bg"])

    for ap in topo.aps:
        color  = _ap_colour(ap)
        marker = _ap_marker(ap)
        ax.scatter(
            ap.pos[0], ap.pos[1], ap.pos[2],
            c=color, marker=marker, s=_SIZE["tx_ap"],
            depthshade=True, edgecolors="white", linewidths=0.5, zorder=4,
        )
        if show_labels:
            ax.text(
                ap.pos[0], ap.pos[1], ap.pos[2] + 5,
                _ap_label(ap), fontsize=7, color=color,
            )

    for ue in topo.ues:
        ax.scatter(
            ue.pos[0], ue.pos[1], ue.pos[2],
            c=_C["ue"], marker=_MARKER["ue"], s=_SIZE["ue"],
            depthshade=True, edgecolors="white", linewidths=0.5,
        )
        if show_labels:
            ax.text(
                ue.pos[0], ue.pos[1], ue.pos[2] + 4,
                f"UE{ue.idx}", fontsize=7, color=_C["ue"],
            )

    for tg in topo.targets:
        ax.scatter(
            tg.pos[0], tg.pos[1], tg.pos[2],
            c=_C["target"], marker=_MARKER["target"], s=_SIZE["target"],
            depthshade=True, edgecolors="white", linewidths=0.5,
        )
        if show_labels:
            ax.text(
                tg.pos[0], tg.pos[1], tg.pos[2] + 4,
                f"T{tg.idx}", fontsize=7, color=_C["target"],
            )

    ax.set_xlabel("x [m]", fontsize=8, labelpad=6)
    ax.set_ylabel("y [m]", fontsize=8, labelpad=6)
    ax.set_zlabel("z [m]", fontsize=8, labelpad=6)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, pad=12)
    ax.legend(handles=_legend_handles(), fontsize=8, loc="upper left")
    fig.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# =============================================================================
# plot_lsf_heatmap
# =============================================================================

def plot_lsf_heatmap(
    topo: NetworkTopology,
    lsf,
    title: str = "Large-Scale Fading  β [dB]",
    figsize: Tuple[float, float] = (7, 5),
    cmap: str = "RdYlGn",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Heatmap of large-scale fading coefficient β_{au} [dB] for all AP-UE pairs.

    Rows = APs (transmit only), Columns = UEs.
    Green = strong link, Red = weak link.

    Parameters
    ----------
    topo : NetworkTopology
    lsf  : LargeScaleFading
    title : str
    figsize : tuple
    cmap : str
        Matplotlib colormap name.
    save_path : str or Path or None
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_tx = topo.n_tx
    n_ue = topo.n_ue

    # Build matrix: rows = TX APs, cols = UEs
    beta_db = np.zeros((n_tx, n_ue))
    tx_labels = []
    for row, ap in enumerate(topo.tx_aps):
        tx_labels.append(_ap_label(ap))
        for col, ue in enumerate(topo.ues):
            b_lin = lsf.beta_lin[ap.idx, ue.idx]
            beta_db[row, col] = 10 * np.log10(max(b_lin, 1e-20))

    ue_labels = [f"UE{u.idx}" for u in topo.ues]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["bg"])
    im = ax.imshow(beta_db, aspect="auto", cmap=cmap)

    # Axis ticks and labels
    ax.set_xticks(range(n_ue))
    ax.set_xticklabels(ue_labels, fontsize=9)
    ax.set_yticks(range(n_tx))
    ax.set_yticklabels(tx_labels, fontsize=9)
    ax.set_xlabel("User Equipment", fontsize=9)
    ax.set_ylabel("Transmit AP", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)

    # Cell annotations
    for i in range(n_tx):
        for j in range(n_ue):
            val = beta_db[i, j]
            # White text on dark cells, dark text on light cells
            norm_val = (val - beta_db.min()) / (beta_db.max() - beta_db.min() + 1e-12)
            txt_color = "white" if norm_val < 0.4 else "black"
            ax.text(
                j, i, f"{val:.0f}",
                ha="center", va="center",
                fontsize=8, color=txt_color, fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("β  [dB]", fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# =============================================================================
# plot_los_matrix
# =============================================================================

def plot_los_matrix(
    topo: NetworkTopology,
    lsf,
    title: str = "LoS / NLoS State",
    figsize: Tuple[float, float] = (7, 5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Binary matrix showing the LoS (blue) / NLoS (grey) state for every
    transmit AP-UE pair.

    Parameters
    ----------
    topo : NetworkTopology
    lsf  : LargeScaleFading
    title : str
    figsize : tuple
    save_path : str or Path or None
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_tx = topo.n_tx
    n_ue = topo.n_ue

    los_mat = np.zeros((n_tx, n_ue))
    tx_labels = []
    for row, ap in enumerate(topo.tx_aps):
        tx_labels.append(_ap_label(ap))
        for col, ue in enumerate(topo.ues):
            los_mat[row, col] = float(lsf.los_state[ap.idx, ue.idx])

    ue_labels = [f"UE{u.idx}" for u in topo.ues]

    # Custom 2-colour map: 0 = NLoS (slate), 1 = LoS (blue)
    cmap = matplotlib.colors.ListedColormap(["#CBD5E1", "#2563EB"])

    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["bg"])
    ax.imshow(los_mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n_ue))
    ax.set_xticklabels(ue_labels, fontsize=9)
    ax.set_yticks(range(n_tx))
    ax.set_yticklabels(tx_labels, fontsize=9)
    ax.set_xlabel("User Equipment", fontsize=9)
    ax.set_ylabel("Transmit AP", fontsize=9)
    ax.set_title(
        f"{title}  "
        f"(LoS fraction = {lsf.los_state.mean():.0%})",
        fontsize=10, pad=10,
    )

    for i in range(n_tx):
        for j in range(n_ue):
            txt = "LoS" if los_mat[i, j] else "NLoS"
            txt_color = "white" if los_mat[i, j] else "#475569"
            ax.text(
                j, i, txt,
                ha="center", va="center", fontsize=8, color=txt_color,
            )

    # Manual legend
    legend_handles = [
        mpatches.Patch(color="#2563EB", label="LoS"),
        mpatches.Patch(color="#CBD5E1", label="NLoS"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.9)
    fig.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# =============================================================================
# plot_pathloss_vs_distance
# =============================================================================

def plot_pathloss_vs_distance(
    topo: NetworkTopology,
    lsf,
    title: str = "Path Loss vs. 2-D Distance",
    figsize: Tuple[float, float] = (7, 5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Scatter plot of path loss [dB] vs. 2-D AP-UE distance.

    LoS and NLoS links are plotted with different markers and colours.

    Parameters
    ----------
    topo : NetworkTopology
    lsf  : LargeScaleFading
    title : str
    figsize : tuple
    save_path : str or Path or None
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    d2d = topo.ap_ue_distances_2d()   # (N_ap, N_ue)

    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["bg"])
    ax.set_facecolor(_C["bg"])
    ax.grid(True, color=_C["grid"], linewidth=0.6, zorder=0)

    pl_los_x,  pl_los_y  = [], []
    pl_nlos_x, pl_nlos_y = [], []

    for ap in topo.tx_aps:
        for ue in topo.ues:
            d   = d2d[ap.idx, ue.idx]
            pl  = lsf.path_loss_db[ap.idx, ue.idx]
            if lsf.los_state[ap.idx, ue.idx]:
                pl_los_x.append(d);  pl_los_y.append(pl)
            else:
                pl_nlos_x.append(d); pl_nlos_y.append(pl)

    if pl_los_x:
        ax.scatter(pl_los_x, pl_los_y, c=_C["tx_ap"], marker="o",
                   s=50, label="LoS", zorder=3, alpha=0.85,
                   edgecolors="white", linewidths=0.5)
    if pl_nlos_x:
        ax.scatter(pl_nlos_x, pl_nlos_y, c="#94A3B8", marker="s",
                   s=50, label="NLoS", zorder=3, alpha=0.85,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("2-D distance  [m]", fontsize=9)
    ax.set_ylabel("Path loss  [dB]", fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save_or_show(fig, save_path, dpi)
    return fig


# =============================================================================
# plot_topology_summary  (2×2 dashboard)
# =============================================================================

def plot_topology_summary(
    topo: NetworkTopology,
    lsf=None,
    title: str = "CORDIS Topology Summary",
    figsize: Tuple[float, float] = (14, 11),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    2×2 dashboard combining the four most useful topology views:
      - Top-left  : 2-D topology map
      - Top-right : β [dB] heatmap        (requires lsf)
      - Bot-left  : LoS/NLoS state matrix (requires lsf)
      - Bot-right : Path loss vs. distance (requires lsf)

    If ``lsf`` is None, only the 2-D topology map is populated and the
    other three panels show a placeholder message.

    Parameters
    ----------
    topo : NetworkTopology
    lsf  : LargeScaleFading or None
    title : str
    figsize : tuple
    save_path : str or Path or None
    dpi : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, facecolor=_C["bg"])
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_topo = fig.add_subplot(gs[0, 0])
    ax_lsf  = fig.add_subplot(gs[0, 1])
    ax_los  = fig.add_subplot(gs[1, 0])
    ax_pl   = fig.add_subplot(gs[1, 1])

    for ax in (ax_topo, ax_lsf, ax_los, ax_pl):
        ax.set_facecolor(_C["bg"])

    # ── Top-left: 2-D topology ────────────────────────────────────────────
    _draw_topology_on_ax(ax_topo, topo, lsf,
                         show_links=(lsf is not None),
                         show_labels=True)

    # ── Top-right: β heatmap ──────────────────────────────────────────────
    if lsf is not None:
        _draw_lsf_heatmap_on_ax(ax_lsf, topo, lsf)
    else:
        ax_lsf.text(0.5, 0.5, "LSF not provided",
                    ha="center", va="center", transform=ax_lsf.transAxes,
                    fontsize=11, color="#94A3B8")
        ax_lsf.set_title("Large-Scale Fading  β [dB]", fontsize=10)

    # ── Bot-left: LoS/NLoS matrix ─────────────────────────────────────────
    if lsf is not None:
        _draw_los_matrix_on_ax(ax_los, topo, lsf)
    else:
        ax_los.text(0.5, 0.5, "LSF not provided",
                    ha="center", va="center", transform=ax_los.transAxes,
                    fontsize=11, color="#94A3B8")
        ax_los.set_title("LoS / NLoS State", fontsize=10)

    # ── Bot-right: Path loss vs. distance ─────────────────────────────────
    if lsf is not None:
        _draw_pathloss_scatter_on_ax(ax_pl, topo, lsf)
    else:
        ax_pl.text(0.5, 0.5, "LSF not provided",
                   ha="center", va="center", transform=ax_pl.transAxes,
                   fontsize=11, color="#94A3B8")
        ax_pl.set_title("Path Loss vs. Distance", fontsize=10)

    _save_or_show(fig, save_path, dpi)
    return fig


# =============================================================================
# Internal drawing helpers (draw onto a pre-existing Axes)
# =============================================================================

def _draw_topology_on_ax(ax, topo, lsf=None,
                          show_links=False, show_labels=True):
    """Draw the 2-D topology onto an existing Axes object."""
    ax.set_aspect("equal")
    ax.grid(True, color=_C["grid"], linewidth=0.5, zorder=0)

    if show_links:
        for ap in topo.tx_aps:
            for ue in topo.ues:
                if lsf is not None:
                    is_los = bool(lsf.los_state[ap.idx, ue.idx])
                    lc = _C["link_los"] if is_los else _C["link_nlos"]
                    lw = 0.7 if is_los else 0.3
                else:
                    lc, lw = _C["link_nlos"], 0.3
                ax.plot([ap.pos[0], ue.pos[0]],
                        [ap.pos[1], ue.pos[1]],
                        color=lc, lw=lw, zorder=1, alpha=0.6)

    for ap in topo.aps:
        ax.scatter(ap.pos[0], ap.pos[1],
                   c=_ap_colour(ap), marker=_ap_marker(ap),
                   s=_SIZE["tx_ap"], zorder=4,
                   edgecolors="white", linewidths=0.7)
        if show_labels:
            ax.annotate(
                _ap_label(ap), xy=(ap.pos[0], ap.pos[1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=6.5, color=_ap_colour(ap), fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    for ue in topo.ues:
        ax.scatter(ue.pos[0], ue.pos[1],
                   c=_C["ue"], marker=_MARKER["ue"],
                   s=_SIZE["ue"], zorder=3,
                   edgecolors="white", linewidths=0.7)
        if show_labels:
            ax.annotate(f"UE{ue.idx}", xy=(ue.pos[0], ue.pos[1]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, color=_C["ue"],
                        path_effects=[pe.withStroke(linewidth=2,
                                                     foreground="white")])

    for tg in topo.targets:
        ax.scatter(tg.pos[0], tg.pos[1],
                   c=_C["target"], marker=_MARKER["target"],
                   s=_SIZE["target"], zorder=3,
                   edgecolors="white", linewidths=0.7)
        if show_labels:
            ax.annotate(f"T{tg.idx}", xy=(tg.pos[0], tg.pos[1]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, color=_C["target"], fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2,
                                                     foreground="white")])

    ax.legend(handles=_legend_handles(), fontsize=7,
              loc="upper right", framealpha=0.9)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.set_title(
        f"Network Layout  |  {topo.n_ap} APs · {topo.n_ue} UEs · "
        f"{topo.n_targets} Targets",
        fontsize=9,
    )


def _draw_lsf_heatmap_on_ax(ax, topo, lsf, cmap="RdYlGn"):
    """Draw β [dB] heatmap onto an existing Axes."""
    n_tx = topo.n_tx
    n_ue = topo.n_ue
    beta_db = np.zeros((n_tx, n_ue))
    tx_labels = []
    for row, ap in enumerate(topo.tx_aps):
        tx_labels.append(_ap_label(ap))
        for col, ue in enumerate(topo.ues):
            b = lsf.beta_lin[ap.idx, ue.idx]
            beta_db[row, col] = 10 * np.log10(max(b, 1e-20))

    ue_labels = [f"UE{u.idx}" for u in topo.ues]
    im = ax.imshow(beta_db, aspect="auto", cmap=cmap)
    ax.set_xticks(range(n_ue)); ax.set_xticklabels(ue_labels, fontsize=8)
    ax.set_yticks(range(n_tx)); ax.set_yticklabels(tx_labels, fontsize=8)

    for i in range(n_tx):
        for j in range(n_ue):
            val = beta_db[i, j]
            norm_val = (val - beta_db.min()) / (beta_db.max() - beta_db.min() + 1e-12)
            tc = "white" if norm_val < 0.4 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7.5, color=tc, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label("β [dB]",
                                                               fontsize=8)
    ax.set_xlabel("User Equipment", fontsize=8)
    ax.set_ylabel("Transmit AP", fontsize=8)
    ax.set_title("Large-Scale Fading  β [dB]", fontsize=9)


def _draw_los_matrix_on_ax(ax, topo, lsf):
    """Draw LoS/NLoS binary matrix onto an existing Axes."""
    n_tx = topo.n_tx
    n_ue = topo.n_ue
    los_mat = np.zeros((n_tx, n_ue))
    tx_labels = []
    for row, ap in enumerate(topo.tx_aps):
        tx_labels.append(_ap_label(ap))
        for col, ue in enumerate(topo.ues):
            los_mat[row, col] = float(lsf.los_state[ap.idx, ue.idx])

    ue_labels = [f"UE{u.idx}" for u in topo.ues]
    cmap = matplotlib.colors.ListedColormap(["#CBD5E1", "#2563EB"])
    ax.imshow(los_mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n_ue)); ax.set_xticklabels(ue_labels, fontsize=8)
    ax.set_yticks(range(n_tx)); ax.set_yticklabels(tx_labels, fontsize=8)

    for i in range(n_tx):
        for j in range(n_ue):
            txt = "LoS" if los_mat[i, j] else "NLoS"
            tc  = "white" if los_mat[i, j] else "#475569"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=tc)

    legend_handles = [
        mpatches.Patch(color="#2563EB", label="LoS"),
        mpatches.Patch(color="#CBD5E1", label="NLoS"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
              framealpha=0.9)
    ax.set_xlabel("User Equipment", fontsize=8)
    ax.set_ylabel("Transmit AP", fontsize=8)
    ax.set_title(
        f"LoS / NLoS State  (LoS = {lsf.los_state.mean():.0%})",
        fontsize=9,
    )


def _draw_pathloss_scatter_on_ax(ax, topo, lsf):
    """Draw path loss vs. distance scatter onto an existing Axes."""
    ax.grid(True, color=_C["grid"], linewidth=0.5, zorder=0)
    d2d = topo.ap_ue_distances_2d()
    los_x, los_y, nlos_x, nlos_y = [], [], [], []
    for ap in topo.tx_aps:
        for ue in topo.ues:
            d  = d2d[ap.idx, ue.idx]
            pl = lsf.path_loss_db[ap.idx, ue.idx]
            if lsf.los_state[ap.idx, ue.idx]:
                los_x.append(d);  los_y.append(pl)
            else:
                nlos_x.append(d); nlos_y.append(pl)

    if los_x:
        ax.scatter(los_x, los_y, c=_C["tx_ap"], marker="o", s=40,
                   label="LoS", alpha=0.85, edgecolors="white",
                   linewidths=0.4, zorder=3)
    if nlos_x:
        ax.scatter(nlos_x, nlos_y, c="#94A3B8", marker="s", s=40,
                   label="NLoS", alpha=0.85, edgecolors="white",
                   linewidths=0.4, zorder=3)

    ax.set_xlabel("2-D distance [m]", fontsize=8)
    ax.set_ylabel("Path loss [dB]", fontsize=8)
    ax.set_title("Path Loss vs. 2-D Distance", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)

