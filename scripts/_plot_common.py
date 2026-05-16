"""
scripts/_plot_common.py
=======================

Shared helpers for the eleven ``scripts/plot_<name>.py`` figure
generators.

Each per-experiment wrapper:

1. parses a uniform plot CLI (``--exp-dir`` to override the auto-latest
   result, ``--output-dir`` for the figure target, ``--formats`` for
   PDF / PGF / PNG output);
2. loads the saved :class:`ExperimentResult`;
3. dispatches to one or more ``cordis.plotting.*`` helpers;
4. saves figures via :func:`cordis.plotting.save_figure` (which writes
   matched PDF + PGF + tex metadata pairs by default).

Per-experiment wrappers stay ~30 lines because all CLI, IO, and
styling logic lives here.  See ``plot_sinr_cdf.py`` for the canonical
single-figure example, ``plot_convergence_trace.py`` for a
multi-axes case, and ``plot_fronthaul_table.py`` for the table case.

Usage in a per-experiment wrapper::

    # scripts/plot_sinr_cdf.py
    from _plot_common import build_plot_parser, load_result, save_paper_figure
    from cordis.plotting import plot_cdf, apply_paper_style
    import matplotlib.pyplot as plt

    args   = build_plot_parser("sinr_cdf").parse_args()
    result = load_result("sinr_cdf", args)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    plot_cdf(result.sim_result, metric="min_sinr_db",
             xlabel=r"min-SINR [dB]", ax=ax)
    save_paper_figure(fig, "sinr_cdf", args)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

log = logging.getLogger(__name__)


def _import_cordis():
    from cordis.experiments import ExperimentResult, figure_dir, latest_result
    from cordis.plotting import save_figure
    return {
        "ExperimentResult": ExperimentResult,
        "figure_dir":       figure_dir,
        "latest_result":    latest_result,
        "save_figure":      save_figure,
    }


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_plot_parser(experiment_name: str) -> argparse.ArgumentParser:
    """Standard plot-script CLI."""
    p = argparse.ArgumentParser(
        prog=f"plot_{experiment_name}",
        description=f"Render paper figures for the {experiment_name!r} "
                    f"experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp-dir", default=None,
                   help="Specific results/exp_<name>/<ts>/ to load. "
                        "If omitted, the most recent run is used.")
    p.add_argument("--results-root", default="results",
                   help="Where to look for results/exp_<name>/ runs.")
    p.add_argument("--output-dir", default=None,
                   help="Where to save figures. Default: "
                        "figures/exp_<name>/.")
    p.add_argument("--figures-root", default="figures",
                   help="Where to put figures/exp_<name>/ when "
                        "--output-dir is not given.")
    p.add_argument("--formats", default="pdf,pgf",
                   help="Comma-separated output formats (pdf,pgf,png,svg).")
    p.add_argument("--no-tex-metadata", action="store_true",
                   help="Skip the .tex metadata sidecar.")
    return p


# ─────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────

def load_result(experiment_name: str, args: argparse.Namespace):
    """Load :class:`ExperimentResult` from --exp-dir or latest run."""
    cordis = _import_cordis()
    if args.exp_dir:
        exp_path = Path(args.exp_dir)
    else:
        exp_path = cordis["latest_result"](experiment_name,
                                           root=args.results_root)
        if exp_path is None:
            raise FileNotFoundError(
                f"No saved runs found for experiment {experiment_name!r} "
                f"under {args.results_root}/exp_{experiment_name}/. "
                f"Run scripts/exp_{experiment_name}.sh first."
            )
    log.info("Loading result from %s", exp_path)
    return cordis["ExperimentResult"].load(exp_path)


def resolve_output_dir(experiment_name: str,
                       args: argparse.Namespace) -> Path:
    """Where figures land."""
    cordis = _import_cordis()
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = cordis["figure_dir"](experiment_name, root=args.figures_root)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_paper_figure(fig,
                      base_name: str,
                      args: argparse.Namespace,
                      experiment_name: Optional[str] = None,
                      metadata: Optional[dict] = None) -> List[Path]:
    """Save a figure in all the requested formats via cordis.plotting.

    ``base_name`` is the filename stem (no extension); the figure is
    written under ``<output-dir>/<base_name>.<ext>`` for each ext
    listed in ``--formats``.
    """
    cordis = _import_cordis()
    if experiment_name is None:
        # Inferred from the parser prog: "plot_<name>".
        prog = args._exp_name if hasattr(args, "_exp_name") else None
        experiment_name = prog or "unknown"

    out_dir   = resolve_output_dir(experiment_name, args)
    base_path = out_dir / base_name
    formats   = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    md = {"experiment": experiment_name, "figure": base_name}
    if metadata:
        md.update(metadata)

    paths = cordis["save_figure"](
        fig, base_path,
        formats=formats,
        metadata=None if args.no_tex_metadata else md,
    )
    for p in paths:
        log.info("Wrote %s", p)
    return paths


# ─────────────────────────────────────────────────────────────────────
# Sweep helpers (used by 7 of the 11 plot scripts)
# ─────────────────────────────────────────────────────────────────────

def sweep_plot_pair(result,
                    metric_top: str,
                    metric_bot: str,
                    ylabel_top: str,
                    ylabel_bot: str,
                    xlabel: Optional[str] = None,
                    only: Optional[Sequence[str]] = None,
                    log_x: bool = False):
    """Stacked top/bottom panels: one metric each vs sweep axis.

    The most common figure layout in the paper for sweep experiments:
    min-SINR on top, sum-SCNR on bottom, sharing the x-axis.
    """
    import matplotlib.pyplot as plt
    from cordis.plotting import apply_paper_style, figsize, plot_sweep

    apply_paper_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5/2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )
    axis = result.sweep_axis
    plot_sweep(result.sweep_results, metric=metric_top, ax=ax_top,
               xlabel="", ylabel=ylabel_top, only=only, log_x=log_x)
    plot_sweep(result.sweep_results, metric=metric_bot, ax=ax_bot,
               xlabel=xlabel or axis.display, ylabel=ylabel_bot,
               only=only, log_x=log_x)
    # Don't duplicate the legend.
    if ax_bot.get_legend() is not None:
        ax_bot.get_legend().remove()
    return fig, (ax_top, ax_bot)

