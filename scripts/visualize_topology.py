"""
scripts/visualize_topology.py
==============================
Standalone script to generate and optionally save all topology visualizations.

Run from the project root:
    python scripts/visualize_topology.py
    python scripts/visualize_topology.py --save --out-dir results/figures
    python scripts/visualize_topology.py --n-ap 8 --n-ue 6 --n-targets 3
    python scripts/visualize_topology.py --seed 99 --topology random --no-show
"""

import argparse
import sys
from pathlib import Path

# ── Make sure project root is on the path regardless of CWD ───────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cordis.utils.logger import setup_logging, get_logger
from cordis.utils.config import load_config
from cordis.utils.io_utils import make_rng
from cordis.channel.topology import generate_topology
from cordis.channel.pathloss import compute_large_scale_fading
from cordis.visualization.topology_viz import (
    plot_topology_2d,
    plot_topology_3d,
    plot_lsf_heatmap,
    plot_los_matrix,
    plot_pathloss_vs_distance,
    plot_topology_summary,
)


SCRIPT_DESCRIPTION = (
    "Generate publication-quality topology plots for a CORDIS network "
    "scenario: 2-D map, 3-D scatter, large-scale fading heatmap, "
    "LoS/NLoS matrix, path-loss vs. distance, and a 2x2 summary dashboard."
)

SCRIPT_EXAMPLES = """
examples:
  # Interactive view with default config (seed 42, 6 APs, 4 UEs, 2 targets)
  python scripts/visualize_topology.py

  # Grid layout with 12 APs (auto-factorised to 3x4), save as PDF
  python scripts/visualize_topology.py --topology grid --n-ap 12 --save --fmt pdf

  # Random layout, more users, save PNG figures, no interactive window
  python scripts/visualize_topology.py --topology random --n-ap 8 --n-ue 10 \\
      --n-targets 4 --seed 7 --save --out-dir results/figures --no-show

  # Only the 2x2 summary dashboard, custom config
  python scripts/visualize_topology.py --summary-only --config configs/exp_scalability.json
"""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="visualize_topology.py",
        description=SCRIPT_DESCRIPTION,
        epilog=SCRIPT_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scenario options
    scenario = parser.add_argument_group("scenario")
    scenario.add_argument(
        "--config", default="configs/default.json", metavar="PATH",
        help="Base JSON config file (default: configs/default.json)",
    )
    scenario.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Master random seed for topology and channel generation (default: 42)",
    )
    scenario.add_argument(
        "--n-ap", type=int, default=None, metavar="INT",
        help="Number of APs.  For grid topology this is auto-factorised into "
             "the most square grid shape (e.g. 10 -> 2x5, 9 -> 3x3)",
    )
    scenario.add_argument(
        "--n-ue", type=int, default=None, metavar="INT",
        help="Number of communication users (UEs)",
    )
    scenario.add_argument(
        "--n-targets", type=int, default=None, metavar="INT",
        help="Number of sensing targets",
    )
    scenario.add_argument(
        "--topology", default=None, metavar="TYPE",
        choices=["circle", "random", "grid"],
        help="AP placement layout: circle (default) | random | grid",
    )

    # Output options
    output = parser.add_argument_group("output")
    output.add_argument(
        "--save", action="store_true",
        help="Save all figures to disk (default: display interactively only)",
    )
    output.add_argument(
        "--out-dir", default="eval/figures/topology", metavar="PATH",
        help="Directory for saved figures (default: eval/figures/topology)",
    )
    output.add_argument(
        "--fmt", default="png", choices=["png", "pdf", "svg"],
        help="File format for saved figures (default: png)",
    )
    output.add_argument(
        "--dpi", type=int, default=150, metavar="INT",
        help="Resolution in dots per inch for raster formats (default: 150)",
    )
    output.add_argument(
        "--no-show", action="store_true",
        help="Skip interactive display; useful in headless environments",
    )
    output.add_argument(
        "--summary-only", action="store_true",
        help="Generate only the 2x2 summary dashboard instead of all 6 plots",
    )

    return parser.parse_args()


def _best_grid(n: int):
    """
    Return (n_rows, n_cols) whose product equals n and whose shape is as
    close to square as possible.  Examples:
        6  -> (2, 3),  9  -> (3, 3),  10 -> (2, 5),  7  -> (1, 7)
    """
    best_rows, best_cols, best_diff = 1, n, n - 1
    for r in range(1, int(n ** 0.5) + 1):
        if n % r == 0:
            c = n // r
            if abs(c - r) < best_diff:
                best_diff = abs(c - r)
                best_rows, best_cols = r, c
    return best_rows, best_cols


def main():
    args = parse_args()
    setup_logging(level="INFO")
    logger = get_logger("visualize_topology")

    # ── Load config and apply CLI overrides ───────────────────────────────
    cfg = load_config(args.config)
    if args.n_ue      is not None: cfg.topology.n_ue      = args.n_ue
    if args.n_targets is not None: cfg.topology.n_targets = args.n_targets
    if args.topology  is not None: cfg.topology.type      = args.topology

    # For grid layout: derive grid_n_rows x grid_n_cols from --n-ap so the
    # AP count matches exactly.  For circle/random, set n_ap directly.
    if args.n_ap is not None:
        cfg.topology.n_ap = args.n_ap
        if cfg.topology.type == "grid":
            rows, cols = _best_grid(args.n_ap)
            cfg.topology.grid_n_rows = rows
            cfg.topology.grid_n_cols = cols
            logger.info(
                "Grid layout: --n-ap %d -> grid_n_rows=%d, grid_n_cols=%d",
                args.n_ap, rows, cols,
            )

    cfg.validate()

    # ── Generate topology and large-scale fading ──────────────────────────
    rng  = make_rng(args.seed)
    topo = generate_topology(cfg, rng)
    logger.info("\n%s", topo.summary())

    rng2 = make_rng(args.seed + 1)   # separate rng for LSF
    lsf  = compute_large_scale_fading(topo, cfg, rng2, include_targets=True)

    # ── Handle matplotlib backend for --no-show ───────────────────────────
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend

    # ── Build output paths ─────────────────────────────────────────────────
    def out(name: str):
        if not args.save:
            return None
        return Path(args.out_dir) / f"{name}.{args.fmt}"

    tag = f"seed{args.seed}_{cfg.topology.type}_{cfg.topology.n_ap}ap"

    # ── Generate plots ────────────────────────────────────────────────────
    if args.summary_only:
        plot_topology_summary(
            topo, lsf,
            title=f"CORDIS Topology Summary  ({tag})",
            save_path=out(f"summary_{tag}"),
            dpi=args.dpi,
        )
    else:
        plot_topology_2d(
            topo, lsf,
            title=f"Network Topology  ({tag})",
            show_links=True,
            save_path=out(f"topology_2d_{tag}"),
            dpi=args.dpi,
        )
        plot_topology_3d(
            topo,
            title=f"Network Topology 3-D  ({tag})",
            save_path=out(f"topology_3d_{tag}"),
            dpi=args.dpi,
        )
        plot_lsf_heatmap(
            topo, lsf,
            save_path=out(f"lsf_heatmap_{tag}"),
            dpi=args.dpi,
        )
        plot_los_matrix(
            topo, lsf,
            save_path=out(f"los_matrix_{tag}"),
            dpi=args.dpi,
        )
        plot_pathloss_vs_distance(
            topo, lsf,
            save_path=out(f"pathloss_vs_distance_{tag}"),
            dpi=args.dpi,
        )
        plot_topology_summary(
            topo, lsf,
            title=f"CORDIS Topology Summary  ({tag})",
            save_path=out(f"summary_{tag}"),
            dpi=args.dpi,
        )

    if args.save:
        logger.info("All figures saved to: %s", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()

