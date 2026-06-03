#!/usr/bin/env python3
"""
paper/figure_src/fig_csi/build_fig_csi.py
=========================================

Build the CSI-robustness figure for the journal paper's Simulation Results
section, from a ``csi_sweep`` result (a sweep over the uplink pilot power
P_p/σ_n², which sets the MMSE channel-estimation quality).

Stacked two-panel, single-column (shared x-axis = pilot power [dB]):

  (top)  worst-user **min-SINR** vs pilot power.
  (bot)  **sum-SCNR** vs pilot power.

Lower pilot power → noisier channel estimates ĥ_{a,u} → every algorithm
degrades.  The figure's point is the *shape* of that degradation: the
centralized bound and the CORDIS variants should bend down gracefully, while
interference-naïve baselines (e.g. MRT) cliff-edge sooner.  That contrast is
why this figure plots **all** algorithms in the run by default (the trio
leading, benchmarks appended), rather than filtering to the proposed pair like
the other figures.

A vertical dashed line marks the deployed operating point **P_p\\*** (the pilot
power the headline figures run at, 128 dB by default ≈ near-perfect CSI), so the
reader sees that the rest of the paper operates in the good-CSI plateau and this
figure explores what happens as CSI degrades.

The knee location (~110 dB for the default ap_radius=650 m, where the predicted
NMSE ≈ 0.1) is tied to the large-scale fading β; if the campaign changes
``topology.ap_radius_m`` the informative sweep range shifts with it.

Filtered curves use the project-wide per-algorithm style (``style_for`` /
``ALGORITHM_STYLE``) so a curve reads identically here and in the sibling
figures.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_sweep``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_csi/build_fig_csi.py
    python3 .../build_fig_csi.py --no-tex
    python3 .../build_fig_csi.py --pilot-star 128
    python3 .../build_fig_csi.py --only "Centralized,CORDIS-ADMM,CORDIS-Split"
    python3 .../build_fig_csi.py --result-dir results/exp_csi_sweep/array_12345_aggregated
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("fig_csi")


# ─────────────────────────────────────────────────────────────────────
#  Path plumbing (mirrors the sibling builders)
# ─────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
FIG_DIR = THIS_FILE.parent
PAPER_ROOT = FIG_DIR.parents[1]
FIGURES_OUT = PAPER_ROOT / "figures"


def find_repo_root(start: Path = THIS_FILE) -> Path:
    for p in [start, *start.parents]:
        if (p / "cordis").is_dir() and (p / "cordis" / "__init__.py").exists():
            return p
    return PAPER_ROOT.parent


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────
#  Which algorithms, which metrics, which operating point
# ─────────────────────────────────────────────────────────────────────

# csi_sweep runs all_algorithms.  Unlike the other figures we plot ALL of
# them (the graceful-vs-cliff contrast IS the figure), but list the proposed
# trio as the LEAD so they take the consistent styles / legend order; any
# benchmarks present are appended after.
LEAD: Tuple[str, ...] = ("Centralized", "CORDIS-ADMM", "CORDIS-Split")

SINR_METRIC = "min_sinr_db"

SCNR_METRIC_PREFERENCE: Tuple[str, ...] = (
    "sum_scnr_db",
    "weighted_sum_scnr_db",
    "mean_scnr_db",
)

# Deployed operating point P_p\* (matches PILOT_POWER_DB in _defaults.sh / the
# pilot power the CDF + tradeoff figures run at).  Marked with a vline.
PILOT_STAR_DEFAULT_DB = 128.0

_SCNR_YLABEL = {
    "sum_scnr_db": r"sum-SCNR [dB]",
    "weighted_sum_scnr_db": r"weighted sum-SCNR [dB]",
    "mean_scnr_db": r"mean SCNR [dB]",
}


# ─────────────────────────────────────────────────────────────────────
#  Result discovery  (array-aggregated first, never `latest`)
# ─────────────────────────────────────────────────────────────────────

import re as _re

_ARRAY_AGG_PATTERN = _re.compile(r"^array_(\d+)_aggregated$")
_ARRAY_TASK_PATTERN = _re.compile(r"^array_(\d+)_task_(\d+)$")


def _discover_result_dir(experiment: str = "csi_sweep",
                         results_root: Optional[Path] = None) -> Optional[Path]:
    """Newest result dir for ``exp_<experiment>``.

    Prefers the highest-numbered ``array_<jobid>_aggregated/``; otherwise the
    most recently modified plain run dir, excluding per-task dirs and the
    unreliable ``latest`` symlink.
    """
    base = (results_root or (REPO_ROOT / "results")) / f"exp_{experiment}"
    if not base.is_dir():
        return None

    aggs: List[Tuple[int, Path]] = []
    for child in base.iterdir():
        if child.is_dir() and _ARRAY_AGG_PATTERN.match(child.name):
            aggs.append((int(_ARRAY_AGG_PATTERN.match(child.name).group(1)), child))
    if aggs:
        aggs.sort()
        return aggs[-1][1]

    plains: List[Path] = []
    for child in base.iterdir():
        if not child.is_dir() or child.is_symlink() or child.name == "latest":
            continue
        if _ARRAY_TASK_PATTERN.match(child.name):
            continue
        if (child / "manifest.json").exists():
            plains.append(child)
    if not plains:
        return None
    plains.sort(key=lambda d: d.stat().st_mtime)
    return plains[-1]


def load_sweep(result_dir: Optional[Path] = None,
               experiment: str = "csi_sweep",
               results_root: Optional[Path] = None):
    """Load a sweep (``kind == 'sweep'``) :class:`ExperimentResult`.

    If ``result_dir`` is given it is loaded verbatim; otherwise the newest run
    of ``exp_<experiment>`` is auto-discovered (aggregated array preferred).
    """
    from cordis.experiments.result import ExperimentResult  # lazy

    if result_dir is None:
        result_dir = _discover_result_dir(experiment, results_root)
        if result_dir is None:
            raise FileNotFoundError(
                f"No usable run under results/exp_{experiment}/.  Run the "
                f"{experiment!r} campaign (or aggregate its array job) first."
            )
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = REPO_ROOT / result_dir

    result = ExperimentResult.load(result_dir)
    if getattr(result, "kind", None) != "sweep":
        raise ValueError(
            f"fig_csi needs a sweep (kind='sweep') result; "
            f"{result_dir} has kind={getattr(result, 'kind', None)!r}."
        )
    return result, result_dir


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _first_point(result):
    keys = sorted(result.sweep_results.keys())
    return result.sweep_results[keys[0]]


def _ordered_algorithms(result,
                        lead: Sequence[str] = LEAD,
                        only: Optional[Sequence[str]] = None) -> List[str]:
    """Algorithms to plot.  With ``only`` set, filter to it (in given order).
    Otherwise plot ALL algorithms present, with ``lead`` first (in lead order)
    and any remaining algorithms appended in their native order."""
    sim = _first_point(result)
    available = list(getattr(sim, "algorithm_names", None)
                     or getattr(sim, "algorithm_results", {}).keys())
    if only:
        picked = [n for n in only if n in available]
        if not picked:
            logger.warning("None of %s present (have: %s); plotting all.",
                           list(only), available)
        else:
            return picked
    lead_present = [n for n in lead if n in available]
    rest = [n for n in available if n not in lead_present]
    return lead_present + rest


def _select_scnr_metric(result,
                        preference: Sequence[str] = SCNR_METRIC_PREFERENCE,
                        only: Optional[Sequence[str]] = None) -> Optional[str]:
    sim = _first_point(result)
    results = getattr(sim, "algorithm_results", {})
    names = list(only) if only else list(results.keys())
    algos = [results[n] for n in names if n in results]
    if not algos:
        return None
    for metric in preference:
        if all(a.has_metric(metric) for a in algos):
            return metric
    return None


def _nearest_pilot(result, pilot_star: float) -> float:
    keys = sorted(result.sweep_results.keys())
    return min(keys, key=lambda k: abs(k - pilot_star))


def _in_range(result, x: float) -> bool:
    keys = sorted(result.sweep_results.keys())
    return keys[0] <= x <= keys[-1]


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary
# ─────────────────────────────────────────────────────────────────────

def collect_summary(result,
                    only: Sequence[str],
                    scnr_metric: Optional[str],
                    pilot_star: float) -> Dict[str, Dict[str, float]]:
    """Per-algorithm robustness numbers: min-SINR (median) at the worst-CSI,
    operating, and best-CSI pilot powers, plus the degradation
    SINR(best)−SINR(worst) — a small drop means a robust, graceful algorithm;
    a large drop is a cliff-edge.  Same for SCNR."""
    keys = sorted(result.sweep_results.keys())
    p_lo, p_hi = keys[0], keys[-1]
    p_op = _nearest_pilot(result, pilot_star)

    def _med(sim, name, metric):
        a = sim.algorithm_results.get(name)
        if a is None or not a.has_metric(metric):
            return None
        return a.percentile(metric, 50)

    out: Dict[str, Dict[str, float]] = {}
    for name in only:
        row: Dict[str, float] = {}
        s_lo = _med(result.sweep_results[p_lo], name, SINR_METRIC)
        s_op = _med(result.sweep_results[p_op], name, SINR_METRIC)
        s_hi = _med(result.sweep_results[p_hi], name, SINR_METRIC)
        if s_lo is not None:
            row["sinr_worst_db"] = s_lo
        if s_op is not None:
            row["sinr_op_db"] = s_op
        if s_hi is not None:
            row["sinr_best_db"] = s_hi
        if s_lo is not None and s_hi is not None:
            row["sinr_drop_db"] = s_hi - s_lo          # degradation, best→worst
        if scnr_metric:
            c_lo = _med(result.sweep_results[p_lo], name, scnr_metric)
            c_hi = _med(result.sweep_results[p_hi], name, scnr_metric)
            if c_lo is not None:
                row["scnr_worst_db"] = c_lo
            if c_lo is not None and c_hi is not None:
                row["scnr_drop_db"] = c_hi - c_lo
        out[name] = row
    return out


def _print_summary(result, summary, scnr_metric, pilot_star) -> None:
    keys = sorted(result.sweep_results.keys())
    p_op = _nearest_pilot(result, pilot_star)
    print(f"[info] pilot grid = {keys} dB")
    print(f"[info] P_p* (mark)= {pilot_star:g} dB  (nearest swept {p_op:g} dB)"
          + ("" if _in_range(result, pilot_star) else "  [OUT OF RANGE — marker skipped]"))
    print(f"[info] SCNR metric= {scnr_metric or '(none)'}")
    hdr = (f"{'algorithm':<14}{'SINR worst':>11}{'SINR op':>9}{'SINR best':>10}"
           f"{'SINR drop':>10}{'SCNR drop':>10}")
    print(hdr); print("-" * len(hdr))
    for name, r in summary.items():
        def f(k):
            v = r.get(k); return (f"{v:.2f}" if v is not None else "—")
        print(f"{name:<14}{f('sinr_worst_db'):>11}{f('sinr_op_db'):>9}"
              f"{f('sinr_best_db'):>10}{f('sinr_drop_db'):>10}{f('scnr_drop_db'):>10}")


# ─────────────────────────────────────────────────────────────────────
#  Figure  (self-contained so the notebook can inline an editable copy)
# ─────────────────────────────────────────────────────────────────────

def build_figure(result,
                 *,
                 only: Optional[Sequence[str]] = None,
                 scnr_metric: Optional[str] = None,
                 pilot_star: float = PILOT_STAR_DEFAULT_DB,
                 log_x: bool = False,
                 use_tex: bool = True):
    """Assemble the stacked CSI-robustness figure; return ``(fig, only, scnr_metric)``.

    Built directly on ``cordis.plotting.plot_sweep`` so the paper builder stays
    decoupled from scripts/.  Reuses the project per-algorithm style.
    """
    import matplotlib
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    from cordis.plotting import apply_paper_style, figsize, plot_sweep

    apply_paper_style()
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False

    only = list(only) if only else _ordered_algorithms(result, LEAD)
    if scnr_metric is None:
        scnr_metric = _select_scnr_metric(result, only=only)

    axis = result.sweep_axis
    xlabel = axis.display or r"Pilot power $P_p/\sigma_n^2$ [dB]"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5 / 2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    # (top) min-SINR vs pilot power
    plot_sweep(result.sweep_results, metric=SINR_METRIC, ax=ax_top,
               xlabel="", ylabel=r"min-SINR [dB]", only=only, log_x=log_x)
    ax_top.set_title("Robustness to imperfect CSI")

    # (bot) SCNR vs pilot power
    if scnr_metric is not None:
        plot_sweep(result.sweep_results, metric=scnr_metric, ax=ax_bot,
                   xlabel=xlabel,
                   ylabel=_SCNR_YLABEL.get(scnr_metric, scnr_metric),
                   only=only, log_x=log_x)
    else:
        logger.warning("No SCNR metric available; bottom panel left empty.")
        ax_bot.set_xlabel(xlabel)

    # operating-point marker P_p* (only if inside the swept range)
    if _in_range(result, pilot_star):
        for ax in (ax_top, ax_bot):
            ax.axvline(pilot_star, ls="--", lw=0.9, color="0.45", zorder=0)
        # get_xaxis_transform() = (data-x, axes-y); tight_layout-safe.
        ax_top.text(pilot_star, 0.96, r"$P_p^\star$",
                    transform=ax_top.get_xaxis_transform(),
                    ha="right", va="top", fontsize="small", color="0.35")
    else:
        logger.warning("pilot_star=%.3g dB is outside the swept range; "
                       "operating-point marker skipped.", pilot_star)

    # one legend only (top); drop the duplicate on the bottom panel
    if ax_bot.get_legend() is not None:
        ax_bot.get_legend().remove()

    # NB: no fig.tight_layout() — conflicts with the shared-x / hspace stacked
    # layout (same reason scripts/_plot_common.sweep_plot_pair omits it).
    # save_figure() uses bbox_inches='tight', which handles the margins.
    return fig, only, scnr_metric


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_fig_csi",
        description="Build the CSI-robustness (pilot-power sweep) figure for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--result-dir", default=None,
                   help="Specific results/exp_csi_sweep/<run>/ to load. "
                        "Default: newest (aggregated array preferred).")
    p.add_argument("--pilot-star", type=float, default=PILOT_STAR_DEFAULT_DB,
                   help="Operating-point pilot power P_p* [dB] marked with a line.")
    p.add_argument("--scnr-metric", default=None,
                   help="SCNR metric for the bottom panel.  Default: first "
                        f"available of {', '.join(SCNR_METRIC_PREFERENCE)}.")
    p.add_argument("--only", default=None,
                   help="Comma-separated algorithm display names. Default: ALL "
                        f"present, with {', '.join(LEAD)} leading.")
    p.add_argument("--log-x", action="store_true",
                   help="Log-scale the pilot-power axis (rarely needed; it is "
                        "already in dB).")
    p.add_argument("--out", default=str(FIGURES_OUT / "fig_csi"),
                   help="Output path stem (no extension).")
    p.add_argument("--no-tex", action="store_true",
                   help="Disable LaTeX text rendering (nodes without pdflatex).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _build_parser().parse_args(argv)

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    result, result_dir = load_sweep(
        Path(args.result_dir) if args.result_dir else None, experiment="csi_sweep")

    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {result_dir}")

    only_resolved = _ordered_algorithms(result, LEAD, only=only)
    scnr_metric = args.scnr_metric or _select_scnr_metric(result, only=only_resolved)

    summary = collect_summary(result, only_resolved, scnr_metric, args.pilot_star)
    _print_summary(result, summary, scnr_metric, args.pilot_star)

    fig, used_only, used_scnr_metric = build_figure(
        result, only=only_resolved, scnr_metric=scnr_metric,
        pilot_star=args.pilot_star, log_x=args.log_x, use_tex=not args.no_tex)

    out_stem = Path(args.out)
    if not out_stem.is_absolute():
        out_stem = Path.cwd() / out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_csi",
            "PilotStarDB": f"{args.pilot_star:g}",
            "Algorithms": ", ".join(used_only),
            "ScnrMetric": str(used_scnr_metric),
            "Run": result_dir.name,
        }))
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    for pth in paths + [png]:
        print(f"[ok] wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

