#!/usr/bin/env python3
"""
paper/figure_src/fig_clutter/build_fig_clutter.py
=================================================

Build the clutter-suppression figure for the journal paper's Simulation
Results section, from a ``kappa_sweep`` result.

Stacked two-panel, single-column (shared x-axis = clutter penalty κ):

  (top)  achieved **SINR** vs κ  — the communication cost of clutter
         avoidance.  As κ grows the beamformer spends degrees of freedom
         steering a null toward the clutter PAS, so the SINR drifts down.
         Metric selectable: 'min' (worst-user) or 'mean' (across users).

  (bot)  achieved **SCNR** vs κ  — the sensing benefit.  As κ grows clutter
         leakage into the receive filter falls and SCNR climbs, saturating
         once the clutter is essentially nulled.  This is the "suppression"
         the figure is named for.  Metric selectable: 'wsum' (weighted target
         SCNR sum, default), 'min' (worst target), or 'mean' (across targets).

Each panel's central line is the across-trial mean; an optional shaded band
shows dispersion (``--band std`` = mean±std, ``iqr`` = 25–75 pct, ``none`` =
no band).

A vertical dashed line marks the deployed operating point **κ\\*** (the κ used
across the headline figures, 0.08 by default), so the reader sees that the
shipped system sits in the gentle-cost / near-saturated-gain region.

Why this story needs the *offset* clutter geometry: with
``clutter_center_strategy='target_centroid'`` the clutter PAS sits on top of
the target, so penalising clutter also penalises the target echo and SCNR
barely moves with κ (geometric, not an implementation issue — see
joint_opt.py).  The ``kappa_sweep`` recipe therefore inherits the ``offset``
strategy from _defaults.sh; keep it that way for this figure.

Both panels filter to the proposed pair vs the centralized ceiling and use the
project-wide per-algorithm style (``style_for`` / ``ALGORITHM_STYLE``), so a
curve reads identically here and in fig_cs_tradeoff / fig_cdf.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_sweep``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_clutter/build_fig_clutter.py
    python3 .../build_fig_clutter.py --no-tex
    python3 .../build_fig_clutter.py --sinr-metric mean --scnr-metric min
    python3 .../build_fig_clutter.py --band none --kappa-star 0.08
    python3 .../build_fig_clutter.py --result-dir results/exp_kappa_sweep/array_12345_aggregated
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("fig_clutter")


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

# kappa_sweep runs the proposed pair vs the centralized ceiling.  Filter to
# these (in this order); robust whether the run carries 3 or 5 algorithms.
PREFERRED: Tuple[str, ...] = ("Centralized", "CORDIS-ADMM", "CORDIS-Split")

# SINR metric for the top panel: worst-user (min) or across-user (mean) per
# trial.  Both are real registry metrics.  -> (registry_name, axis_label)
SINR_METRIC_CHOICES: Dict[str, Tuple[str, str]] = {
    "min":  ("min_sinr_db",  "min-user SINR"),
    "mean": ("mean_sinr_db", "mean-user SINR"),
}
SINR_METRIC_DEFAULT = "min"

# SCNR metric for the bottom panel.  -> (registry_name, axis_label, legend_short)
# 'wsum' = the WEIGHTED target-SCNR sum Σ_t ω_t·SCNR_{a_r,t} (there is no plain
# "sum_scnr_db" in the registry; this is "weighted_sum_scnr_db").
SCNR_METRIC_CHOICES: Dict[str, Tuple[str, str, str]] = {
    "min":  ("min_scnr_db",          "min-target SCNR",   "min-target SCNR"),
    "mean": ("mean_scnr_db",         "mean-target SCNR",  "mean SCNR"),
    "wsum": ("weighted_sum_scnr_db", "weighted-sum SCNR", "w-sum SCNR"),
}
SCNR_METRIC_DEFAULT = "wsum"

# Dispersion band drawn around each mean curve (passed to plot_sweep's `error`).
BAND_DEFAULT = "std"   # 'std' | 'iqr' | 'none'

# Deployed operating point κ\* (matches ADMM_KAPPA in _defaults.sh / the κ used
# by the CDF + tradeoff figures).  Marked with a vline; overridable.
KAPPA_STAR_DEFAULT = 0.08


# ─────────────────────────────────────────────────────────────────────
#  Result discovery  (array-aggregated first, never `latest`)
# ─────────────────────────────────────────────────────────────────────

import re as _re

_ARRAY_AGG_PATTERN = _re.compile(r"^array_(\d+)_aggregated$")
_ARRAY_TASK_PATTERN = _re.compile(r"^array_(\d+)_task_(\d+)$")


def _discover_result_dir(experiment: str = "kappa_sweep",
                         results_root: Optional[Path] = None) -> Optional[Path]:
    """Newest result dir for ``exp_<experiment>``.

    Prefers the highest-numbered ``array_<jobid>_aggregated/`` (the merged
    campaign); otherwise the most recently modified plain run dir, excluding
    per-task ``array_*_task_*`` dirs and the unreliable ``latest`` symlink.
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
               experiment: str = "kappa_sweep",
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
            f"fig_clutter needs a sweep (kind='sweep') result; "
            f"{result_dir} has kind={getattr(result, 'kind', None)!r}."
        )
    return result, result_dir


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _first_point(result):
    """The SimResult at the first sweep point (for algorithm/metric probing)."""
    keys = sorted(result.sweep_results.keys())
    return result.sweep_results[keys[0]]


def _present_algorithms(result, preferred: Sequence[str]) -> List[str]:
    sim = _first_point(result)
    available = list(getattr(sim, "algorithm_names", None)
                     or getattr(sim, "algorithm_results", {}).keys())
    picked = [n for n in preferred if n in available]
    if picked:
        return picked
    logger.warning("None of %s found in run (have: %s); plotting all.",
                   list(preferred), available)
    return available


def _resolve_sinr_metric(key: str) -> Tuple[str, str]:
    """Map a --sinr-metric key ('min'|'mean') to (registry_metric, label)."""
    if key not in SINR_METRIC_CHOICES:
        raise ValueError(f"--sinr-metric must be one of "
                         f"{list(SINR_METRIC_CHOICES)}, got {key!r}.")
    return SINR_METRIC_CHOICES[key]


def _resolve_scnr_metric(key: str,
                         result=None,
                         only: Optional[Sequence[str]] = None
                         ) -> Tuple[Optional[str], str, str]:
    """Map a --scnr-metric key ('min'|'mean'|'wsum') to
    (registry_metric, axis_label, legend_short).

    If ``result`` is given and the chosen metric is absent from the present
    algorithms, fall back to the first available choice (with a warning); the
    metric is ``None`` if none is available (bottom panel then left empty)."""
    if key not in SCNR_METRIC_CHOICES:
        raise ValueError(f"--scnr-metric must be one of "
                         f"{list(SCNR_METRIC_CHOICES)}, got {key!r}.")
    name, full, short = SCNR_METRIC_CHOICES[key]
    if result is None:
        return name, full, short

    sim = _first_point(result)
    results = getattr(sim, "algorithm_results", {})
    names = list(only) if only else list(results.keys())
    ars = [results[n] for n in names if n in results]
    if ars and all(a.has_metric(name) for a in ars):
        return name, full, short
    # graceful fallback
    for k2, (n2, f2, s2) in SCNR_METRIC_CHOICES.items():
        if ars and all(a.has_metric(n2) for a in ars):
            logger.warning("SCNR metric %r not in run; using %r instead.", name, n2)
            return n2, f2, s2
    logger.warning("No SCNR metric available in run; bottom panel left empty.")
    return None, full, short


def _nearest_kappa(result, kappa_star: float) -> float:
    """Sweep point closest to κ\\* (the swept grid may not contain it exactly)."""
    keys = sorted(result.sweep_results.keys())
    return min(keys, key=lambda k: abs(k - kappa_star))


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary
# ─────────────────────────────────────────────────────────────────────

def collect_summary(result,
                    only: Sequence[str],
                    sinr_metric: str,
                    scnr_metric: Optional[str],
                    kappa_star: float) -> Dict[str, Dict[str, float]]:
    """Per-algorithm headline numbers: SCNR (median) at κ_min, κ\\*, κ_max, the
    suppression gain SCNR(κ_max)−SCNR(κ_min), and the SINR cost at κ\\*.

    ``sinr_metric`` / ``scnr_metric`` are RESOLVED registry names."""
    keys = sorted(result.sweep_results.keys())
    k_lo, k_hi = keys[0], keys[-1]
    k_star = _nearest_kappa(result, kappa_star)

    def _med(sim, name, metric):
        a = sim.algorithm_results.get(name)
        if a is None or metric is None or not a.has_metric(metric):
            return None
        return a.percentile(metric, 50)

    out: Dict[str, Dict[str, float]] = {}
    for name in only:
        row: Dict[str, float] = {}
        if scnr_metric:
            s_lo = _med(result.sweep_results[k_lo], name, scnr_metric)
            s_st = _med(result.sweep_results[k_star], name, scnr_metric)
            s_hi = _med(result.sweep_results[k_hi], name, scnr_metric)
            if s_lo is not None:
                row["scnr_kmin_db"] = s_lo
            if s_st is not None:
                row["scnr_kstar_db"] = s_st
            if s_hi is not None:
                row["scnr_kmax_db"] = s_hi
            if s_lo is not None and s_hi is not None:
                row["scnr_gain_db"] = s_hi - s_lo
        sinr_st = _med(result.sweep_results[k_star], name, sinr_metric)
        sinr_lo = _med(result.sweep_results[k_lo], name, sinr_metric)
        if sinr_st is not None:
            row["sinr_kstar_db"] = sinr_st
        if sinr_lo is not None and sinr_st is not None:
            row["sinr_cost_db"] = sinr_st - sinr_lo   # negative = comm cost
        out[name] = row
    return out


def _print_summary(result, summary, sinr_metric, scnr_metric, kappa_star) -> None:
    keys = sorted(result.sweep_results.keys())
    k_star = _nearest_kappa(result, kappa_star)
    print(f"[info] κ grid     = {keys}")
    print(f"[info] κ* (marked)= {kappa_star:g}  (nearest swept point {k_star:g})")
    print(f"[info] SINR metric= {sinr_metric}")
    print(f"[info] SCNR metric= {scnr_metric or '(none)'}")
    hdr = (f"{'algorithm':<14}{'SCNR κmin':>10}{'SCNR κ*':>9}{'SCNR κmax':>10}"
           f"{'gain':>8}{'SINR κ*':>9}{'SINR cost':>10}")
    print(hdr); print("-" * len(hdr))
    for name, r in summary.items():
        def g(k): return r.get(k)
        def f(v): return (f"{v:.2f}" if v is not None else "—")
        print(f"{name:<14}"
              f"{f(g('scnr_kmin_db')):>10}{f(g('scnr_kstar_db')):>9}"
              f"{f(g('scnr_kmax_db')):>10}{f(g('scnr_gain_db')):>8}"
              f"{f(g('sinr_kstar_db')):>9}{f(g('sinr_cost_db')):>10}")


# ─────────────────────────────────────────────────────────────────────
#  Figure  (kept dependency-light + self-contained so the notebook can
#  inline an editable copy and override style)
# ─────────────────────────────────────────────────────────────────────

def build_figure(result,
                 *,
                 only: Optional[Sequence[str]] = None,
                 sinr_metric: str = SINR_METRIC_DEFAULT,
                 scnr_metric: str = SCNR_METRIC_DEFAULT,
                 band: str = BAND_DEFAULT,
                 kappa_star: float = KAPPA_STAR_DEFAULT,
                 log_x: bool = False,
                 use_tex: bool = True):
    """Assemble the stacked κ-sweep figure and return
    ``(fig, only, scnr_metric_name)``.

    ``sinr_metric`` is 'min'|'mean'; ``scnr_metric`` is 'min'|'mean'|'wsum';
    ``band`` is 'std'|'iqr'|'none' (the dispersion shading around each mean
    curve).  Built directly on ``cordis.plotting.plot_sweep``.
    """
    import matplotlib
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    from cordis.plotting import apply_paper_style, figsize, plot_sweep

    apply_paper_style()
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False

    only = list(only) if only else _present_algorithms(result, PREFERRED)
    sinr_name, sinr_label = _resolve_sinr_metric(sinr_metric)
    scnr_name, scnr_full, _scnr_short = _resolve_scnr_metric(scnr_metric, result, only)

    axis = result.sweep_axis
    if log_x and 0.0 in result.sweep_results:
        logger.warning("log_x requested but κ=0 is in the grid; using linear x.")
        log_x = False

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5 / 2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    # (top) SINR vs κ  — communication cost
    plot_sweep(result.sweep_results, metric=sinr_name, ax=ax_top, error=band,
               xlabel="", ylabel=f"{sinr_label} [dB]", only=only, log_x=log_x)
    ax_top.set_title(r"Clutter suppression vs penalty $\kappa$")

    # (bot) SCNR vs κ  — sensing benefit (the suppression)
    if scnr_name is not None:
        plot_sweep(result.sweep_results, metric=scnr_name, ax=ax_bot, error=band,
                   xlabel=axis.display or r"$\kappa$",
                   ylabel=f"{scnr_full} [dB]", only=only, log_x=log_x)
    else:
        logger.warning("No SCNR metric available; bottom panel left empty.")
        ax_bot.set_xlabel(axis.display or r"$\kappa$")

    # operating-point marker κ* on both panels (label once, on top).
    # get_xaxis_transform() = (data-x, axes-y); tight_layout-safe.
    for ax in (ax_top, ax_bot):
        ax.axvline(kappa_star, ls="--", lw=0.9, color="0.45", zorder=0)
    ax_top.text(kappa_star, 0.96, r"$\kappa^\star$",
                transform=ax_top.get_xaxis_transform(),
                ha="left", va="top", fontsize="small", color="0.35")

    # one legend only (top); drop the duplicate on the bottom panel
    if ax_bot.get_legend() is not None:
        ax_bot.get_legend().remove()

    # NB: no fig.tight_layout() — it conflicts with the shared-x / hspace
    # stacked layout (same reason scripts/_plot_common.sweep_plot_pair omits
    # it).  save_figure() uses bbox_inches='tight', which handles the margins.
    return fig, only, scnr_name


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_fig_clutter",
        description="Build the clutter-suppression κ-sweep figure for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--result-dir", default=None,
                   help="Specific results/exp_kappa_sweep/<run>/ to load. "
                        "Default: newest (aggregated array preferred).")
    p.add_argument("--sinr-metric", choices=list(SINR_METRIC_CHOICES),
                   default=SINR_METRIC_DEFAULT,
                   help="Top-panel SINR metric: 'min' (worst-user) or 'mean'.")
    p.add_argument("--scnr-metric", choices=list(SCNR_METRIC_CHOICES),
                   default=SCNR_METRIC_DEFAULT,
                   help="Bottom-panel SCNR metric: 'wsum' (weighted target-SCNR "
                        "sum), 'min' (worst target), or 'mean'.")
    p.add_argument("--band", choices=("std", "iqr", "none"), default=BAND_DEFAULT,
                   help="Dispersion band around each mean curve: standard "
                        "deviation, inter-quartile range, or none.")
    p.add_argument("--kappa-star", type=float, default=KAPPA_STAR_DEFAULT,
                   help="Operating-point κ* marked with a vertical line.")
    p.add_argument("--only", default=None,
                   help="Comma-separated algorithm display names. "
                        f"Default: {', '.join(PREFERRED)} (filtered to those present).")
    p.add_argument("--log-x", action="store_true",
                   help="Log-scale the κ axis (ignored if κ=0 is in the grid).")
    p.add_argument("--out", default=str(FIGURES_OUT / "fig_clutter"),
                   help="Output path stem (no extension).")
    p.add_argument("--no-tex", action="store_true",
                   help="Disable LaTeX text rendering (nodes without pdflatex).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _build_parser().parse_args(argv)

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    result, result_dir = load_sweep(
        Path(args.result_dir) if args.result_dir else None, experiment="kappa_sweep")

    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {result_dir}")

    only_resolved = list(only) if only else _present_algorithms(result, PREFERRED)
    sinr_name, _sinr_label = _resolve_sinr_metric(args.sinr_metric)
    scnr_name, _scnr_full, _scnr_short = _resolve_scnr_metric(
        args.scnr_metric, result, only_resolved)

    summary = collect_summary(result, only_resolved, sinr_name, scnr_name,
                              args.kappa_star)
    _print_summary(result, summary, sinr_name, scnr_name, args.kappa_star)

    fig, used_only, used_scnr_metric = build_figure(
        result, only=only_resolved, sinr_metric=args.sinr_metric,
        scnr_metric=args.scnr_metric, band=args.band,
        kappa_star=args.kappa_star, log_x=args.log_x, use_tex=not args.no_tex)

    out_stem = Path(args.out)
    if not out_stem.is_absolute():
        out_stem = Path.cwd() / out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_clutter",
            "KappaStar": f"{args.kappa_star:g}",
            "SinrMetric": sinr_name,
            "ScnrMetric": str(used_scnr_metric),
            "Band": args.band,
            "Algorithms": ", ".join(used_only),
            "Run": result_dir.name,
        }))
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    for pth in paths + [png]:
        print(f"[ok] wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

