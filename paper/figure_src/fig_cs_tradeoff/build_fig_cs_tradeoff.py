#!/usr/bin/env python3
"""
paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py
=========================================================

Build the communication-sensing tradeoff / feasible-region figure for the
journal paper's Simulation Results section, from a `gamma_sweep` result.

Two panels, double-column:

  (a) Tradeoff vs the per-user SINR target gamma:
        - achieved worst-user min-SINR  (left axis, solid + marker)
        - achieved sum-SCNR             (right axis, dashed + marker),
          conditioned on feasibility (median over trials that met gamma)
      A grey y = x guide marks the "target met" line on the SINR axis;
      where an algorithm's min-SINR curve peels below it, it is starting
      to miss the constraint.

  (b) Feasibility rate vs gamma: fraction of trials where the worst user
      meets the target (= 1 - infeasibility_rate). This is where the
      operating region is read off; with the Stage-23 fixes + favorable
      config, CORDIS-ADMM tracks Centralized at low-mid gamma and falls
      off at high gamma -- a positive characterization, not a problem.

Feasibility uses the literature-standard served/feasible definition
(min-SINR >= gamma per trial). SCNR is reported conditional on feasibility
(sensing performance on the trials the algorithm actually solved); min-SINR
and feasibility are reported over all successful trials. Centralized is the
one-shot ceiling; CORDIS-ADMM/Split return their best feasible iterate.

Loader + builder only -- no runner. Point it at a `gamma_sweep` result you
copied from the cluster (auto-discovers the newest one under
results/exp_gamma_sweep/ if --result-dir is omitted).

This file is BOTH a CLI builder and an importable module so the sibling
notebook reuses `load_sweep`, `collect_curves`, and `build_figure`.

Usage::

    python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py
    python3 .../build_fig_cs_tradeoff.py --no-tex
    python3 .../build_fig_cs_tradeoff.py --result-dir results/exp_gamma_sweep/<ts>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Path plumbing (mirrors fig_convergence so every builder behaves alike)
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
#  Which algorithms, and which metrics
# ─────────────────────────────────────────────────────────────────────

# C-S tradeoff figure shows the proposed pair against the centralized
# ceiling. gamma_sweep is run with all_algorithms, so we filter down.
PREFERRED = ("Centralized", "CORDIS-ADMM", "CORDIS-Split")
SINR_METRIC = "min_sinr_db"
SCNR_METRICS = ("sum_scnr_db", "weighted_sum_scnr_db")  # first available wins


# ─────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────

def _discover_sweep_dir(experiment: str = "gamma_sweep") -> Optional[Path]:
    base = REPO_ROOT / "results" / f"exp_{experiment}"
    if not base.is_dir():
        return None
    cands = [d for d in base.iterdir()
             if d.is_dir() and (d / "manifest.json").exists()]
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_sweep(result_dir: Optional[Path] = None,
               experiment: str = "gamma_sweep"):
    """Load a sweep ExperimentResult. If `result_dir` is None, auto-discover
    the newest run under results/exp_<experiment>/."""
    from cordis.experiments import ExperimentResult
    if result_dir is None:
        result_dir = _discover_sweep_dir(experiment)
        if result_dir is None:
            raise FileNotFoundError(
                f"No {experiment} result found under "
                f"{REPO_ROOT/'results'/('exp_'+experiment)}. "
                f"Copy a run there or pass --result-dir."
            )
    res = ExperimentResult.load(Path(result_dir))
    if res.kind != "sweep":
        raise ValueError(
            f"{result_dir} holds a {res.kind!r} result, expected 'sweep'."
        )
    return res, Path(result_dir)


# ─────────────────────────────────────────────────────────────────────
#  Curve extraction (pure data; no matplotlib)
# ─────────────────────────────────────────────────────────────────────

def _scnr_metric_for(ar) -> Optional[str]:
    for m in SCNR_METRICS:
        if ar.has_metric(m):
            return m
    return None


def collect_curves(result,
                   only: Optional[Sequence[str]] = None,
                   scnr_conditional: bool = True) -> Tuple[List[float], List[str], Dict]:
    """Build per-algorithm curves vs gamma from a sweep result.

    Returns (gammas, names, data) where data[name] has aligned lists:
      gamma, sinr_med, sinr_lo (p25), sinr_hi (p75),
      scnr_gamma, scnr_med   (only at points with >=1 feasible trial),
      feas_gamma, feas        (feasibility rate in [0, 1]).
    """
    gammas = sorted(float(g) for g in result.sweep_results.keys())

    # Algorithms present anywhere, intersected with `only` and ordered.
    present = set()
    for sr in result.sweep_results.values():
        present.update(sr.algorithm_results.keys())
    wanted = list(only) if only else list(PREFERRED)
    names = [n for n in wanted if n in present]

    data: Dict[str, Dict[str, list]] = {
        n: {"gamma": [], "sinr_med": [], "sinr_lo": [], "sinr_hi": [],
            "scnr_gamma": [], "scnr_med": [],
            "feas_gamma": [], "feas": []}
        for n in names
    }

    for g in gammas:
        sr = result.sweep_results[g]
        for n in names:
            ar = sr.algorithm_results.get(n)
            if ar is None or not ar.has_metric(SINR_METRIC):
                continue
            d = data[n]
            sinr = np.asarray(ar._samples(SINR_METRIC), dtype=float)
            if sinr.size:
                d["gamma"].append(g)
                d["sinr_med"].append(float(np.median(sinr)))
                d["sinr_lo"].append(float(np.percentile(sinr, 25)))
                d["sinr_hi"].append(float(np.percentile(sinr, 75)))
                d["feas_gamma"].append(g)
                d["feas"].append(1.0 - float(ar.infeasibility_rate(g, metric=SINR_METRIC)))

            sm = _scnr_metric_for(ar)
            if sm is not None:
                sc = np.asarray(ar._samples(sm), dtype=float)
                if scnr_conditional:
                    try:
                        mask = np.asarray(
                            ar.feasible_trials_mask(g, metric=SINR_METRIC), dtype=bool)
                    except Exception:
                        mask = np.ones(sc.shape, dtype=bool)
                    if mask.size == sc.size and mask.any():
                        d["scnr_gamma"].append(g)
                        d["scnr_med"].append(float(np.median(sc[mask])))
                elif sc.size:
                    d["scnr_gamma"].append(g)
                    d["scnr_med"].append(float(np.median(sc)))

    return gammas, names, data


# ─────────────────────────────────────────────────────────────────────
#  Figure
# ─────────────────────────────────────────────────────────────────────

def build_figure(result, *, only: Optional[Sequence[str]] = None,
                 scnr_conditional: bool = True,
                 show_sinr_band: bool = True,
                 feas_as_percent: bool = False,
                 use_tex: bool = True):
    """Two-panel C-S tradeoff figure. Returns the matplotlib Figure."""
    import matplotlib
    if not use_tex:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from cordis.plotting import apply_paper_style, figsize, style_for

    apply_paper_style(use_latex=use_tex)
    if not use_tex:
        matplotlib.rcParams["text.usetex"] = False

    gammas, names, data = collect_curves(result, only=only,
                                         scnr_conditional=scnr_conditional)
    xlabel = getattr(result.sweep_axis, "display", None) or r"$\gamma$ [dB]"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize("double", aspect=2.4))
    axA2 = axA.twinx()

    # ── Panel (a): SINR (left, solid) + SCNR (right, dashed) ─────────
    for n in names:
        st = style_for(n)
        color = st.get("color", None)
        marker = st.get("marker", "o")
        d = data[n]
        if d["gamma"]:
            axA.plot(d["gamma"], d["sinr_med"], color=color, marker=marker,
                     ls="-", lw=1.3, ms=3.5)
            if show_sinr_band:
                axA.fill_between(d["gamma"], d["sinr_lo"], d["sinr_hi"],
                                 color=color, alpha=0.12, linewidth=0)
        if d["scnr_gamma"]:
            axA2.plot(d["scnr_gamma"], d["scnr_med"], color=color, marker=marker,
                      ls="--", lw=1.3, ms=3.5)

    if gammas:
        axA.plot(gammas, gammas, ls=":", color="#888888", lw=0.9, zorder=0)

    axA.set_xlabel(xlabel)
    axA.set_ylabel("achieved min-SINR [dB]")
    axA2.set_ylabel("sum-SCNR [dB]" + (" (feasible)" if scnr_conditional else ""))
    axA.set_title("(a) Communication–sensing tradeoff")
    axA.grid(True, alpha=0.3)

    # Legend: algorithm colors + linestyle key (solid=SINR, dashed=SCNR).
    algo_handles = [Line2D([0], [0], color=style_for(n).get("color"),
                           marker=style_for(n).get("marker", "o"),
                           ls="-", lw=1.3, ms=3.5,
                           label=style_for(n).get("label", n))
                    for n in names]
    key_handles = [
        Line2D([0], [0], color="#444444", ls="-", lw=1.3, label="min-SINR (L)"),
        Line2D([0], [0], color="#444444", ls="--", lw=1.3, label="sum-SCNR (R)"),
        Line2D([0], [0], color="#888888", ls=":", lw=0.9, label=r"$\gamma$ target"),
    ]
    axA.legend(handles=algo_handles + key_handles, fontsize=6.0,
               loc="upper left", ncol=1, framealpha=0.9)

    # ── Panel (b): feasibility rate vs gamma ─────────────────────────
    scale = 100.0 if feas_as_percent else 1.0
    for n in names:
        st = style_for(n)
        d = data[n]
        if d["feas_gamma"]:
            axB.plot(d["feas_gamma"], np.asarray(d["feas"]) * scale,
                     color=st.get("color"), marker=st.get("marker", "o"),
                     ls="-", lw=1.3, ms=3.5, label=st.get("label", n))
    axB.set_xlabel(xlabel)
    axB.set_ylabel("feasibility rate" + (" [%]" if feas_as_percent else ""))
    axB.set_ylim(0, (100 * 1.02) if feas_as_percent else 1.02)
    axB.set_title("(b) Feasible region")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=6.5, loc="lower left")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _print_summary(names_data) -> None:
    gammas, names, data = names_data
    print("\nfeasibility rate by gamma:")
    hdr = "  gamma | " + " | ".join(f"{n:>13}" for n in names)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for g in gammas:
        cells = []
        for n in names:
            d = data[n]
            if g in d["feas_gamma"]:
                cells.append(f"{d['feas'][d['feas_gamma'].index(g)]*100:>12.1f}%")
            else:
                cells.append(f"{'n/a':>13}")
        print(f"  {g:>5.1f} | " + " | ".join(cells))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", default=None,
                    help="gamma_sweep result dir (default: newest under "
                         "results/exp_gamma_sweep/)")
    ap.add_argument("--only", default=",".join(PREFERRED),
                    help="comma-separated algorithm names to plot")
    ap.add_argument("--scnr-all-trials", action="store_true",
                    help="report SCNR over all trials instead of feasible-only")
    ap.add_argument("--no-sinr-band", action="store_true",
                    help="hide the SINR IQR band in panel (a)")
    ap.add_argument("--feas-percent", action="store_true",
                    help="plot feasibility as percent instead of [0,1]")
    ap.add_argument("--out", default=str(FIGURES_OUT / "fig_cs_tradeoff"),
                    help="output path stem (no extension)")
    ap.add_argument("--no-tex", action="store_true",
                    help="disable LaTeX text rendering (headless nodes)")
    args = ap.parse_args(argv)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    result, rdir = load_sweep(Path(args.result_dir) if args.result_dir else None)
    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {rdir}")
    print(f"[info] gamma pts  : "
          f"{sorted(float(g) for g in result.sweep_results.keys())}")

    names_data = collect_curves(result, only=only,
                                scnr_conditional=not args.scnr_all_trials)
    _print_summary(names_data)

    fig = build_figure(result, only=only,
                       scnr_conditional=not args.scnr_all_trials,
                       show_sinr_band=not args.no_sinr_band,
                       feas_as_percent=args.feas_percent,
                       use_tex=not args.no_tex)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    from cordis.plotting import save_figure
    paths = list(save_figure(fig, out_stem, formats=("pdf",)))
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    for p in paths + [png]:
        print(f"[ok] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

