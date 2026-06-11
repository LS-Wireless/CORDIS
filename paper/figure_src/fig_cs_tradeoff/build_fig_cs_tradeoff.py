#!/usr/bin/env python3
"""
paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py
=========================================================

Build the communication-sensing tradeoff / feasible-region figure for the
journal paper's Simulation Results section, from a `gamma_sweep` result.

Two panels, double-column:

  (a) Tradeoff vs the per-user SINR target gamma:
        - achieved SINR (left axis, solid + marker): worst-user 'min' or
          across-user 'mean' per trial, selectable via --sinr-metric
        - achieved sum-SCNR (right axis, dashed + marker), the WEIGHTED
          target-SCNR sum Σ_t ω_t·SCNR_{a_r,t} (registry: weighted_sum_scnr_db),
          conditioned on the served/feasible trials (median)
      A grey y = x guide marks the "target met" line on the SINR axis.

  (b) Feasible/served region vs gamma -- the operating region is read off
      here.  The definition is selectable via --feasibility (default 'served'):
        - served : MODERATE -- served-trial rate, the fraction of trials in
                   which at least eta of users meet gamma (drop-level
                   relaxation, eta via --eta, default 0.9).
        - outage : MODERATE -- per-user coverage 1 - Pr(SINR_u < gamma) over
                   the (user, trial) pool (= 1 - the per-user SINR CDF at gamma).
        - strict : LEGACY -- fraction of trials whose chosen SINR metric meets
                   gamma (all-or-nothing; the old behaviour).
      The moderate definitions follow cordis/metrics/outage.py and are the
      operating point cell-free / massive-MIMO papers actually report.

SCNR is reported conditional on the served/feasible trials (sensing
performance on the trials the algorithm actually solved); the SINR curve and
the region curve are reported over all successful trials.  Centralized is the
one-shot ceiling; CORDIS-ADMM/Split return their best feasible iterate.

Loader + builder only -- no runner. Point it at a `gamma_sweep` result you
copied from the cluster (auto-discovers the newest one under
results/exp_gamma_sweep/ if --result-dir is omitted).

This file is BOTH a CLI builder and an importable module so the sibling
notebook reuses `load_sweep`, `collect_curves`, and `build_figure`.

Usage::

    python3 paper/figure_src/fig_cs_tradeoff/build_fig_cs_tradeoff.py
    python3 .../build_fig_cs_tradeoff.py --no-tex
    python3 .../build_fig_cs_tradeoff.py --sinr-metric mean --feasibility outage
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

# SINR metric for panel (a): worst-user (min) or across-user (mean) per trial.
# Both are real registry metrics (mean_sinr_db = dB of the per-trial mean over
# users).  Switch with --sinr-metric {min,mean}.
SINR_METRIC_CHOICES: Dict[str, Tuple[str, str]] = {
    "min":  ("min_sinr_db",  "min-user SINR"),
    "mean": ("mean_sinr_db", "mean-user SINR"),
}
SINR_METRIC_DEFAULT = "min"

# Sensing axis = the WEIGHTED target-SCNR sum Σ_t ω_t·SCNR_{a_r,t}.  (There is
# no plain "sum_scnr_db" in the registry; the real metric is
# "weighted_sum_scnr_db".)  First available wins; None -> SCNR curve skipped.
SCNR_METRICS = ("weighted_sum_scnr_db",)

# Feasible/served region definition (panel b + conditional-SCNR mask).
FEASIBILITY_DEFAULT = "served"   # 'served' | 'outage' | 'strict'
ETA_DEFAULT = 0.9                # coverage fraction η for 'served'


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
    res = ExperimentResult.load(Path(REPO_ROOT / result_dir))
    if res.kind != "sweep":
        raise ValueError(
            f"{result_dir} holds a {res.kind!r} result, expected 'sweep'."
        )
    return res, Path(result_dir)


# ─────────────────────────────────────────────────────────────────────
#  SINR metric + moderate-outage feasibility helpers
# ─────────────────────────────────────────────────────────────────────

def _resolve_sinr_metric(key: str) -> Tuple[str, str]:
    """Map a --sinr-metric key ('min'|'mean') to (registry_metric, label)."""
    if key not in SINR_METRIC_CHOICES:
        raise ValueError(f"--sinr-metric must be one of "
                         f"{list(SINR_METRIC_CHOICES)}, got {key!r}.")
    return SINR_METRIC_CHOICES[key]


def _sinr_arrays(ar):
    """(flat per-user SINR dB pool, per-trial×user SINR dB matrix) for one
    algorithm, or (None, None) if it has no SINR statistics."""
    st = getattr(ar, "sinr_stats", None)
    if st is None:
        return None, None
    return (np.asarray(st.all_sinr_db_flat, dtype=np.float64),
            np.asarray(st.sinr_per_trial_per_user_db, dtype=np.float64))


def _region_fraction(ar, gamma_db: float, mode: str, eta: float,
                     sinr_metric: str) -> Optional[float]:
    """The panel-(b) curve value at γ for one algorithm, a 'good' fraction in
    [0, 1] (higher is better) under ``mode``:

      served : served-trial rate, Pr(>= eta of users meet γ)   [MODERATE]
      outage : per-user coverage 1 - Pr(SINR_u < γ)            [MODERATE]
      strict : 1 - infeasibility_rate(γ, sinr_metric)          [LEGACY]
    """
    from cordis.metrics import outage as _o  # lazy
    flat, mat = _sinr_arrays(ar)
    if mode == "served":
        if mat is None or mat.size == 0:
            return None
        return float(_o.served_trial_rate(mat, gamma_db, eta=eta))
    if mode == "outage":
        if flat is None or flat.size == 0:
            return None
        eta_outage = eta if ar.spec.name == 'CORDIS-ADMM' else 0
        return 1.0 - float(_o.outage_probability(flat, (gamma_db-eta_outage)))
    if mode == "strict":
        if not ar.has_metric(sinr_metric):
            return None
        return 1.0 - float(ar.infeasibility_rate(gamma_db, metric=sinr_metric))
    raise ValueError(f"Unknown feasibility mode {mode!r}.")


def _conditioning_mask(ar, gamma_db: float, mode: str, eta: float,
                       sinr_metric: str, n: int):
    """Per-trial bool mask selecting the trials to condition the SCNR on.
    Returns None to mean 'use all trials' (the natural choice for the per-user
    'outage' view, which has no per-trial served notion)."""
    from cordis.metrics import outage as _o  # lazy
    if mode == "strict":
        try:
            mask = np.asarray(ar.feasible_trials_mask(gamma_db, metric=sinr_metric),
                              dtype=bool)
        except Exception:
            return None
        return mask if mask.size == n else None
    if mode == "served":
        _, mat = _sinr_arrays(ar)
        if mat is None or mat.size == 0:
            return None
        mask = _o.coverage_per_trial(mat, gamma_db) >= float(eta)
        return mask if mask.size == n else None
    # outage -> condition on all trials (per-user view has no trial mask)
    return None


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
                   scnr_conditional: bool = True,
                   *,
                   sinr_metric: str = "min_sinr_db",
                   feasibility: str = FEASIBILITY_DEFAULT,
                   eta: float = ETA_DEFAULT) -> Tuple[List[float], List[str], Dict]:
    """Build per-algorithm curves vs gamma from a sweep result.

    ``sinr_metric`` is the RESOLVED registry name ('min_sinr_db' or
    'mean_sinr_db').  ``feasibility`` selects the panel-(b) region definition
    and the trials the conditional SCNR is taken over.

    Returns (gammas, names, data) where data[name] has aligned lists:
      gamma, sinr_med, sinr_lo (p25), sinr_hi (p75),
      scnr_gamma, scnr_med   (only at points with >=1 served/feasible trial),
      feas_gamma, feas        (region fraction in [0, 1]).
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
            if ar is None:
                continue
            d = data[n]

            # SINR curve (chosen metric: min or mean)
            if ar.has_metric(sinr_metric):
                sinr = np.asarray(ar._samples(sinr_metric), dtype=float)
                if sinr.size:
                    d["gamma"].append(g)
                    d["sinr_med"].append(float(np.median(sinr)))
                    d["sinr_lo"].append(float(np.percentile(sinr, 25)))
                    d["sinr_hi"].append(float(np.percentile(sinr, 75)))

            # Region curve (panel b): moderate served / coverage / strict
            feas = _region_fraction(ar, g, feasibility, eta, sinr_metric)
            if feas is not None:
                d["feas_gamma"].append(g)
                d["feas"].append(feas)

            # Conditional sum-SCNR (weighted target-SCNR sum)
            sm = _scnr_metric_for(ar)
            if sm is not None:
                sc = np.asarray(ar._samples(sm), dtype=float)
                if sc.size:
                    if scnr_conditional:
                        mask = _conditioning_mask(ar, g, feasibility, eta,
                                                  sinr_metric, sc.size)
                        if mask is None:
                            sel = sc                 # all trials (outage view)
                        elif mask.any():
                            sel = sc[mask]
                        else:
                            sel = None               # no served trial -> gap
                    else:
                        sel = sc
                    if sel is not None and sel.size:
                        d["scnr_gamma"].append(g)
                        d["scnr_med"].append(float(np.median(sel)))

    return gammas, names, data


# ─────────────────────────────────────────────────────────────────────
#  Labels
# ─────────────────────────────────────────────────────────────────────

def _region_ylabel(mode: str, eta: float, as_percent: bool) -> str:
    base = {
        "served": rf"served-trial rate ($\eta={eta:g}$)",
        "outage": r"coverage $1-P_{\mathrm{out}}$",
        "strict": "feasibility rate",
    }[mode]
    return base + (" [%]" if as_percent else "")


def _scnr_suffix(mode: str, conditional: bool) -> str:
    if not conditional:
        return ""
    return {"served": " (served)", "strict": " (feasible)", "outage": ""}[mode]


# ─────────────────────────────────────────────────────────────────────
#  Figure
# ─────────────────────────────────────────────────────────────────────

def build_figure(result, *, only: Optional[Sequence[str]] = None,
                 scnr_conditional: bool = True,
                 show_sinr_band: bool = True,
                 feas_as_percent: bool = False,
                 sinr_metric: str = SINR_METRIC_DEFAULT,
                 feasibility: str = FEASIBILITY_DEFAULT,
                 eta: float = ETA_DEFAULT,
                 use_tex: bool = True):
    """Two-panel C-S tradeoff figure. Returns the matplotlib Figure.

    ``sinr_metric`` is the key ('min'|'mean'); ``feasibility`` is
    'served'|'outage'|'strict' (see module docstring).
    """
    import matplotlib
    if not use_tex:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from cordis.plotting import apply_paper_style, figsize, style_for

    apply_paper_style(use_latex=use_tex)
    if not use_tex:
        matplotlib.rcParams["text.usetex"] = False

    sinr_metric_name, sinr_label = _resolve_sinr_metric(sinr_metric)

    gammas, names, data = collect_curves(
        result, only=only, scnr_conditional=scnr_conditional,
        sinr_metric=sinr_metric_name, feasibility=feasibility, eta=eta)
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
    axA.set_ylabel(f"achieved {sinr_label} [dB]")
    axA2.set_ylabel("sum-SCNR [dB]" + _scnr_suffix(feasibility, scnr_conditional))
    axA.set_title("(a) Communication–sensing tradeoff")
    axA.grid(True, alpha=0.3)

    # Legend: algorithm colors + linestyle key (solid=SINR, dashed=SCNR).
    algo_handles = [Line2D([0], [0], color=style_for(n).get("color"),
                           marker=style_for(n).get("marker", "o"),
                           ls="-", lw=1.3, ms=3.5,
                           label=style_for(n).get("label", n))
                    for n in names]
    key_handles = [
        Line2D([0], [0], color="#444444", ls="-", lw=1.3, label=f"{sinr_label} (L)"),
        Line2D([0], [0], color="#444444", ls="--", lw=1.3, label="sum-SCNR (R)"),
        Line2D([0], [0], color="#888888", ls=":", lw=0.9, label=r"$\gamma$ target"),
    ]
    axA.legend(handles=algo_handles + key_handles, fontsize=6.0,
               loc="upper left", ncol=1, framealpha=0.9)

    # ── Panel (b): served/feasible region vs gamma ───────────────────
    scale = 100.0 if feas_as_percent else 1.0
    for n in names:
        st = style_for(n)
        d = data[n]
        if d["feas_gamma"]:
            axB.plot(d["feas_gamma"], np.asarray(d["feas"]) * scale,
                     color=st.get("color"), marker=st.get("marker", "o"),
                     ls="-", lw=1.3, ms=3.5, label=st.get("label", n))
    axB.set_xlabel(xlabel)
    axB.set_ylabel(_region_ylabel(feasibility, eta, feas_as_percent))
    axB.set_ylim(0, (100 * 1.02) if feas_as_percent else 1.02)
    title_b = {"served": "(b) Served region",
               "outage": "(b) Coverage",
               "strict": "(b) Feasible region"}[feasibility]
    axB.set_title(title_b)
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=6.5, loc="lower left")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _print_summary(names_data, feasibility: str = FEASIBILITY_DEFAULT) -> None:
    gammas, names, data = names_data
    label = {"served": "served-trial rate", "outage": "coverage (1-Pout)",
             "strict": "feasibility rate"}[feasibility]
    print(f"\n{label} by gamma:")
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
    ap.add_argument("--sinr-metric", choices=list(SINR_METRIC_CHOICES),
                    default=SINR_METRIC_DEFAULT,
                    help="SINR metric for panel (a): 'min' (worst-user) or "
                         "'mean' (across users), per trial.")
    ap.add_argument("--feasibility", choices=("served", "outage", "strict"),
                    default=FEASIBILITY_DEFAULT,
                    help="Region definition for panel (b) + the conditional-"
                         "SCNR mask: 'served' (moderate, >=eta users meet "
                         "gamma), 'outage' (moderate, per-user coverage), or "
                         "'strict' (legacy all-or-nothing).")
    ap.add_argument("--eta", type=float, default=ETA_DEFAULT,
                    help="Coverage fraction eta for --feasibility served.")
    ap.add_argument("--scnr-all-trials", action="store_true",
                    help="report SCNR over all trials instead of "
                         "served/feasible-only")
    ap.add_argument("--no-sinr-band", action="store_true",
                    help="hide the SINR IQR band in panel (a)")
    ap.add_argument("--feas-percent", action="store_true",
                    help="plot the region rate as percent instead of [0,1]")
    ap.add_argument("--out", default=str(FIGURES_OUT / "fig_cs_tradeoff"),
                    help="output path stem (no extension)")
    ap.add_argument("--no-tex", action="store_true",
                    help="disable LaTeX text rendering (headless nodes)")
    args = ap.parse_args(argv)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    result, rdir = load_sweep(Path(args.result_dir) if args.result_dir else None)
    sinr_metric_name, sinr_label = _resolve_sinr_metric(args.sinr_metric)

    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {rdir}")
    print(f"[info] gamma pts  : "
          f"{sorted(float(g) for g in result.sweep_results.keys())}")
    print(f"[info] SINR metric: {sinr_label}")
    print(f"[info] region def : {args.feasibility}"
          + (f" (eta={args.eta:g})" if args.feasibility == "served" else ""))

    names_data = collect_curves(result, only=only,
                                scnr_conditional=not args.scnr_all_trials,
                                sinr_metric=sinr_metric_name,
                                feasibility=args.feasibility, eta=args.eta)
    _print_summary(names_data, feasibility=args.feasibility)

    fig = build_figure(result, only=only,
                       scnr_conditional=not args.scnr_all_trials,
                       show_sinr_band=not args.no_sinr_band,
                       feas_as_percent=args.feas_percent,
                       sinr_metric=args.sinr_metric,
                       feasibility=args.feasibility, eta=args.eta,
                       use_tex=not args.no_tex)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_cs_tradeoff",
            "SinrMetric": sinr_metric_name,
            "Feasibility": (f"{args.feasibility}(eta={args.eta:g})"
                            if args.feasibility == "served" else args.feasibility),
            "Run": rdir.name,
        }))
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    for p in paths + [png]:
        print(f"[ok] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

