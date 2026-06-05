#!/usr/bin/env python3
"""
paper/figure_src/fig_csi/build_fig_csi.py
=========================================

Build the CSI-robustness figure for the journal paper's Simulation Results
section, from a ``csi_sweep`` result (a sweep over the uplink pilot power
P_p/σ_n², which sets the MMSE channel-estimation quality).

Stacked two-panel, single-column (shared x-axis = pilot power [dB]):

  (top)  the **communication** metric vs pilot power — selectable
         (``--comm-metric``):
           * 'min'    : worst-user SINR  [dB]   (default)
           * 'mean'   : across-user SINR [dB]
           * 'outage' : per-user outage probability Pr(SINR_u < γ\\*)
  (bot)  the achieved **SCNR** vs pilot power — selectable (``--scnr-metric``):
           'wsum' (weighted target-SCNR sum, default), 'min', or 'mean'.

Each curve's central line is the across-trial mean (the outage line is the
per-user pool fraction); an optional shaded band shows dispersion
(``--band std|iqr|none``).

Lower pilot power → noisier channel estimates ĥ_{a,u} → every algorithm
degrades.  The figure's point is the *shape* of that degradation: the
centralized bound and the CORDIS variants bend down gracefully (or their
outage rises gently), while interference-naïve baselines (e.g. MRT) cliff-edge
sooner.  That contrast is why this figure plots **all** algorithms in the run
by default (the trio leading, benchmarks appended).

A vertical dashed line marks the deployed operating point **P_p\\*** (the pilot
power the headline figures run at, 128 dB by default ≈ near-perfect CSI).  The
knee (~110 dB for the default ap_radius=650 m, where the predicted NMSE ≈ 0.1)
is tied to the large-scale fading β; if the campaign changes
``topology.ap_radius_m`` the informative sweep range shifts with it.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_sweep``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_csi/build_fig_csi.py
    python3 .../build_fig_csi.py --comm-metric outage --scnr-metric min
    python3 .../build_fig_csi.py --comm-metric mean --band none --no-tex
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

# Top-panel communication metric.  'min'/'mean' are SINR (dB); 'outage' is the
# per-user outage probability Pr(SINR_u < γ\*).  -> (registry_metric|None, label)
SINR_METRIC_CHOICES: Dict[str, Tuple[str, str]] = {
    "min":  ("min_sinr_db",  "min-user SINR"),
    "mean": ("mean_sinr_db", "mean-user SINR"),
}
COMM_METRIC_DEFAULT = "min"   # 'min' | 'mean' | 'outage'

# Bottom-panel SCNR metric.  -> (registry_name, axis_label, legend_short)
# 'wsum' = the WEIGHTED target-SCNR sum Σ_t ω_t·SCNR_{a_r,t} ("weighted_sum_scnr_db";
# there is no plain "sum_scnr_db" in the registry).
SCNR_METRIC_CHOICES: Dict[str, Tuple[str, str, str]] = {
    "min":  ("min_scnr_db",          "min-target SCNR",   "min-target SCNR"),
    "mean": ("mean_scnr_db",         "mean-target SCNR",  "mean SCNR"),
    "wsum": ("weighted_sum_scnr_db", "weighted-sum SCNR", "w-sum SCNR"),
}
SCNR_METRIC_DEFAULT = "wsum"

# Dispersion band drawn around each curve (passed to plot_sweep's `error`, and
# applied to the manual outage curve too).
BAND_DEFAULT = "std"   # 'std' | 'iqr' | 'none'

# Deployed operating point P_p\* (matches PILOT_POWER_DB in _defaults.sh).
PILOT_STAR_DEFAULT_DB = 128.0

# SINR target γ\* used by --comm-metric outage (auto-detected from metadata).
GAMMA_DB_DEFAULT = 5.0


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
    """Load a sweep (``kind == 'sweep'``) :class:`ExperimentResult`."""
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


def _resolve_sinr_metric(key: str) -> Tuple[str, str]:
    """Map a SINR key ('min'|'mean') to (registry_metric, label)."""
    if key not in SINR_METRIC_CHOICES:
        raise ValueError(f"SINR metric must be one of "
                         f"{list(SINR_METRIC_CHOICES)}, got {key!r}.")
    return SINR_METRIC_CHOICES[key]


def _resolve_scnr_metric(key: str,
                         result=None,
                         only: Optional[Sequence[str]] = None
                         ) -> Tuple[Optional[str], str, str]:
    """Map a --scnr-metric key ('min'|'mean'|'wsum') to
    (registry_metric, axis_label, legend_short), with graceful fallback if the
    chosen metric is absent from the present algorithms."""
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
    for k2, (n2, f2, s2) in SCNR_METRIC_CHOICES.items():
        if ars and all(a.has_metric(n2) for a in ars):
            logger.warning("SCNR metric %r not in run; using %r instead.", name, n2)
            return n2, f2, s2
    logger.warning("No SCNR metric available in run; bottom panel left empty.")
    return None, full, short


def _nearest_pilot(result, pilot_star: float) -> float:
    keys = sorted(result.sweep_results.keys())
    return min(keys, key=lambda k: abs(k - pilot_star))


def _in_range(result, x: float) -> bool:
    keys = sorted(result.sweep_results.keys())
    return keys[0] <= x <= keys[-1]


def _detect_gamma_db(result, fallback: float = GAMMA_DB_DEFAULT) -> Tuple[float, bool]:
    """Recover the SINR target γ\\* (dB) from run metadata (cfg_summary's
    ``gamma_u_db`` / ``gamma_db``).  Returns ``(gamma_db, detected)``."""
    meta = getattr(result, "metadata", None) or {}

    def _scan(d: Any) -> Optional[float]:
        if not isinstance(d, dict):
            return None
        for key in ("gamma_u_db", "gamma_db", "gamma"):
            if key in d:
                try:
                    fv = float(d[key])
                    if np.isfinite(fv):
                        return fv
                except (TypeError, ValueError):
                    pass
        for k, v in d.items():
            kl = str(k).lower()
            if kl.endswith("gamma_u_db") or kl.endswith("gamma_db"):
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        return fv
                except (TypeError, ValueError):
                    continue
        return None

    g = _scan(meta)
    if g is None:
        g = _scan(meta.get("cfg_summary"))
    if g is None:
        return float(fallback), False
    return float(g), True


def _sinr_arrays(ar):
    """(flat per-user SINR dB pool, per-trial×user SINR dB matrix) or (None, None)."""
    st = getattr(ar, "sinr_stats", None)
    if st is None:
        return None, None
    return (np.asarray(st.all_sinr_db_flat, dtype=np.float64),
            np.asarray(st.sinr_per_trial_per_user_db, dtype=np.float64))


def _outage_xy(result, name: str, gamma_db: float, band: str):
    """Per-algorithm outage curve vs the sweep axis: ``(xs, line, lo, hi)``.

    ``line`` is the per-user outage probability Pr(SINR_u < γ) over the pool at
    each sweep point; the band (if any) is the std / IQR of the per-trial outage
    fraction (1 − coverage), clipped to [0, 1]."""
    from cordis.metrics import outage as _o  # lazy
    xs: List[float] = []
    line: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    for x in sorted(result.sweep_results.keys()):
        ar = result.sweep_results[x].algorithm_results.get(name)
        if ar is None:
            continue
        flat, mat = _sinr_arrays(ar)
        if flat is None or flat.size == 0:
            continue
        p = float(_o.outage_probability(flat, gamma_db))
        xs.append(x); line.append(p)
        if band in ("std", "iqr") and mat is not None and mat.size:
            per_trial = 1.0 - np.asarray(_o.coverage_per_trial(mat, gamma_db), float)
            if per_trial.size:
                if band == "std":
                    s = float(np.std(per_trial))
                    lo.append(max(0.0, p - s)); hi.append(min(1.0, p + s))
                else:
                    lo.append(float(np.percentile(per_trial, 25)))
                    hi.append(float(np.percentile(per_trial, 75)))
    return xs, line, lo, hi


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary
# ─────────────────────────────────────────────────────────────────────

def _comm_value(sim, name, comm_metric, sinr_name, gamma_db):
    a = sim.algorithm_results.get(name)
    if a is None:
        return None
    if comm_metric == "outage":
        flat, _ = _sinr_arrays(a)
        if flat is None or flat.size == 0:
            return None
        from cordis.metrics import outage as _o
        return float(_o.outage_probability(flat, gamma_db))
    if not a.has_metric(sinr_name):
        return None
    return a.percentile(sinr_name, 50)


def collect_summary(result,
                    only: Sequence[str],
                    comm_metric: str,
                    sinr_name: str,
                    scnr_metric: Optional[str],
                    gamma_db: float,
                    pilot_star: float) -> Dict[str, Dict[str, float]]:
    """Per-algorithm robustness numbers: the communication metric (SINR or
    outage) at the worst-CSI, operating, and best-CSI pilot powers, the
    best→worst change, and the SCNR change.  A small comm change = robust /
    graceful; a large one = cliff-edge."""
    keys = sorted(result.sweep_results.keys())
    p_lo, p_hi = keys[0], keys[-1]
    p_op = _nearest_pilot(result, pilot_star)

    def _med(sim, name, metric):
        a = sim.algorithm_results.get(name)
        if a is None or metric is None or not a.has_metric(metric):
            return None
        return a.percentile(metric, 50)

    out: Dict[str, Dict[str, float]] = {}
    for name in only:
        row: Dict[str, float] = {}
        c_lo = _comm_value(result.sweep_results[p_lo], name, comm_metric, sinr_name, gamma_db)
        c_op = _comm_value(result.sweep_results[p_op], name, comm_metric, sinr_name, gamma_db)
        c_hi = _comm_value(result.sweep_results[p_hi], name, comm_metric, sinr_name, gamma_db)
        if c_lo is not None:
            row["comm_worst"] = c_lo
        if c_op is not None:
            row["comm_op"] = c_op
        if c_hi is not None:
            row["comm_best"] = c_hi
        if c_lo is not None and c_hi is not None:
            row["comm_delta"] = c_hi - c_lo     # best − worst CSI
        if scnr_metric:
            s_lo = _med(result.sweep_results[p_lo], name, scnr_metric)
            s_hi = _med(result.sweep_results[p_hi], name, scnr_metric)
            if s_lo is not None and s_hi is not None:
                row["scnr_drop_db"] = s_hi - s_lo
        out[name] = row
    return out


def _print_summary(result, summary, comm_metric, comm_label, scnr_metric,
                   gamma_db, pilot_star) -> None:
    keys = sorted(result.sweep_results.keys())
    p_op = _nearest_pilot(result, pilot_star)
    print(f"[info] pilot grid = {keys} dB")
    print(f"[info] P_p* (mark)= {pilot_star:g} dB  (nearest swept {p_op:g} dB)"
          + ("" if _in_range(result, pilot_star) else "  [OUT OF RANGE — marker skipped]"))
    print(f"[info] comm metric= {comm_label}"
          + (f"  (γ*={gamma_db:g} dB)" if comm_metric == "outage" else ""))
    print(f"[info] SCNR metric= {scnr_metric or '(none)'}")
    cw = comm_label[:11]
    hdr = (f"{'algorithm':<14}{cw+' worst':>16}{'op':>9}{'best':>9}"
           f"{'Δ(best-worst)':>15}{'SCNR drop':>11}")
    print(hdr); print("-" * len(hdr))
    for name, r in summary.items():
        def f(k):
            v = r.get(k); return (f"{v:.3f}" if v is not None else "—")
        print(f"{name:<14}{f('comm_worst'):>16}{f('comm_op'):>9}{f('comm_best'):>9}"
              f"{f('comm_delta'):>15}{f('scnr_drop_db'):>11}")


# ─────────────────────────────────────────────────────────────────────
#  Figure  (self-contained so the notebook can inline an editable copy)
# ─────────────────────────────────────────────────────────────────────

def build_figure(result,
                 *,
                 only: Optional[Sequence[str]] = None,
                 comm_metric: str = COMM_METRIC_DEFAULT,
                 scnr_metric: str = SCNR_METRIC_DEFAULT,
                 band: str = BAND_DEFAULT,
                 gamma_db: float = GAMMA_DB_DEFAULT,
                 pilot_star: float = PILOT_STAR_DEFAULT_DB,
                 log_x: bool = False,
                 use_tex: bool = True):
    """Assemble the stacked CSI-robustness figure; return
    ``(fig, only, scnr_metric_name)``.

    ``comm_metric`` is 'min'|'mean' (SINR) or 'outage' (per-user Pr(SINR<γ\\*));
    ``scnr_metric`` is 'min'|'mean'|'wsum'; ``band`` is 'std'|'iqr'|'none'.
    """
    import matplotlib
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    from cordis.plotting import apply_paper_style, figsize, plot_sweep, style_for

    apply_paper_style()
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False

    only = list(only) if only else _ordered_algorithms(result, LEAD)
    scnr_name, scnr_full, _scnr_short = _resolve_scnr_metric(scnr_metric, result, only)

    axis = result.sweep_axis
    xlabel = axis.display or r"Pilot power $P_p/\sigma_n^2$ [dB]"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5 / 2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    # (top) communication metric vs pilot power
    if comm_metric == "outage":
        for n in only:
            xs, line, lo, hi = _outage_xy(result, n, gamma_db, band)
            if not xs:
                continue
            st = style_for(n)
            ax_top.plot(xs, line, label=st.get("label", n),
                        color=st.get("color"), marker=st.get("marker", "o"),
                        ls=st.get("linestyle", "-"))
            if band in ("std", "iqr") and lo:
                ax_top.fill_between(xs, lo, hi, color=st.get("color"),
                                    alpha=0.15, linewidth=0)
        if log_x:
            ax_top.set_xscale("log")
        ax_top.set_ylim(bottom=0.0)
        ax_top.set_ylabel(rf"$P_{{\mathrm{{out}}}}$ at $\gamma\!=\!{gamma_db:g}$ dB")
        ax_top.grid(True, alpha=0.3)
        if ax_top.get_legend_handles_labels()[0]:
            ax_top.legend(loc="best", fontsize=6)
    else:
        sinr_name, sinr_label = _resolve_sinr_metric(comm_metric)
        plot_sweep(result.sweep_results, metric=sinr_name, ax=ax_top, error=band,
                   xlabel="", ylabel=f"{sinr_label} [dB]", only=only, log_x=log_x)
    ax_top.set_title("Robustness to imperfect CSI")

    # (bot) SCNR vs pilot power
    if scnr_name is not None:
        plot_sweep(result.sweep_results, metric=scnr_name, ax=ax_bot, error=band,
                   xlabel=xlabel, ylabel=f"{scnr_full} [dB]", only=only, log_x=log_x)
    else:
        logger.warning("No SCNR metric available; bottom panel left empty.")
        ax_bot.set_xlabel(xlabel)

    # operating-point marker P_p* (only if inside the swept range)
    if _in_range(result, pilot_star):
        for ax in (ax_top, ax_bot):
            ax.axvline(pilot_star, ls="--", lw=0.9, color="0.45", zorder=0)
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
    # layout.  save_figure() uses bbox_inches='tight'.
    return fig, only, scnr_name


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
    p.add_argument("--comm-metric", choices=("min", "mean", "outage"),
                   default=COMM_METRIC_DEFAULT,
                   help="Top-panel communication metric: 'min'/'mean' SINR [dB], "
                        "or 'outage' = per-user Pr(SINR<γ*).")
    p.add_argument("--scnr-metric", choices=list(SCNR_METRIC_CHOICES),
                   default=SCNR_METRIC_DEFAULT,
                   help="Bottom-panel SCNR metric: 'wsum' (weighted target-SCNR "
                        "sum), 'min' (worst target), or 'mean'.")
    p.add_argument("--band", choices=("std", "iqr", "none"), default=BAND_DEFAULT,
                   help="Dispersion band around each curve: std, iqr, or none.")
    p.add_argument("--gamma-db", type=float, default=None,
                   help="SINR target γ* [dB] for --comm-metric outage. "
                        "Default: auto-detected from run metadata, else "
                        f"{GAMMA_DB_DEFAULT:g}.")
    p.add_argument("--pilot-star", type=float, default=PILOT_STAR_DEFAULT_DB,
                   help="Operating-point pilot power P_p* [dB] marked with a line.")
    p.add_argument("--only", default=None,
                   help="Comma-separated algorithm display names. Default: ALL "
                        f"present, with {', '.join(LEAD)} leading.")
    p.add_argument("--log-x", action="store_true",
                   help="Log-scale the pilot-power axis (rarely needed; dB already).")
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
    scnr_name, _scnr_full, _scnr_short = _resolve_scnr_metric(
        args.scnr_metric, result, only_resolved)

    gamma_db, detected = (args.gamma_db, True) if args.gamma_db is not None \
        else _detect_gamma_db(result)
    if args.comm_metric == "outage" and not detected:
        logger.warning("γ* not found in metadata; using %.4g dB for outage. "
                       "Override with --gamma-db.", gamma_db)

    if args.comm_metric == "outage":
        sinr_name, comm_label = None, rf"outage Pout"
    else:
        sinr_name, comm_label = _resolve_sinr_metric(args.comm_metric)

    summary = collect_summary(result, only_resolved, args.comm_metric,
                              sinr_name or "min_sinr_db", scnr_name,
                              gamma_db, args.pilot_star)
    _print_summary(result, summary, args.comm_metric, comm_label, scnr_name,
                   gamma_db, args.pilot_star)

    fig, used_only, used_scnr_metric = build_figure(
        result, only=only_resolved, comm_metric=args.comm_metric,
        scnr_metric=args.scnr_metric, band=args.band, gamma_db=gamma_db,
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
            "CommMetric": args.comm_metric
                          + (f"(gamma={gamma_db:g})" if args.comm_metric == "outage" else ""),
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

