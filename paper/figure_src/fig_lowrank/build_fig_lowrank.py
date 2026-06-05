#!/usr/bin/env python3
"""
paper/figure_src/fig_lowrank/build_fig_lowrank.py
=================================================

Build the locally-low-rank scalability figure for the journal paper's
Simulation Results section — CORDIS-ADMM's headline advantage — from an
``n_ue_sweep`` result run in the few-antennas-per-AP regime.

Stacked two-panel, single-column (shared x-axis = number of UEs N_ue):

  (top)  the **communication** metric vs N_ue — selectable (``--comm-metric``):
           * 'min'    : worst-user SINR  [dB]   (default)
           * 'mean'   : across-user SINR [dB]
           * 'outage' : per-user outage probability Pr(SINR_u < γ\\*)
  (bot)  the achieved **SCNR** vs N_ue — selectable (``--scnr-metric``):
           'wsum' (weighted target-SCNR sum, default), 'min', or 'mean'.

The campaign fixes a small per-AP array (M = n_ant antennas per AP, with the
total Tx-antenna budget held constant) and increases the user load N_ue.  The
per-AP spatial degrees of freedom are M, so once **N_ue > M** the local
channels are *locally low-rank*: the fixed-local-beamformer Algorithm 1
(CORDIS-Split) can no longer null all co-users from a single AP and its QoS
degrades / diverges (its SINR drops, its outage spikes), whereas the
consensus-ADMM Algorithm 2 (CORDIS-ADMM) keeps meeting the SINR floor by
coordinating across APs — tracking the centralized ceiling.  That divergence
in the shaded ``N_ue > M`` region is the figure's money shot.

The locally-low-rank region (N_ue > M) is shaded on both panels; the
full-rank reference point is N_ue = M.

Each curve's central line is the across-trial mean (the outage line is the
per-user pool fraction); an optional shaded band shows dispersion
(``--band std|iqr|none``).

Filtered to the proposed pair vs the centralized ceiling
(``Centralized``, ``CORDIS-ADMM``, ``CORDIS-Split``) using the project-wide
per-algorithm style, so a curve reads identically here and in the siblings.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_sweep``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py
    python3 .../build_fig_lowrank.py --comm-metric outage --scnr-metric min
    python3 .../build_fig_lowrank.py --comm-metric mean --band none --rank-threshold 3
    python3 .../build_fig_lowrank.py --result-dir results/exp_n_ue_sweep/array_12345_aggregated
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("fig_lowrank")


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
#  Which algorithms, which metrics, which regime boundary
# ─────────────────────────────────────────────────────────────────────

# The proposed pair vs the centralized ceiling.  CORDIS-Split is the one that
# diverges in the low-rank region — keeping it in the figure is the point.
PREFERRED: Tuple[str, ...] = ("Centralized", "CORDIS-ADMM", "CORDIS-Split")

# Top-panel communication metric.  'min'/'mean' are SINR (dB); 'outage' is the
# per-user outage probability Pr(SINR_u < γ\*).  -> (registry_metric, label)
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

# Dispersion band drawn around each curve.
BAND_DEFAULT = "std"   # 'std' | 'iqr' | 'none'

# Fallback per-AP DoF (antennas per AP, M) if it can't be read from metadata.
# The locally-low-rank campaign runs at n_ant=3.
RANK_THRESHOLD_DEFAULT = 3

# SINR target γ\* used by --comm-metric outage (auto-detected from metadata).
GAMMA_DB_DEFAULT = 5.0


# ─────────────────────────────────────────────────────────────────────
#  Result discovery  (array-aggregated first, never `latest`)
# ─────────────────────────────────────────────────────────────────────

import re as _re

_ARRAY_AGG_PATTERN = _re.compile(r"^array_(\d+)_aggregated$")
_ARRAY_TASK_PATTERN = _re.compile(r"^array_(\d+)_task_(\d+)$")


def _discover_result_dir(experiment: str = "n_ue_sweep",
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
               experiment: str = "n_ue_sweep",
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
                f"low-rank {experiment!r} campaign (or aggregate its array "
                f"job) first."
            )
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = REPO_ROOT / result_dir

    result = ExperimentResult.load(result_dir)
    if getattr(result, "kind", None) != "sweep":
        raise ValueError(
            f"fig_lowrank needs a sweep (kind='sweep') result; "
            f"{result_dir} has kind={getattr(result, 'kind', None)!r}."
        )
    return result, result_dir


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _first_point(result):
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


def _detect_rank_threshold(result, fallback: int = RANK_THRESHOLD_DEFAULT) -> Tuple[int, bool]:
    """Recover the per-AP DoF M (= antennas per AP) from run metadata.

    Looks for an ``n_ant`` entry in metadata / cfg_summary.  Returns
    ``(M, detected)``; ``detected`` is False when we fell back.
    """
    meta = getattr(result, "metadata", None) or {}

    def _scan(d: Any) -> Optional[int]:
        if not isinstance(d, dict):
            return None
        for key in ("n_ant", "M", "n_antennas"):
            if key in d:
                try:
                    return int(d[key])
                except (TypeError, ValueError):
                    pass
        for k, v in d.items():
            kl = str(k).lower()
            if kl == "n_ant" or kl.endswith(".n_ant"):
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
        return None

    m = _scan(meta)
    if m is None:
        m = _scan(meta.get("cfg_summary"))
    if m is None:
        return int(fallback), False
    return int(m), True


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
    """Per-algorithm outage curve vs N_ue: ``(xs, line, lo, hi)``.

    ``line`` is the per-user outage probability Pr(SINR_u < γ) at each N_ue; the
    band (if any) is the std / IQR of the per-trial outage fraction, in [0, 1]."""
    eta = 0.85 if name == "CORDIS-ADMM" else 1.0
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
        p = float(_o.outage_probability(flat, eta*gamma_db))
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
                    rank_threshold: int) -> Dict[str, Dict[str, float]]:
    """Per-algorithm numbers at the full-rank reference (N_ue ≈ M) and the most
    loaded point (max N_ue): the communication metric (SINR or outage) and SCNR,
    plus the reference→max-load change."""
    keys = sorted(result.sweep_results.keys())
    n_hi = keys[-1]
    n_ref = min(keys, key=lambda k: abs(k - rank_threshold))

    def _med(sim, name, metric):
        a = sim.algorithm_results.get(name)
        if a is None or metric is None or not a.has_metric(metric):
            return None
        return a.percentile(metric, 50)

    out: Dict[str, Dict[str, float]] = {}
    for name in only:
        row: Dict[str, float] = {}
        c_ref = _comm_value(result.sweep_results[n_ref], name, comm_metric, sinr_name, gamma_db)
        c_hi = _comm_value(result.sweep_results[n_hi], name, comm_metric, sinr_name, gamma_db)
        if c_ref is not None:
            row["comm_ref"] = c_ref
        if c_hi is not None:
            row["comm_maxload"] = c_hi
        if c_ref is not None and c_hi is not None:
            row["comm_delta"] = c_hi - c_ref   # SINR: <0 degrade; outage: >0 degrade
        if scnr_metric:
            s_ref = _med(result.sweep_results[n_ref], name, scnr_metric)
            s_hi = _med(result.sweep_results[n_hi], name, scnr_metric)
            if s_ref is not None:
                row["scnr_ref_db"] = s_ref
            if s_hi is not None:
                row["scnr_maxload_db"] = s_hi
        out[name] = row
    return out


def _print_summary(result, summary, comm_metric, comm_label, scnr_metric,
                   gamma_db, rank_threshold, detected) -> None:
    keys = sorted(result.sweep_results.keys())
    n_ref = min(keys, key=lambda k: abs(k - rank_threshold))
    n_hi = keys[-1]
    print(f"[info] N_ue grid     = {[int(k) for k in keys]}")
    print(f"[info] per-AP DoF M  = {rank_threshold}"
          + ("" if detected else "  (fallback — set --rank-threshold)"))
    print(f"[info] low-rank region: N_ue > {rank_threshold}  "
          f"(full-rank ref N_ue={int(n_ref)}, max-load N_ue={int(n_hi)})")
    print(f"[info] comm metric   = {comm_label}"
          + (f"  (γ*={gamma_db:g} dB)" if comm_metric == "outage" else ""))
    print(f"[info] SCNR metric   = {scnr_metric or '(none)'}")
    cw = comm_label[:11]
    hdr = (f"{'algorithm':<14}{cw+'@ref':>15}{'@max':>9}{'Δ(max-ref)':>12}"
           f"{'SCNR@ref':>9}{'SCNR@max':>9}")
    print(hdr); print("-" * len(hdr))
    for name, r in summary.items():
        def f(k):
            v = r.get(k); return (f"{v:.3f}" if v is not None else "—")
        print(f"{name:<14}{f('comm_ref'):>15}{f('comm_maxload'):>9}{f('comm_delta'):>12}"
              f"{f('scnr_ref_db'):>9}{f('scnr_maxload_db'):>9}")
    # headline: CORDIS-ADMM advantage over CORDIS-Split at max load
    admm = summary.get("CORDIS-ADMM", {}).get("comm_maxload")
    split = summary.get("CORDIS-Split", {}).get("comm_maxload")
    if admm is not None and split is not None:
        if comm_metric == "outage":
            adv = split - admm    # Split has higher outage -> positive = ADMM better
            unit = ""
        else:
            adv = admm - split    # ADMM has higher SINR -> positive = ADMM better
            unit = " dB"
        print(f"[headline] CORDIS-ADMM advantage over CORDIS-Split at N_ue={int(n_hi)}: "
              f"{adv:+.3f}{unit}  (the low-rank divergence)")


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
                 rank_threshold: int = RANK_THRESHOLD_DEFAULT,
                 shade_lowrank: bool = True,
                 use_tex: bool = True):
    """Assemble the stacked low-rank scalability figure; return
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

    only = list(only) if only else _present_algorithms(result, PREFERRED)
    scnr_name, scnr_full, _scnr_short = _resolve_scnr_metric(scnr_metric, result, only)

    axis = result.sweep_axis
    xlabel = axis.display or r"$N_{\rm UE}$"
    keys = sorted(result.sweep_results.keys())
    x_lo, x_hi = keys[0], keys[-1]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5 / 2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    # (top) communication metric vs N_ue
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
        ax_top.set_ylim(bottom=0.0)
        ax_top.set_ylabel(rf"$P_{{\mathrm{{out}}}}$ at $\gamma\!=\!{gamma_db:g}$ dB")
        ax_top.grid(True, alpha=0.3)
        if ax_top.get_legend_handles_labels()[0]:
            ax_top.legend(loc="best", fontsize=6)
    else:
        sinr_name, sinr_label = _resolve_sinr_metric(comm_metric)
        plot_sweep(result.sweep_results, metric=sinr_name, ax=ax_top, error=band,
                   xlabel="", ylabel=f"{sinr_label} [dB]", only=only, log_x=False)
    ax_top.set_title("Scalability in the locally low-rank regime")

    # (bot) SCNR vs N_ue
    if scnr_name is not None:
        plot_sweep(result.sweep_results, metric=scnr_name, ax=ax_bot, error=band,
                   xlabel=xlabel, ylabel=f"{scnr_full} [dB]", only=only, log_x=False)
    else:
        logger.warning("No SCNR metric available; bottom panel left empty.")
        ax_bot.set_xlabel(xlabel)

    # integer N_ue ticks + a little x-margin
    for ax in (ax_top, ax_bot):
        ax.set_xticks([int(k) for k in keys])
        ax.set_xlim(x_lo - 0.3, x_hi + 0.3)

    # shade the locally-low-rank region  N_ue > M  on both panels
    shade_left = rank_threshold + 0.5
    if shade_lowrank and shade_left < x_hi + 0.3:
        for ax in (ax_top, ax_bot):
            ax.axvspan(max(shade_left, x_lo - 0.3), x_hi + 0.3,
                       color="0.85", alpha=0.5, zorder=0, linewidth=0)
        ax_top.text(0.99, 0.04,
                    r"locally low-rank ($N_{\rm UE} > M$)",
                    transform=ax_top.transAxes, ha="right", va="bottom",
                    fontsize="x-small", color="0.35")

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
        prog="build_fig_lowrank",
        description="Build the locally-low-rank scalability figure (n_ue sweep) "
                    "for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--result-dir", default=None,
                   help="Specific results/exp_n_ue_sweep/<run>/ to load. "
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
                        "Default: auto-detected from metadata, else "
                        f"{GAMMA_DB_DEFAULT:g}.")
    p.add_argument("--rank-threshold", type=int, default=None,
                   help="Per-AP DoF M (antennas/AP); N_ue > M is shaded "
                        "locally-low-rank.  Default: auto-detected from run "
                        f"metadata, else {RANK_THRESHOLD_DEFAULT}.")
    p.add_argument("--no-shade", action="store_true",
                   help="Do not shade the locally-low-rank region.")
    p.add_argument("--only", default=None,
                   help="Comma-separated algorithm display names. "
                        f"Default: {', '.join(PREFERRED)} (filtered to those present).")
    p.add_argument("--out", default=str(FIGURES_OUT / "fig_lowrank"),
                   help="Output path stem (no extension).")
    p.add_argument("--no-tex", action="store_true",
                   help="Disable LaTeX text rendering (nodes without pdflatex).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _build_parser().parse_args(argv)

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    result, result_dir = load_sweep(
        Path(args.result_dir) if args.result_dir else None, experiment="n_ue_sweep")

    rank_threshold, rank_detected = (args.rank_threshold, True) if args.rank_threshold is not None \
        else _detect_rank_threshold(result)
    gamma_db, gamma_detected = (args.gamma_db, True) if args.gamma_db is not None \
        else _detect_gamma_db(result)
    if args.comm_metric == "outage" and not gamma_detected:
        logger.warning("γ* not found in metadata; using %.4g dB for outage. "
                       "Override with --gamma-db.", gamma_db)

    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {result_dir}")

    only_resolved = list(only) if only else _present_algorithms(result, PREFERRED)
    scnr_name, _scnr_full, _scnr_short = _resolve_scnr_metric(
        args.scnr_metric, result, only_resolved)

    if args.comm_metric == "outage":
        sinr_name, comm_label = None, "outage Pout"
    else:
        sinr_name, comm_label = _resolve_sinr_metric(args.comm_metric)

    summary = collect_summary(result, only_resolved, args.comm_metric,
                              sinr_name or "min_sinr_db", scnr_name,
                              gamma_db, rank_threshold)
    _print_summary(result, summary, args.comm_metric, comm_label, scnr_name,
                   gamma_db, rank_threshold, rank_detected)

    fig, used_only, used_scnr_metric = build_figure(
        result, only=only_resolved, comm_metric=args.comm_metric,
        scnr_metric=args.scnr_metric, band=args.band, gamma_db=gamma_db,
        rank_threshold=rank_threshold, shade_lowrank=not args.no_shade,
        use_tex=not args.no_tex)

    out_stem = Path(args.out)
    if not out_stem.is_absolute():
        out_stem = Path.cwd() / out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_lowrank",
            "RankThresholdM": str(rank_threshold),
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

