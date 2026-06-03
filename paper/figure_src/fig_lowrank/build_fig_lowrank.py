#!/usr/bin/env python3
"""
paper/figure_src/fig_lowrank/build_fig_lowrank.py
=================================================

Build the locally-low-rank scalability figure for the journal paper's
Simulation Results section — CORDIS-ADMM's headline advantage — from an
``n_ue_sweep`` result run in the few-antennas-per-AP regime.

Stacked two-panel, single-column (shared x-axis = number of UEs N_ue):

  (top)  worst-user **min-SINR** vs N_ue.
  (bot)  **sum-SCNR** vs N_ue.

The campaign fixes a small per-AP array (M = n_ant antennas per AP, with the
total Tx-antenna budget held constant) and increases the user load N_ue.  The
per-AP spatial degrees of freedom are M, so once **N_ue > M** the local
channels are *locally low-rank*: the fixed-local-beamformer Algorithm 1
(CORDIS-Split) can no longer null all co-users from a single AP and its QoS
degrades / diverges, whereas the consensus-ADMM Algorithm 2 (CORDIS-ADMM)
keeps meeting the SINR floor by coordinating across APs — tracking the
centralized ceiling.  That divergence in the shaded ``N_ue > M`` region is the
figure's money shot.

The locally-low-rank region (N_ue > M) is shaded on both panels; the
full-rank reference point is N_ue = M.

Filtered to the proposed pair vs the centralized ceiling
(``Centralized``, ``CORDIS-ADMM``, ``CORDIS-Split``) using the project-wide
per-algorithm style, so a curve reads identically here and in the siblings.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_sweep``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_lowrank/build_fig_lowrank.py
    python3 .../build_fig_lowrank.py --no-tex
    python3 .../build_fig_lowrank.py --rank-threshold 3
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

SINR_METRIC = "min_sinr_db"

SCNR_METRIC_PREFERENCE: Tuple[str, ...] = (
    "sum_scnr_db",
    "weighted_sum_scnr_db",
    "mean_scnr_db",
)

# Fallback per-AP DoF (antennas per AP, M) if it can't be read from metadata.
# The locally-low-rank campaign runs at n_ant=3.
RANK_THRESHOLD_DEFAULT = 3

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


def _detect_rank_threshold(result, fallback: int = RANK_THRESHOLD_DEFAULT) -> Tuple[int, bool]:
    """Recover the per-AP DoF M (= antennas per AP) from run metadata.

    Looks for an ``n_ant`` entry in metadata / cfg_summary.  Returns
    ``(M, detected)``; ``detected`` is False when we fell back.
    """
    meta = getattr(result, "metadata", None) or {}

    def _scan(d: Any) -> Optional[int]:
        if not isinstance(d, dict):
            return None
        # exact key first, then anything that looks like an antenna count
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


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary
# ─────────────────────────────────────────────────────────────────────

def collect_summary(result,
                    only: Sequence[str],
                    scnr_metric: Optional[str],
                    rank_threshold: int) -> Dict[str, Dict[str, float]]:
    """Per-algorithm numbers at the full-rank reference (N_ue = M, or the
    smallest swept N_ue if M isn't a grid point) and at the most-loaded point
    (max N_ue, deepest into the low-rank regime): median min-SINR and SCNR,
    plus the min-SINR drop from reference to max-load."""
    keys = sorted(result.sweep_results.keys())
    n_lo, n_hi = keys[0], keys[-1]
    # full-rank reference = nearest swept N_ue to M
    n_ref = min(keys, key=lambda k: abs(k - rank_threshold))

    def _med(sim, name, metric):
        a = sim.algorithm_results.get(name)
        if a is None or not a.has_metric(metric):
            return None
        return a.percentile(metric, 50)

    out: Dict[str, Dict[str, float]] = {}
    for name in only:
        row: Dict[str, float] = {}
        s_ref = _med(result.sweep_results[n_ref], name, SINR_METRIC)
        s_hi = _med(result.sweep_results[n_hi], name, SINR_METRIC)
        if s_ref is not None:
            row["sinr_ref_db"] = s_ref
        if s_hi is not None:
            row["sinr_maxload_db"] = s_hi
        if s_ref is not None and s_hi is not None:
            row["sinr_drop_db"] = s_hi - s_ref       # negative = degradation
        if scnr_metric:
            c_ref = _med(result.sweep_results[n_ref], name, scnr_metric)
            c_hi = _med(result.sweep_results[n_hi], name, scnr_metric)
            if c_ref is not None:
                row["scnr_ref_db"] = c_ref
            if c_hi is not None:
                row["scnr_maxload_db"] = c_hi
        out[name] = row
    return out


def _print_summary(result, summary, scnr_metric, rank_threshold, detected) -> None:
    keys = sorted(result.sweep_results.keys())
    n_ref = min(keys, key=lambda k: abs(k - rank_threshold))
    n_hi = keys[-1]
    print(f"[info] N_ue grid     = {[int(k) for k in keys]}")
    print(f"[info] per-AP DoF M  = {rank_threshold}"
          + ("" if detected else "  (fallback — set --rank-threshold)"))
    print(f"[info] low-rank region: N_ue > {rank_threshold}  "
          f"(full-rank ref N_ue={int(n_ref)}, max-load N_ue={int(n_hi)})")
    print(f"[info] SCNR metric   = {scnr_metric or '(none)'}")
    hdr = (f"{'algorithm':<14}{'SINR@ref':>9}{'SINR@max':>9}{'SINR drop':>10}"
           f"{'SCNR@ref':>9}{'SCNR@max':>9}")
    print(hdr); print("-" * len(hdr))
    for name, r in summary.items():
        def f(k):
            v = r.get(k); return (f"{v:.2f}" if v is not None else "—")
        print(f"{name:<14}{f('sinr_ref_db'):>9}{f('sinr_maxload_db'):>9}"
              f"{f('sinr_drop_db'):>10}{f('scnr_ref_db'):>9}{f('scnr_maxload_db'):>9}")
    # headline gap: ADMM vs Split at max load (the divergence)
    admm = summary.get("CORDIS-ADMM", {}).get("sinr_maxload_db")
    split = summary.get("CORDIS-Split", {}).get("sinr_maxload_db")
    if admm is not None and split is not None:
        print(f"[headline] min-SINR gap at N_ue={int(n_hi)}: "
              f"CORDIS-ADMM − CORDIS-Split = {admm - split:+.2f} dB "
              f"(the low-rank divergence)")


# ─────────────────────────────────────────────────────────────────────
#  Figure  (self-contained so the notebook can inline an editable copy)
# ─────────────────────────────────────────────────────────────────────

def build_figure(result,
                 *,
                 only: Optional[Sequence[str]] = None,
                 scnr_metric: Optional[str] = None,
                 rank_threshold: int = RANK_THRESHOLD_DEFAULT,
                 shade_lowrank: bool = True,
                 use_tex: bool = True):
    """Assemble the stacked low-rank scalability figure; return
    ``(fig, only, scnr_metric)``.

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

    only = list(only) if only else _present_algorithms(result, PREFERRED)
    if scnr_metric is None:
        scnr_metric = _select_scnr_metric(result, only=only)

    axis = result.sweep_axis
    xlabel = axis.display or r"$N_{\rm UE}$"
    keys = sorted(result.sweep_results.keys())
    x_lo, x_hi = keys[0], keys[-1]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5 / 2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    # (top) min-SINR vs N_ue
    plot_sweep(result.sweep_results, metric=SINR_METRIC, ax=ax_top,
               xlabel="", ylabel=r"min-SINR [dB]", only=only, log_x=False)
    ax_top.set_title("Scalability in the locally low-rank regime")

    # (bot) SCNR vs N_ue
    if scnr_metric is not None:
        plot_sweep(result.sweep_results, metric=scnr_metric, ax=ax_bot,
                   xlabel=xlabel,
                   ylabel=_SCNR_YLABEL.get(scnr_metric, scnr_metric),
                   only=only, log_x=False)
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
    return fig, only, scnr_metric


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
    p.add_argument("--rank-threshold", type=int, default=None,
                   help="Per-AP DoF M (antennas/AP); N_ue > M is shaded "
                        "locally-low-rank.  Default: auto-detected from run "
                        f"metadata, else {RANK_THRESHOLD_DEFAULT}.")
    p.add_argument("--no-shade", action="store_true",
                   help="Do not shade the locally-low-rank region.")
    p.add_argument("--scnr-metric", default=None,
                   help="SCNR metric for the bottom panel.  Default: first "
                        f"available of {', '.join(SCNR_METRIC_PREFERENCE)}.")
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

    rank_threshold, detected = (args.rank_threshold, True) if args.rank_threshold is not None \
        else _detect_rank_threshold(result)

    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {result_dir}")

    only_resolved = list(only) if only else _present_algorithms(result, PREFERRED)
    scnr_metric = args.scnr_metric or _select_scnr_metric(result, only=only_resolved)

    summary = collect_summary(result, only_resolved, scnr_metric, rank_threshold)
    _print_summary(result, summary, scnr_metric, rank_threshold, detected)

    fig, used_only, used_scnr_metric = build_figure(
        result, only=only_resolved, scnr_metric=scnr_metric,
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

