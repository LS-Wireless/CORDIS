#!/usr/bin/env python3
"""
paper/figure_src/fig_cdf/build_fig_cdf.py
=========================================

Build the empirical-CDF figure for the journal paper's Simulation Results
section: the worst-user **min-SINR CDF** and the **SCNR CDF** at the chosen
operating point γ\\*.

Two panels, double-column:

  (a) min-SINR CDF at γ\\*  (communication / worst-user reliability):
        empirical CDF of the per-trial worst-user SINR for each algorithm,
        with a vertical dashed line at γ\\*.  The mass to the LEFT of γ\\* is
        the per-user outage / infeasibility fraction; ``plot_cdf`` annotates
        each algorithm's infeasibility rate in the legend (Stage-20 feature).
        Read it as: how reliably does each algorithm seat the worst user at
        or above the requested floor.

  (b) SCNR CDF at γ\\*  (sensing):
        empirical CDF of the per-trial sensing SCNR for each algorithm at the
        SAME operating point.  No γ line here — γ\\* is an SINR floor, not an
        SCNR target; this panel shows the sensing performance you actually get
        once the communication floor is being enforced at γ\\*.

Both panels share the same algorithm set and the project-wide per-algorithm
style (``style_for`` / ``ALGORITHM_STYLE``) so a curve reads identically here
and in fig_cs_tradeoff.

Data sources (the figure index): the SINR panel is drawn from a ``sinr_cdf``
run and the SCNR panel from a ``scnr_cdf`` run.  Both experiments compute the
full per-trial SINR *and* SCNR statistics, so if only one campaign has
finished the builder reuses that single run for both panels and says so.
Monte-Carlo campaigns for the paper are SLURM job arrays, so discovery prefers
the merged ``array_<jobid>_aggregated/`` directory and never trusts the
``latest`` symlink (rsync ordering can leave it on a per-task dir).

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_cdf``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_cdf/build_fig_cdf.py
    python3 .../build_fig_cdf.py --no-tex
    python3 .../build_fig_cdf.py --gamma-db 5
    python3 .../build_fig_cdf.py --scnr-metric weighted_sum_scnr_db
    python3 .../build_fig_cdf.py --sinr-dir results/exp_sinr_cdf/array_12345_aggregated
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("fig_cdf")


# ─────────────────────────────────────────────────────────────────────
#  Path plumbing (mirrors fig_convergence / fig_cs_tradeoff so every
#  builder behaves alike and survives renaming paper/)
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

# Same trio as the C-S tradeoff figure: the proposed pair vs the centralized
# ceiling.  CDF experiments run with all_algorithms, so we filter down.
PREFERRED: Tuple[str, ...] = ("Centralized", "CORDIS-ADMM", "CORDIS-Split")

SINR_METRIC = "min_sinr_db"

# For the sensing panel, prefer the worst-target SCNR (the dual of the
# worst-user SINR — both are "worst-case" CDFs, the cleanest pairing for the
# one-target scenario in the draft).  Fall back through sum / mean if a run
# doesn't carry it.  First metric the result actually has wins.
SCNR_METRIC_PREFERENCE: Tuple[str, ...] = (
    "min_scnr_db",
    "weighted_sum_scnr_db",
    "mean_scnr_db",
)

# Last-resort default for γ\* if it can't be recovered from run metadata.
# (Matches the cross-figure operating point agreed for fig_cs_tradeoff.)
GAMMA_DEFAULT_DB = 5.0

_SCNR_XLABEL = {
    "min_scnr_db": r"worst-target SCNR [dB]",
    "weighted_sum_scnr_db": r"sum-SCNR [dB]",
    "mean_scnr_db": r"mean SCNR [dB]",
}


# ─────────────────────────────────────────────────────────────────────
#  Result discovery  (array-aggregated first, never `latest`)
# ─────────────────────────────────────────────────────────────────────

import re as _re

_ARRAY_AGG_PATTERN = _re.compile(r"^array_(\d+)_aggregated$")
_ARRAY_TASK_PATTERN = _re.compile(r"^array_(\d+)_task_(\d+)$")


def _discover_result_dir(experiment: str,
                         results_root: Optional[Path] = None) -> Optional[Path]:
    """Newest result dir for ``exp_<experiment>``.

    Preference order, robust to HPC sync quirks:
      1. the highest-numbered ``array_<jobid>_aggregated/`` (the merged
         campaign — this is what paper figures should use);
      2. otherwise the most recently modified plain run dir (timestamped),
         excluding per-task ``array_*_task_*`` dirs and the ``latest`` symlink.
    Returns None if nothing usable is found.
    """
    base = (results_root or (REPO_ROOT / "results")) / f"exp_{experiment}"
    if not base.is_dir():
        return None

    # 1) aggregated array runs
    aggs: List[Tuple[int, Path]] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        m = _ARRAY_AGG_PATTERN.match(child.name)
        if m:
            aggs.append((int(m.group(1)), child))
    if aggs:
        aggs.sort()
        return aggs[-1][1]

    # 2) plain timestamped runs (skip per-task dirs and the `latest` symlink)
    plains: List[Path] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == "latest" or child.is_symlink():
            continue
        if _ARRAY_TASK_PATTERN.match(child.name):
            continue
        if not (child / "manifest.json").exists():
            continue
        plains.append(child)
    if not plains:
        return None
    plains.sort(key=lambda d: d.stat().st_mtime)
    return plains[-1]


def load_cdf(result_dir: Optional[Path] = None,
             experiment: str = "sinr_cdf",
             results_root: Optional[Path] = None):
    """Load a CDF (``kind == 'single'``) :class:`ExperimentResult`.

    If ``result_dir`` is given, that directory is loaded verbatim; otherwise
    the newest run of ``exp_<experiment>`` is auto-discovered (aggregated
    array run preferred).  Raises ``FileNotFoundError`` if nothing is found.
    """
    from cordis.experiments.result import ExperimentResult  # lazy: keep import light

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
    if getattr(result, "kind", None) != "single":
        raise ValueError(
            f"fig_cdf needs a CDF (kind='single') result; "
            f"{result_dir} has kind={getattr(result, 'kind', None)!r}."
        )
    return result, result_dir


# ─────────────────────────────────────────────────────────────────────
#  Helpers: γ\* detection, algorithm filtering, SCNR metric selection
# ─────────────────────────────────────────────────────────────────────

def _detect_gamma_db(result, fallback: float = GAMMA_DEFAULT_DB) -> Tuple[float, bool]:
    """Recover the operating-point γ\\* (dB) from a result's metadata.

    Searches metadata (and its ``cfg_summary`` block) for any key whose name
    contains "gamma" and whose value is a finite number.  Returns
    ``(gamma_db, detected)`` — ``detected`` is False when we fell back.
    """
    meta = getattr(result, "metadata", None) or {}

    def _scan(d: Any) -> Optional[float]:
        if not isinstance(d, dict):
            return None
        for k, v in d.items():
            if "gamma" in str(k).lower():
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


def _present_algorithms(result, preferred: Sequence[str]) -> List[str]:
    """Intersection of ``preferred`` with the algorithms actually in the run,
    preserving the preferred order.  Falls back to every algorithm in the run
    (in its native order) if none of the preferred names are present."""
    sim = result.sim_result
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
    """First metric in ``preference`` that every selected algorithm can
    report.  Returns None if the run carries no SCNR stats at all (e.g. a
    no-target scenario) — the caller then drops panel (b)."""
    sim = result.sim_result
    results = getattr(sim, "algorithm_results", {})
    names = list(only) if only else list(results.keys())
    algos = [results[n] for n in names if n in results]
    if not algos:
        return None
    for metric in preference:
        if all(a.has_metric(metric) for a in algos):
            return metric
    return None


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary (printed to stdout + reusable in the notebook)
# ─────────────────────────────────────────────────────────────────────

def collect_summary(sinr_result,
                    scnr_result,
                    gamma_db: float,
                    only: Sequence[str],
                    scnr_metric: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Per-algorithm headline numbers for the caption / sanity check:
    median & 5th-pct min-SINR, infeasibility at γ\\*, and median SCNR."""
    out: Dict[str, Dict[str, float]] = {}
    sinr_res = sinr_result.sim_result.algorithm_results
    scnr_res = scnr_result.sim_result.algorithm_results
    for name in only:
        row: Dict[str, float] = {}
        a = sinr_res.get(name)
        if a is not None and a.has_metric(SINR_METRIC):
            row["sinr_p50_db"] = a.percentile(SINR_METRIC, 50)
            row["sinr_p05_db"] = a.percentile(SINR_METRIC, 5)
            row["infeas_at_gamma"] = a.infeasibility_rate(gamma_db, SINR_METRIC)
        b = scnr_res.get(name)
        if scnr_metric and b is not None and b.has_metric(scnr_metric):
            row["scnr_p50_db"] = b.percentile(scnr_metric, 50)
        out[name] = row
    return out


def _print_summary(summary: Dict[str, Dict[str, float]],
                   gamma_db: float, scnr_metric: Optional[str]) -> None:
    print(f"[info] operating point γ* = {gamma_db:g} dB")
    print(f"[info] SCNR metric        = {scnr_metric or '(none — panel b dropped)'}")
    hdr = f"{'algorithm':<14}{'minSINR p50':>12}{'p05':>8}{'infeas@γ*':>11}{'SCNR p50':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, row in summary.items():
        p50 = row.get("sinr_p50_db");  p05 = row.get("sinr_p05_db")
        inf = row.get("infeas_at_gamma");  sc = row.get("scnr_p50_db")
        print(f"{name:<14}"
              f"{(f'{p50:.2f}' if p50 is not None else '—'):>12}"
              f"{(f'{p05:.2f}' if p05 is not None else '—'):>8}"
              f"{(f'{inf*100:.1f}%' if inf is not None else '—'):>11}"
              f"{(f'{sc:.2f}' if sc is not None else '—'):>11}")


# ─────────────────────────────────────────────────────────────────────
#  Figure
# ─────────────────────────────────────────────────────────────────────

def build_figure(sinr_result,
                 scnr_result,
                 *,
                 gamma_db: float,
                 only: Optional[Sequence[str]] = None,
                 scnr_metric: Optional[str] = None,
                 use_tex: bool = True):
    """Assemble the two-panel CDF figure and return the matplotlib Figure.

    Reuses ``cordis.plotting.plot_cdf`` for both panels so styling, the γ
    marker, and the infeasibility annotation match the rest of the toolkit.
    """
    import matplotlib
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    from cordis.plotting import apply_paper_style, figsize, plot_cdf

    apply_paper_style()
    if use_tex is False:
        # apply_paper_style may flip usetex back on; force it off for nodes
        # without a TeX install.
        matplotlib.rcParams["text.usetex"] = False

    only = list(only) if only else _present_algorithms(sinr_result, PREFERRED)
    if scnr_metric is None:
        scnr_metric = _select_scnr_metric(scnr_result, only=only)

    have_scnr = scnr_metric is not None
    ncols = 2 if have_scnr else 1
    fig, axes = plt.subplots(
        1, ncols,
        figsize=figsize(width="double" if have_scnr else "single",
                        aspect=(7.16 / 2.8) if have_scnr else (3.5 / 2.6)),
    )
    axes = np.atleast_1d(axes)

    # ── Panel (a): min-SINR CDF at γ* ───────────────────────────────
    ax_a = axes[0]
    plot_cdf(
        sinr_result.sim_result,
        metric=SINR_METRIC,
        ax=ax_a,
        xlabel=r"min-SINR [dB]",
        only=only,
        gamma_db=gamma_db,            # vertical γ* line + infeasibility annotation
        annotate_infeasibility=True,
        legend_loc="lower right",     # any CDF is empty in the lower-right corner
    )
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_title(r"(a) min-SINR CDF at $\gamma^\star$")

    # ── Panel (b): SCNR CDF at γ* ───────────────────────────────────
    if have_scnr:
        ax_b = axes[1]
        plot_cdf(
            scnr_result.sim_result,
            metric=scnr_metric,
            ax=ax_b,
            xlabel=_SCNR_XLABEL.get(scnr_metric, scnr_metric),
            only=only,
            legend_loc="lower right",
        )
        ax_b.set_ylim(0.0, 1.0)
        ax_b.set_title(r"(b) SCNR CDF at $\gamma^\star$")

    fig.tight_layout()
    return fig, only, scnr_metric


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_fig_cdf",
        description="Build the min-SINR / SCNR CDF figure at γ* for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sinr-dir", default=None,
                   help="Specific results/exp_sinr_cdf/<run>/ to load for "
                        "panel (a).  Default: newest (aggregated array preferred).")
    p.add_argument("--scnr-dir", default=None,
                   help="Specific results/exp_scnr_cdf/<run>/ to load for "
                        "panel (b).  Default: newest; falls back to the SINR "
                        "run if no scnr_cdf run exists.")
    p.add_argument("--gamma-db", type=float, default=None,
                   help="Operating point γ* in dB.  Default: auto-detected "
                        f"from run metadata, else {GAMMA_DEFAULT_DB:g}.")
    p.add_argument("--scnr-metric", default=None,
                   help="SCNR metric for panel (b).  Default: first available "
                        f"of {', '.join(SCNR_METRIC_PREFERENCE)}.")
    p.add_argument("--only", default=None,
                   help="Comma-separated algorithm display names to plot. "
                        f"Default: {', '.join(PREFERRED)} (filtered to those present).")
    p.add_argument("--out", default=str(FIGURES_OUT / "fig_cdf"),
                   help="Output path stem (no extension).")
    p.add_argument("--no-tex", action="store_true",
                   help="Disable LaTeX text rendering (for nodes without pdflatex).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _build_parser().parse_args(argv)

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    # Panel (a): SINR run.
    sinr_result, sinr_dir = load_cdf(
        Path(args.sinr_dir) if args.sinr_dir else None, experiment="sinr_cdf")

    # Panel (b): SCNR run — reuse the SINR run if no scnr_cdf campaign exists.
    if args.scnr_dir:
        scnr_result, scnr_dir = load_cdf(Path(args.scnr_dir), experiment="scnr_cdf")
    else:
        try:
            scnr_result, scnr_dir = load_cdf(None, experiment="scnr_cdf")
        except FileNotFoundError:
            logger.info("No scnr_cdf run found; reusing the sinr_cdf run for "
                        "panel (b) (it carries SCNR stats too).")
            scnr_result, scnr_dir = sinr_result, sinr_dir

    gamma_db, detected = (args.gamma_db, True) if args.gamma_db is not None \
        else _detect_gamma_db(sinr_result)
    if not detected:
        logger.warning("γ* not found in metadata; using %.3g dB. Override "
                       "with --gamma-db.", gamma_db)

    print(f"[info] repo root : {REPO_ROOT}")
    print(f"[info] SINR run  : {sinr_dir}")
    print(f"[info] SCNR run  : {scnr_dir}")

    only_resolved = list(only) if only else _present_algorithms(sinr_result, PREFERRED)
    scnr_metric = args.scnr_metric or _select_scnr_metric(scnr_result, only=only_resolved)

    summary = collect_summary(sinr_result, scnr_result, gamma_db,
                              only_resolved, scnr_metric)
    _print_summary(summary, gamma_db, scnr_metric)

    fig, used_only, used_scnr_metric = build_figure(
        sinr_result, scnr_result,
        gamma_db=gamma_db, only=only_resolved,
        scnr_metric=scnr_metric, use_tex=not args.no_tex)

    out_stem = Path(args.out)
    if not out_stem.is_absolute():
        out_stem = (Path.cwd() / out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_cdf",
            "GammaStarDB": f"{gamma_db:g}",
            "Algorithms": ", ".join(used_only),
            "ScnrMetric": str(used_scnr_metric),
            "SinrRun": sinr_dir.name,
            "ScnrRun": scnr_dir.name,
        }))
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    for pth in paths + [png]:
        print(f"[ok] wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

