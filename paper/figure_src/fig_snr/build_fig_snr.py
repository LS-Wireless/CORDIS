#!/usr/bin/env python3
"""
paper/figure_src/fig_snr/build_fig_snr.py
=========================================

Build the optional appendix SNR/power-sweep figure for the journal paper, from
an ``snr_sweep`` result.  A standard sanity check that both metrics scale
sensibly with the transmit power budget P_max/σ_n².

Stacked two-panel, single-column (shared x-axis = SNR = P_max/σ_n² [dB]):

  (top)  worst-user **min-SINR** vs SNR.
  (bot)  **sum-SCNR** vs SNR.

More power → both communication and sensing improve monotonically and the
algorithm ordering (Centralized ≥ CORDIS-ADMM ≳ CORDIS-Split) is preserved
across the range — the reviewer-reassurance figure.

A vertical dashed line marks the deployed operating point **SNR\\*** (the
``channel.snr_db`` the headline figures run at, 137 dB by default — a
transmit-side P_max/σ_n²; after ~117 dB path loss at the default 650 m cell
this is the ~20 dB *received* SNR quoted in the system model).

Filtered to the proposed pair vs the centralized ceiling
(``Centralized``, ``CORDIS-ADMM``, ``CORDIS-Split``), styled via ``style_for`` /
``ALGORITHM_STYLE`` so a curve reads identically across every figure.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_sweep``,
``collect_summary``, and ``build_figure``.

Usage::

    python3 paper/figure_src/fig_snr/build_fig_snr.py
    python3 .../build_fig_snr.py --no-tex
    python3 .../build_fig_snr.py --snr-star 137
    python3 .../build_fig_snr.py --result-dir results/exp_snr_sweep/array_12345_aggregated
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("fig_snr")


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

PREFERRED: Tuple[str, ...] = ("Centralized", "CORDIS-ADMM", "CORDIS-Split")

SINR_METRIC = "min_sinr_db"

SCNR_METRIC_PREFERENCE: Tuple[str, ...] = (
    "sum_scnr_db",
    "weighted_sum_scnr_db",
    "mean_scnr_db",
)

# Deployed operating point SNR\* (matches channel.snr_db in the favorable
# config / the SNR the headline figures run at).  Marked with a vline.
SNR_STAR_DEFAULT_DB = 137.0

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


def _discover_result_dir(experiment: str = "snr_sweep",
                         results_root: Optional[Path] = None) -> Optional[Path]:
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
               experiment: str = "snr_sweep",
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
            f"fig_snr needs a sweep (kind='sweep') result; "
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


def _detect_snr_star(result, fallback: float = SNR_STAR_DEFAULT_DB) -> Tuple[float, bool]:
    """Recover the operating-point SNR\\* (dB) from run metadata (the config's
    ``channel.snr_db``).  Returns ``(snr_db, detected)``."""
    meta = getattr(result, "metadata", None) or {}

    def _scan(d: Any) -> Optional[float]:
        if not isinstance(d, dict):
            return None
        for key in ("snr_db", "channel.snr_db"):
            if key in d:
                try:
                    fv = float(d[key])
                    if np.isfinite(fv):
                        return fv
                except (TypeError, ValueError):
                    pass
        for k, v in d.items():
            kl = str(k).lower()
            if kl == "snr_db" or kl.endswith(".snr_db"):
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


def _nearest(result, x: float) -> float:
    keys = sorted(result.sweep_results.keys())
    return min(keys, key=lambda k: abs(k - x))


def _in_range(result, x: float) -> bool:
    keys = sorted(result.sweep_results.keys())
    return keys[0] <= x <= keys[-1]


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary
# ─────────────────────────────────────────────────────────────────────

def collect_summary(result,
                    only: Sequence[str],
                    scnr_metric: Optional[str],
                    snr_star: float) -> Dict[str, Dict[str, float]]:
    """Per-algorithm min-SINR & SCNR (median) at the lowest, operating, and
    highest SNR — enough to confirm both metrics rise monotonically with power."""
    keys = sorted(result.sweep_results.keys())
    s_lo, s_hi = keys[0], keys[-1]
    s_op = _nearest(result, snr_star)

    def _med(sim, name, metric):
        a = sim.algorithm_results.get(name)
        if a is None or not a.has_metric(metric):
            return None
        return a.percentile(metric, 50)

    out: Dict[str, Dict[str, float]] = {}
    for name in only:
        row: Dict[str, float] = {}
        for tag, k in (("lo", s_lo), ("op", s_op), ("hi", s_hi)):
            v = _med(result.sweep_results[k], name, SINR_METRIC)
            if v is not None:
                row[f"sinr_{tag}_db"] = v
            if scnr_metric:
                c = _med(result.sweep_results[k], name, scnr_metric)
                if c is not None:
                    row[f"scnr_{tag}_db"] = c
        out[name] = row
    return out


def _print_summary(result, summary, scnr_metric, snr_star) -> None:
    keys = sorted(result.sweep_results.keys())
    s_op = _nearest(result, snr_star)
    print(f"[info] SNR grid   = {keys} dB")
    print(f"[info] SNR* (mark)= {snr_star:g} dB  (nearest swept {s_op:g} dB)"
          + ("" if _in_range(result, snr_star) else "  [OUT OF RANGE — marker skipped]"))
    print(f"[info] SCNR metric= {scnr_metric or '(none)'}")
    hdr = (f"{'algorithm':<14}{'SINR lo':>9}{'SINR op':>9}{'SINR hi':>9}"
           f"{'SCNR lo':>9}{'SCNR op':>9}{'SCNR hi':>9}")
    print(hdr); print("-" * len(hdr))
    for name, r in summary.items():
        def f(k):
            v = r.get(k); return (f"{v:.2f}" if v is not None else "—")
        print(f"{name:<14}{f('sinr_lo_db'):>9}{f('sinr_op_db'):>9}{f('sinr_hi_db'):>9}"
              f"{f('scnr_lo_db'):>9}{f('scnr_op_db'):>9}{f('scnr_hi_db'):>9}")


# ─────────────────────────────────────────────────────────────────────
#  Figure  (self-contained so the notebook can inline an editable copy)
# ─────────────────────────────────────────────────────────────────────

def build_figure(result,
                 *,
                 only: Optional[Sequence[str]] = None,
                 scnr_metric: Optional[str] = None,
                 snr_star: float = SNR_STAR_DEFAULT_DB,
                 use_tex: bool = True):
    """Assemble the stacked SNR-sweep figure; return ``(fig, only, scnr_metric)``.

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
    xlabel = axis.display or r"SNR $P_{\max}/\sigma_n^2$ [dB]"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize(width="single", aspect=3.5 / 2.6),
        sharex=True, gridspec_kw={"hspace": 0.12},
    )

    plot_sweep(result.sweep_results, metric=SINR_METRIC, ax=ax_top,
               xlabel="", ylabel=r"min-SINR [dB]", only=only, log_x=False)
    ax_top.set_title("Scaling with transmit power")

    if scnr_metric is not None:
        plot_sweep(result.sweep_results, metric=scnr_metric, ax=ax_bot,
                   xlabel=xlabel,
                   ylabel=_SCNR_YLABEL.get(scnr_metric, scnr_metric),
                   only=only, log_x=False)
    else:
        logger.warning("No SCNR metric available; bottom panel left empty.")
        ax_bot.set_xlabel(xlabel)

    # operating-point marker SNR* (only if inside the swept range)
    if _in_range(result, snr_star):
        for ax in (ax_top, ax_bot):
            ax.axvline(snr_star, ls="--", lw=0.9, color="0.45", zorder=0)
        ax_top.text(snr_star, 0.04, r"$\mathrm{SNR}^\star$",
                    transform=ax_top.get_xaxis_transform(),
                    ha="right", va="bottom", fontsize="small", color="0.35")
    else:
        logger.warning("snr_star=%.3g dB is outside the swept range; "
                       "operating-point marker skipped.", snr_star)

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
        prog="build_fig_snr",
        description="Build the optional SNR/power-sweep appendix figure for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--result-dir", default=None,
                   help="Specific results/exp_snr_sweep/<run>/ to load. "
                        "Default: newest (aggregated array preferred).")
    p.add_argument("--snr-star", type=float, default=None,
                   help="Operating-point SNR* [dB] marked with a line. "
                        "Default: auto-detected from run metadata, else "
                        f"{SNR_STAR_DEFAULT_DB:g}.")
    p.add_argument("--scnr-metric", default=None,
                   help="SCNR metric for the bottom panel.  Default: first "
                        f"available of {', '.join(SCNR_METRIC_PREFERENCE)}.")
    p.add_argument("--only", default=None,
                   help="Comma-separated algorithm display names. "
                        f"Default: {', '.join(PREFERRED)} (filtered to those present).")
    p.add_argument("--out", default=str(FIGURES_OUT / "fig_snr"),
                   help="Output path stem (no extension).")
    p.add_argument("--no-tex", action="store_true",
                   help="Disable LaTeX text rendering (nodes without pdflatex).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _build_parser().parse_args(argv)

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    result, result_dir = load_sweep(
        Path(args.result_dir) if args.result_dir else None, experiment="snr_sweep")

    snr_star, detected = (args.snr_star, True) if args.snr_star is not None \
        else _detect_snr_star(result)
    if not detected:
        logger.warning("SNR* not found in metadata; using %.4g dB. Override "
                       "with --snr-star.", snr_star)

    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {result_dir}")

    only_resolved = list(only) if only else _present_algorithms(result, PREFERRED)
    scnr_metric = args.scnr_metric or _select_scnr_metric(result, only=only_resolved)

    summary = collect_summary(result, only_resolved, scnr_metric, snr_star)
    _print_summary(result, summary, scnr_metric, snr_star)

    fig, used_only, used_scnr_metric = build_figure(
        result, only=only_resolved, scnr_metric=scnr_metric,
        snr_star=snr_star, use_tex=not args.no_tex)

    out_stem = Path(args.out)
    if not out_stem.is_absolute():
        out_stem = Path.cwd() / out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_snr",
            "SnrStarDB": f"{snr_star:g}",
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

