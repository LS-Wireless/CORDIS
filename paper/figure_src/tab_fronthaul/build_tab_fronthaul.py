#!/usr/bin/env python3
"""
paper/figure_src/tab_fronthaul/build_tab_fronthaul.py
=====================================================

Build the fronthaul coordination-overhead deliverables for the journal paper,
from a ``fronthaul_table`` result (``kind == 'table'``):

  1. **The table** — per-AP fronthaul payload for Centralized vs CORDIS-Split
     vs CORDIS-ADMM, written as both LaTeX (``tab_fronthaul.tex``, for
     ``\\input{}``) and Markdown (``tab_fronthaul.md``, for previews).  Columns:
     data shared, size, real scalars / round, iterations (= T_ADMM for ADMM,
     1 otherwise), total / solve, what it scales with, and scalable (Y/N).

  2. **The bar figure** — ``fig_fronthaul.pdf``: per-AP, per-coordination-round
     real-scalar payload as grouped bars at two array sizes (M_lo, M_hi).  The
     headline reads straight off the bars: Centralized's payload grows with the
     per-AP array size M, while CORDIS-Split and CORDIS-ADMM are **independent
     of M** (their two bars are the same height).  T_ADMM is annotated on the
     ADMM bars (per-round payload is apples-to-apples; ADMM simply runs T_ADMM
     rounds per solve, reported in the table's total/solve column).

Why the bars are per-round: the framework reports ``real_scalars`` per
coordination round for every algorithm so the comparison is apples-to-apples
(a Stage-17 fix forbids the ×T_ADMM total here, which would otherwise make ADMM
look larger than Centralized).  The ×T_ADMM total lives in the table.

Why M_hi can be extrapolated from one run: the payloads are deterministic
formulas, not Monte-Carlo estimates — only T_ADMM is measured.  Centralized's
payload is exactly linear in M (``2·M·(N_ue+|D|)``), so its bar at M_hi is the
loaded value × (M_hi / M_lo); the M-independent rows are unchanged.  Pass
``--result-dir-hi`` to use a *measured* second run instead.

Loader + builder only -- no runner.  This file is BOTH a CLI builder and an
importable module so the sibling notebook reuses ``load_table``,
``render_markdown``, ``render_latex``, and ``build_bar_figure``.

Usage::

    python3 paper/figure_src/tab_fronthaul/build_tab_fronthaul.py
    python3 .../build_tab_fronthaul.py --no-tex
    python3 .../build_tab_fronthaul.py --m-hi 32
    python3 .../build_tab_fronthaul.py --result-dir results/exp_fronthaul_table/<run>
    python3 .../build_tab_fronthaul.py --result-dir-hi results/exp_fronthaul_table/<run_M32>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("tab_fronthaul")


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
#  Ordering + the second-array-size default
# ─────────────────────────────────────────────────────────────────────

# Bar / table order: the centralized baseline first, then the proposed pair.
PREFERRED: Tuple[str, ...] = ("Centralized", "CORDIS-Split", "CORDIS-ADMM")

# Second per-AP array size for the M-independence bars (extrapolated unless a
# measured --result-dir-hi run is supplied).
M_HI_DEFAULT = 64


# ─────────────────────────────────────────────────────────────────────
#  Result discovery  (array-aggregated first, never `latest`)
# ─────────────────────────────────────────────────────────────────────

import re as _re

_ARRAY_AGG_PATTERN = _re.compile(r"^array_(\d+)_aggregated$")
_ARRAY_TASK_PATTERN = _re.compile(r"^array_(\d+)_task_(\d+)$")


def _discover_result_dir(experiment: str = "fronthaul_table",
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


def load_table(result_dir: Optional[Path] = None,
               experiment: str = "fronthaul_table",
               results_root: Optional[Path] = None):
    """Load a table (``kind == 'table'``) :class:`ExperimentResult`."""
    from cordis.experiments.result import ExperimentResult  # lazy

    if result_dir is None:
        result_dir = _discover_result_dir(experiment, results_root)
        if result_dir is None:
            raise FileNotFoundError(
                f"No usable run under results/exp_{experiment}/.  Run the "
                f"{experiment!r} experiment first."
            )
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = REPO_ROOT / result_dir

    result = ExperimentResult.load(result_dir)
    if getattr(result, "kind", None) != "table":
        raise ValueError(
            f"tab_fronthaul needs a table (kind='table') result; "
            f"{result_dir} has kind={getattr(result, 'kind', None)!r}."
        )
    return result, result_dir


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _ordered_algos(table: Dict[str, Any],
                   preferred: Sequence[str] = PREFERRED) -> List[str]:
    lead = [n for n in preferred if n in table]
    rest = [n for n in table if n not in lead]
    return lead + rest


def _is_m_dependent(row: Dict[str, Any]) -> bool:
    """True if this algorithm's per-round payload grows with the per-AP array
    size M.  Detected from the ``scales_with`` formula (contains an ``M`` term);
    falls back to ``not scalable`` (Centralized is the only M-dependent row)."""
    sw = str(row.get("scales_with", ""))
    if "M_" in sw or r"M_{" in sw or sw.strip().startswith("$M"):
        return True
    return not bool(row.get("scalable", True))


def _scalars_at_M(row: Dict[str, Any], m_lo: float, m_hi: float) -> float:
    """Per-round real scalars for ``row`` rescaled from M_lo to M_hi.
    Exact: the only M-dependent payload (Centralized) is linear in M."""
    rs = row.get("real_scalars")
    if rs is None:
        return float("nan")
    rs = float(rs)
    if _is_m_dependent(row) and m_lo and m_hi:
        return rs * (float(m_hi) / float(m_lo))
    return rs


def _fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{int(round(xf))}" if float(xf).is_integer() else f"{xf:.1f}"


# ─────────────────────────────────────────────────────────────────────
#  Table renderers (Markdown + LaTeX)
# ─────────────────────────────────────────────────────────────────────

def render_markdown(result, preferred: Sequence[str] = PREFERRED) -> str:
    table = result.table_data
    md = result.metadata or {}
    t_admm = md.get("admm_avg_iters", float("nan"))
    lines = [
        "# Fronthaul Coordination Overhead per AP",
        "",
        f"_Scenario: M={md.get('M')}, N_UE={md.get('n_ue')}, "
        f"N_targets={md.get('n_targets')}, |D|={md.get('n_streams')}_",
        "",
        f"_Average ADMM iterations: T_ADMM = {t_admm:.2f}_",
        "",
        "| Algorithm | Data shared | Size | Real scalars / round | "
        "Iterations | Total / solve | Scales with | Scalable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for algo in _ordered_algos(table, preferred):
        row = table[algo]
        rs = row.get("real_scalars")
        it = row.get("iterations")
        tot = (None if (rs is None or it is None) else float(rs) * float(it))
        lines.append(
            f"| {algo} | {row.get('data_to_share','')} | {row.get('size','')} | "
            f"{_fmt_num(rs)} | {_fmt_num(it)} | {_fmt_num(tot)} | "
            f"{row.get('scales_with','')} | "
            f"{'✓' if row.get('scalable') else '✗'} |"
        )
    return "\n".join(lines) + "\n"


def render_latex(result, preferred: Sequence[str] = PREFERRED) -> str:
    table = result.table_data
    md = result.metadata or {}
    t_admm = md.get("admm_avg_iters", float("nan"))
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-AP fronthaul coordination overhead. Real scalars are "
        r"reported per coordination round; CORDIS-ADMM runs $T_{\rm ADMM}$ "
        rf"rounds per solve ($T_{{\rm ADMM}}\approx{t_admm:.1f}$ here).}}",
        r"\label{tab:fronthaul}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Algorithm & Data shared & Size & Real/round & "
        r"Iter. & Scalable \\",
        r"\midrule",
    ]
    for algo in _ordered_algos(table, preferred):
        row = table[algo]
        scal = r"$\checkmark$" if row.get("scalable") else r"$\times$"
        out.append(
            f"{algo} & {row.get('data_to_share','')} & {row.get('size','')} & "
            f"{_fmt_num(row.get('real_scalars'))} & "
            f"{_fmt_num(row.get('iterations'))} & {scal} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


# ─────────────────────────────────────────────────────────────────────
#  Numeric summary
# ─────────────────────────────────────────────────────────────────────

def collect_summary(result, m_lo: float, m_hi: float,
                    preferred: Sequence[str] = PREFERRED) -> Dict[str, Dict[str, float]]:
    table = result.table_data
    out: Dict[str, Dict[str, float]] = {}
    for algo in _ordered_algos(table, preferred):
        row = table[algo]
        out[algo] = {
            "per_round_m_lo": _scalars_at_M(row, m_lo, m_lo),
            "per_round_m_hi": _scalars_at_M(row, m_lo, m_hi),
            "iterations": (float(row["iterations"])
                           if row.get("iterations") is not None else float("nan")),
            "m_dependent": float(_is_m_dependent(row)),
            "scalable": float(bool(row.get("scalable"))),
        }
    return out


def _print_summary(result, summary, m_lo, m_hi) -> None:
    md = result.metadata or {}
    print(f"[info] scenario   : M={md.get('M')}, N_UE={md.get('n_ue')}, "
          f"N_targets={md.get('n_targets')}, |D|={md.get('n_streams')}")
    print(f"[info] T_ADMM      : {md.get('admm_avg_iters', float('nan')):.2f}")
    print(f"[info] bar M values: M_lo={m_lo:g}, M_hi={m_hi:g}")
    hdr = (f"{'algorithm':<14}{'per-round@Mlo':>15}{'per-round@Mhi':>15}"
           f"{'iters':>8}{'M-indep':>9}{'scalable':>9}")
    print(hdr); print("-" * len(hdr))
    for algo, r in summary.items():
        print(f"{algo:<14}{_fmt_num(r['per_round_m_lo']):>15}"
              f"{_fmt_num(r['per_round_m_hi']):>15}"
              f"{_fmt_num(r['iterations']):>8}"
              f"{('yes' if not r['m_dependent'] else 'NO'):>9}"
              f"{('✓' if r['scalable'] else '✗'):>9}")


# ─────────────────────────────────────────────────────────────────────
#  Bar figure  (self-contained so the notebook can inline an editable copy)
# ─────────────────────────────────────────────────────────────────────

def build_bar_figure(result,
                     *,
                     m_lo: Optional[float] = None,
                     m_hi: Optional[float] = M_HI_DEFAULT,
                     table_hi=None,
                     preferred: Sequence[str] = PREFERRED,
                     use_tex: bool = True):
    """Grouped per-round fronthaul-payload bars at M_lo and M_hi.

    ``m_lo`` defaults to the run's M (from metadata).  ``m_hi`` adds a second
    bar per algorithm to expose M-independence; pass ``m_hi=None`` for a single
    bar per algorithm.  If ``table_hi`` (a second result's ``table_data``) is
    given, the M_hi bars use those *measured* values instead of extrapolating.

    Returns ``(fig, m_lo, m_hi)``.
    """
    import matplotlib
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from cordis.plotting import apply_paper_style, figsize, style_for

    apply_paper_style()
    if use_tex is False:
        matplotlib.rcParams["text.usetex"] = False

    table = result.table_data
    md = result.metadata or {}
    if m_lo is None:
        m_lo = float(md.get("M") or 0) or 1.0
    algos = _ordered_algos(table, preferred)
    t_admm = md.get("admm_avg_iters", None)

    two_m = (m_hi is not None) and (float(m_hi) != float(m_lo))

    def _hi_value(algo, row):
        if table_hi is not None and algo in table_hi:
            return float(table_hi[algo].get("real_scalars", float("nan")))
        return _scalars_at_M(row, m_lo, m_hi)

    fig, ax = plt.subplots(figsize=figsize(width="single", aspect=3.5 / 2.6))
    x = np.arange(len(algos))
    width = 0.38 if two_m else 0.56

    def _color(name):
        c = style_for(name).get("color")
        return c or "0.5"

    lo_vals = [_scalars_at_M(table[a], m_lo, m_lo) for a in algos]
    if two_m:
        hi_vals = [_hi_value(a, table[a]) for a in algos]
        bars_lo = ax.bar(x - width / 2, lo_vals, width,
                         color=[_color(a) for a in algos], edgecolor="black",
                         linewidth=0.5, zorder=3)
        bars_hi = ax.bar(x + width / 2, hi_vals, width,
                         color=[_color(a) for a in algos], edgecolor="black",
                         linewidth=0.5, alpha=0.55, hatch="///", zorder=3)
        # M legend (shade/hatch encodes M; color encodes algorithm)
        leg = [Patch(facecolor="0.6", edgecolor="black", label=f"$M={int(m_lo)}$"),
               Patch(facecolor="0.6", edgecolor="black", alpha=0.55,
                     hatch="///", label=f"$M={int(m_hi)}$")]
        ax.legend(handles=leg, loc="best", framealpha=0.9)
        bar_groups = [(bars_lo, lo_vals), (bars_hi, hi_vals)]
    else:
        bars_lo = ax.bar(x, lo_vals, width,
                         color=[_color(a) for a in algos], edgecolor="black",
                         linewidth=0.5, zorder=3)
        bar_groups = [(bars_lo, lo_vals)]

    # value labels on top of each bar
    for bars, vals in bar_groups:
        for rect, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax.annotate(f"{int(round(v))}",
                        xy=(rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize="xx-small")

    # annotate T_ADMM just above the ADMM group (per-round bars; ×T per solve).
    # Multiplicative offset = consistent visual gap on the log axis.
    if t_admm is not None and "CORDIS-ADMM" in algos:
        i = algos.index("CORDIS-ADMM")
        y_admm = lo_vals[i] if np.isfinite(lo_vals[i]) else 1.0
        ax.text(i, y_admm * 1.7,
                rf"$\times\,T_{{\rm ADMM}}\!\approx\!{float(t_admm):.0f}$/solve",
                ha="center", va="bottom", fontsize="xx-small", color="0.3")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=12, ha="right")
    ax.set_ylabel("Real scalars / AP / round")
    ax.set_title("Fronthaul coordination overhead")
    ax.grid(True, axis="y", which="both", alpha=0.3, zorder=0)

    fig.tight_layout()
    return fig, float(m_lo), (float(m_hi) if two_m else None)


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_tab_fronthaul",
        description="Build the fronthaul-overhead table (LaTeX+MD) and the "
                    "M-independence bar figure for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--result-dir", default=None,
                   help="Specific results/exp_fronthaul_table/<run>/ to load. "
                        "Default: newest.")
    p.add_argument("--result-dir-hi", default=None,
                   help="Optional second fronthaul_table run (e.g. at n_ant=32) "
                        "whose MEASURED payloads supply the M_hi bars instead "
                        "of extrapolating.")
    p.add_argument("--m-hi", type=float, default=M_HI_DEFAULT,
                   help="Second per-AP array size for the M-independence bars. "
                        "Use a value == M_lo to draw a single bar per algorithm.")
    p.add_argument("--out-fig", default=str(FIGURES_OUT / "fig_fronthaul"),
                   help="Output path stem for the bar figure (no extension).")
    p.add_argument("--out-tex", default=str(FIGURES_OUT / "tab_fronthaul.tex"),
                   help="Output path for the LaTeX table.")
    p.add_argument("--out-md", default=str(FIGURES_OUT / "tab_fronthaul.md"),
                   help="Output path for the Markdown table.")
    p.add_argument("--no-tex", action="store_true",
                   help="Disable LaTeX text rendering in the figure (nodes "
                        "without pdflatex).  Does not affect the .tex table.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = _build_parser().parse_args(argv)

    result, result_dir = load_table(
        Path(args.result_dir) if args.result_dir else None)
    print(f"[info] repo root  : {REPO_ROOT}")
    print(f"[info] result dir : {result_dir}")

    m_lo = float((result.metadata or {}).get("M") or 0) or 1.0
    table_hi = None
    if args.result_dir_hi:
        result_hi, dir_hi = load_table(Path(args.result_dir_hi))
        table_hi = result_hi.table_data
        m_hi_meta = float((result_hi.metadata or {}).get("M") or args.m_hi)
        args.m_hi = m_hi_meta
        print(f"[info] M_hi run   : {dir_hi}  (measured, M={m_hi_meta:g})")

    summary = collect_summary(result, m_lo, args.m_hi)
    _print_summary(result, summary, m_lo, args.m_hi)

    # ── write the table (LaTeX + Markdown) ───────────────────────────
    tex = render_latex(result)
    md = render_markdown(result)
    for path, text, label in ((Path(args.out_tex), tex, "LaTeX"),
                              (Path(args.out_md), md, "Markdown")):
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ok] wrote {path}  ({label} table)")

    # ── build + save the bar figure ──────────────────────────────────
    fig, used_m_lo, used_m_hi = build_bar_figure(
        result, m_lo=m_lo, m_hi=args.m_hi, table_hi=table_hi,
        use_tex=not args.no_tex)

    out_stem = Path(args.out_fig)
    if not out_stem.is_absolute():
        out_stem = Path.cwd() / out_stem
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    from cordis.plotting import save_figure
    paths = list(save_figure(
        fig, out_stem, formats=("pdf",),
        metadata={
            "Figure": "fig_fronthaul",
            "Mlo": f"{used_m_lo:g}",
            "Mhi": (f"{used_m_hi:g}" if used_m_hi else "—"),
            "TADMM": f"{(result.metadata or {}).get('admm_avg_iters', float('nan')):.2f}",
            "Run": result_dir.name,
        }))
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    for pth in paths + [png]:
        print(f"[ok] wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

