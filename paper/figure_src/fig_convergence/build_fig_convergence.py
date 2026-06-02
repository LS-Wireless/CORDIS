#!/usr/bin/env python3
"""
paper/figure_src/fig_convergence/build_fig_convergence.py
=========================================================

Build the CORDIS-ADMM convergence figure for the journal paper's
Simulation Results section.

Two panels, double-column:

  (a) Residual convergence (log-y): primal r_pri and dual r_dual vs
      ADMM iteration, overlaying the *fixed-rho* run (the working
      default) and the *adaptive-rho* run (the ablation).  Fixed-rho
      settles monotonically; adaptive-rho perturbs the in-loop SCA and
      swings -- this is what justifies adaptive_rho=false being the
      shipped default.

  (b) Feasibility dynamics for the default (fixed-rho) run: the SOC
      slack max_u eps_u (left, log) collapsing toward zero -- the ALM
      driving feasibility -- together with the worst-user min-SINR
      (right, dB) climbing across the gamma requirement line.

Why these quantities: the `convergence_trace` experiment persists the
primal/dual residual histories, the per-user slack history, and the
per-user SINR history (in *linear* units), plus `best_iter`.  It does
NOT persist the rho multiplier trajectory, so we deliberately tell the
convergence story with slack + min-SINR rather than a rho-vs-iter panel.

This file is BOTH a CLI builder and an importable module: the sibling
notebook (fig_convergence.ipynb) imports `load_trace`, `extract_gamma_db`,
and `build_figure` so the notebook and the headless build stay in sync.

Usage (from anywhere; repo root is auto-detected)::

    python3 paper/figure_src/fig_convergence/build_fig_convergence.py
    python3 .../build_fig_convergence.py --no-tex          # headless / no pdflatex
    python3 .../build_fig_convergence.py \
        --fixed-dir results/exp_convergence_trace/<ts_fixed> \
        --adaptive-dir results/exp_convergence_trace/<ts_adaptive>

By default it reads the two trace directories prepared by
`run_traces.sh` under this figure's ``data/`` folder
(``data/trace_fixed`` and ``data/trace_adaptive``) and writes
``paper/figures/fig_convergence.pdf`` (+ a .png preview).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Path plumbing
# ─────────────────────────────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
FIG_DIR = THIS_FILE.parent                 # .../fig_convergence
PAPER_ROOT = FIG_DIR.parents[1]            # .../paper  (renamable, e.g. twc_paper)
FIGURES_OUT = PAPER_ROOT / "figures"       # committed PDFs live here


def find_repo_root(start: Path = THIS_FILE) -> Path:
    """Walk up until we find the directory that contains the `cordis/`
    package, so `import cordis` works no matter where this is called
    from and regardless of whether `paper/` was renamed."""
    for p in [start, *start.parents]:
        if (p / "cordis").is_dir() and (p / "cordis" / "__init__.py").exists():
            return p
    # Fallback: assume paper/ sits at the repo root.
    return PAPER_ROOT.parent


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────
#  Colours (figure-local; this is a single-algorithm diagnostic, so we
#  don't pull from ALGORITHM_STYLE which is for cross-algorithm plots)
# ─────────────────────────────────────────────────────────────────────

C_FIXED = "#1f3b73"   # navy   — fixed-rho (default)
C_ADAPT = "#d1791f"   # orange — adaptive-rho (ablation)
C_SLACK = "#6a51a3"   # purple — SOC slack
C_SINR = "#1f3b73"    # navy   — min-SINR
C_GUIDE = "#666666"   # grey   — best-iter / gamma guides


# ─────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────

def load_trace(exp_dir: Path):
    """Load a `convergence_trace` ExperimentResult and return its
    LoadedADMMResult plus metadata. Raises if the dir is not a trace."""
    from cordis.experiments import ExperimentResult
    res = ExperimentResult.load(Path(exp_dir))
    if res.kind != "trace":
        raise ValueError(
            f"{exp_dir} holds a {res.kind!r} result, expected 'trace' "
            f"(run the convergence_trace experiment)."
        )
    return res.admm_result, (res.metadata or {})


def _newest_trace_dirs(results_root: Path, n: int = 2):
    """Return up to `n` newest convergence_trace result dirs (newest
    first), each containing a trace manifest."""
    base = results_root / "exp_convergence_trace"
    if not base.is_dir():
        return []
    cands = [d for d in base.iterdir()
             if d.is_dir() and (d / "manifest.json").exists()]
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return cands[:n]


def resolve_trace_dirs(args) -> Tuple[Optional[Path], Optional[Path]]:
    """Figure out which directories to read for the fixed-rho and
    adaptive-rho runs, in priority order:

      1. explicit --fixed-dir / --adaptive-dir
      2. data/trace_fixed and data/trace_adaptive (written by run_traces.sh)
      3. the two newest results/exp_convergence_trace/* dirs (a warning
         is printed because we can't tell which is which)
    """
    fixed = Path(args.fixed_dir) if args.fixed_dir else None
    adapt = Path(args.adaptive_dir) if args.adaptive_dir else None

    data_dir = Path(args.data_dir)
    if fixed is None and (data_dir / "trace_fixed" / "manifest.json").exists():
        fixed = data_dir / "trace_fixed"
    if adapt is None and (data_dir / "trace_adaptive" / "manifest.json").exists():
        adapt = data_dir / "trace_adaptive"

    if fixed is None and adapt is None:
        newest = _newest_trace_dirs(REPO_ROOT / "results", n=2)
        if newest:
            print("[warn] No data/trace_{fixed,adaptive} found; falling back "
                  "to the two newest results/exp_convergence_trace runs.\n"
                  "       Pass --fixed-dir/--adaptive-dir to be explicit.")
            adapt = newest[0] if len(newest) >= 1 else None       # newest = last run
            fixed = newest[1] if len(newest) >= 2 else None
    return fixed, adapt


# ─────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────

def extract_gamma_db(metadata: dict, fallback: float = 5.0) -> float:
    """Pull the per-user SINR target gamma (dB) from trace metadata.
    `run_convergence_trace` stores it under 'gamma_u_db_uniform' (the
    _uniform suffix is added for ndarray params); older runs may use
    'gamma_db', or it may live inside cfg_summary."""
    for key in ("gamma_u_db_uniform", "gamma_db"):
        v = metadata.get(key)
        if v is not None:
            try:
                arr = np.atleast_1d(np.asarray(v, dtype=float))
                return float(arr.flat[0])
            except (TypeError, ValueError):
                pass
    cfg = metadata.get("cfg_summary") or {}
    if isinstance(cfg, dict):
        v = cfg.get("gamma_db")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return float(fallback)


def _min_sinr_db(admm) -> Optional[np.ndarray]:
    """Per-iteration worst-user SINR in dB, from the (linear) sinr_history."""
    sh = getattr(admm, "sinr_history", None)
    if sh is None or len(sh) == 0:
        return None
    lin = np.stack([np.asarray(s, dtype=float).ravel() for s in sh])  # [it, n_ue]
    worst_lin = np.maximum(lin.min(axis=1), 1e-30)
    return 10.0 * np.log10(worst_lin)


def _max_slack(admm) -> Optional[np.ndarray]:
    """Per-iteration max-over-users SOC slack (the ALM feasibility gauge)."""
    sl = getattr(admm, "slack_history", None)
    if sl is None or len(sl) == 0:
        return None
    arr = np.asarray([np.asarray(s, dtype=float) for s in sl], dtype=object) \
        if any(np.ndim(s) for s in sl) else np.asarray(sl, dtype=float)
    arr = np.stack([np.atleast_1d(np.asarray(s, dtype=float)) for s in sl])
    return arr.max(axis=1) if arr.ndim == 2 else arr.ravel()


# ─────────────────────────────────────────────────────────────────────
#  The figure
# ─────────────────────────────────────────────────────────────────────

def build_figure(fixed, adaptive, *, gamma_db: float, use_tex: bool = True):
    """Build the two-panel convergence figure.

    `fixed` / `adaptive` are LoadedADMMResult objects (adaptive may be
    None — then panel (a) shows the default run only). Returns the
    matplotlib Figure."""
    import matplotlib
    if not use_tex:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cordis.plotting import apply_paper_style, figsize

    apply_paper_style(use_latex=use_tex)
    if not use_tex:
        matplotlib.rcParams["text.usetex"] = False

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=figsize("double", aspect=2.4),
    )

    # ── Panel (a): residual convergence ──────────────────────────────
    def _plot_res(admm, color, tag):
        pr = np.asarray(admm.primal_res_history, dtype=float)
        du = np.asarray(admm.dual_res_history, dtype=float)
        it = np.arange(1, len(pr) + 1)
        axA.semilogy(it, np.maximum(pr, 1e-30), color=color, ls="-",
                     lw=1.3, label=rf"{tag}: $r_{{\mathrm{{pri}}}}$")
        axA.semilogy(np.arange(1, len(du) + 1), np.maximum(du, 1e-30),
                     color=color, ls="--", lw=1.3,
                     label=rf"{tag}: $r_{{\mathrm{{dual}}}}$")

    _plot_res(fixed, C_FIXED, r"fixed $\rho$ (default)")
    if adaptive is not None:
        _plot_res(adaptive, C_ADAPT, r"adaptive $\rho$")

    bi = getattr(fixed, "best_iter", None)
    if bi:
        axA.axvline(int(bi), color=C_GUIDE, ls=":", lw=0.8,
                    label=f"best iter ({int(bi)})")
    axA.set_xlabel("ADMM iteration")
    axA.set_ylabel("consensus residual")
    axA.set_title("(a) Residual convergence")
    axA.grid(True, which="both", alpha=0.3)
    axA.legend(fontsize=6.5, loc="upper right", ncol=1)

    # ── Panel (b): slack -> 0 and min-SINR crossing gamma ────────────
    slack = _max_slack(fixed)
    sinr_db = _min_sinr_db(fixed)

    handles, labels = [], []
    if slack is not None:
        it_s = np.arange(1, len(slack) + 1)
        (h_sl,) = axB.semilogy(it_s, np.maximum(slack, 1e-30),
                               color=C_SLACK, ls="-", lw=1.3,
                               label=r"max$_u\,\varepsilon_u$ (SOC slack)")
        handles.append(h_sl); labels.append(h_sl.get_label())
    axB.set_xlabel("ADMM iteration")
    axB.set_ylabel(r"max$_u\,\varepsilon_u$  (SOC slack)", color=C_SLACK)
    axB.tick_params(axis="y", labelcolor=C_SLACK)
    axB.grid(True, which="both", alpha=0.3)

    axB2 = axB.twinx()
    if sinr_db is not None:
        it_g = np.arange(1, len(sinr_db) + 1)
        (h_g,) = axB2.plot(it_g, sinr_db, color=C_SINR, ls="-", lw=1.3,
                           label=r"min-SINR")
        handles.append(h_g); labels.append(h_g.get_label())
    h_gamma = axB2.axhline(gamma_db, color="k", ls="--", lw=0.8,
                           label=rf"$\gamma = {gamma_db:g}$ dB")
    handles.append(h_gamma); labels.append(h_gamma.get_label())
    axB2.set_ylabel("min-SINR [dB]", color=C_SINR)
    axB2.tick_params(axis="y", labelcolor=C_SINR)

    if bi:
        axB.axvline(int(bi), color=C_GUIDE, ls=":", lw=0.8)
    axB.set_title("(b) Feasibility dynamics (default run)")
    axB2.legend(handles, labels, fontsize=6.5, loc="center right")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed-dir", default=None,
                    help="trace dir for the fixed-rho (default) run")
    ap.add_argument("--adaptive-dir", default=None,
                    help="trace dir for the adaptive-rho (ablation) run")
    ap.add_argument("--data-dir", default=str(FIG_DIR / "data"),
                    help="where run_traces.sh stashed the traces "
                         "(default: this figure's data/)")
    ap.add_argument("--gamma-db", type=float, default=5.0,
                    help="fallback gamma [dB] if absent from trace metadata")
    ap.add_argument("--out", default=str(FIGURES_OUT / "fig_convergence"),
                    help="output path stem (no extension)")
    ap.add_argument("--no-tex", action="store_true",
                    help="disable LaTeX text rendering (headless nodes)")
    args = ap.parse_args(argv)

    fixed_dir, adapt_dir = resolve_trace_dirs(args)
    if fixed_dir is None:
        print("[error] No fixed-rho trace found. Run run_traces.sh first, "
              "or pass --fixed-dir.", file=sys.stderr)
        return 1

    print(f"[info] repo root      : {REPO_ROOT}")
    print(f"[info] fixed-rho dir   : {fixed_dir}")
    print(f"[info] adaptive-rho dir: {adapt_dir if adapt_dir else '(none — panel (a) shows default only)'}")

    fixed, md = load_trace(fixed_dir)
    adaptive = None
    if adapt_dir is not None:
        adaptive, _ = load_trace(adapt_dir)

    gamma_db = extract_gamma_db(md, fallback=args.gamma_db)
    print(f"[info] gamma           : {gamma_db:g} dB")

    fig = build_figure(fixed, adaptive, gamma_db=gamma_db, use_tex=not args.no_tex)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # Vector PDF (paper) + PNG preview (quick look / GitHub).
    from cordis.plotting import save_figure
    paths = save_figure(fig, out_stem, formats=("pdf",))
    png_path = out_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    paths = list(paths) + [png_path]
    for p in paths:
        print(f"[ok] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

