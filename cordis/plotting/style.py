"""
Stage 8a — Plotting style and per-algorithm visual conventions.

Two responsibilities, deliberately kept separate:

1.  :func:`apply_paper_style` sets ``matplotlib.rcParams`` for IEEE
    two-column journal figures.  Idempotent; safe to call repeatedly.
    ``use_latex=False`` falls back to pure-Python rendering (mathtext)
    when ``pdflatex`` is not on PATH (e.g. headless compute nodes
    without TeX).

2.  :data:`ALGORITHM_STYLE` and :func:`style_for` provide per-algorithm
    color/marker/linestyle so the same algorithm reads the same way
    across every figure in the paper.

Figure widths
-------------
IEEE journal columns are ≈ 3.5 in (single-column) or 7.16 in
(double-column spread).  :func:`apply_paper_style` does NOT set a
default ``figure.figsize`` — callers pass ``figsize=(w, h)`` explicitly
to ``plt.subplots`` to make their choice visible.  Use
:func:`figsize(width=, aspect=)` to compose a tuple from a named width.

Public API
~~~~~~~~~~
``apply_paper_style``, ``figsize``, ``WIDTHS``, ``style_for``,
``register_algorithm_style``, ``ALGORITHM_STYLE``.
"""
from __future__ import annotations

import logging
import shutil
from typing import Dict, Tuple, Union

import matplotlib as mpl

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Figure widths (inches) for IEEE two-column layout
# ─────────────────────────────────────────────────────────────────────

#: Canonical column widths in inches.  ``half`` is an alias for
#: ``single`` since IEEE single-column figures *span* one column ≈ half
#: the text block.  ``third`` is useful for three-up panels.
WIDTHS: Dict[str, float] = {
    "single": 3.5,
    "half":   3.5,
    "double": 7.16,
    "third":  2.40,
}

#: Default aspect ratio (width / height) used by :func:`figsize` when
#: the caller does not specify one.
GOLDEN_RATIO: float = 1.618


def figsize(width: Union[str, float] = "single",
            aspect: float = GOLDEN_RATIO) -> Tuple[float, float]:
    """
    Return ``(w, h)`` in inches for ``plt.subplots(figsize=...)``.

    Parameters
    ----------
    width : str | float
        Either a key in :data:`WIDTHS` (``"single"``, ``"double"``,
        ``"third"``, ``"half"``) or a numeric width in inches.
    aspect : float
        Width / height ratio.  Default is the golden ratio so figures
        feel proportioned without explicit tuning.

    Examples
    --------
    >>> figsize("single")
    (3.5, 2.163...)
    >>> figsize("double", aspect=2.5)
    (7.16, 2.864)
    >>> figsize(4.0, aspect=1.0)
    (4.0, 4.0)
    """
    if isinstance(width, str):
        if width not in WIDTHS:
            raise KeyError(
                f"Unknown width key {width!r}.  Valid keys: "
                f"{sorted(WIDTHS)} — or pass a numeric width in inches."
            )
        w = WIDTHS[width]
    else:
        w = float(width)
    if aspect <= 0:
        raise ValueError(f"aspect must be positive, got {aspect}")
    return (w, w / aspect)


# ─────────────────────────────────────────────────────────────────────
#  Matplotlib rcParams
# ─────────────────────────────────────────────────────────────────────

def apply_paper_style(use_latex: bool = True) -> None:
    """
    Configure matplotlib for IEEE two-column journal figures.

    Idempotent: calling repeatedly does no harm.

    Parameters
    ----------
    use_latex : bool, default True
        If True **and** ``pdflatex`` is on PATH, engage matplotlib's
        ``text.usetex=True`` plus the PGF backend's ``pdflatex`` engine.
        If False or pdflatex is missing, fall back to mathtext (built-in
        Python LaTeX-subset renderer) and log a warning.  PGF *file*
        output still works in the fallback mode — it just won't have
        Computer Modern fonts.
    """
    latex_ok = use_latex and shutil.which("pdflatex") is not None
    if use_latex and not latex_ok:
        logger.warning(
            "pdflatex not found on PATH; falling back to mathtext. "
            "PGF .pgf output still works, but rendered fonts will not "
            "match a real LaTeX run."
        )

    rc: Dict[str, object] = {
        # Fonts
        "font.family":      "serif" if latex_ok else "sans-serif",
        "font.serif":       ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size":         8.0,
        "axes.labelsize":    8.0,
        "axes.titlesize":    9.0,
        "xtick.labelsize":   7.0,
        "ytick.labelsize":   7.0,
        "legend.fontsize":   7.0,
        # Math font: in fallback (no-LaTeX) mode use DejaVu Serif which
        # is bundled with matplotlib.  In LaTeX mode mathtext is bypassed
        # since text.usetex=True routes math through LaTeX directly.
        "mathtext.fontset":  "cm" if latex_ok else "dejavuserif",
        # Lines and markers
        "lines.linewidth":      1.2,
        "lines.markersize":     3.5,
        "lines.markeredgewidth": 0.7,
        # Axes
        "axes.linewidth":   0.8,
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.4,
        # Ticks
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.minor.size":  1.5,
        "ytick.minor.size":  1.5,
        # Legend
        "legend.frameon":       True,
        "legend.framealpha":    0.85,
        "legend.fancybox":      False,
        "legend.edgecolor":     "0.5",
        "legend.borderaxespad": 0.4,
        # Savefig
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "savefig.dpi":          300,
    }

    if latex_ok:
        rc.update({
            "text.usetex":   True,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts":   False,                       # honour rc fonts above
            "pgf.preamble":  r"\usepackage{amsmath}\usepackage{amssymb}",
        })
    else:
        # Pin the PGF engine to pdflatex even when LaTeX is currently
        # unavailable.  Without this, matplotlib's PGF backend defaults
        # to xelatex, which makes the eventual "command not found" error
        # confusing ("xelatex not found" when the user expected pdflatex).
        rc["pgf.texsystem"] = "pdflatex"

    mpl.rcParams.update(rc)


# ─────────────────────────────────────────────────────────────────────
#  Per-algorithm visual style
# ─────────────────────────────────────────────────────────────────────

#: Color / marker / linestyle / label for every algorithm that appears
#: in the paper.  Centralised here so every figure renders the same
#: algorithm identically.
#:
#: Override or extend at runtime with :func:`register_algorithm_style`.
#:
#: Keys MUST match :data:`cordis.experiments.specs._DISPLAY` exactly,
#: because :func:`style_for` does a direct dict lookup.  Raw strings
#: with ``$\rho$`` math-mode are used for the LR-MMSE variants so the
#: same key works under usetex=True (LaTeX renders ρ via mathmode) and
#: usetex=False (mathtext fallback).
ALGORITHM_STYLE: Dict[str, Dict[str, object]] = {
    "CORDIS-Split":             dict(color="#1f77b4", marker="o", linestyle="-",  label="CORDIS-Split"),
    "CORDIS-ADMM":              dict(color="#d62728", marker="s", linestyle="-",  label="CORDIS-ADMM"),
    "Centralized":              dict(color="#2ca02c", marker="^", linestyle="--", label="Centralized"),
    "MRT-Split":                dict(color="#7f7f7f", marker="x", linestyle=":",  label="MRT"),
    # The default LR-MMSE curve is now LR-MMSE Phase-I + fixed ρ=0.5 (NOT
    # the legacy LR-MMSE + P-Split PA, which is identical to CORDIS-Split).
    # The two variants below sit alongside it for the paper's PSR
    # sensitivity figure (psr_baselines spec set).
    r"LR-MMSE ($\rho$=0.5)":    dict(color="#9467bd", marker="D", linestyle="-.", label=r"LR-MMSE ($\rho$=0.5)"),
    r"LR-MMSE ($\rho$=0.2)":    dict(color="#9467bd", marker="<", linestyle=":",  label=r"LR-MMSE ($\rho$=0.2)"),
    r"LR-MMSE ($\rho$=0.8)":    dict(color="#9467bd", marker=">", linestyle="--", label=r"LR-MMSE ($\rho$=0.8)"),
    # Legacy alias — LR-MMSE + optimal P-Split PA (numerically identical
    # to CORDIS-Split).  Not in any default spec set; kept so notebooks
    # that explicitly reference the old behaviour don't crash.
    "LR-MMSE (P-Split PA)":     dict(color="#9467bd", marker="D", linestyle="-",  label="LR-MMSE (P-Split PA)"),
    "RZF-Split":                dict(color="#8c564b", marker="v", linestyle="--", label="RZF"),
    "ZF-Split":                 dict(color="#e377c2", marker="P", linestyle=":",  label="ZF"),
    "Global-MRT":               dict(color="#bcbd22", marker="<", linestyle="-",  label="Global MRT"),
    "Global-ZF":                dict(color="#17becf", marker=">", linestyle="-",  label="Global ZF"),
}

#: Greyscale palette used for unknown algorithms.  Indexed deterministically
#: by ``hash(name)`` so the same unknown name always lands on the same
#: shade but distinct unknown names get distinct shades.
_FALLBACK_COLORS = ["#1a1a1a", "#555555", "#888888", "#aaaaaa"]


def style_for(name: str) -> Dict[str, object]:
    """
    Return a kwargs-dict suitable for ``ax.plot(..., **style)``.

    Known algorithms get their registered style; unknown names get a
    deterministic greyscale fallback (different unknowns → different
    shades, same unknown → same shade across calls).  The returned dict
    is a copy, so mutating it does not affect future lookups.
    """
    if name in ALGORITHM_STYLE:
        return dict(ALGORITHM_STYLE[name])
    idx = abs(hash(name)) % len(_FALLBACK_COLORS)
    return dict(
        color=_FALLBACK_COLORS[idx],
        marker="",
        linestyle="-",
        label=name,
    )


def register_algorithm_style(name: str, **style) -> None:
    """
    Register or override a per-algorithm style at runtime.

    Useful for one-off experiments that introduce a new algorithm
    name without modifying this file.  Keys are passed straight to
    ``ax.plot``: ``color``, ``marker``, ``linestyle``, ``label``,
    etc.

    Examples
    --------
    >>> register_algorithm_style("MyAlgo", color="purple", marker="*",
    ...                          linestyle="-", label="My Algorithm")
    """
    ALGORITHM_STYLE[name] = dict(style)

