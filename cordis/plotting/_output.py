"""
Stage 8a — Figure saving with provenance metadata.

The PGF backend is selected *per call* via ``savefig(format=...)`` so
the same Python process can emit PDF and PGF side by side without
switching matplotlib's global backend.

Provenance
~~~~~~~~~~
Every saved figure embeds:

* ``Creator``       — ``"CORDIS <version>"``
* ``CreationDate``  — ISO-8601 timestamp at second resolution
* ``GitSHA``        — short hash of HEAD, or ``"?"`` if not in a repo

Plus any user-supplied metadata.

For PDFs this rides matplotlib's ``savefig(metadata=...)`` kwarg and is
readable via ``pypdf.PdfReader(...)``.  For PGFs (which are TeX text
files matplotlib writes directly) we prepend ``% Key: value`` comment
lines after writing; LaTeX ignores them but ``grep '^%' fig.pgf`` makes
the metadata auditable from the shell.

Public API
~~~~~~~~~~
:func:`save_figure`.
"""
from __future__ import annotations

import datetime as _dt
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


#: Bumped alongside the package; surfaced in figure metadata.
_CORDIS_VERSION: str = "0.1.0"

#: Default formats written by :func:`save_figure`.
_DEFAULT_FORMATS: tuple = ("pdf", "pgf")

#: Formats for which we know how to embed metadata.  Other formats
#: still save fine but skip the metadata embedding step.
_METADATA_FORMATS: frozenset = frozenset({"pdf", "pgf"})


def _git_sha(short: bool = True) -> str:
    """
    Return ``git rev-parse [--short] HEAD`` of the current working
    directory's repo, or ``"?"`` if anything goes wrong (no repo, no
    git on PATH, timeout, ...).
    """
    cmd = ["git", "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        return "?"


def _prepend_pgf_comments(path: Path, metadata: Dict[str, str]) -> None:
    """Prepend metadata as TeX comments at the top of a PGF file.

    Comments are written as ``% Key: value`` lines.  Idempotent in the
    sense that callers should only invoke this once per save; the
    function does not check for existing headers.
    """
    contents = path.read_text(encoding="utf-8")
    header = "\n".join(f"% {k}: {v}" for k, v in metadata.items())
    path.write_text(header + "\n" + contents, encoding="utf-8")


def save_figure(
    fig: Figure,
    base_path: Union[str, Path],
    formats: Sequence[str] = _DEFAULT_FORMATS,
    metadata: Optional[Dict[str, str]] = None,
    bbox_inches: Optional[str] = "tight",
    pad_inches: float = 0.02,
) -> List[Path]:
    """
    Save ``fig`` once per requested format alongside each other.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    base_path : str | Path
        Path **without** extension.  The format suffix is appended.
        E.g. ``base_path='figures/sinr_cdf'`` with
        ``formats=('pdf','pgf')`` writes
        ``figures/sinr_cdf.pdf`` and ``figures/sinr_cdf.pgf``.
        Parent directories are created if missing.
    formats : sequence of str
        Subset of ``{'pdf', 'pgf', 'png', 'svg'}``.  Default
        ``('pdf', 'pgf')`` covers the paper workflow (LaTeX inclusion +
        general preview).  ``'pdf'`` and ``'pgf'`` get full metadata
        embedding; other formats save fine but without metadata.
    metadata : dict | None
        User key/value pairs to embed.  Merged on top of the standard
        provenance fields (Creator, CreationDate, GitSHA).
    bbox_inches, pad_inches : matplotlib ``savefig`` arguments.
        Sensible defaults for tight paper figures.

    Returns
    -------
    list[Path]
        Paths of files actually written, in the order they were
        emitted (matches the ``formats`` order).

    Raises
    ------
    Any exception ``matplotlib.figure.Figure.savefig`` raises is
    logged and re-raised; this function does not silently swallow
    save failures.
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    now = _dt.datetime.now()
    sha = _git_sha(short=True)

    # PGF gets every key as-is (we control the comment writer).
    # PDF must respect matplotlib's allowlist: Creator, Title, Subject,
    # Keywords, Author, ModDate, Trapped, Producer, CreationDate.
    # Custom keys (GitSHA, Experiment, ...) are packed into Keywords as
    # ``key=value`` pairs so the PDF metadata stays standards-compliant.
    _PDF_ALLOWED = {"Title", "Subject", "Author", "Creator",
                    "Producer", "Keywords", "CreationDate",
                    "ModDate", "Trapped"}

    base_meta: Dict[str, object] = {
        "Creator":      f"CORDIS {_CORDIS_VERSION}",
        "CreationDate": now,                          # datetime, not str
        "GitSHA":       sha,
    }
    if metadata:
        base_meta.update(metadata)

    # PGF metadata: string-valued copy of everything
    pgf_meta = {k: str(v) if not isinstance(v, _dt.datetime) else v.isoformat(timespec="seconds")
                for k, v in base_meta.items()}

    # PDF metadata: standard fields go in directly; custom fields are
    # serialised into Keywords.
    pdf_meta: Dict[str, object] = {}
    custom_pairs = []
    for k, v in base_meta.items():
        if k in _PDF_ALLOWED:
            pdf_meta[k] = v
        else:
            custom_pairs.append(f"{k}={v}")
    if custom_pairs:
        existing = pdf_meta.get("Keywords", "")
        joined = "; ".join(custom_pairs)
        pdf_meta["Keywords"] = f"{existing}; {joined}".lstrip("; ")

    written: List[Path] = []
    for fmt in formats:
        out = base_path.with_suffix(f".{fmt}")
        try:
            if fmt == "pdf":
                fig.savefig(
                    out, format="pdf", metadata=pdf_meta,
                    bbox_inches=bbox_inches, pad_inches=pad_inches,
                )
            elif fmt == "pgf":
                fig.savefig(
                    out, format="pgf",
                    bbox_inches=bbox_inches, pad_inches=pad_inches,
                )
                _prepend_pgf_comments(out, pgf_meta)
            else:
                fig.savefig(
                    out, format=fmt,
                    bbox_inches=bbox_inches, pad_inches=pad_inches,
                )
                if fmt not in _METADATA_FORMATS:
                    logger.debug(
                        "Format %r does not support metadata embedding; "
                        "wrote %s without provenance fields.", fmt, out,
                    )
        except Exception as e:
            logger.error("Failed to save %s as %s: %s",
                         base_path, fmt, type(e).__name__)
            raise
        written.append(out)
    return written

