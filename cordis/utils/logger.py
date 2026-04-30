"""
cordis/utils/logger.py
======================
Centralized logging configuration for the CORDIS simulation framework.

All modules obtain their logger via:
    from cordis.utils.logger import get_logger
    logger = get_logger(__name__)

This ensures a single logging hierarchy that can be configured once
(in a script entry point) and respected everywhere.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ── ANSI color codes for console output ─────────────────────────────────────
_RESET   = "\033[0m"
_GREY    = "\033[90m"
_CYAN    = "\033[96m"
_YELLOW  = "\033[93m"
_RED     = "\033[91m"
_BOLD_RED = "\033[1;91m"

_LEVEL_COLOURS = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _CYAN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BOLD_RED,
}


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI color to the level name on TTY streams."""

    _FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        if self._use_colour:
            colour = _LEVEL_COLOURS.get(record.levelno, _RESET)
            record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file handlers (no ANSI codes)."""

    _FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colour: bool = True,
) -> None:
    """
    Configure the root *cordis* logger.

    Call this **once** at the top of every entry-point script (e.g.
    ``scripts/cordis_admm.py``) before importing anything else.  All
    child loggers (``cordis.channel``, ``cordis.algorithms``, …) will
    automatically inherit the level and handlers set here.

    Parameters
    ----------
    level : str
        Logging level string.  One of ``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.  Default ``"INFO"``.
    log_file : str or None
        Optional path to a log file.  If supplied, log records are
        written to the file **in addition** to the console.
    use_colour : bool
        Whether to add ANSI colour codes to the console output.
        Automatically disabled when stdout is not a TTY.

    Examples
    --------
    >>> from cordis.utils.logger import setup_logging
    >>> setup_logging(level="DEBUG", log_file="results/run.log")
    """
    root = logging.getLogger("cordis")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any handlers that may have been added by a previous call
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    console_handler.setFormatter(_ColourFormatter(use_colour=use_colour and is_tty))
    root.addHandler(console_handler)

    # ── Optional file handler ─────────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(_PlainFormatter())
        root.addHandler(file_handler)

    # Prevent propagation to the root Python logger (avoids duplicate output)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the *cordis* namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module, e.g.
        ``"cordis.channel.rician"``.  If ``name`` does not already start
        with ``"cordis"``, the prefix is added automatically so the
        logger is always part of the cordis hierarchy.

    Returns
    -------
    logging.Logger

    Examples
    --------
    >>> from cordis.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Channel generated for %d APs", 6)
    """
    if not name.startswith("cordis"):
        name = f"cordis.{name}"
    return logging.getLogger(name)

