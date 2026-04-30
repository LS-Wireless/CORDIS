"""
scripts/list_config_params.py
==============================
Print a complete, human-readable reference for every CORDIS config parameter.

For each parameter the output shows:
  - JSON key path  (e.g. topology.n_ap)
  - Python type
  - Default value
  - Unit
  - Valid range / choices
  - Description

The information is pulled directly from the dataclass definitions and the
PARAM_REGISTRY in cordis/utils/config.py, so it is always in sync with
the actual code.

Usage
-----
    # Full reference (all sections)
    python scripts/list_config_params.py

    # Filter to one or more sections
    python scripts/list_config_params.py --section topology
    python scripts/list_config_params.py --section channel sensing

    # Search by keyword in name or description
    python scripts/list_config_params.py --search kappa
    python scripts/list_config_params.py --search "pilot"

    # Export as Markdown table (for README / paper appendix)
    python scripts/list_config_params.py --format markdown > docs/config_reference.md

    # Export as CSV (for spreadsheet)
    python scripts/list_config_params.py --format csv > docs/config_reference.csv
"""

from __future__ import annotations

SCRIPT_DESCRIPTION = (
    "Print a complete reference for every CORDIS config parameter: "
    "type, default, unit, valid range, and description. "
    "Supports section filtering, keyword search, and Markdown/CSV export."
)

import argparse
import csv
import dataclasses
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cordis.utils.config import (
    CORDISConfig,
    TopologyConfig,
    FrequencyConfig,
    ChannelConfig,
    SensingConfig,
    ADMMConfig,
    SplitOptConfig,
    AlgorithmConfig,
    SimulationConfig,
    PARAM_REGISTRY,
)


# =============================================================================
# Section map: JSON key → (dataclass, display title)
# =============================================================================

SECTIONS = {
    "topology":          (TopologyConfig,  "Topology"),
    "frequency":         (FrequencyConfig, "Frequency & Waveform"),
    "channel":           (ChannelConfig,   "Channel Model"),
    "sensing":           (SensingConfig,   "Sensing & Clutter"),
    "algorithm.admm":    (ADMMConfig,      "Algorithm: CORDIS-ADMM"),
    "algorithm.split":   (SplitOptConfig,  "Algorithm: CORDIS-Split"),
    "simulation":        (SimulationConfig,"Simulation Control"),
}


# =============================================================================
# Parameter record
# =============================================================================

def _type_name(t: Any) -> str:
    """Return a clean human-readable type name."""
    if t is type(None):
        return "None"
    if hasattr(t, "__origin__"):              # Optional[X], List[X], …
        origin = t.__origin__
        args   = getattr(t, "__args__", ())
        if origin is type(None):
            return "None"
        # Optional[X] is Union[X, None]
        import typing
        if origin is typing.Union:
            non_none = [a for a in args if a is not type(None)]
            inner = " | ".join(_type_name(a) for a in non_none)
            if type(None) in args:
                return f"Optional[{inner}]"
            return inner
        return str(t)
    return getattr(t, "__name__", str(t))


def _default_str(f: dataclasses.Field) -> str:
    """Return a short string representation of the field's default value."""
    if f.default is not dataclasses.MISSING:
        val = f.default
        if val is None:
            return "null"
        if isinstance(val, bool):
            return str(val).lower()
        return str(val)
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            val = f.default_factory()
            return str(val)
        except Exception:
            return "<factory>"
    return "(required)"


def collect_params(
    section_filter: Optional[List[str]] = None,
    search: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Return a list of dicts, one per parameter, with keys:
      path, type, default, unit, range, description
    """
    rows: List[Dict[str, str]] = []
    search_lower = search.lower() if search else None

    for section_key, (cls, _title) in SECTIONS.items():
        if section_filter:
            # Accept "topology", "channel", "admm", "split", etc.
            top = section_key.split(".")[-1]  # "admm" from "algorithm.admm"
            if not any(
                section_key.startswith(f) or top.startswith(f)
                for f in section_filter
            ):
                continue

        for f in dataclasses.fields(cls):
            path = f"{section_key}.{f.name}"
            info = PARAM_REGISTRY.get(path, {})

            desc  = info.get("help",  "")
            unit  = info.get("unit",  "-")
            rng   = info.get("range", "-")
            typ   = _type_name(f.type)
            dflt  = _default_str(f)

            if search_lower:
                haystack = (path + desc + rng).lower()
                if search_lower not in haystack:
                    continue

            rows.append({
                "path":        path,
                "type":        typ,
                "default":     dflt,
                "unit":        unit,
                "range":       rng,
                "description": desc,
            })

    return rows


# =============================================================================
# Renderers
# =============================================================================

_W = {                   # Column widths for terminal table
    "path":    42,   # wide enough for "algorithm.admm.field_name"
    "type":    10,
    "default": 10,
    "unit":    8,
    "range":   26,
}
_DESC_WIDTH = 52


def _wrap(text: str, width: int, indent: int = 0) -> List[str]:
    """Word-wrap text to width, returning a list of lines."""
    words = text.split()
    lines, current = [], ""
    prefix = " " * indent
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = (current + " " + word).lstrip()
        else:
            if current:
                lines.append(prefix + current)
            current = word
    if current:
        lines.append(prefix + current)
    return lines or [""]


def render_terminal(rows: List[Dict[str, str]]) -> str:
    """Render as a colored, sectioned terminal table."""
    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    DIM   = "\033[2m"
    RESET = "\033[0m"
    SEP   = "─" * 110
    THICK = "═" * 110

    buf = io.StringIO()

    def w(s: str) -> None:
        buf.write(s + "\n")

    w("")
    w(BOLD + THICK + RESET)
    w(BOLD + "  CORDIS Configuration Parameter Reference" + RESET)
    w(BOLD + THICK + RESET)

    current_section = None
    for row in rows:
        section = ".".join(row["path"].split(".")[:2] if "admm" in row["path"]
                           or "split" in row["path"] else row["path"].split(".")[:1])

        # Section header
        if section != current_section:
            current_section = section
            # Find display title
            for sk, (_, title) in SECTIONS.items():
                top = sk.split(".")[-1]
                if row["path"].startswith(sk):
                    title_str = title
                    break
            else:
                title_str = section

            w("")
            w(BOLD + CYAN + f"  ▸ {title_str}" + RESET)
            w(DIM + "  " + SEP + RESET)
            # Column headers
            w(
                DIM
                + f"  {'--set path':<{_W['path']}} "
                + f"{'Type':<{_W['type']}} "
                + f"{'Default':<{_W['default']}} "
                + f"{'Unit':<{_W['unit']}} "
                + f"{'Range':<{_W['range']}}"
                + RESET
            )
            w(DIM + "  " + SEP + RESET)

        # Parameter row — show full dot-separated path so it can be
        # pasted directly into create_config.py --set path=value
        full_path = row["path"]
        type_str  = row["type"][:_W["type"]]
        dflt_str  = row["default"][:_W["default"]]
        unit_str  = row["unit"][:_W["unit"]]

        w(
            BOLD
            + f"  {full_path:<{_W['path']}} "
            + RESET
            + f"{type_str:<{_W['type']}} "
            + f"{dflt_str:<{_W['default']}} "
            + f"{unit_str:<{_W['unit']}} "
            + f"{row['range']}"
        )

        # Description (wrapped, indented)
        desc_lines = _wrap(row["description"], _DESC_WIDTH)
        for dl in desc_lines:
            w(f"  {DIM}{dl}{RESET}")

        w("")

    w(BOLD + THICK + RESET)
    w(f"  {len(rows)} parameters listed.")
    w(BOLD + THICK + RESET)
    w("")
    return buf.getvalue()


def render_markdown(rows: List[Dict[str, str]]) -> str:
    """Render as a GitHub-flavored Markdown table, sectioned by config block."""
    buf = io.StringIO()

    def w(s: str) -> None:
        buf.write(s + "\n")

    w("# CORDIS Configuration Parameter Reference\n")
    w("Generated automatically from `cordis/utils/config.py`.\n")

    current_section = None
    for row in rows:
        section_key = ".".join(
            row["path"].split(".")[:2]
            if ("admm" in row["path"] or "split" in row["path"])
            else row["path"].split(".")[:1]
        )

        if section_key != current_section:
            current_section = section_key
            for sk, (_, title) in SECTIONS.items():
                if row["path"].startswith(sk):
                    title_str = title
                    break
            else:
                title_str = section_key

            w(f"\n## {title_str}\n")
            w("| Parameter | Type | Default | Unit | Range | Description |")
            w("|-----------|------|---------|------|-------|-------------|")

        name = row["path"].split(".")[-1]
        cols = [
            f"`{name}`",
            row["type"],
            f"`{row['default']}`",
            row["unit"],
            row["range"],
            row["description"],
        ]
        w("| " + " | ".join(cols) + " |")

    return buf.getvalue()


def render_csv(rows: List[Dict[str, str]]) -> str:
    """Render as CSV."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["path", "type", "default", "unit", "range", "description"],
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def render_paths_only(rows: List[Dict[str, str]]) -> str:
    """
    Render only the dot-separated path for each parameter, one per line.
    Useful for quick copy-paste into create_config.py --set.
    """
    buf = io.StringIO()
    current_section = None
    for row in rows:
        section_key = ".".join(
            row["path"].split(".")[:2]
            if ("admm" in row["path"] or "split" in row["path"])
            else row["path"].split(".")[:1]
        )
        if section_key != current_section:
            current_section = section_key
            for sk, (_, title) in SECTIONS.items():
                if row["path"].startswith(sk):
                    title_str = title
                    break
            else:
                title_str = section_key
            buf.write(f"\n# {title_str}\n")
        buf.write(row["path"] + "\n")
    return buf.getvalue()


def render_sections() -> str:
    """
    List all valid section key prefixes with their display titles.
    These are the values you can pass to --section.
    """
    BOLD  = "\033[1m"
    CYAN  = "\033[96m"
    DIM   = "\033[2m"
    RESET = "\033[0m"
    THICK = "\u2550" * 60

    buf = io.StringIO()
    buf.write("\n")
    buf.write(BOLD + THICK + RESET + "\n")
    buf.write(BOLD + "  Config sections (use with --section or --set)\n" + RESET)
    buf.write(BOLD + THICK + RESET + "\n\n")
    buf.write(
        DIM
        + f"  {'--section value':<20} {'--set prefix':<26} Description\n"
        + RESET
    )
    buf.write(DIM + "  " + "\u2500" * 56 + RESET + "\n")

    SECTION_SHORTNAMES = {
        "topology":        "topology",
        "frequency":       "frequency",
        "channel":         "channel",
        "sensing":         "sensing",
        "algorithm.admm":  "admm",
        "algorithm.split": "split",
        "simulation":      "simulation",
    }

    for section_key, (_, title) in SECTIONS.items():
        short = SECTION_SHORTNAMES[section_key]
        buf.write(
            f"  {BOLD}{short:<20}{RESET}"
            f"{DIM}{section_key + '.<field>':<26}{RESET}"
            f"{title}\n"
        )

    buf.write("\n")
    buf.write(DIM + "  Example:\n" + RESET)
    buf.write(DIM + "    --section admm\n" + RESET)
    buf.write(DIM + "    --set algorithm.admm.kappa=2.0\n" + RESET)
    buf.write("\n")
    buf.write(BOLD + THICK + RESET + "\n\n")
    return buf.getvalue()


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="list_config_params.py",
        description=SCRIPT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Full reference (shows full --set paths in first column)
  python scripts/list_config_params.py

  # List section names and --set prefixes
  python scripts/list_config_params.py --sections

  # Only topology and sensing sections
  python scripts/list_config_params.py --section topology sensing

  # Find all parameters related to pilots
  python scripts/list_config_params.py --search pilot

  # Print only paths — paste directly into create_config.py --set
  python scripts/list_config_params.py --paths-only
  python scripts/list_config_params.py --section admm --paths-only

  # Export Markdown reference doc
  python scripts/list_config_params.py --format markdown > docs/config_reference.md

  # Export CSV for a spreadsheet
  python scripts/list_config_params.py --format csv > docs/config_reference.csv
""",
    )
    parser.add_argument(
        "--section", nargs="+", default=None, metavar="NAME",
        help=(
            "Show only these config sections. "
            "Choices: topology frequency channel sensing admm split simulation"
        ),
    )
    parser.add_argument(
        "--search", default=None, metavar="KEYWORD",
        help="Show only parameters whose name or description contain KEYWORD",
    )
    parser.add_argument(
        "--format", default="terminal", choices=["terminal", "markdown", "csv"],
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--paths-only", action="store_true",
        help=(
            "Print only the dot-separated parameter paths, one per line. "
            "Ideal for copy-pasting into create_config.py --set"
        ),
    )
    parser.add_argument(
        "--sections", action="store_true",
        help=(
            "List all config section names with their --section shortnames "
            "and --set prefixes, then exit"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --sections: just print the section map and exit
    if args.sections:
        output = render_sections()
        if not sys.stdout.isatty():
            import re
            output = re.sub(r"\033\[[0-9;]*m", "", output)
        sys.stdout.write(output)
        return

    rows = collect_params(
        section_filter=[s.lower() for s in args.section] if args.section else None,
        search=args.search,
    )

    if not rows:
        print("No parameters matched your filters.")
        sys.exit(0)

    # --paths-only: bare paths for copy-paste into --set
    if args.paths_only:
        sys.stdout.write(render_paths_only(rows))
        return

    if args.format == "markdown":
        sys.stdout.write(render_markdown(rows))
    elif args.format == "csv":
        sys.stdout.write(render_csv(rows))
    else:
        # Terminal: disable colors when piped
        output = render_terminal(rows)
        if not sys.stdout.isatty():
            import re
            output = re.sub(r"\033\[[0-9;]*m", "", output)
        sys.stdout.write(output)


if __name__ == "__main__":
    main()

