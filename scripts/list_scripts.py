"""
scripts/list_scripts.py
========================
Discovery tool for the CORDIS simulation framework.

Lists every callable script in the scripts/ directory with its description
and available command-line options.

Usage
-----
    # List all scripts with one-line descriptions
    python scripts/list_scripts.py

    # Show full options for every script
    python scripts/list_scripts.py --full

    # Show full options for one specific script
    python scripts/list_scripts.py --script visualize_topology.py
"""

import argparse
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"

# Scripts to skip when auto-discovering
_SKIP = {"list_scripts.py", "__init__.py"}


# =============================================================================
# Helpers
# =============================================================================

def _discover_scripts() -> list[Path]:
    """Return all .py files in scripts/ except this file and skipped ones."""
    return sorted(
        p for p in SCRIPTS_DIR.glob("*.py")
        if p.name not in _SKIP
    )


def _extract_description(script_path: Path) -> str:
    """
    Extract the SCRIPT_DESCRIPTION constant from a script file without
    importing it (safe for scripts that have side effects at module level).

    Falls back to the first non-empty line of the module docstring if
    SCRIPT_DESCRIPTION is not defined.
    """
    try:
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return "(could not parse file)"

    # Look for:  SCRIPT_DESCRIPTION = "..."  or  SCRIPT_DESCRIPTION = (...)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SCRIPT_DESCRIPTION"
        ):
            try:
                return ast.literal_eval(node.value)
            except Exception:
                pass

    # Fall back to module docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        docstring = tree.body[0].value.value
        # Return the first non-empty line of the docstring
        for line in docstring.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(script_path.stem):
                return stripped

    return "(no description available)"


def _get_full_help(script_path: Path) -> str:
    """
    Run the script with --help and capture the output.
    Returns the help text or an error message.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(ROOT),
        )
        return (result.stdout or result.stderr).strip()
    except subprocess.TimeoutExpired:
        return "(timed out — script may have side effects at import time)"
    except Exception as e:
        return f"(error running --help: {e})"


# =============================================================================
# Formatting
# =============================================================================

_DIVIDER       = "─" * 72
_THICK_DIVIDER = "═" * 72

def _print_summary_table(scripts: list[Path]) -> None:
    """Print a compact one-line-per-script table."""
    print()
    print(_THICK_DIVIDER)
    print("  CORDIS Simulation Framework — Available Scripts")
    print(_THICK_DIVIDER)
    print()

    col_w = max(len(p.name) for p in scripts) + 2
    fmt   = f"  {{:<{col_w}}} {{}}"

    print(fmt.format("Script", "Description"))
    print(fmt.format("─" * col_w, "─" * (68 - col_w)))

    for script in scripts:
        desc = _extract_description(script)
        # Truncate description to fit one line
        max_desc = 72 - col_w - 4
        if len(desc) > max_desc:
            desc = desc[: max_desc - 3] + "..."
        print(fmt.format(script.name, desc))

    print()
    print("  Run any script with --help for full option details.")
    print("  Run this tool with --full to show all options at once.")
    print("  Run this tool with --script <name> for one specific script.")
    print()
    print(_THICK_DIVIDER)
    print()


def _print_full_help(scripts: list[Path]) -> None:
    """Print --help output for every script."""
    for script in scripts:
        desc = _extract_description(script)
        print()
        print(_THICK_DIVIDER)
        print(f"  {script.name}")
        print(f"  {desc}")
        print(_THICK_DIVIDER)
        help_text = _get_full_help(script)
        # Indent the help text slightly for readability
        for line in help_text.splitlines():
            print(f"  {line}")
        print()


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog="list_scripts.py",
        description=(
            "Discover all callable scripts in the CORDIS scripts/ directory, "
            "showing their descriptions and available command-line options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Compact table of all scripts and their one-line descriptions
  python scripts/list_scripts.py

  # Full --help output for every script
  python scripts/list_scripts.py --full

  # Full --help output for one script
  python scripts/list_scripts.py --script visualize_topology.py
""",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show full --help output for every discovered script",
    )
    parser.add_argument(
        "--script", default=None, metavar="FILENAME",
        help="Show full --help for a single named script (e.g. visualize_topology.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scripts = _discover_scripts()

    if not scripts:
        print("No scripts found in scripts/")
        return

    if args.script:
        # Single-script mode
        matches = [s for s in scripts if s.name == args.script]
        if not matches:
            available = ", ".join(s.name for s in scripts)
            print(f"Script '{args.script}' not found.")
            print(f"Available scripts: {available}")
            sys.exit(1)
        _print_full_help(matches)

    elif args.full:
        _print_full_help(scripts)

    else:
        _print_summary_table(scripts)


if __name__ == "__main__":
    main()

