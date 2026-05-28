"""
scripts/list_benchmarks.py
============================
Discovery / documentation tool for the CORDIS benchmark registry.

Renders ``cordis.algorithms.benchmarks.BENCHMARK_REGISTRY`` in several
formats so it can be browsed from the terminal or committed to docs/.

Usage
-----
    # Terminal-friendly listing
    python scripts/list_benchmarks.py

    # Markdown reference (regenerates docs/benchmarks_reference.md)
    python scripts/list_benchmarks.py --format markdown > docs/benchmarks_reference.md

    # CSV (spreadsheet-friendly)
    python scripts/list_benchmarks.py --format csv > docs/benchmarks_reference.csv

    # Filter by name substring
    python scripts/list_benchmarks.py --search mrt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cordis.algorithms.benchmarks import BENCHMARK_REGISTRY  # noqa: E402


SCRIPT_DESCRIPTION = (
    "List all CORDIS benchmark algorithms (BF + PA baselines) registered "
    "in cordis.algorithms.benchmarks.BENCHMARK_REGISTRY."
)


# =============================================================================
# Row extraction
# =============================================================================

# Internal record type: (name, comm_bf, pa_mode, rho_str, description)
Row = Tuple[str, str, str, str, str]


def _rows(search: str | None = None) -> List[Row]:
    out: List[Row] = []
    for name, (desc, comm_bf, pa_opt, fixed_psr) in BENCHMARK_REGISTRY.items():
        if search and (search.lower() not in name.lower()
                       and search.lower() not in desc.lower()
                       and search.lower() not in comm_bf.lower()):
            continue
        pa_mode = "optimised (P-Split)" if pa_opt else "fixed"
        rho_str = "—" if pa_opt else f"{fixed_psr:.2f}"
        out.append((name, comm_bf, pa_mode, rho_str, desc))
    return out


def _split_groups(rows: List[Row]) -> Tuple[List[Row], List[Row]]:
    """Partition rows into (optimised-PA, fixed-PA) groups, preserving order."""
    opt = [r for r in rows if r[2].startswith("optimised")]
    fix = [r for r in rows if r[2] == "fixed"]
    return opt, fix


# =============================================================================
# Renderers
# =============================================================================

def render_terminal(rows: List[Row]) -> str:
    """Plain-text table for stdout."""
    if not rows:
        return "(no benchmarks matched the filter)\n"

    w_name = max(len(r[0]) for r in rows + [("name", "", "", "", "")])
    w_bf   = max(len(r[1]) for r in rows + [("", "comm BF", "", "", "")])
    w_pa   = max(len(r[2]) for r in rows + [("", "", "PA mode", "", "")])
    w_rho  = max(len(r[3]) for r in rows + [("", "", "", "ρ", "")])

    lines: List[str] = []
    lines.append("=" * 76)
    lines.append("  CORDIS Benchmark Registry")
    lines.append("=" * 76)
    lines.append("")
    lines.append("  NOTE on LR-MMSE benchmarks (paper revision):")
    lines.append("    'lr_mmse_split' (LR-MMSE Phase-I + optimal P-Split PA) is")
    lines.append("    numerically identical to CORDIS-Split, so the default benchmark")
    lines.append("    surface (cordis_vs_benchmarks / all_algorithms) now uses")
    lines.append("    'lr_mmse_fixed' (PSR ρ=0.5, no PA optimisation) as the LR-MMSE")
    lines.append("    representative.  The 'lr_mmse_split' entry is retained for")
    lines.append("    debugging / legacy access but is not in any default spec set.")
    lines.append("    The two extra variants 'lr_mmse_fixed_p020' and")
    lines.append("    'lr_mmse_fixed_p080' (ρ=0.2 and ρ=0.8) support the paper's")
    lines.append("    'benefit of optimal PSR' figure via the 'psr_baselines' spec set.")
    lines.append("")
    lines.append(
        f"  {'name':<{w_name}}  {'comm BF':<{w_bf}}  "
        f"{'PA mode':<{w_pa}}  {'ρ':<{w_rho}}  description"
    )
    lines.append(
        f"  {'-' * w_name}  {'-' * w_bf}  {'-' * w_pa}  {'-' * w_rho}  {'-' * 40}"
    )
    for name, bf, pa, rho, desc in rows:
        lines.append(
            f"  {name:<{w_name}}  {bf:<{w_bf}}  "
            f"{pa:<{w_pa}}  {rho:<{w_rho}}  {desc}"
        )
    lines.append("")
    lines.append(f"  {len(rows)} benchmark{'s' if len(rows) != 1 else ''} listed.")
    lines.append("")
    lines.append("  Run any benchmark via:")
    lines.append("    from cordis.algorithms.benchmarks import run_benchmark")
    lines.append("    res = run_benchmark('<name>', topo, cfg, est, ...)")
    lines.append("")
    lines.append("=" * 76)
    return "\n".join(lines) + "\n"


def render_markdown(rows: List[Row]) -> str:
    """Markdown reference, grouped by PA mode."""
    lines: List[str] = []
    lines.append("# CORDIS Benchmark Registry Reference")
    lines.append("")
    lines.append(
        "Generated automatically from "
        "`cordis.algorithms.benchmarks.BENCHMARK_REGISTRY`."
    )
    lines.append("Regenerate with: `./scripts/list_benchmarks.sh`")
    lines.append("")
    lines.append(
        "Each benchmark pairs a communication beamformer (varied across "
        "rows) with a power-allocation strategy (Phase II of CORDIS-Split, "
        "or a uniform fixed ratio ρ). All benchmarks share the same "
        "interface and return a `BenchmarkResult` for the simulation "
        "runner."
    )
    lines.append("")
    lines.append("> **Note on LR-MMSE benchmarks (paper revision).**")
    lines.append(">")
    lines.append("> `lr_mmse_split` (LR-MMSE Phase-I + optimal P-Split PA) is "
                 "numerically identical to CORDIS-Split (which is precisely "
                 "LR-MMSE BF + optimal PSR by construction).  To avoid "
                 "duplicating the CORDIS-Split curve in figures, the default "
                 "benchmark surface (`cordis_vs_benchmarks`, `all_algorithms`) "
                 "now uses `lr_mmse_fixed` (PSR ρ=0.5, no PA optimisation) "
                 "as the LR-MMSE representative.  The `lr_mmse_split` entry "
                 "is retained for debugging / legacy access but is not in any "
                 "default spec set.")
    lines.append(">")
    lines.append("> The two extra variants `lr_mmse_fixed_p020` (ρ=0.2, "
                 "sensing-biased) and `lr_mmse_fixed_p080` (ρ=0.8, comm-biased) "
                 "support the paper's *benefit of optimal PSR* figure via the "
                 "`psr_baselines` spec set "
                 "(`cordis.experiments.specs.psr_baselines`).")
    lines.append("")
    lines.append("## Quick usage")
    lines.append("")
    lines.append("```python")
    lines.append("from cordis.algorithms.benchmarks import (")
    lines.append("    BENCHMARK_REGISTRY, run_benchmark, run_all_benchmarks,")
    lines.append(")")
    lines.append("")
    lines.append("# Single benchmark")
    lines.append("res = run_benchmark(")
    lines.append('    "mrt_split", topo, cfg, est, sensing_stats,')
    lines.append("    association, sigma_n_sq, Pmax,")
    lines.append(")")
    lines.append("# res.W_tx, res.phase_i, res.split_res, res.psr")
    lines.append("")
    lines.append("# Whole sweep on one scenario")
    lines.append("all_res = run_all_benchmarks(")
    lines.append("    topo, cfg, est, sensing_stats, association,")
    lines.append("    sigma_n_sq, Pmax, skip_failures=True,")
    lines.append(")")
    lines.append("```")
    lines.append("")

    opt_rows, fix_rows = _split_groups(rows)

    def _table(rs: List[Row]) -> List[str]:
        out = ["| Name | Comm BF | ρ | Description |",
               "|------|---------|---|-------------|"]
        for name, bf, _pa, rho, desc in rs:
            # Escape pipes in descriptions just in case
            desc_clean = desc.replace("|", "\\|")
            out.append(f"| `{name}` | `{bf}` | {rho} | {desc_clean} |")
        return out

    if opt_rows:
        lines.append(
            "## PA optimised via P-Split (Phase II of CORDIS-Split)"
        )
        lines.append("")
        lines.append(
            "Each AP's power-splitting ratio ρ_a is chosen by the "
            "projected-gradient / CVXPY Phase II solver to satisfy the "
            "per-user SINR target γ while maximising the sensing utility. "
            "These variants isolate the *choice of communication "
            "beamformer*."
        )
        lines.append("")
        lines.extend(_table(opt_rows))
        lines.append("")

    if fix_rows:
        lines.append("## PA fixed (no Phase II optimisation)")
        lines.append("")
        lines.append(
            "All APs use a uniform ρ (no PA optimisation). These variants "
            "isolate the *value of the PA optimisation itself* by holding "
            "the beamformer fixed and skipping Phase II."
        )
        lines.append("")
        lines.extend(_table(fix_rows))
        lines.append("")

    lines.append("## Adding a benchmark")
    lines.append("")
    lines.append(
        "Add a new entry to `BENCHMARK_REGISTRY` in "
        "`cordis/algorithms/benchmarks.py`. The tuple is "
        "`(description, comm_bf_method, pa_optimized, fixed_psr)`. "
        "Then run `./scripts/list_benchmarks.sh` to refresh this file."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_csv(rows: List[Row]) -> str:
    """CSV for spreadsheets."""
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "comm_bf", "pa_mode", "rho", "description"])
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="list_benchmarks.py",
        description=SCRIPT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Terminal table
  python scripts/list_benchmarks.py

  # Markdown reference doc
  python scripts/list_benchmarks.py --format markdown > docs/benchmarks_reference.md

  # CSV export
  python scripts/list_benchmarks.py --format csv > docs/benchmarks_reference.csv

  # Filter by substring (matches name, BF method, or description)
  python scripts/list_benchmarks.py --search mrt
  python scripts/list_benchmarks.py --search fixed
""",
    )
    parser.add_argument(
        "--format", default="terminal",
        choices=["terminal", "markdown", "csv"],
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--search", default=None, metavar="KEYWORD",
        help=(
            "Show only benchmarks whose name, BF method, or description "
            "contains KEYWORD (case-insensitive)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _rows(search=args.search)

    if args.format == "terminal":
        sys.stdout.write(render_terminal(rows))
    elif args.format == "markdown":
        sys.stdout.write(render_markdown(rows))
    elif args.format == "csv":
        sys.stdout.write(render_csv(rows))


if __name__ == "__main__":
    main()

