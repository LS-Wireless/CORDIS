"""
scripts/regenerate_experiment_scripts.py
========================================

Regenerate the per-experiment runner and plot scripts from a single
metadata table.

Outputs FOUR files per experiment, into a target directory tree::

    configs/recipes/exp_<name>.sh      — config recipe (sources _defaults.sh)
    scripts/exp_<name>.py              — runner (thin wrapper over _exp_common)
    scripts/exp_<name>.sh              — shell launcher for runner
    scripts/plot_<name>.py             — plot wrapper (thin wrapper over _plot_common)

Run from the repo root::

    python3 scripts/regenerate_experiment_scripts.py

Re-running is safe — files are overwritten in place.  The generated
files are checked-in deliverables; this tool exists so that adding
a new experiment to :data:`cordis.experiments.REGISTRY` is a single
metadata-table edit rather than four hand-written files.
"""
from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Metadata for the 11 experiments
# ─────────────────────────────────────────────────────────────────────
#
# Each entry produces FOUR files.  Fields:
#
#   summary       : one-line description for headers
#   kind          : "single" | "sweep" | "trace" | "table"
#                   (matches ExperimentResult.kind; controls plot type)
#   default_n_drops, default_n_real : Monte-Carlo size for the
#                   ``run_*`` call, set as the runner-script default
#   recipe_overrides : list of (VAR, value, comment) — config-recipe
#                   overrides relative to _defaults.sh
#   sweep_cli     : list of extra CLI args the runner script exposes
#                   for sweep ranges (used by sweep / convergence /
#                   table experiments only)
#   extra_run_kwargs : extra kwargs to forward to the registry's
#                   ``run_*`` function beyond n_drops / n_realizations
#   plot_call     : Python source snippet that builds & saves the
#                   figure(s); references ``result`` and ``args``

EXPERIMENTS: Dict[str, Dict[str, Any]] = {

    # =================================================================
    # Single-config CDFs (kind="single")
    # =================================================================
    "sinr_cdf": {
        "summary": "Empirical CDF of per-user min-SINR across many trials.",
        "kind": "single",
        "default_n_drops": 50,
        "default_n_real":  4,
        "recipe_overrides": [
            ("N_TRIALS", "200", "Baseline; final-paper figure may want 500+"),
        ],
        "sweep_cli": [],
        "extra_run_kwargs": [],
        "plot_call": '''\
apply_paper_style()
fig, ax = plt.subplots(figsize=figsize(width="single", aspect=3.5/2.4))
plot_cdf(result.sim_result, metric="min_sinr_db",
         xlabel=r"min-SINR [dB]", ax=ax)
save_paper_figure(fig, "sinr_cdf", args, experiment_name="sinr_cdf")
plt.close(fig)
''',
    },

    "scnr_cdf": {
        "summary": "Empirical CDF of sum-SCNR (sensing performance) across trials.",
        "kind": "single",
        "default_n_drops": 50,
        "default_n_real":  4,
        "recipe_overrides": [
            ("N_TRIALS", "200", "Baseline; final-paper figure may want 500+"),
        ],
        "sweep_cli": [],
        "extra_run_kwargs": [],
        "plot_call": '''\
apply_paper_style()
fig, ax = plt.subplots(figsize=figsize(width="single", aspect=3.5/2.4))
plot_cdf(result.sim_result, metric="sum_scnr_db",
         xlabel=r"sum-SCNR [dB]", ax=ax)
save_paper_figure(fig, "scnr_cdf", args, experiment_name="scnr_cdf")
plt.close(fig)
''',
    },

    # =================================================================
    # Sweep experiments (kind="sweep")
    # =================================================================
    "gamma_sweep": {
        "summary": "min-SINR & sum-SCNR vs per-user SINR target γ [dB].",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS",   "100",  "Smaller than CDF — many sweep points"),
            ("SPLIT_GAMMA_DB", "10.0", "Baseline; sweep overrides at runtime"),
        ],
        "sweep_cli": [
            ("--gamma-values", "str", "''",
             "Comma- or space-separated γ values in dB (e.g. '-3,0,3,6,10,14,18'). "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("gamma_values",
             "parse_value_list(args.gamma_values, float) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"$\\gamma$ [dB]",
)
save_paper_figure(fig, "gamma_sweep", args, experiment_name="gamma_sweep")
plt.close(fig)
''',
    },

    "kappa_sweep": {
        "summary": "min-SINR & sum-SCNR vs clutter penalty κ.",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS",   "100", ""),
            ("ADMM_KAPPA", "1.0", "Baseline; sweep overrides at runtime"),
        ],
        "sweep_cli": [
            ("--kappa-values", "str", "''",
             "Comma- or space-separated κ values (e.g. '0,0.1,0.5,1.0,1.5,2.0'). "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("kappa_values",
             "parse_value_list(args.kappa_values, float) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"$\\kappa$",
)
save_paper_figure(fig, "kappa_sweep", args, experiment_name="kappa_sweep")
plt.close(fig)
''',
    },

    "clutter_cnr_sweep": {
        "summary": "min-SINR & sum-SCNR vs clutter-to-noise ratio (CNR) [dB].",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS",                "100", ""),
            ("CLUTTER_CNR_DB",          "-10.0", "Baseline; swept at runtime"),
            ("CLUTTER_CENTER_STRATEGY", '"target_centroid"',
             "Default placement; switch to offset to decouple geometry"),
        ],
        "sweep_cli": [
            ("--cnr-values-db", "str", "''",
             "Comma- or space-separated CNR values in dB. "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("cnr_values_db",
             "parse_value_list(args.cnr_values_db, float) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"Clutter CNR [dB]",
)
save_paper_figure(fig, "clutter_cnr_sweep", args,
                  experiment_name="clutter_cnr_sweep")
plt.close(fig)
''',
    },

    "snr_sweep": {
        "summary": "min-SINR & sum-SCNR vs operating SNR P_max/σ² [dB].",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS", "100", ""),
            ("SNR_DB",   "20.0", "Baseline; swept at runtime"),
        ],
        "sweep_cli": [
            ("--snr-values-db", "str", "''",
             "Comma- or space-separated SNR values in dB. "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("snr_values_db",
             "parse_value_list(args.snr_values_db, float) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"$P_{\\max}/\\sigma^2$ [dB]",
)
save_paper_figure(fig, "snr_sweep", args, experiment_name="snr_sweep")
plt.close(fig)
''',
    },

    "n_ue_sweep": {
        "summary": "min-SINR & sum-SCNR vs number of UEs.",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS", "100", ""),
            ("N_UE",     "4",   "Baseline; swept at runtime"),
        ],
        "sweep_cli": [
            ("--n-ue-values", "str", "''",
             "Comma- or space-separated UE counts (e.g. '2,4,6,8,10'). "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("n_ue_values",
             "parse_value_list(args.n_ue_values, int) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"$N_{\\rm UE}$",
)
save_paper_figure(fig, "n_ue_sweep", args, experiment_name="n_ue_sweep")
plt.close(fig)
''',
    },

    "n_ap_sweep": {
        "summary": "min-SINR & sum-SCNR vs number of APs.",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS", "100", ""),
            ("N_AP",     "6",   "Baseline; swept at runtime"),
        ],
        "sweep_cli": [
            ("--n-ap-values", "str", "''",
             "Comma- or space-separated AP counts (e.g. '4,6,8,10,12'). "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("n_ap_values",
             "parse_value_list(args.n_ap_values, int) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"$N_{\\rm AP}$",
)
save_paper_figure(fig, "n_ap_sweep", args, experiment_name="n_ap_sweep")
plt.close(fig)
''',
    },

    "antennas_sweep": {
        "summary": "min-SINR & sum-SCNR vs antennas per AP M.",
        "kind": "sweep",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS", "100", ""),
            ("N_ANT",    "10",  "Baseline; swept at runtime"),
            ("N_RF_CHAINS", "10", "Matches N_ANT"),
        ],
        "sweep_cli": [
            ("--n-ant-values", "str", "''",
             "Comma- or space-separated antenna counts (e.g. '4,6,8,10,12'). "
             "Empty → use registry default."),
        ],
        "extra_run_kwargs": [
            ("n_ant_values",
             "parse_value_list(args.n_ant_values, int) or None"),
        ],
        "plot_call": '''\
fig, _ = sweep_plot_pair(
    result,
    metric_top="min_sinr_db",  ylabel_top=r"min-SINR [dB]",
    metric_bot="sum_scnr_db",  ylabel_bot=r"sum-SCNR [dB]",
    xlabel=r"$M$",
)
save_paper_figure(fig, "antennas_sweep", args, experiment_name="antennas_sweep")
plt.close(fig)
''',
    },

    # =================================================================
    # Trace experiment (kind="trace")
    # =================================================================
    "convergence_trace": {
        "summary": "ADMM primal/dual residual + objective trajectory on one trial.",
        "kind": "trace",
        "default_n_drops": 1,    # ignored — uses fixed seed pair
        "default_n_real":  1,
        "recipe_overrides": [
            ("ADMM_N_MAX", "80",  "Headroom past convergence to see plateau"),
        ],
        "sweep_cli": [
            ("--drop-seed",       "int", "42",
             "Topology drop seed for the single trial."),
            ("--realization-seed", "int", "43",
             "Channel realisation seed for the single trial."),
            ("--n-admm-max",       "int", "80",
             "Cap on ADMM iterations."),
        ],
        "extra_run_kwargs": [
            ("drop_seed",         "args.drop_seed"),
            ("realization_seed",  "args.realization_seed"),
            ("n_admm_max",        "args.n_admm_max"),
        ],
        "plot_call": '''\
apply_paper_style()
ax_top, ax_bot = plot_admm_convergence(result.admm_result, log_y=True)
fig = ax_top.figure
save_paper_figure(fig, "convergence_trace", args,
                  experiment_name="convergence_trace")
plt.close(fig)
''',
    },

    # =================================================================
    # Table experiment (kind="table")
    # =================================================================
    "fronthaul_table": {
        "summary": "Per-AP fronthaul coordination overhead for each algorithm.",
        "kind": "table",
        "default_n_drops": 20,
        "default_n_real":  2,
        "recipe_overrides": [
            ("N_TRIALS", "100", "Fewer drops — only used to estimate T_ADMM"),
        ],
        "sweep_cli": [
            ("--n-admm-drops", "int", "20",
             "Drops used to estimate average ADMM iteration count."),
        ],
        "extra_run_kwargs": [
            ("n_admm_drops", "args.n_admm_drops"),
        ],
        "plot_call": '''\
# fronthaul_table emits a Markdown + LaTeX table rather than a figure.
import json

out_dir = resolve_output_dir("fronthaul_table", args)
table   = result.table_data
md      = result.metadata

# ── Markdown ─────────────────────────────────────────────────────────
md_lines = ["# Fronthaul Coordination Overhead per AP",
            "",
            f"_Scenario: M={md.get('M')}, "
            f"N_UE={md.get('n_ue')}, "
            f"N_targets={md.get('n_targets')}, "
            f"|D|={md.get('n_streams')}_",
            "",
            f"_Average ADMM iterations: T_ADMM = "
            f"{md.get('admm_avg_iters', float('nan')):.2f}_",
            "",
            "| Algorithm | Data shared | Size | Real scalars / round | Scales with | Scalable |",
            "|---|---|---|---|---|---|"]
for algo, row in table.items():
    rs = row.get("real_scalars")
    rs_str = "—" if rs is None else (
        f"{rs:.1f}" if isinstance(rs, float) else f"{rs:d}"
    )
    md_lines.append(
        f"| {algo} | {row['data_to_share']} | {row['size']} | "
        f"{rs_str} | {row['scales_with']} | "
        f"{'✓' if row['scalable'] else '✗'} |"
    )
md_path = out_dir / "fronthaul_table.md"
md_path.write_text("\\n".join(md_lines) + "\\n", encoding="utf-8")
log.info("Wrote %s", md_path)

# ── LaTeX ────────────────────────────────────────────────────────────
tex = ["\\\\begin{table}[t]",
       "\\\\centering",
       "\\\\caption{Per-AP fronthaul coordination overhead.}",
       "\\\\label{tab:fronthaul}",
       "\\\\begin{tabular}{lcccc}",
       "\\\\toprule",
       "Algorithm & Data shared & Size & Real / iter & Scalable \\\\\\\\",
       "\\\\midrule"]
for algo, row in table.items():
    rs = row.get("real_scalars")
    rs_str = "--" if rs is None else (
        f"{rs:.1f}" if isinstance(rs, float) else f"{rs:d}"
    )
    tex.append(
        f"{algo} & {row['data_to_share']} & {row['size']} & "
        f"{rs_str} & {'$\\\\checkmark$' if row['scalable'] else '$\\\\times$'} \\\\\\\\"
    )
tex += ["\\\\bottomrule", "\\\\end{tabular}", "\\\\end{table}"]
tex_path = out_dir / "fronthaul_table.tex"
tex_path.write_text("\\n".join(tex) + "\\n", encoding="utf-8")
log.info("Wrote %s", tex_path)

# ── JSON dump for downstream tooling ─────────────────────────────────
json_path = out_dir / "fronthaul_table.json"
json_path.write_text(json.dumps({"table": table, "metadata": md},
                                 indent=2, default=str),
                     encoding="utf-8")
log.info("Wrote %s", json_path)
''',
    },
}


# ─────────────────────────────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────────────────────────────

CONFIG_RECIPE_TEMPLATE = """\
#!/usr/bin/env bash
# =============================================================================
# configs/recipes/exp_{name}.sh
# =============================================================================
#
# {summary}
#
# Usage:
#   bash configs/recipes/exp_{name}.sh                 # write configs/exp_{name}.json
#   bash configs/recipes/exp_{name}.sh --dry-run       # preview the diff
#
# This recipe sources configs/recipes/_defaults.sh, then overrides only
# the parameters that matter for this experiment.  Adjust the values in
# the OVERRIDES block below; everything else inherits from defaults.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR/../.." || {{ echo "ERROR: could not find repo root"; exit 1; }}

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi

# ─── OVERRIDES — edit these ──────────────────────────────────────────
NAME="exp_{name}"
{overrides}

# ─── Defaults for everything else ────────────────────────────────────
source "$SCRIPT_DIR/_defaults.sh"

# ─── Build the config ────────────────────────────────────────────────
python3 scripts/create_config.py \\
    --name "$NAME" \\
    $DRY_RUN_FLAG \\
    --set "${{SET_ARGS[@]}}"
"""

RUNNER_PY_TEMPLATE = '''\
#!/usr/bin/env python3
"""
scripts/exp_{name}.py
{name_underline}

{summary}

Run via the matching shell wrapper::

    bash scripts/exp_{name}.sh

…or directly::

    python3 scripts/exp_{name}.py --help

Generated by scripts/regenerate_experiment_scripts.py — edit the metadata
table in that file (not this one) if behaviour needs to change.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _exp_common import (    # noqa: E402
    build_base_parser,
    add_{add_args_helper},
    parse_value_list,
    run_experiment,
)


def main() -> None:
    parser = build_base_parser("{name}")
    add_{add_args_helper}(parser,
                          default_n_drops={n_drops},
                          default_n_real={n_real})
{sweep_cli_block}
    args = parser.parse_args()

    extra_kwargs = {{
        "n_drops":        args.n_drops,
        "n_realizations": args.n_realizations,
{extra_kwargs_block}
    }}
    # Drop any None-valued sweep ranges so the registry uses its defaults.
    extra_kwargs = {{k: v for k, v in extra_kwargs.items() if v is not None}}

    run_experiment("{name}", args, extra_kwargs)


if __name__ == "__main__":
    main()
'''

RUNNER_SH_TEMPLATE = """\
#!/usr/bin/env bash
# =============================================================================
# scripts/exp_{name}.sh
# =============================================================================
#
# {summary}
#
# Wrapper around scripts/exp_{name}.py that sets defaults appropriate
# for the journal paper.  Edit the OPTIONS block to customise.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR/.." || {{ echo "ERROR: could not find repo root"; exit 1; }}

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi

# ─── OPTIONS — edit these ────────────────────────────────────────────
BASE_CONFIG="${{BASE_CONFIG:-configs/default.json}}"
EXP_CONFIG="${{EXP_CONFIG:-configs/exp_{name}.json}}"   # generated by configs/recipes/exp_{name}.sh
N_DROPS="${{N_DROPS:-{n_drops}}}"
N_REAL="${{N_REAL:-{n_real}}}"
N_WORKERS="${{N_WORKERS:--1}}"          # -1 = all cores; 1 = sequential
SEED="${{SEED:-42}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-results}}"
{sweep_options_block}

# ─── Run ─────────────────────────────────────────────────────────────
EXP_CONFIG_FLAG=""
if [ -f "$EXP_CONFIG" ]; then EXP_CONFIG_FLAG="--exp-config $EXP_CONFIG"; fi

python3 scripts/exp_{name}.py \\
    --base-config   "$BASE_CONFIG"   \\
    $EXP_CONFIG_FLAG                 \\
    --n-drops       "$N_DROPS"       \\
    --n-realizations "$N_REAL"       \\
    --n-workers     "$N_WORKERS"     \\
    --seed          "$SEED"          \\
    --output-root   "$OUTPUT_ROOT"   \\
{sweep_args_block}"$@"
"""

PLOT_PY_TEMPLATE = '''\
#!/usr/bin/env python3
"""
scripts/plot_{name}.py
{name_underline}

Render the paper figure for the {name!r} experiment.

Loads the most recent results/exp_{name}/<ts>/ run (override with
``--exp-dir``) and writes figures into figures/exp_{name}/.

Generated by scripts/regenerate_experiment_scripts.py.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make sibling helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plot_common import (   # noqa: E402
    build_plot_parser, load_result, save_paper_figure,
    resolve_output_dir, sweep_plot_pair,
)

# Common matplotlib + cordis.plotting bits.
import matplotlib.pyplot as plt    # noqa: E402
from cordis.plotting import (      # noqa: E402
    apply_paper_style, figsize,
    plot_cdf, plot_sweep, plot_admm_convergence,
)

log = logging.getLogger(__name__)


def main() -> None:
    args   = build_plot_parser("{name}").parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    result = load_result("{name}", args)

{plot_call_indented}

if __name__ == "__main__":
    main()
'''


# ─────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────

def _format_overrides(overrides: List) -> str:
    lines = []
    for var, val, comment in overrides:
        if comment:
            lines.append(f"{var}={val}    # {comment}")
        else:
            lines.append(f"{var}={val}")
    return "\n".join(lines) if lines else "# (no overrides — uses pure defaults)"


def _format_sweep_cli(sweep_cli: List) -> str:
    if not sweep_cli:
        return ""
    lines = []
    for flag, kind, default, helptext in sweep_cli:
        if kind == "str":
            lines.append(
                f'    parser.add_argument("{flag}", default={default},'
                f'\n                        help={helptext!r})'
            )
        elif kind == "int":
            lines.append(
                f'    parser.add_argument("{flag}", type=int, default={default},'
                f'\n                        help={helptext!r})'
            )
        elif kind == "float":
            lines.append(
                f'    parser.add_argument("{flag}", type=float, default={default},'
                f'\n                        help={helptext!r})'
            )
        else:
            raise ValueError(f"unknown sweep_cli kind {kind!r}")
    return "\n".join(lines)


def _format_extra_kwargs(extra_kwargs: List) -> str:
    if not extra_kwargs:
        return ""
    return "\n".join(
        f'        "{k}": {expr},' for k, expr in extra_kwargs
    )


def _format_sweep_options_block(sweep_cli: List) -> str:
    """Shell snippet defining variables for each sweep flag."""
    if not sweep_cli:
        return ""
    lines = []
    for flag, kind, default, _help in sweep_cli:
        var = flag.lstrip("-").upper().replace("-", "_")
        default_val = default.strip("'\"") if isinstance(default, str) else default
        lines.append(f'{var}="${{{var}:-{default_val}}}"')
    return "\n".join(lines)


def _format_sweep_args_block(sweep_cli: List) -> str:
    """Shell snippet appending each sweep flag to the python call."""
    if not sweep_cli:
        return "    "
    lines = []
    for flag, _kind, _default, _help in sweep_cli:
        var = flag.lstrip("-").upper().replace("-", "_")
        lines.append(f'    {flag}    "${var}"   \\')
    return "\n".join(lines) + "\n    "


def _indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join((pad + line) if line.strip() else line
                     for line in s.splitlines())


def render_files(name: str, meta: Dict[str, Any]) -> Dict[Path, str]:
    """Produce the four files for one experiment as a dict path → text."""
    out: Dict[Path, str] = {}
    summary = meta["summary"]
    n_drops = meta["default_n_drops"]
    n_real  = meta["default_n_real"]

    # Pick the right add_*_args helper based on kind.
    add_helper = {
        "single": "cdf_args",
        "sweep":  "sweep_args",
        "trace":  "drops_args",
        "table":  "sweep_args",
    }[meta["kind"]]

    # ── configs/recipes/exp_<name>.sh ────────────────────────────────
    overrides_text = _format_overrides(meta["recipe_overrides"])
    out[REPO_ROOT / "configs" / "recipes" / f"exp_{name}.sh"] = (
        CONFIG_RECIPE_TEMPLATE.format(
            name=name,
            summary=summary,
            overrides=overrides_text,
        )
    )

    # ── scripts/exp_<name>.py ────────────────────────────────────────
    sweep_cli_block   = _format_sweep_cli(meta["sweep_cli"])
    extra_kwargs_blk  = _format_extra_kwargs(meta["extra_run_kwargs"])
    out[REPO_ROOT / "scripts" / f"exp_{name}.py"] = (
        RUNNER_PY_TEMPLATE.format(
            name=name,
            name_underline="=" * (len(f"scripts/exp_{name}.py")),
            summary=summary,
            add_args_helper=add_helper,
            n_drops=n_drops,
            n_real=n_real,
            sweep_cli_block=sweep_cli_block,
            extra_kwargs_block=extra_kwargs_blk,
        )
    )

    # ── scripts/exp_<name>.sh ────────────────────────────────────────
    out[REPO_ROOT / "scripts" / f"exp_{name}.sh"] = (
        RUNNER_SH_TEMPLATE.format(
            name=name,
            summary=summary,
            n_drops=n_drops,
            n_real=n_real,
            sweep_options_block=_format_sweep_options_block(meta["sweep_cli"]),
            sweep_args_block=_format_sweep_args_block(meta["sweep_cli"]),
        )
    )

    # ── scripts/plot_<name>.py ───────────────────────────────────────
    plot_call_indented = _indent(meta["plot_call"], 4)
    out[REPO_ROOT / "scripts" / f"plot_{name}.py"] = (
        PLOT_PY_TEMPLATE.format(
            name=name,
            name_underline="=" * (len(f"scripts/plot_{name}.py")),
            plot_call_indented=plot_call_indented,
        )
    )

    return out


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    n_written = 0
    for name, meta in EXPERIMENTS.items():
        files = render_files(name, meta)
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            # Make .sh files executable.
            if path.suffix == ".sh":
                mode = path.stat().st_mode
                path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            n_written += 1
        print(f"  generated 4 files for {name}")
    print(f"\nTotal: {n_written} files for {len(EXPERIMENTS)} experiments.")


if __name__ == "__main__":
    main()

