"""
scripts/regenerate_experiment_scripts.py
========================================

Regenerate the per-experiment runner, plot, and SLURM scripts from a
single metadata table.

Outputs FIVE files per experiment, into a target directory tree::

    configs/recipes/exp_<name>.sh      — config recipe (sources _defaults.sh)
    scripts/exp_<name>.py              — runner (thin wrapper over _exp_common)
    scripts/exp_<name>.sh              — shell launcher for runner
    scripts/plot_<name>.py             — plot wrapper (thin wrapper over _plot_common)
    scripts/slurm/exp_<name>.sbatch    — SLURM submission script

Run from the repo root::

    python3 scripts/regenerate_experiment_scripts.py

Re-running is safe — files are overwritten in place.  The generated
files are checked-in deliverables; this tool exists so that adding
a new experiment to :data:`cordis.experiments.REGISTRY` is a single
metadata-table edit rather than five hand-written files.
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
    add_{add_args_helper}(parser)
{sweep_cli_block}
    args = parser.parse_args()

    extra_kwargs = {{
{extra_kwargs_block}
    }}
    # Drop any None-valued sweep ranges so the registry uses its defaults.
    extra_kwargs = {{k: v for k, v in extra_kwargs.items() if v is not None}}

    # Per-experiment fallback drops × real (used only if CLI args don't
    # specify trial counts and config has no simulation.n_trials).
    run_experiment("{name}", args, extra_kwargs,
                   fallback_n_drops={n_drops},
                   fallback_n_real={n_real})


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

# Ensure cordis is importable without requiring `pip install -e .`
export PYTHONPATH="$(pwd):${{PYTHONPATH:-}}"

# ─── OPTIONS — edit these ────────────────────────────────────────────
BASE_CONFIG="${{BASE_CONFIG:-configs/default.json}}"
EXP_CONFIG="${{EXP_CONFIG:-configs/exp_{name}.json}}"   # generated by configs/recipes/exp_{name}.sh
# Trial-count knobs — set EITHER N_TRIALS (auto-decompose into drops×real)
# OR N_DROPS and N_REAL explicitly.  See `python3 scripts/exp_{name}.py --help`.
# Per-experiment fallback when nothing else is set: {n_drops} drops × {n_real} real.
N_DROPS="${{N_DROPS:-}}"
N_REAL="${{N_REAL:-}}"
N_TRIALS="${{N_TRIALS:-}}"
N_WORKERS="${{N_WORKERS:--1}}"          # -1 = all cores; 1 = sequential
SEED="${{SEED:-42}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-results}}"
# Algorithm selection — leave empty to use the per-experiment default
# (sinr_cdf/scnr_cdf → all_algorithms; gamma/kappa/clutter sweeps →
# cordis_vs_centralized; antennas_sweep → cordis_vs_benchmarks).  Set
# to one of: cordis_only, cordis_vs_centralized, cordis_vs_benchmarks,
# all_algorithms — to override.
SPECS="${{SPECS:-}}"
{sweep_options_block}

# ─── Run ─────────────────────────────────────────────────────────────
EXP_CONFIG_FLAG=""
if [ -f "$EXP_CONFIG" ]; then EXP_CONFIG_FLAG="--exp-config $EXP_CONFIG"; fi

# Conditionally include trial-count flags so the resolver can pick the
# right precedence path (explicit → n_trials → config → defaults).
TRIAL_ARGS=()
[ -n "$N_DROPS"  ] && TRIAL_ARGS+=(--n-drops       "$N_DROPS")
[ -n "$N_REAL"   ] && TRIAL_ARGS+=(--n-realizations "$N_REAL")
[ -n "$N_TRIALS" ] && TRIAL_ARGS+=(--n-trials      "$N_TRIALS")

# Conditionally include --specs.  Empty means "use per-experiment default".
SPEC_ARGS=()
[ -n "$SPECS" ] && SPEC_ARGS+=(--specs "$SPECS")

python3 scripts/exp_{name}.py \\
    --base-config   "$BASE_CONFIG"   \\
    $EXP_CONFIG_FLAG                 \\
    "${{TRIAL_ARGS[@]}}"             \\
    "${{SPEC_ARGS[@]}}"              \\
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

# Make `cordis` and the sibling `_plot_common` helpers importable when
# this script is run directly (e.g. `python3 scripts/plot_{name}.py`).
# The runner picks up this script's directory (scripts/) for siblings
# AND its parent (the repo root) for the in-tree cordis package.
_HERE      = Path(__file__).resolve().parent          # scripts/
_REPO_ROOT = _HERE.parent                              # repo root
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT))

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
# SLURM submission template
# ─────────────────────────────────────────────────────────────────────
#
# Per-experiment-kind defaults for cluster resource sizing.  These
# are reasonable starting points — tune per your cluster's hardware
# and the N_TRIALS you intend to run.

SLURM_DEFAULTS_BY_KIND: Dict[str, Dict[str, str]] = {
    # Stage 21 — UCI HPC3 standard partition (40-core/192GB Cascade Lake):
    # asking for 40 CPUs gives ~192 GB RAM proportionally; specifying
    # --mem artificially caps it.  All single-job experiments use the
    # same conservative envelope (40 CPUs, 8h) to be safe even when
    # final-paper figures bump N_TRIALS.  Override at submit time.
    # CDFs:   N_TRIALS ≈ 200-500, single config
    "single": {"time": "08:00:00", "cpus": "40"},
    # Sweeps: N_TRIALS × N_sweep_points
    "sweep":  {"time": "08:00:00", "cpus": "40"},
    # Trace:  single trial, ADMM iters traced — still small, but use
    # same envelope to keep submit scripts uniform.
    "trace":  {"time": "08:00:00", "cpus": "40"},
    # Table:  small sweep over n_ap
    "table":  {"time": "08:00:00", "cpus": "40"},
}

# Array-task envelope: each task does TOTAL_TRIALS / N_ARRAY_TASKS
# trials, so wall time per task is ~1/N_ARRAY_TASKS of the single-job
# wall time.  4h is more than enough for the default 10 tasks but
# allows headroom if the user bumps N_TRIALS or shrinks N_ARRAY_TASKS.
SLURM_ARRAY_DEFAULTS = {
    "array_time":         "04:00:00",
    "cpus":               "40",
    "default_n_tasks":    10,
    "default_array_last": 9,    # zero-indexed → 0-9 means 10 tasks
}

SLURM_TEMPLATE = """\
#!/usr/bin/env bash
# =============================================================================
# scripts/slurm/exp_{name}.sbatch
# =============================================================================
#
# SLURM submission script for the {name!r} experiment.
#
# Submit from the repo root:
#
#     sbatch scripts/slurm/exp_{name}.sbatch
#
# Resource directives are starting points — adjust ``--time``, ``--mem``,
# ``--cpus-per-task``, and add ``--partition`` / ``--account`` per your
# cluster's policy.  ``N_WORKERS`` inside the launcher honours
# ``$SLURM_CPUS_PER_TASK`` automatically.
#
# Generated by scripts/regenerate_experiment_scripts.py.
# =============================================================================
#SBATCH --job-name=cordis-{name}
#SBATCH --output=logs/slurm/{name}-%j.out
#SBATCH --error=logs/slurm/{name}-%j.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
# Uncomment and customise for your cluster:
# #SBATCH --partition=standard
# #SBATCH --account=your-account
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com

set -euo pipefail

# Ensure logs/slurm/ exists before SLURM tries to redirect into it.
# (The directives above are evaluated by sbatch BEFORE this script runs,
#  so on a fresh clone you may need to `mkdir -p logs/slurm` once by hand.)
cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"
mkdir -p logs/slurm

# Activate environment — adjust to your setup.
# module load python/3.12
# source venv/bin/activate

# Worker count tracks the SLURM allocation.
export N_WORKERS="${{SLURM_CPUS_PER_TASK:-1}}"

bash scripts/exp_{name}.sh
"""


# ─────────────────────────────────────────────────────────────────────
# UCI HPC3 — site-specific SLURM scripts (Stage 11)
#
# The generic SLURM_TEMPLATE above is portable but unopinionated.
# These UCI-specific .sub files mirror the per-experiment resource
# metadata but add: cluster account, partition, module load,
# venv activation, and a fixed repo path under DFS public storage.
#
# Cluster-wide and personal settings (account, email, repo dir,
# python module, venv name) live in scripts/slurm/uci-hpc3/_config.sh
# so they're shared across all 11 .sub files and easy to override.
# ─────────────────────────────────────────────────────────────────────

UCI_HPC3_CONFIG = '''\
# =============================================================================
# scripts/slurm/uci-hpc3/_config.sh
# =============================================================================
# Cluster-wide and personal settings sourced by every UCI HPC3 .sub script.
#
# Override any of these at submit time:
#
#     CORDIS_ACCOUNT=other_lab sbatch scripts/slurm/uci-hpc3/exp_sinr_cdf.sub
#
# Generated by scripts/regenerate_experiment_scripts.py.
# =============================================================================

# Cluster settings.
export CORDIS_ACCOUNT="${CORDIS_ACCOUNT:-swindle_lab}"
export CORDIS_PARTITION="${CORDIS_PARTITION:-standard}"

# Personal settings.
export CORDIS_MAIL_USER="${CORDIS_MAIL_USER:-mzafarid@uci.edu}"
export CORDIS_MAIL_TYPE="${CORDIS_MAIL_TYPE:-END,FAIL}"

# Environment.
export CORDIS_REPO_DIR="${CORDIS_REPO_DIR:-/pub/$USER/CORDIS}"
export CORDIS_PYTHON_MODULE="${CORDIS_PYTHON_MODULE:-python/3.14.3}"
export CORDIS_VENV="${CORDIS_VENV:-cordis_venv}"
'''


UCI_HPC3_ARRAY_COMMON = '''\
#!/usr/bin/env bash
# =============================================================================
# scripts/slurm/uci-hpc3/array/_array_common.sh
# =============================================================================
# Shared per-task setup sourced by every UCI HPC3 exp_*.array.sub script.
#
# Each array task gets a unique SEED so numpy SeedSequence yields a
# disjoint trial stream, and writes to its own subdirectory under
# results/array_<jobid>/task_<task_id>/ so concurrent tasks don't
# collide on output filenames.  After the array completes, the
# per-task pickles are merged via scripts/aggregate_array_batch.py.
#
# Knobs (set in the calling .array.sub via env, or override at submit
# time with VAR=value sbatch ...):
#
#   EXP_NAME           experiment name (set by caller, required)
#   N_TRIALS_TOTAL     total trials across the whole array (default 500)
#   N_ARRAY_TASKS      total task count (must match --array=0-(N-1))
#                      (default $SLURM_ARRAY_TASK_COUNT)
#   BASE_SEED          base seed; per-task SEED = BASE_SEED + task_id
#                      (default 42)
#   OUTPUT_ROOT        per-task output root.  Default places all tasks
#                      under results/array_<arrayjobid>/task_<id>/
#
# Generated by scripts/regenerate_experiment_scripts.py.
# =============================================================================

set -euo pipefail

# ── Locate repo + source _config.sh (cluster+personal settings) ─────
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    UCI_DIR="$SLURM_SUBMIT_DIR/scripts/slurm/uci-hpc3"
else
    # _array_common.sh is at .../uci-hpc3/array/, _config.sh at .../uci-hpc3/
    UCI_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
fi
# shellcheck source=../_config.sh
source "$UCI_DIR/_config.sh"

# ── Repo + venv + modules ───────────────────────────────────────────
cd "$CORDIS_REPO_DIR"
mkdir -p logs/slurm

module purge
module load "$CORDIS_PYTHON_MODULE"
# shellcheck source=/dev/null
source "$CORDIS_VENV/bin/activate"

# ── Compute per-task trial count + seed ─────────────────────────────
# N_ARRAY_TASKS defaults to whatever SLURM allocated (always available
# inside an array job; falls back to 1 if user invoked the script
# outside SLURM for local debugging).
: "${N_ARRAY_TASKS:=${SLURM_ARRAY_TASK_COUNT:-1}}"
: "${N_TRIALS_TOTAL:=500}"
: "${BASE_SEED:=42}"
: "${ARRAY_TASK_ID:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${ARRAY_JOB_ID:=${SLURM_ARRAY_JOB_ID:-local}}"

# Ceiling division so the final task picks up the remainder when
# N_TRIALS_TOTAL is not divisible by N_ARRAY_TASKS.  Last task may
# do slightly fewer trials than its share; that's fine.
export N_TRIALS=$(( (N_TRIALS_TOTAL + N_ARRAY_TASKS - 1) / N_ARRAY_TASKS ))
# Each task uses a unique base seed; SeedSequence(SEED).spawn(...) is
# prefix-stable so different SEED values yield disjoint streams.
export SEED=$(( BASE_SEED + ARRAY_TASK_ID ))
# Each task writes to its own subdirectory so concurrent writers don't
# collide on auto-generated timestamps.  The aggregator globs the
# array_<jobid>/ parent dir to find them.
export OUTPUT_ROOT="results/array_${ARRAY_JOB_ID}/task_${ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_ROOT"

# Worker count tracks the SLURM allocation (40 by default on HPC3).
export N_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

# Ensure cordis is importable without `pip install -e .`.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Tag the run so it's identifiable in the per-task output.
export TAG="array_${ARRAY_JOB_ID}_t${ARRAY_TASK_ID}"
'''


UCI_HPC3_SUB_TEMPLATE = """\
#!/usr/bin/env bash
# =============================================================================
# scripts/slurm/uci-hpc3/exp_{name}.sub
# =============================================================================
#
# SLURM submission script for the {name!r} experiment on UCI HPC3.
#
# Submit from anywhere (uses absolute repo path from _config.sh):
#
#     sbatch scripts/slurm/uci-hpc3/exp_{name}.sub
#
# Cluster-wide and personal settings (account, email, repo path,
# python module, venv name) come from scripts/slurm/uci-hpc3/_config.sh.
# Override any of them per-submit via env vars, e.g.:
#
#     CORDIS_PARTITION=free sbatch scripts/slurm/uci-hpc3/exp_{name}.sub
#
# Generated by scripts/regenerate_experiment_scripts.py.
# =============================================================================
#SBATCH --job-name=cordis-{name}
#SBATCH --output=logs/slurm/{name}-%j.out
#SBATCH --error=logs/slurm/{name}-%j.err
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition=standard
#SBATCH --account=swindle_lab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mzafarid@uci.edu
# NOTE: the four directives above (partition/account/mail-*) are
# parsed by sbatch BEFORE the script runs, so they cannot reference
# the env vars set in _config.sh.  They reflect the default values
# for the swindle_lab group; if you need to override, do so at
# submit time:  sbatch --partition=free --account=...  exp_{name}.sub

set -euo pipefail

# ── 1. Source shared cluster + personal settings ─────────────────────
# When run via sbatch, SLURM COPIES the script to its spool directory
# before execution, so ${{BASH_SOURCE[0]}} resolves to the spool path,
# not the .sub file's original location.  Use SLURM_SUBMIT_DIR (the
# directory from which `sbatch` was invoked, normally the repo root)
# to find _config.sh.  Fall back to BASH_SOURCE-based resolution for
# direct invocation outside SLURM (e.g. `bash exp_{name}.sub` for
# local debugging).
if [ -n "${{SLURM_SUBMIT_DIR:-}}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/scripts/slurm/uci-hpc3"
else
    SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
fi
# shellcheck source=_config.sh
source "$SCRIPT_DIR/_config.sh"

# ── 2. Navigate to the repo on DFS public storage ────────────────────
cd "$CORDIS_REPO_DIR"
mkdir -p logs/slurm

# ── 3. Load HPC3 modules + activate venv ─────────────────────────────
module purge
module load "$CORDIS_PYTHON_MODULE"
# shellcheck source=/dev/null
source "$CORDIS_VENV/bin/activate"

# ── 4. Worker count tracks the SLURM allocation ──────────────────────
export N_WORKERS="${{SLURM_CPUS_PER_TASK:-1}}"

# ── 5. Run the experiment ────────────────────────────────────────────
echo "=========================================================="
echo "CORDIS experiment: {name}"
echo "Node:              ${{SLURMD_NODENAME:-unknown}}"
echo "Job ID:            ${{SLURM_JOB_ID:-unknown}}"
echo "Allocated cores:   $N_WORKERS"
echo "Repo dir:          $CORDIS_REPO_DIR"
echo "Account/partition: $CORDIS_ACCOUNT / $CORDIS_PARTITION"
echo "=========================================================="

bash scripts/exp_{name}.sh

echo "=========================================================="
echo "Experiment complete: {name}"
echo "=========================================================="
"""


# ─────────────────────────────────────────────────────────────────────
# Stage 21 — Job-array variants for UCI HPC3
#
# Layout:
#     scripts/slurm/uci-hpc3/array/_array_common.sh   # shared per-task setup
#     scripts/slurm/uci-hpc3/array/exp_<name>.array.sub
#
# Each array task runs with a different ``SEED`` (= base + task_id) so
# numpy.random.SeedSequence yields disjoint trial streams.  Each task
# writes to its own ``results/array_<jobid>/task_<task_id>/...`` so
# concurrent writers don't collide.  After the array completes, the
# per-task pickles are merged via ``scripts/aggregate_array_batch.py``.
#
# Single-job .sub scripts above remain unchanged and continue to work.
# ─────────────────────────────────────────────────────────────────────

# Experiments that benefit from per-trial array splitting:
ARRAY_ENABLED_EXPERIMENTS = {
    "antennas_sweep", "clutter_cnr_sweep", "gamma_sweep",
    "kappa_sweep", "n_ap_sweep", "n_ue_sweep", "snr_sweep",
    "sinr_cdf", "scnr_cdf",
}
# Excluded:  convergence_trace (single trial — not parallelizable across
# trials), fronthaul_table (deterministic small sweep — overhead > gain).

UCI_HPC3_ARRAY_SUB_TEMPLATE = """\
#!/usr/bin/env bash
# =============================================================================
# scripts/slurm/uci-hpc3/array/exp_{name}.array.sub
# =============================================================================
#
# SLURM JOB-ARRAY submission script for the {name!r} experiment.
#
# Splits ``N_TRIALS_TOTAL`` trials across ``N_ARRAY_TASKS`` parallel
# tasks (default {default_n_tasks}).  Each task uses a unique seed so
# trial streams are disjoint.  After all tasks complete, aggregate
# per-task pickles via:
#
#     python3 scripts/aggregate_array_batch.py \\
#         results/array_<jobid>/
#
# Submit:
#
#     sbatch scripts/slurm/uci-hpc3/array/exp_{name}.array.sub
#
# Override knobs at submit time:
#
#     N_TRIALS_TOTAL=1000 N_ARRAY_TASKS=20  \\
#     sbatch scripts/slurm/uci-hpc3/array/exp_{name}.array.sub
#
# Generated by scripts/regenerate_experiment_scripts.py.
# =============================================================================
#SBATCH --job-name=cordis-{name}-array
#SBATCH --output=logs/slurm/{name}-array-%A_%a.out
#SBATCH --error=logs/slurm/{name}-array-%A_%a.err
#SBATCH --time={array_time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --array=0-{default_array_last}
#SBATCH --partition=standard
#SBATCH --account=swindle_lab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mzafarid@uci.edu
# NOTE: the directives above (time / cpus / array range / partition /
# account / mail-*) are parsed by sbatch BEFORE the script runs and
# cannot reference shell variables.  To change the array size at
# submit time use --array, e.g.:
#     sbatch --array=0-19 scripts/slurm/uci-hpc3/array/exp_{name}.array.sub

set -euo pipefail

# ── Locate the repo + sub-scripts (same logic as single-job .sub) ────
if [ -n "${{SLURM_SUBMIT_DIR:-}}" ]; then
    ARRAY_DIR="$SLURM_SUBMIT_DIR/scripts/slurm/uci-hpc3/array"
else
    ARRAY_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"
fi

# Source the shared per-task setup (computes SEED, N_TRIALS per task,
# OUTPUT_ROOT, loads modules, activates venv).
EXP_NAME="{name}"
# shellcheck source=_array_common.sh
source "$ARRAY_DIR/_array_common.sh"

# ── Run the experiment (env vars set by _array_common.sh) ────────────
echo "=========================================================="
echo "CORDIS experiment (array): $EXP_NAME"
echo "Array job ID:      ${{SLURM_ARRAY_JOB_ID:-unknown}}"
echo "Task ID:           ${{SLURM_ARRAY_TASK_ID:-unknown}} / ${{SLURM_ARRAY_TASK_COUNT:-?}}"
echo "Node:              ${{SLURMD_NODENAME:-unknown}}"
echo "Allocated cores:   $N_WORKERS"
echo "Seed (this task):  $SEED"
echo "Trials this task:  $N_TRIALS"
echo "Output root:       $OUTPUT_ROOT"
echo "=========================================================="

bash scripts/exp_${{EXP_NAME}}.sh

echo "=========================================================="
echo "Task complete: $EXP_NAME  (task ${{SLURM_ARRAY_TASK_ID:-unknown}})"
echo "After all tasks finish, aggregate with:"
echo "  python3 scripts/aggregate_array_batch.py $OUTPUT_ROOT/.."
echo "=========================================================="
"""


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

    # ── scripts/slurm/exp_<name>.sbatch ──────────────────────────────
    slurm_res = meta.get("slurm_resources",
                         SLURM_DEFAULTS_BY_KIND[meta["kind"]])
    out[REPO_ROOT / "scripts" / "slurm" / f"exp_{name}.sbatch"] = (
        SLURM_TEMPLATE.format(
            name=name,
            time=slurm_res["time"],
            cpus=slurm_res["cpus"],
        )
    )

    # ── scripts/slurm/uci-hpc3/exp_<name>.sub (Stage 11) ─────────────
    out[REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / f"exp_{name}.sub"] = (
        UCI_HPC3_SUB_TEMPLATE.format(
            name=name,
            time=slurm_res["time"],
            cpus=slurm_res["cpus"],
        )
    )

    # ── scripts/slurm/uci-hpc3/array/exp_<name>.array.sub (Stage 21) ─
    if name in ARRAY_ENABLED_EXPERIMENTS:
        out[REPO_ROOT / "scripts" / "slurm" / "uci-hpc3" / "array"
            / f"exp_{name}.array.sub"] = (
            UCI_HPC3_ARRAY_SUB_TEMPLATE.format(
                name=name,
                array_time=SLURM_ARRAY_DEFAULTS["array_time"],
                cpus=SLURM_ARRAY_DEFAULTS["cpus"],
                default_n_tasks=SLURM_ARRAY_DEFAULTS["default_n_tasks"],
                default_array_last=SLURM_ARRAY_DEFAULTS["default_array_last"],
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
        n_for_exp = len(files)
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            # Make .sh, .sbatch, and .sub files executable.
            if path.suffix in (".sh", ".sbatch", ".sub"):
                mode = path.stat().st_mode
                path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            n_written += 1
        print(f"  generated {n_for_exp} files for {name}")

    # ── Emit UCI HPC3 _config.sh once (Stage 11) ─────────────────────
    # Shared cluster + personal settings sourced by every exp_*.sub.
    # Always overwrite so the canonical defaults stay in sync with this
    # script's UCI_HPC3_CONFIG constant.
    uci_dir = REPO_ROOT / "scripts" / "slurm" / "uci-hpc3"
    uci_dir.mkdir(parents=True, exist_ok=True)
    config_path = uci_dir / "_config.sh"
    config_path.write_text(UCI_HPC3_CONFIG, encoding="utf-8")
    n_written += 1
    print(f"  generated 1 file for uci-hpc3/_config.sh")

    # ── Emit UCI HPC3 array/_array_common.sh once (Stage 21) ─────────
    # Shared per-task setup sourced by every exp_*.array.sub.
    array_dir = uci_dir / "array"
    array_dir.mkdir(parents=True, exist_ok=True)
    array_common_path = array_dir / "_array_common.sh"
    array_common_path.write_text(UCI_HPC3_ARRAY_COMMON, encoding="utf-8")
    n_written += 1
    print(f"  generated 1 file for uci-hpc3/array/_array_common.sh")

    print(f"\nTotal: {n_written} files for {len(EXPERIMENTS)} experiments.")


if __name__ == "__main__":
    main()

