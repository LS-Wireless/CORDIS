#!/usr/bin/env bash
# =============================================================================
# paper/figure_src/fig_convergence/run_traces.sh
# =============================================================================
#
# Generate the two single-trial ADMM convergence traces this figure needs:
#
#   1. fixed-rho   (the shipped default: adaptive_rho=false)   -> data/trace_fixed
#   2. adaptive-rho (the ablation:       adaptive_rho=true )   -> data/trace_adaptive
#
# Both use the SAME drop/realization seed so the two curves are comparable,
# and both disable patience early-stop (ADMM_EARLY_STOP_PATIENCE=0) so the
# full n_max trajectory is captured.
#
# The env overrides flow into the run through the recipe's SET_ARGS
# (configs/recipes/_defaults.sh, the ADMM_-prefixed knobs added in Stage 22b).
#
# ── IMPORTANT ────────────────────────────────────────────────────────────────
# Set RUN_CMD below to however YOU run a single experiment in this repo
# (the Makefile target, your SLURM submit wrapper, or the python runner).
# The default assumes the recipe writes configs/exp_convergence_trace.json and
# `make convergence_trace` consumes it. Adjust if your pipeline differs.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

# Walk up to the repo root (the dir containing cordis/).
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -d "$REPO_ROOT/cordis" ]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [ ! -d "$REPO_ROOT/cordis" ]; then
    echo "ERROR: could not find repo root (no cordis/ above $SCRIPT_DIR)"; exit 1
fi
cd "$REPO_ROOT"
echo "[info] repo root: $REPO_ROOT"

# ── Knobs you may want to edit ───────────────────────────────────────────────
RUN_CMD="${RUN_CMD:-make convergence_trace}"   # how to run one experiment
RESULTS_BASE="results/exp_convergence_trace"   # where runs land
mkdir -p "$DATA_DIR"

# Grab the newest run directory under RESULTS_BASE (the one we just produced).
newest_run() {
    ls -1dt "$RESULTS_BASE"/*/ 2>/dev/null | head -n 1
}

run_one() {
    # $1 = adaptive_rho value (true|false), $2 = destination subdir name
    local adaptive="$1" dest="$2"
    echo
    echo "=============================================================="
    echo "  Trace: adaptive_rho=$adaptive  ->  data/$dest"
    echo "=============================================================="

    # 1) (Re)build the experiment config with our overrides baked in.
    ADMM_ADAPTIVE_RHO="$adaptive" \
    ADMM_EARLY_STOP_PATIENCE=0 \
        bash configs/recipes/exp_convergence_trace.sh

    # 2) Run it. We also export the overrides in case your runner reads
    #    them from the environment directly rather than from the config.
    ADMM_ADAPTIVE_RHO="$adaptive" \
    ADMM_EARLY_STOP_PATIENCE=0 \
        ${RUN_CMD}

    # 3) Copy the freshly-produced run into this figure's data/ dir.
    local src; src="$(newest_run)"
    if [ -z "${src:-}" ] || [ ! -f "${src}manifest.json" ]; then
        echo "ERROR: no run dir with manifest.json under $RESULTS_BASE" >&2
        exit 1
    fi
    rm -rf "${DATA_DIR:?}/$dest"
    mkdir -p "$DATA_DIR/$dest"
    cp -f "${src}manifest.json" "${src}trace.npz" "$DATA_DIR/$dest/"
    echo "[ok] stashed $(basename "$src") -> data/$dest"
}

run_one "false" "trace_fixed"
run_one "true"  "trace_adaptive"

echo
echo "Done. Build the figure with:"
echo "    python3 paper/figure_src/fig_convergence/build_fig_convergence.py"
echo "(add --no-tex on a node without pdflatex)"

