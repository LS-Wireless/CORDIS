#!/usr/bin/env bash
# =============================================================================
# configs/recipes/exp_convergence_trace.sh
# =============================================================================
#
# ADMM primal/dual residual + objective trajectory on one trial.
#
# Usage:
#   bash configs/recipes/exp_convergence_trace.sh                 # write configs/exp_convergence_trace.json
#   bash configs/recipes/exp_convergence_trace.sh --dry-run       # preview the diff
#
# This recipe sources configs/recipes/_defaults.sh, then overrides only
# the parameters that matter for this experiment.  Adjust the values in
# the OVERRIDES block below; everything else inherits from defaults.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi

# ─── OVERRIDES — edit these ──────────────────────────────────────────
NAME="exp_convergence_trace"
# (ADMM_N_MAX inherits from _defaults.sh — 200 iterations gives ample
# headroom past the usual convergence window to see plateau / drift.)

# ─── Defaults for everything else ────────────────────────────────────
source "$SCRIPT_DIR/_defaults.sh"

# ─── Build the config ────────────────────────────────────────────────
python3 scripts/create_config.py \
    --name "$NAME" \
    $DRY_RUN_FLAG \
    --set "${SET_ARGS[@]}"
