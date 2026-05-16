#!/usr/bin/env bash
# =============================================================================
# configs/recipes/exp_n_ue_sweep.sh
# =============================================================================
#
# min-SINR & sum-SCNR vs number of UEs.
#
# Usage:
#   bash configs/recipes/exp_n_ue_sweep.sh                 # write configs/exp_n_ue_sweep.json
#   bash configs/recipes/exp_n_ue_sweep.sh --dry-run       # preview the diff
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
NAME="exp_n_ue_sweep"
N_TRIALS=100
N_UE=4    # Baseline; swept at runtime

# ─── Defaults for everything else ────────────────────────────────────
source "$SCRIPT_DIR/_defaults.sh"

# ─── Build the config ────────────────────────────────────────────────
python3 scripts/create_config.py \
    --name "$NAME" \
    $DRY_RUN_FLAG \
    --set "${SET_ARGS[@]}"
