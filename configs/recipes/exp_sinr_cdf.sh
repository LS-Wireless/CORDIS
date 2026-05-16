#!/usr/bin/env bash
# =============================================================================
# configs/recipes/exp_sinr_cdf.sh
# =============================================================================
#
# Empirical CDF of per-user min-SINR across many trials.
#
# Usage:
#   bash configs/recipes/exp_sinr_cdf.sh                 # write configs/exp_sinr_cdf.json
#   bash configs/recipes/exp_sinr_cdf.sh --dry-run       # preview the diff
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
NAME="exp_sinr_cdf"
N_TRIALS=200    # Baseline; final-paper figure may want 500+

# ─── Defaults for everything else ────────────────────────────────────
source "$SCRIPT_DIR/_defaults.sh"

# ─── Build the config ────────────────────────────────────────────────
python3 scripts/create_config.py \
    --name "$NAME" \
    $DRY_RUN_FLAG \
    --set "${SET_ARGS[@]}"
