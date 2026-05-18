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

# ─── Topology ────────────────────────────────────────────────────────
TOPOLOGY_TYPE="${TOPOLOGY_TYPE:-circle}"
N_AP=8
N_UE=10
N_TARGETS=4
N_ANT=16
N_RF_CHAINS=16
#ARRAY_TYPE="${ARRAY_TYPE:-UCA}"
#N_SENSING_RX="${N_SENSING_RX:-1}"
#AP_RADIUS_M="${AP_RADIUS_M:-650.0}"
#UE_MIN_RADIUS_M="${UE_MIN_RADIUS_M:-35.0}"
#UE_MAX_RADIUS_M="${UE_MAX_RADIUS_M:-1000.0}"
#TG_MIN_RADIUS_M="${TG_MIN_RADIUS_M:-35.0}"
#TG_MAX_RADIUS_M="${TG_MAX_RADIUS_M:-1000.0}"

# ─── Channel ─────────────────────────────────────────────────────────
SNR_DB=140
PILOT_POWER_DB=120
TAU_P=15

# ─── Algorithm — CORDIS-Split ────────────────────────────────────────
#SPLIT_GAMMA_DB="${SPLIT_GAMMA_DB:-10.0}"
#SPLIT_EPSILON_REG="${SPLIT_EPSILON_REG:-0.01}"
#SPLIT_EPSILON_NSC="${SPLIT_EPSILON_NSC:-0.001}"
#SPLIT_KAPPA="${SPLIT_KAPPA:-1.0}"
#SPLIT_XI_PENALTY="${SPLIT_XI_PENALTY:-10000.0}"

# ─── Algorithm — CORDIS-ADMM ─────────────────────────────────────────
ADMM_KAPPA=0.5 # "${ADMM_KAPPA:-1.0}"
#ADMM_RHO="${ADMM_RHO:-1.0}"
#ADMM_N_MAX="${ADMM_N_MAX:-50}"
#ADMM_EPS_PRI="${ADMM_EPS_PRI:-0.001}"
#ADMM_EPS_DUAL="${ADMM_EPS_DUAL:-0.001}"
#ADMM_XI_SLACK="${ADMM_XI_SLACK:-10000.0}"

# ─── Defaults for everything else ────────────────────────────────────
source "$SCRIPT_DIR/_defaults.sh"

# ─── Build the config ────────────────────────────────────────────────
python3 scripts/create_config.py \
    --name "$NAME" \
    $DRY_RUN_FLAG \
    --set "${SET_ARGS[@]}"
