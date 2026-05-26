#!/usr/bin/env bash
# =============================================================================
# configs/recipes/_defaults.sh
# =============================================================================
#
# Shared default values for the CORDIS config parameters, sourced by
# every configs/recipes/exp_<name>.sh recipe.
#
# Per-experiment recipes set ONLY the variables they override, then
# source this file, then call create_config.py.  Variables that are
# already set in the caller's scope are preserved; everything else
# falls back to its default.json value via the ${VAR:-default} idiom.
#
# How to use in an experiment recipe::
#
#     #!/usr/bin/env bash
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#
#     # Experiment-specific overrides FIRST (so they "win" against
#     # _defaults.sh's ${VAR:-default} expansion).
#     NAME="exp_kappa_sweep"
#     N_TRIALS=200
#     ADMM_KAPPA=1.0          # baseline κ; the sweep itself happens at runtime
#
#     # Pull in defaults for everything else.
#     source "$SCRIPT_DIR/_defaults.sh"
#
#     # Standard create_config.py invocation.
#     python3 scripts/create_config.py --name "$NAME" $DRY_RUN_FLAG \
#         --set "${SET_ARGS[@]}"
#
# Whenever a new parameter is added to CORDISConfig, update it HERE and
# every recipe picks it up automatically.  Whenever a parameter's
# default in default.json changes, update it here too.
# =============================================================================

# ─── Required: experiment name (caller MUST set this) ────────────────
NAME="${NAME:-exp_unnamed}"

# ─── Topology ────────────────────────────────────────────────────────
TOPOLOGY_TYPE="${TOPOLOGY_TYPE:-circle}"
N_AP="${N_AP:-6}"
N_UE="${N_UE:-4}"
N_TARGETS="${N_TARGETS:-2}"
N_ANT="${N_ANT:-10}"
N_RF_CHAINS="${N_RF_CHAINS:-10}"
ARRAY_TYPE="${ARRAY_TYPE:-UCA}"
ANTENNA_SPACING_FACTOR="${ANTENNA_SPACING_FACTOR:-0.5}"
N_SENSING_RX="${N_SENSING_RX:-1}"
AP_RADIUS_M="${AP_RADIUS_M:-650.0}"
UE_MIN_RADIUS_M="${UE_MIN_RADIUS_M:-35.0}"
UE_MAX_RADIUS_M="${UE_MAX_RADIUS_M:-1000.0}"
TG_MIN_RADIUS_M="${TG_MIN_RADIUS_M:-35.0}"
TG_MAX_RADIUS_M="${TG_MAX_RADIUS_M:-1000.0}"
H_AP_M="${H_AP_M:-10.0}"
H_UE_M="${H_UE_M:-1.5}"
H_TG_M="${H_TG_M:-1.5}"

# ─── Frequency ───────────────────────────────────────────────────────
CARRIER_FREQ_GHZ="${CARRIER_FREQ_GHZ:-3.5}"
BANDWIDTH_MHZ="${BANDWIDTH_MHZ:-20.0}"

# ─── Channel ─────────────────────────────────────────────────────────
SNR_DB="${SNR_DB:-137.0}"        # ≈ 20 W per AP at B=20 MHz, NF=7 dB
PILOT_POWER_DB="${PILOT_POWER_DB:-120.0}"
TAU_P="${TAU_P:-10}"
ESTIMATION_METHOD="${ESTIMATION_METHOD:-MMSE}"
RICIAN_K_DB_MEAN="${RICIAN_K_DB_MEAN:-9.0}"
RICIAN_K_DB_STD="${RICIAN_K_DB_STD:-5.0}"
NOISE_FIGURE_DB="${NOISE_FIGURE_DB:-7.0}"
NOISE_TEMP_K="${NOISE_TEMP_K:-290.0}"
SHADOW_CORR_DIST_M="${SHADOW_CORR_DIST_M:-50.0}"

# ─── Sensing ─────────────────────────────────────────────────────────
RX_STRATEGY="${RX_STRATEGY:-single_closest_centroid}"
SIGMA_RCS_SQ_DB="${SIGMA_RCS_SQ_DB:--3.0}"
N_SNAPSHOTS="${N_SNAPSHOTS:-20}"
# Clutter — set CLUTTER_CNR_DB (default).  For unit tests or
# noise-independent studies, set SIGMA_CLT to a numeric value (e.g.
# "0.1") to override.  Leave SIGMA_CLT as "null" to use the CNR formula.
CLUTTER_CNR_DB="${CLUTTER_CNR_DB:--10.0}"
SIGMA_CLT="${SIGMA_CLT:-null}"
CLUTTER_AS_DEG="${CLUTTER_AS_DEG:-15.0}"
CLUTTER_CENTER_STRATEGY="${CLUTTER_CENTER_STRATEGY:-target_centroid}"
CLUTTER_OFFSET_AZ_DEG="${CLUTTER_OFFSET_AZ_DEG:-60.0}"
LOS_MODEL="${LOS_MODEL:-always}"  # paper assumes s_t=1 (rank-1 LoS target model)
LOS_PROBABILITY_OVERRIDE="${LOS_PROBABILITY_OVERRIDE:-null}"
MAX_TX_APS_PER_TARGET="${MAX_TX_APS_PER_TARGET:-5}"
MAX_RX_APS_PER_TARGET="${MAX_RX_APS_PER_TARGET:-3}"

# ─── Algorithm — shared QoS target (γ_u for every algorithm) ─────────
GAMMA_DB="${GAMMA_DB:-5.0}"

# ─── Algorithm — CORDIS-Split ────────────────────────────────────────
SPLIT_EPSILON_REG="${SPLIT_EPSILON_REG:-0.01}"
SPLIT_EPSILON_NSC="${SPLIT_EPSILON_NSC:-0.001}"
SPLIT_KAPPA="${SPLIT_KAPPA:-1.0}"
SPLIT_XI_PENALTY="${SPLIT_XI_PENALTY:-10000.0}"

# ─── Algorithm — CORDIS-ADMM ─────────────────────────────────────────
ADMM_KAPPA="${ADMM_KAPPA:-1.0}"
ADMM_RHO="${ADMM_RHO:-1.0}"
ADMM_N_MAX="${ADMM_N_MAX:-200}"
ADMM_EPS_PRI="${ADMM_EPS_PRI:-1.0}"   # auto-rho settles residuals near 1
ADMM_EPS_DUAL="${ADMM_EPS_DUAL:-1.0}"  # auto-rho settles residuals near 1
ADMM_XI_SLACK="${ADMM_XI_SLACK:-10000.0}"
# Stage 22a: best-iterate selection criterion when ADMM hits n_max
# without converging.  Allowed values: "residual_norm" (default, picks
# smallest r_pri+r_dual) | "min_sinr" (legacy, picks highest worst-user
# SINR).  See ADMMConfig.best_iter_criterion docstring.
ADMM_BEST_ITER_CRITERION="${ADMM_BEST_ITER_CRITERION:-residual_norm}"

# ─── Simulation ──────────────────────────────────────────────────────
# Trial count can be set in two ways:
#   • N_TRIALS  → auto-decomposed into closest factor pair (drops × real)
#   • N_DROPS and N_REAL → explicit (export them at make-time, NOT here)
# Both reach the launcher via env vars and the Python resolver picks one
# (explicit drops×real beats n_trials).  See scripts/_exp_common.py
# resolve_drops_real for the full precedence chain.
N_TRIALS="${N_TRIALS:-100}"
SEED="${SEED:-42}"
N_JOBS="${N_JOBS:--1}"
SAVE_DIR="${SAVE_DIR:-results/}"
TAG="${TAG:-}"

# ─── Build the --set arg array for create_config.py ──────────────────
# Sourcing a recipe gets you ``SET_ARGS`` ready to splat into the
# python invocation.  No recipe needs to repeat this list.
SET_ARGS=(
    topology.type="$TOPOLOGY_TYPE"
    topology.n_ap="$N_AP"
    topology.n_ue="$N_UE"
    topology.n_targets="$N_TARGETS"
    topology.n_ant="$N_ANT"
    topology.n_rf_chains="$N_RF_CHAINS"
    topology.array_type="$ARRAY_TYPE"
    topology.antenna_spacing_factor="$ANTENNA_SPACING_FACTOR"
    topology.n_sensing_rx="$N_SENSING_RX"
    topology.ap_radius_m="$AP_RADIUS_M"
    topology.ue_min_radius_m="$UE_MIN_RADIUS_M"
    topology.ue_max_radius_m="$UE_MAX_RADIUS_M"
    topology.tg_min_radius_m="$TG_MIN_RADIUS_M"
    topology.tg_max_radius_m="$TG_MAX_RADIUS_M"
    topology.h_ap_m="$H_AP_M"
    topology.h_ue_m="$H_UE_M"
    topology.h_tg_m="$H_TG_M"
    frequency.carrier_freq_ghz="$CARRIER_FREQ_GHZ"
    frequency.bandwidth_mhz="$BANDWIDTH_MHZ"
    channel.snr_db="$SNR_DB"
    channel.pilot_power_db="$PILOT_POWER_DB"
    channel.tau_p="$TAU_P"
    channel.estimation_method="$ESTIMATION_METHOD"
    channel.rician_k_db_mean="$RICIAN_K_DB_MEAN"
    channel.rician_k_db_std="$RICIAN_K_DB_STD"
    channel.noise_figure_db="$NOISE_FIGURE_DB"
    channel.noise_temp_k="$NOISE_TEMP_K"
    channel.shadow_corr_distance_m="$SHADOW_CORR_DIST_M"
    sensing.rx_strategy="$RX_STRATEGY"
    sensing.sigma_rcs_sq_db="$SIGMA_RCS_SQ_DB"
    sensing.n_snapshots="$N_SNAPSHOTS"
    sensing.clutter_cnr_db="$CLUTTER_CNR_DB"
    sensing.sigma_clt="$SIGMA_CLT"
    sensing.clutter_as_deg="$CLUTTER_AS_DEG"
    sensing.clutter_center_strategy="$CLUTTER_CENTER_STRATEGY"
    sensing.clutter_offset_az_deg="$CLUTTER_OFFSET_AZ_DEG"
    sensing.los_model="$LOS_MODEL"
    sensing.los_probability_override="$LOS_PROBABILITY_OVERRIDE"
    sensing.max_tx_aps_per_target="$MAX_TX_APS_PER_TARGET"
    sensing.max_rx_aps_per_target="$MAX_RX_APS_PER_TARGET"
    algorithm.gamma_db="$GAMMA_DB"
    algorithm.split.epsilon_reg="$SPLIT_EPSILON_REG"
    algorithm.split.epsilon_nsc="$SPLIT_EPSILON_NSC"
    algorithm.split.kappa="$SPLIT_KAPPA"
    algorithm.split.xi_penalty="$SPLIT_XI_PENALTY"
    algorithm.admm.kappa="$ADMM_KAPPA"
    algorithm.admm.rho="$ADMM_RHO"
    algorithm.admm.n_max="$ADMM_N_MAX"
    algorithm.admm.eps_pri="$ADMM_EPS_PRI"
    algorithm.admm.eps_dual="$ADMM_EPS_DUAL"
    algorithm.admm.xi_slack="$ADMM_XI_SLACK"
    algorithm.admm.best_iter_criterion="$ADMM_BEST_ITER_CRITERION"
    simulation.n_trials="$N_TRIALS"
    simulation.seed="$SEED"
    simulation.n_jobs="$N_JOBS"
    simulation.save_dir="$SAVE_DIR"
    simulation.tag="$TAG"
)

# ─── Honour --dry-run from the caller ────────────────────────────────
DRY_RUN_FLAG="${DRY_RUN_FLAG:-}"
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN_FLAG="--dry-run"; fi

