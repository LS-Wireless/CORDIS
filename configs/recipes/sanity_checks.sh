#!/usr/bin/env bash
# =============================================================================
# configs/recipes/create_config.sh   (template — copy, rename, then run)
# =============================================================================
#
# HOW TO USE
# ----------
# This is a template. Do NOT run it directly. Instead:
#
#   1. Copy it and give it your experiment name:
#        cp configs/recipes/create_config.sh configs/recipes/exp_myname.sh
#
#   2. Edit the copy: change NAME= and adjust any parameters you need.
#      All parameters are pre-set to their default.json values — just
#      edit the ones that matter for this experiment.
#
#   3. Dry-run first to preview the diff from default.json:
#        bash configs/recipes/exp_myname.sh --dry-run
#
#   4. Run for real:
#        bash configs/recipes/exp_myname.sh
#        → saves configs/<NAME>.json
#
#   5. Commit both the recipe and the generated config together:
#        git add configs/recipes/exp_myname.sh configs/exp_myname.json
#
# To list every available parameter path:
#        python3 scripts/list_config_params.py --paths-only
# To browse parameters by section:
#        python3 scripts/list_config_params.py --sections
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi

DRY_RUN_FLAG=""
if [ "${1}" = "--dry-run" ]; then DRY_RUN_FLAG="--dry-run"; fi


# =============================================================================
# EXPERIMENT NAME  (required — becomes configs/<NAME>.json)
# =============================================================================

NAME="sanity_checks"


# =============================================================================
# TOPOLOGY
# =============================================================================

TOPOLOGY_TYPE="circle"       # AP placement geometry: circle | random | grid
N_AP=8                       # Total number of APs
N_UE=8                       # Number of communication users
N_TARGETS=2                  # Number of sensing targets
N_ANT=10                     # Antennas per AP (M_t = M_r = M)
N_RF_CHAINS=10               # RF chains per AP (≤ N_ANT)
ARRAY_TYPE="UCA"             # Antenna array geometry: UCA | ULA
ANTENNA_SPACING_FACTOR=0.5   # Inter-element spacing d/λ
N_SENSING_RX=1               # Number of dedicated Rx APs

AP_RADIUS_M=650.0            # AP placement circle radius [m]
UE_MIN_RADIUS_M=35.0         # Min UE–origin distance [m]
UE_MAX_RADIUS_M=1000.0       # Max UE–origin distance [m]
TG_MIN_RADIUS_M=35.0         # Min target–origin distance [m]
TG_MAX_RADIUS_M=1000.0       # Max target–origin distance [m]

H_AP_M=10.0                  # AP height [m]
H_UE_M=1.5                   # UE height [m]
H_TG_M=1.5                   # Target height [m]


# =============================================================================
# FREQUENCY
# =============================================================================

CARRIER_FREQ_GHZ=3.5         # Carrier frequency [GHz]
BANDWIDTH_MHZ=20.0           # System bandwidth [MHz]


# =============================================================================
# CHANNEL
# =============================================================================

SNR_DB=20.0                  # Data SNR: P_max / sigma_n^2 [dB]
                             # NOTE: with 3GPP UMi path loss (β ≈ 1e-8 at 650 m)
                             # realistic values are 80–120 dB. Use 20 dB only for
                             # toy / normalised-channel models.

PILOT_POWER_DB=120.0         # Pilot SNR: P_p / sigma_n^2 [dB]
                             # Received SNR = PILOT_POWER_DB + 10*log10(τ_p * β)
                             # At 650 m (β ≈ 7e-11) need ≥ 110 dB for NMSE < 0.1

TAU_P=10                     # Pilot sequence length [OFDM symbols]
ESTIMATION_METHOD="MMSE"     # Channel estimation: MMSE | perfect

RICIAN_K_DB_MEAN=9.0         # Mean Rician K-factor [dB]
RICIAN_K_DB_STD=5.0          # Std dev of Rician K-factor [dB]
NOISE_FIGURE_DB=7.0          # Receiver noise figure [dB]
NOISE_TEMP_K=290.0           # Noise reference temperature [K]
SHADOW_CORR_DIST_M=50.0      # Shadow fading decorrelation distance [m]


# =============================================================================
# SENSING
# =============================================================================

RX_STRATEGY="single_closest_centroid"
                             # Rx AP selection strategy:
                             #   single_closest_centroid | single_farthest_centroid
                             #   per_target_closest | assent_file | assent_live

SIGMA_RCS_SQ_DB=-3.0         # Target RCS variance σ²_RCS [dB]
N_SNAPSHOTS=20               # STAP snapshot count (T)
CLUTTER_CNR_DB=-10.0         # Clutter-to-noise ratio [dB]
CLUTTER_AS_DEG=15.0          # Clutter angular spread [deg]
CLUTTER_CENTER_STRATEGY="target_centroid"
                             # Spatial placement of the clutter PAS at each Tx AP.
                             # Controls how κ interacts with beam steering:
                             #   target_centroid → PAS aligned with AP→target
                             #     direction; high κ pushes the beam away from
                             #     the target (sensing-vs-comm coupling).
                             #   offset          → PAS at (target azimuth +
                             #     CLUTTER_OFFSET_AZ_DEG); decouples clutter
                             #     avoidance from comm coverage.
                             #   uniform         → PAS spans full azimuth; κ
                             #     becomes a global power regulariser (no
                             #     directional steering effect).
                             #   random          → per-AP random U[-π, π];
                             #     useful for MC over clutter geometries.

CLUTTER_OFFSET_AZ_DEG=60.0   # Azimuth offset added to target azimuth when
                             # CLUTTER_CENTER_STRATEGY="offset". Typical
                             # 30–90°. Ignored by other strategies.

LOS_MODEL="3gpp_umi"         # LoS probability model:
                             #   3gpp_umi | always | never | custom
LOS_PROBABILITY_OVERRIDE="null"  # Fixed LoS prob in [0,1]; "null" = use model

MAX_TX_APS_PER_TARGET=5      # Max TX APs illuminating one target
MAX_RX_APS_PER_TARGET=3      # Max RX APs listening to one target


# =============================================================================
# ALGORITHM — CORDIS-Split
# =============================================================================

SPLIT_GAMMA_DB=10.0          # Min-SINR constraint γ [dB]
SPLIT_EPSILON_REG=0.01       # LR-MMSE regularisation ε
SPLIT_EPSILON_NSC=0.001      # NS-C regularisation ε
SPLIT_KAPPA=1.0              # Clutter penalty κ
SPLIT_XI_PENALTY=10000.0     # Slack variable penalty ξ


# =============================================================================
# ALGORITHM — CORDIS-ADMM
# =============================================================================

ADMM_KAPPA=1.0               # Clutter penalty κ
ADMM_RHO=1.0                 # ADMM step size ρ
ADMM_N_MAX=50                # Max ADMM iterations
ADMM_EPS_PRI=0.001           # Primal residual convergence tolerance
ADMM_EPS_DUAL=0.001          # Dual residual convergence tolerance
ADMM_XI_SLACK=10000.0        # Slack variable penalty ξ


# =============================================================================
# SIMULATION
# =============================================================================

N_TRIALS=500                 # Monte Carlo trials
SEED=42                      # Global random seed
N_JOBS=-1                    # Parallel workers (-1 = all CPUs, 1 = sequential)
SAVE_DIR="results/"          # Output directory for .npz simulation data
TAG=""                       # Free-text label embedded in output filenames


# =============================================================================
# (nothing below this line normally needs editing)
# =============================================================================

python3 scripts/create_config.py \
    --name "$NAME" \
    $DRY_RUN_FLAG \
    --set \
        topology.type="$TOPOLOGY_TYPE"                        \
        topology.n_ap="$N_AP"                                 \
        topology.n_ue="$N_UE"                                 \
        topology.n_targets="$N_TARGETS"                       \
        topology.n_ant="$N_ANT"                               \
        topology.n_rf_chains="$N_RF_CHAINS"                   \
        topology.array_type="$ARRAY_TYPE"                     \
        topology.antenna_spacing_factor="$ANTENNA_SPACING_FACTOR" \
        topology.n_sensing_rx="$N_SENSING_RX"                 \
        topology.ap_radius_m="$AP_RADIUS_M"                   \
        topology.ue_min_radius_m="$UE_MIN_RADIUS_M"           \
        topology.ue_max_radius_m="$UE_MAX_RADIUS_M"           \
        topology.tg_min_radius_m="$TG_MIN_RADIUS_M"           \
        topology.tg_max_radius_m="$TG_MAX_RADIUS_M"           \
        topology.h_ap_m="$H_AP_M"                             \
        topology.h_ue_m="$H_UE_M"                             \
        topology.h_tg_m="$H_TG_M"                             \
        frequency.carrier_freq_ghz="$CARRIER_FREQ_GHZ"        \
        frequency.bandwidth_mhz="$BANDWIDTH_MHZ"              \
        channel.snr_db="$SNR_DB"                              \
        channel.pilot_power_db="$PILOT_POWER_DB"              \
        channel.tau_p="$TAU_P"                                \
        channel.estimation_method="$ESTIMATION_METHOD"        \
        channel.rician_k_db_mean="$RICIAN_K_DB_MEAN"          \
        channel.rician_k_db_std="$RICIAN_K_DB_STD"            \
        channel.noise_figure_db="$NOISE_FIGURE_DB"            \
        channel.noise_temp_k="$NOISE_TEMP_K"                  \
        channel.shadow_corr_distance_m="$SHADOW_CORR_DIST_M"  \
        sensing.rx_strategy="$RX_STRATEGY"                    \
        sensing.sigma_rcs_sq_db="$SIGMA_RCS_SQ_DB"            \
        sensing.n_snapshots="$N_SNAPSHOTS"                    \
        sensing.clutter_cnr_db="$CLUTTER_CNR_DB"              \
        sensing.clutter_as_deg="$CLUTTER_AS_DEG"              \
        sensing.clutter_center_strategy="$CLUTTER_CENTER_STRATEGY" \
        sensing.clutter_offset_az_deg="$CLUTTER_OFFSET_AZ_DEG" \
        sensing.los_model="$LOS_MODEL"                        \
        sensing.los_probability_override="$LOS_PROBABILITY_OVERRIDE" \
        sensing.max_tx_aps_per_target="$MAX_TX_APS_PER_TARGET" \
        sensing.max_rx_aps_per_target="$MAX_RX_APS_PER_TARGET" \
        algorithm.split.gamma_db="$SPLIT_GAMMA_DB"            \
        algorithm.split.epsilon_reg="$SPLIT_EPSILON_REG"      \
        algorithm.split.epsilon_nsc="$SPLIT_EPSILON_NSC"      \
        algorithm.split.kappa="$SPLIT_KAPPA"                  \
        algorithm.split.xi_penalty="$SPLIT_XI_PENALTY"        \
        algorithm.admm.kappa="$ADMM_KAPPA"                    \
        algorithm.admm.rho="$ADMM_RHO"                        \
        algorithm.admm.n_max="$ADMM_N_MAX"                    \
        algorithm.admm.eps_pri="$ADMM_EPS_PRI"                \
        algorithm.admm.eps_dual="$ADMM_EPS_DUAL"              \
        algorithm.admm.xi_slack="$ADMM_XI_SLACK"              \
        simulation.n_trials="$N_TRIALS"                       \
        simulation.seed="$SEED"                               \
        simulation.n_jobs="$N_JOBS"                           \
        simulation.save_dir="$SAVE_DIR"                       \
        simulation.tag="$TAG"

