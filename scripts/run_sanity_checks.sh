#!/usr/bin/env bash
# =============================================================================
# scripts/run_sanity_checks.sh
# =============================================================================
# Wrapper script for run_sanity_checks.py.
# Edit the OPTIONS section below, then run:
#
#   chmod +x scripts/run_sanity_checks.sh   # first time only
#   ./scripts/run_sanity_checks.sh
#
# =============================================================================

# ── Navigate to the repo root regardless of where the script is called from ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "ERROR: could not find repo root"; exit 1; }


# =============================================================================
# OPTIONS — edit these
# =============================================================================

# Number of Monte Carlo trials per sweep point.
# Quick check: 5–10   |   Validation: 50–100   |   Publication: 200–500
N_TRIALS=2

# Scenarios to run (space-separated subset of: 1 2 3)
#   1 = SINR vs SNR        (all-LoS, no targets)
#   2 = SINR vs pilot SNR  (3GPP LoS, no targets)
#   3 = SINR & SCNR vs pilot SNR  (3GPP LoS, with targets)
SCENARIOS="1 2 3"

# BF methods to compare (space-separated subset of: lr_mmse rzf mrt global_zf)
METHODS="lr_mmse rzf mrt global_zf"

# ── Scenario 1: data SNR sweep ────────────────────────────────────────────────
# Maps to P_max / sigma_n^2 on the x-axis of Figure 1.
# At 50 m cell (β ≈ 1e-8):
#   70 dB  → start of positive SINR for ZF/RZF
#   90 dB  → clear method separation
#  120 dB  → MRT is firmly plateaued; ZF still climbing
SNR_MIN=50          # [dB]
SNR_MAX=150         # [dB]
SNR_STEP=5          # [dB]  11 sweep points with defaults

# ── Scenarios 2 & 3: pilot SNR sweep ─────────────────────────────────────────
# Maps to P_p / sigma_n^2 on the x-axis of Figures 2 & 3.
# Received pilot SNR = (P_p/sigma_n^2) x tau_p x beta_au
# At 50 m (β_avg ≈ 1e-8, tau_p = 10):
#   60 dB  → SNR_rx ≈ 0.1  → NMSE ≈ 0.9  (poor estimation)
#   80 dB  → SNR_rx ≈ 10   → NMSE ≈ 0.09 (decent)
#  100 dB  → SNR_rx ≈ 1000 → NMSE ≈ 0.001 (near-perfect)
#  120 dB  → essentially perfect CSI
PILOT_SNR_MIN=50    # [dB]
PILOT_SNR_MAX=150   # [dB]
PILOT_SNR_STEP=5    # [dB]  13 sweep points with defaults

# Fixed pilot SNR used in scenarios 1 (if not set, pilot SNR = data SNR + 70 dB at each sweep point)
FIXED_PILOT_SNR=90  # [dB]

# Fixed data SNR used in scenarios 2 & 3 (90 dB shows clear method differences)
FIXED_SNR=90        # [dB]

# Power splitting ratio rho for scenario 3 (1 = all comm, 0 = all sensing)
PSR=0.7

# ── Parallel execution ────────────────────────────────────────────────────────
# Number of parallel joblib workers.
#   1       = sequential (safest, works everywhere)
#  -1       = use all available CPU cores
#   4, 8 …  = specific core count
# Requires: pip install joblib
N_JOBS=1

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR="eval/figures/sanity"
# Log files go in eval/logs/sanity/ (mirroring OUT_DIR under eval/logs/)
# Leave empty to use the default derived from OUT_DIR.
LOG_DIR=""

# ── Log file ──────────────────────────────────────────────────────────────────
# By default a timestamped log file is saved alongside the figures:
#   eval/figures/sanity/YYYYMMDD_HHMMSS_run.log
#
# Set SAVE_LOG=false to disable log saving (passes --no-save-log to the script).
SAVE_LOG=true


# =============================================================================
# (nothing below this line normally needs editing)
# =============================================================================

# Activate virtual environment if one exists in the repo root
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Build the optional --no-save-log flag
LOG_FLAG=""
if [ "$SAVE_LOG" = "false" ]; then
    LOG_FLAG="--no-save-log"
fi

echo ""
echo "Running run_sanity_checks.py with the following settings:"
echo "  N_TRIALS        = $N_TRIALS"
echo "  SCENARIOS       = $SCENARIOS"
echo "  METHODS         = $METHODS"
echo "  SNR sweep       = ${SNR_MIN}–${SNR_MAX} dB  step=${SNR_STEP} dB"
echo "  Pilot sweep     = ${PILOT_SNR_MIN}–${PILOT_SNR_MAX} dB  step=${PILOT_SNR_STEP} dB"
echo "  Fixed SNR       = ${FIXED_SNR} dB"
echo "  PSR             = ${PSR}"
echo "  N_JOBS          = ${N_JOBS}"
echo "  Output dir      = ${OUT_DIR}"
echo "  Log dir         = ${LOG_DIR}"
echo "  Save log        = ${SAVE_LOG}"
echo ""

python3 scripts/run_sanity_checks.py \
    --n-trials        "$N_TRIALS"        \
    --scenario        $SCENARIOS         \
    --methods         $METHODS           \
    --snr-min         "$SNR_MIN"         \
    --snr-max         "$SNR_MAX"         \
    --snr-step        "$SNR_STEP"        \
    --pilot-snr-min   "$PILOT_SNR_MIN"   \
    --pilot-snr-max   "$PILOT_SNR_MAX"   \
    --pilot-snr-step  "$PILOT_SNR_STEP"  \
    --fixed-snr       "$FIXED_SNR"       \
    --fixed-pilot-snr "$FIXED_PILOT_SNR" \
    --psr             "$PSR"             \
    --n-jobs          "$N_JOBS"          \
    --out-dir         "$OUT_DIR"         \
    --log-dir         "$LOG_DIR"         \
    $LOG_FLAG

