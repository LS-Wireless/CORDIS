#!/usr/bin/env bash
# =============================================================================
# scripts/visualize_topology.sh
# =============================================================================
# Wrapper for visualize_topology.py.
# Edit the OPTIONS section below, then run:
#
#   chmod +x scripts/visualize_topology.sh   # first time only
#   ./scripts/visualize_topology.sh
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi


# =============================================================================
# OPTIONS — edit these
# =============================================================================

# ── Scenario ──────────────────────────────────────────────────────────────────

# Config file to load (leave blank to use configs/default.json)
CONFIG=""                     # e.g. "configs/exp_scalability.json"

# Random seed for topology and channel generation
SEED=42

# Network layout (overrides config if set):
#   circle  — APs equally spaced on a circle (default)
#   random  — APs dropped uniformly in the cell
#   grid    — APs on a regular grid (auto-factorised, e.g. 12 → 3×4)
TOPOLOGY="circle"

# Node counts (overrides config if set; leave at 0 to use the config value)
N_AP=0                        # Number of APs         (0 = use config)
N_UE=0                        # Number of UEs         (0 = use config)
N_TARGETS=0                   # Number of targets     (0 = use config)

# ── Output ────────────────────────────────────────────────────────────────────

# Save figures to disk (true/false)
SAVE=true

# Output directory for saved figures
OUT_DIR="eval/figures/topology"

# File format: png | pdf | svg
FMT="png"

# Resolution [dpi] — only applies to raster formats (png)
DPI=150

# Show figures interactively after generating (true/false)
# Set to false for headless/server environments
SHOW=true

# Generate only the 2×2 summary dashboard instead of all 6 plots (true/false)
SUMMARY_ONLY=false


# =============================================================================
# (nothing below this line normally needs editing)
# =============================================================================

# Build optional flags
ARGS=()

[ -n "$CONFIG" ]            && ARGS+=(--config "$CONFIG")
[ "$TOPOLOGY" != "" ]       && ARGS+=(--topology "$TOPOLOGY")
[ "$N_AP"      -gt 0 ]      && ARGS+=(--n-ap      "$N_AP")
[ "$N_UE"      -gt 0 ]      && ARGS+=(--n-ue      "$N_UE")
[ "$N_TARGETS" -gt 0 ]      && ARGS+=(--n-targets  "$N_TARGETS")
[ "$SAVE"         = "true" ] && ARGS+=(--save)
[ "$SHOW"         = "false" ] && ARGS+=(--no-show)
[ "$SUMMARY_ONLY" = "true" ] && ARGS+=(--summary-only)

ARGS+=(--seed   "$SEED")
ARGS+=(--out-dir "$OUT_DIR")
ARGS+=(--fmt    "$FMT")
ARGS+=(--dpi    "$DPI")

mkdir -p "$OUT_DIR"

echo ""
echo "Visualising topology with the following settings:"
echo "  Config        = ${CONFIG:-configs/default.json (default)}"
echo "  Topology      = $TOPOLOGY"
echo "  Seed          = $SEED"
echo "  N_AP          = ${N_AP:-from config}"
echo "  N_UE          = ${N_UE:-from config}"
echo "  N_TARGETS     = ${N_TARGETS:-from config}"
echo "  Save          = $SAVE"
echo "  Output dir    = $OUT_DIR"
echo "  Format        = $FMT  (${DPI} dpi)"
echo "  Show          = $SHOW"
echo "  Summary only  = $SUMMARY_ONLY"
echo ""

python3 scripts/visualize_topology.py "${ARGS[@]}"

