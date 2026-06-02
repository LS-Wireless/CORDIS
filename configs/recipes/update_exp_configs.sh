#!/usr/bin/env bash
# =============================================================================
# configs/recipes/update_exp_configs.sh
# =============================================================================
#
# Updates all 'configs/exp_*.json' files by re-sourcing the corresponding
# 'configs/recipes/exp_*.sh' scripts.  This is useful for propagating changes
# in the default parameters (configs/recipes/_defaults.sh) to all experiment
# configs, while still allowing each experiment to override only the parameters
# that matter for that experiment.
#
# Run this after making changes to _defaults.sh, or after making any changes
# to the recipe scripts that should be reflected in the JSON configs.
#
# Usage:
#   bash configs/recipes/update_exp_configs.sh
#   ./configs/recipes/update_exp_configs.sh     # (from repo root)
#
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then source venv/bin/activate
fi

echo "=========================================================="
echo "Updating all CORDIS experiment configurations..."
echo "=========================================================="

# Loop through all scripts matching the pattern
for script in configs/recipes/exp_*.sh; do
    # Check if the file actually exists to prevent errors if the directory is empty
    if [[ -f "$script" ]]; then
        echo "Generating: $script"
        bash "$script"
        echo "----------------------------------------------------------"
    else
        echo "No experiment scripts found in configs/recipes/"
        exit 1
    fi
done

echo "=========================================================="
echo "All experiment configurations updated successfully!"
echo "=========================================================="
