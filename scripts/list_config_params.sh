#!/usr/bin/env bash
# =============================================================================
# scripts/list_config_params.sh
# =============================================================================
# Generates the configuration parameter reference and saves it to
# docs/config_reference.md.
#
# Run this whenever you add, remove, or modify a config parameter.
# The output file IS committed to git — it is project documentation.
#
#   chmod +x scripts/list_config_params.sh   # first time only
#   ./scripts/list_config_params.sh
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

OUTPUT="docs/config_reference.md"
mkdir -p docs

python3 scripts/list_config_params.py --format markdown > "$OUTPUT"
python3 scripts/list_config_params.py

echo ""
echo "Config reference saved to: $OUTPUT"
echo "Commit this file alongside any config parameter additions or changes."

