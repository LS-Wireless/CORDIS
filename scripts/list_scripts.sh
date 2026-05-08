#!/usr/bin/env bash
# =============================================================================
# scripts/list_scripts.sh
# =============================================================================
# Generates the scripts reference documentation and saves it to
# docs/scripts_reference.md.
#
# Run this whenever you add or modify a script to keep the docs current.
# The output file IS committed to git — it is project documentation, not
# a throwaway artifact.
#
#   chmod +x scripts/list_scripts.sh   # first time only
#   ./scripts/list_scripts.sh
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

OUTPUT="docs/scripts_reference.md"
mkdir -p docs

{
  echo "# CORDIS Scripts Reference"
  echo ""
  echo "Generated automatically by \`list_scripts.py\`."
  echo "Regenerate with: \`./scripts/list_scripts.sh\`"
  echo ""
  echo '```'
  python3 scripts/list_scripts.py
  echo '```'
} | tee "$OUTPUT"

echo ""
echo "Scripts reference saved to: $OUTPUT"
echo "Commit this file alongside any script additions or changes."

