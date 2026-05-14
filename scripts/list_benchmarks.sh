#!/usr/bin/env bash
# =============================================================================
# scripts/list_benchmarks.sh
# =============================================================================
# Generates the benchmark registry reference and saves it to
# docs/benchmarks_reference.md.
#
# Run this whenever you add, remove, or modify an entry in
# cordis/algorithms/benchmarks.py:BENCHMARK_REGISTRY.
# The output file IS committed to git — it is project documentation,
# not a throwaway artifact.
#
#   chmod +x scripts/list_benchmarks.sh   # first time only
#   ./scripts/list_benchmarks.sh
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || { echo "ERROR: could not find repo root"; exit 1; }

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

OUTPUT="docs/benchmarks_reference.md"
mkdir -p docs

python3 scripts/list_benchmarks.py --format markdown > "$OUTPUT"
python3 scripts/list_benchmarks.py

echo ""
echo "Benchmark reference saved to: $OUTPUT"
echo "Commit this file alongside any BENCHMARK_REGISTRY additions or changes."

