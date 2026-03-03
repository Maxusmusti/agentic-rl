#!/usr/bin/env bash
# Generate comparison report from ART and AGL results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Use either venv (comparison code has no framework-specific deps)
if [ -d "$PROJECT_DIR/venv-art" ]; then
    source "$PROJECT_DIR/venv-art/bin/activate"
elif [ -d "$PROJECT_DIR/venv-agl" ]; then
    source "$PROJECT_DIR/venv-agl/bin/activate"
else
    echo "ERROR: No virtual environment found. Run install.sh first."
    exit 1
fi

cd "$PROJECT_DIR"

echo "=== Generating Comparison ==="

# Run comparison analysis
python -m src.comparison.compare \
    --results-dir results \
    --output-dir results/comparison

# Generate report with plots
python -m src.comparison.report \
    --results-dir results \
    --output-dir results/comparison

echo "=== Comparison complete ==="
echo "Results:"
echo "  JSON:   results/comparison/comparison.json"
echo "  Report: results/comparison/report.md"
echo "  Plots:  results/comparison/*.png"
