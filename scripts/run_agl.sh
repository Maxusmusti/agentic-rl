#!/usr/bin/env bash
# Run Agent Lightning VERL training on tau-bench.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HARDWARE="${HARDWARE:-configs/hardware/8xh100.yaml}"

source "$PROJECT_DIR/venv-agl/bin/activate"
cd "$PROJECT_DIR"

echo "=== Agent Lightning Training ==="
echo "Hardware config: $HARDWARE"

# Run AGL training (VERL manages Ray and vLLM internally)
python -m src.agl_training.train \
    --output-dir results/agl \
    --hardware "$HARDWARE"

# Run post-training evaluation
echo "Running post-training evaluation..."
python -m src.eval.evaluate \
    --config configs/base.yaml \
    --output-dir results/agl/post_eval \
    --label "agl-post-training"

echo "=== AGL training + eval complete. Results in results/agl/ ==="
