#!/usr/bin/env bash
# Run ART GRPO training on tau-bench.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HARDWARE="${HARDWARE:-configs/hardware/8xh100.yaml}"
ITERATIONS="${ITERATIONS:-50}"

source "$PROJECT_DIR/venv-art/bin/activate"
cd "$PROJECT_DIR"

echo "=== ART Training ==="
echo "Hardware config: $HARDWARE"
echo "Iterations: $ITERATIONS"

# Run ART training (ART manages its own vLLM server internally)
python -m src.art_training.train \
    --output-dir results/art \
    --hardware "$HARDWARE" \
    --iterations "$ITERATIONS"

# Run post-training evaluation
echo "Running post-training evaluation..."
# ART saves the LoRA adapter; start vLLM with it and evaluate
python -m src.eval.evaluate \
    --config configs/base.yaml \
    --output-dir results/art/post_eval \
    --label "art-post-training"

echo "=== ART training + eval complete. Results in results/art/ ==="
