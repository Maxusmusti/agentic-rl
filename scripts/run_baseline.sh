#!/usr/bin/env bash
# Run baseline evaluation: start vLLM server and evaluate Qwen3-4B on tau-bench.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
CONFIG="${CONFIG:-configs/base.yaml}"

source "$PROJECT_DIR/venv-art/bin/activate"
cd "$PROJECT_DIR"

echo "=== Baseline Evaluation ==="
echo "Model: $MODEL"
echo "Config: $CONFIG"

# Start vLLM server in the background
echo "Starting vLLM server on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: vLLM server failed to start."
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

# Run baseline evaluation
echo "Running baseline evaluation..."
python -m src.eval.baseline \
    --config "$CONFIG" \
    --output-dir results/baseline \
    --vllm-url "http://localhost:$VLLM_PORT/v1"

# Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=== Baseline complete. Results in results/baseline/ ==="
