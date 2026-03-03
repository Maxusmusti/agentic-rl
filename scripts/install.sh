#!/usr/bin/env bash
# Install script: creates two uv virtual environments for ART and AGL
# to avoid dependency conflicts (different PyTorch/CUDA requirements).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Agentic RL PoC Setup ==="
echo "Project directory: $PROJECT_DIR"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# --- ART virtual environment ---
echo ""
echo "--- Creating ART virtual environment (venv-art) ---"
uv venv "$PROJECT_DIR/venv-art"
source "$PROJECT_DIR/venv-art/bin/activate"

uv pip install -e "$PROJECT_DIR[tau,art,dev]"

echo "ART venv created. Testing imports..."
python3 -c "import src.agent.react_agent; print('  src.agent OK')"
python3 -c "import src.eval.reward; print('  src.eval OK')"
python3 -c "import src.art_training.config; print('  src.art_training OK')"
deactivate

# --- AGL virtual environment ---
echo ""
echo "--- Creating AGL virtual environment (venv-agl) ---"
uv venv "$PROJECT_DIR/venv-agl"
source "$PROJECT_DIR/venv-agl/bin/activate"

uv pip install -e "$PROJECT_DIR[tau,agl,dev]"

echo "AGL venv created. Testing imports..."
python3 -c "import src.agent.react_agent; print('  src.agent OK')"
python3 -c "import src.eval.reward; print('  src.eval OK')"
python3 -c "import src.agl_training.config; print('  src.agl_training OK')"
deactivate

# --- Verify GPU access ---
echo ""
echo "--- Checking GPU availability ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  ART training:  source venv-art/bin/activate && python -m src.art_training.train"
echo "  AGL training:  source venv-agl/bin/activate && python -m src.agl_training.train"
echo "  Baseline eval: source venv-art/bin/activate && python -m src.eval.baseline"
