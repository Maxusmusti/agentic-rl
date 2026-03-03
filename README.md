# Agentic RLVR PoC: ART vs Agent Lightning

Proof-of-concept showing RLVR (Reinforcement Learning from Verifiable Rewards) can improve a local model's agent capabilities, comparing two RL frameworks side-by-side.

## Setup

- **Base model**: Qwen3-4B
- **Agent**: ReAct tool-calling agent
- **Evaluation**: tau-bench 2 (verified), retail domain
- **RL frameworks**: ART (OpenPipe) + Agent Lightning (Microsoft)

## Results

10 test tasks, 4 trials each (train/test splits are disjoint):

| Model | pass@1 | pass@4 | Training |
|-------|--------|--------|----------|
| Baseline (Qwen3-4B) | 0.175 | 0.279 | — |
| ART mini (3 iterations, 60 rollouts) | 0.350 | 0.392 | 15 min |
| ART full (7 iterations, 1120 rollouts) | 0.325 | 0.529 | 4.4h |
| Agent Lightning | N/A | N/A | See notes below |

- **ART doubled pass@1** from 17.5% to 35.0% with just 3 GRPO iterations
- **ART full run achieved 0.529 pass@4** — a 90% relative improvement over baseline
- **Agent Lightning** executed rollouts successfully but VERL's training pipeline couldn't form training batches from multi-turn tool-calling traces

## Key Findings

1. **ART** was significantly easier to integrate — its `model.openai_client()` API lets any existing agent code work as-is. Training ran end-to-end with GRPO.
2. **Agent Lightning** has a heavier setup (Ray, FSDP, VERL) and requires LLM calls routed through its instrumented proxy with per-turn token ID capture — a fundamental architectural requirement not met by multi-turn tool-calling conversations.
3. **Dependency compatibility** was the biggest practical challenge: vLLM version mismatches, flash-attn binary incompatibility, multiprocessing spawn issues.

## Project Structure

```
agentic-rl/
├── pyproject.toml                    # Dependency groups: [tau], [art], [agl], [dev]
├── configs/                          # YAML configs (base, art, agl, hardware)
├── src/
│   ├── agent/                        # Shared ReAct agent + tau-bench adapter
│   ├── eval/                         # Baseline/post-training eval + metrics
│   ├── art_training/                 # ART GRPO training pipeline
│   ├── agl_training/                 # Agent Lightning VERL pipeline
│   └── comparison/                   # Side-by-side analysis + report generation
├── scripts/                          # Install + run shell scripts
├── run_*.py                          # Executable training/eval scripts
├── results/                          # Evaluation results + plots
└── tests/                            # Unit tests (31 passing)
```

## Quick Start

```bash
# Install (creates separate venvs for ART and AGL)
bash scripts/install.sh

# Run baseline evaluation
source venv-art/bin/activate
python run_baseline_eval.py

# Run ART training
python run_art_training.py    # mini (3 iterations)
python run_art_full.py        # full (15 iterations)

# Run post-training evaluation
python run_post_eval.py --model-name art-trained --output-dir results/art/post_eval

# Generate comparison report
python -m src.comparison.compare --results-dir results --output-dir results/comparison
python -m src.comparison.report --results-dir results --output-dir results/comparison
```

## Requirements

- Python >= 3.10
- CUDA 12.x + 1-8 GPUs (tested on 8x H100 80GB)
- OpenAI API key (for tau-bench user simulator)
- tau-bench data: `git clone https://github.com/amazon-agi/tau2-bench-verified` and set `TAU2_DATA_DIR`
