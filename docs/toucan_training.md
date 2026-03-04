# Toucan Single-Turn Tool-Calling Training

## Context

After validating the ART+tau-bench pipeline (multi-turn, GPT-4 user simulator), we tested a simpler, faster setup: single-turn tool-calling with the [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) dataset. Each rollout is one LLM call — no user simulator, no multi-turn conversation. Reward is deterministic string matching on tool call correctness.

## Setup

- **Data**: 5,000 train / 500 val samples from Toucan-1.5M (Qwen3 config)
- **Task**: Given a user question and available tools, produce the correct tool call
- **Reward**: 1.0 (correct name + args), 0.5 (correct name, wrong args), 0.0 (wrong/no tool call)
- **Training**: ART GRPO, 15 iterations, group_size=8, 100 tasks/iter, 12,000 total rollouts
- **Time**: 36 minutes total (vs ~10 hours for tau-bench equivalent)

## Results

| Metric | Baseline | Post-Training | Change |
|--------|----------|---------------|--------|
| Full match (name+args) | 31.4% | 38.2% | +6.8pp (+22% relative) |
| Name match | 48.2% | 68.6% | +20.4pp (+42% relative) |
| No tool call produced | 51.8% | 31.4% | -20.4pp (halved) |
| Mean reward | 0.398 | 0.534 | +0.136 |

## Training Curve

```
Iter  Reward  Full Match
  1   0.460    37.5%
  2   0.499    39.0%
  3   0.564    48.4%
  4   0.535    43.0%
  5   0.468    38.1%
  6   0.516    42.5%
  7   0.610    53.6%   ← peak
  8   0.558    44.0%
  9   0.588    49.9%
 10   0.628    52.5%
 11   0.511    39.0%
 12   0.544    41.8%
 13   0.501    38.1%
 14   0.599    44.9%
 15   0.630    46.5%
```

## What Changed

The biggest improvement was in **tool call production rate**: the base model failed to produce any tool call 52% of the time, and GRPO training cut that in half to 31%. The model learned to use tools instead of trying to answer directly.

Name match accuracy jumped +20pp, indicating the model learned to select the right tool for a given question. Full match improvement was more modest (+7pp) since argument accuracy requires understanding the exact parameter format, which is harder to learn from binary reward signals alone.

## Comparison: Toucan vs tau-bench

| | tau-bench | Toucan |
|--|-----------|--------|
| Turns per rollout | 5-15 | 1 |
| External API calls | GPT-4 user sim (~$0.05/rollout) | None |
| Reward computation | DB replay + LLM judge | String match |
| Time per rollout | ~15s | ~1-2s |
| Rollouts per iteration | 160 | 800 |
| Wall time for 15 iters | ~10 hours | 36 minutes |
| Eval improvement (main metric) | +90% pass@4 | +42% name match |

## Data Format

Each Toucan sample contains:
- `question`: user's natural language query
- `available_tools`: list of tool definitions in OpenAI format
- `target_tool_name`: expected tool name (ground truth)
- `target_arguments`: expected arguments dict (ground truth)
- `system_prompt`: system prompt with tool definitions embedded

The gold tool call comes from the dataset's `messages[2].function_call` field.

## Files

- `src/toucan/dataset.py` — Data loading, filtering, caching
- `src/toucan/reward.py` — Tool call correctness reward (name match + argument match)
- `run_toucan_training.py` — ART GRPO training script
- `run_toucan_eval.py` — Evaluation script
- `results/toucan/` — Training results and eval metrics
