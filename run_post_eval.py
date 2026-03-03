"""Run post-training evaluation for ART (and optionally AGL).

Starts vLLM with the base model + LoRA adapter and runs
the same evaluation as the baseline.
"""
import asyncio
import json
import time
import os
import logging

logging.basicConfig(level=logging.WARNING)
for name in ["LiteLLM", "litellm", "httpx", "openai", "tau2"]:
    logging.getLogger(name).setLevel(logging.ERROR)

os.environ.setdefault("TAU2_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

from openai import AsyncOpenAI
from src.agent.react_agent import ReActConfig
from src.agent.tau_adapter import TauBenchRolloutEnv, get_task_ids
from src.eval.metrics import EvalMetrics, TaskMetrics
from src.eval.reward import binary_reward

# Same tasks and trials as baseline for fair comparison
NUM_TASKS = 10
TRIALS_PER_TASK = 4


async def run_eval(
    vllm_url: str,
    model_name: str,
    output_dir: str,
    label: str,
):
    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    config = ReActConfig(
        model=model_name, max_turns=15, temperature=0.7, max_tokens=1024
    )
    rollout_env = TauBenchRolloutEnv(
        client=client,
        domain="retail",
        model=model_name,
        user_model="gpt-4",
        config=config,
    )

    test_ids = get_task_ids("retail", "test")[:NUM_TASKS]
    print(f"[{label}] Eval: {len(test_ids)} tasks, {TRIALS_PER_TASK} trials each")

    all_metrics = []
    start = time.time()

    for i, task_id in enumerate(test_ids):
        tm = TaskMetrics(task_id=task_id)
        for trial in range(TRIALS_PER_TASK):
            try:
                episode = await rollout_env.run_episode(task_id=task_id)
                reward = binary_reward(episode.reward)
                tm.trials.append(reward)
                tm.num_turns.append(episode.react_result.num_turns)
            except Exception as e:
                print(f"  [{label}] Task {task_id} trial {trial}: FAILED ({e})")
                tm.trials.append(0.0)
        all_metrics.append(tm)
        elapsed = time.time() - start
        print(
            f"[{label}] [{i+1}/{len(test_ids)}] Task {task_id}: "
            f"p@1={tm.pass_at_1:.2f} mean={tm.mean_reward:.2f} "
            f"turns={tm.avg_turns:.1f} ({elapsed:.0f}s)"
        )

    total_time = time.time() - start
    metrics = EvalMetrics(
        task_metrics=all_metrics,
        wall_clock_seconds=total_time,
        rollouts_per_second=sum(len(t.trials) for t in all_metrics) / max(total_time, 0.1),
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print(f"\n[{label}] Complete!")
    print(f"  pass@1: {metrics.pass_at_1:.3f}")
    print(f"  pass@k: {metrics.pass_at_k:.3f}")
    print(f"  mean_reward: {metrics.mean_reward:.3f}")
    print(f"  avg_turns: {metrics.avg_turns:.1f}")
    print(f"  time: {total_time:.1f}s")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="results/art/post_eval")
    parser.add_argument("--label", default="art-post-training")
    args = parser.parse_args()

    asyncio.run(
        run_eval(args.vllm_url, args.model_name, args.output_dir, args.label)
    )
