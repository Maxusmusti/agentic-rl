"""Pre-training baseline evaluation of Qwen3-4B on tau-bench."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from src.agent.react_agent import ReActConfig
from src.agent.tau_adapter import TauBenchRolloutEnv, get_task_ids
from src.eval.metrics import EvalMetrics, TaskMetrics
from src.eval.reward import binary_reward, graded_reward, RewardType

logger = logging.getLogger(__name__)


async def run_baseline(
    config_path: str = "configs/base.yaml",
    output_dir: str = "results/baseline",
    vllm_base_url: str = "http://localhost:8000/v1",
) -> EvalMetrics:
    """Run baseline evaluation of the base model on tau-bench.

    Uses AgentGymEnv for step-by-step control so we get proper tau2
    evaluation (user simulator, environment, reward computation).

    Args:
        config_path: Path to base config YAML.
        output_dir: Directory to save results.
        vllm_base_url: URL of the vLLM server.

    Returns:
        EvalMetrics with pass@1, pass@4, etc.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    domain = config["tau_bench"]["domain"]
    trials_per_task = config["eval"]["trials_per_task"]
    reward_type = config["reward"]["type"]

    client = AsyncOpenAI(base_url=vllm_base_url, api_key="dummy")
    react_config = ReActConfig(
        model=model_name,
        max_turns=config["agent"]["max_turns"],
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["max_tokens"],
    )

    rollout_env = TauBenchRolloutEnv(
        client=client,
        domain=domain,
        model=model_name,
        user_model=config["tau_bench"]["user_model"],
        config=react_config,
    )

    # Get task IDs from the test split
    task_ids = get_task_ids(domain, split=config["tau_bench"]["splits"]["test"])
    if not task_ids:
        logger.error("No tasks found for domain=%s split=%s", domain, config["tau_bench"]["splits"]["test"])
        return EvalMetrics()

    reward_fn = binary_reward if reward_type == "binary" else graded_reward

    logger.info(
        "Running baseline: %d tasks, %d trials each, domain=%s",
        len(task_ids),
        trials_per_task,
        domain,
    )

    start_time = time.time()
    all_task_metrics: list[TaskMetrics] = []

    for task_id in task_ids:
        task_metric = TaskMetrics(task_id=task_id)

        for trial in range(trials_per_task):
            logger.info("Task %s, trial %d/%d", task_id, trial + 1, trials_per_task)
            try:
                episode = await rollout_env.run_episode(task_id=task_id)
                # Apply reward function to the raw gym reward
                reward = reward_fn(episode.reward)
                task_metric.trials.append(reward)
                task_metric.num_turns.append(episode.react_result.num_turns)
            except Exception as e:
                logger.error("Task %s trial %d failed: %s", task_id, trial, e)
                task_metric.trials.append(0.0)

        all_task_metrics.append(task_metric)
        logger.info(
            "Task %s: pass@1=%.2f, mean_reward=%.2f",
            task_id,
            task_metric.pass_at_1,
            task_metric.mean_reward,
        )

    wall_clock = time.time() - start_time
    peak_vram = _get_peak_vram()

    metrics = EvalMetrics(
        task_metrics=all_task_metrics,
        wall_clock_seconds=wall_clock,
        peak_vram_mb=peak_vram,
        rollouts_per_second=sum(len(t.trials) for t in all_task_metrics) / max(wall_clock, 0.1),
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    logger.info(
        "Baseline complete: pass@1=%.3f, pass@4=%.3f, %.1fs",
        metrics.pass_at_1,
        metrics.pass_at_k,
        wall_clock,
    )

    return metrics


def _get_peak_vram() -> float:
    """Get peak VRAM usage in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--output-dir", default="results/baseline", help="Output directory")
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1", help="vLLM server URL"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_baseline(args.config, args.output_dir, args.vllm_url))


if __name__ == "__main__":
    main()
