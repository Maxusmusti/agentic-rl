"""Post-training evaluation — runs the same eval as baseline on a trained model."""

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
from src.eval.reward import binary_reward, graded_reward

logger = logging.getLogger(__name__)


async def evaluate_model(
    config_path: str = "configs/base.yaml",
    output_dir: str = "results/eval",
    vllm_base_url: str = "http://localhost:8000/v1",
    model_name: str | None = None,
    label: str = "post-training",
) -> EvalMetrics:
    """Evaluate a model (base or fine-tuned) on tau-bench test split.

    Args:
        config_path: Path to base config YAML.
        output_dir: Directory to save results.
        vllm_base_url: URL of the vLLM server serving the model.
        model_name: Override model name for the client.
        label: Label for this evaluation run.

    Returns:
        EvalMetrics with pass@1, pass@k, etc.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if model_name is None:
        model_name = config["model"]["name"]

    domain = config["tau_bench"]["domain"]
    trials_per_task = config["eval"]["trials_per_task"]
    reward_type = config["reward"]["type"]

    reward_fn = binary_reward if reward_type == "binary" else graded_reward

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

    # Get test split task IDs
    task_ids = get_task_ids(domain, split=config["tau_bench"]["splits"]["test"])
    if not task_ids:
        logger.error("No tasks found for evaluation")
        return EvalMetrics()

    logger.info(
        "Evaluating [%s]: %d tasks, %d trials each, domain=%s, model=%s",
        label,
        len(task_ids),
        trials_per_task,
        domain,
        model_name,
    )

    start_time = time.time()
    all_task_metrics: list[TaskMetrics] = []

    for task_id in task_ids:
        task_metric = TaskMetrics(task_id=task_id)

        for trial in range(trials_per_task):
            logger.info("[%s] Task %s, trial %d/%d", label, task_id, trial + 1, trials_per_task)
            try:
                episode = await rollout_env.run_episode(task_id=task_id)
                reward = reward_fn(episode.reward)
                task_metric.trials.append(reward)
                task_metric.num_turns.append(episode.react_result.num_turns)
            except Exception as e:
                logger.error("[%s] Task %s trial %d failed: %s", label, task_id, trial, e)
                task_metric.trials.append(0.0)

        all_task_metrics.append(task_metric)

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

    # Save full config for reproducibility
    with open(output_path / "eval_config.json", "w") as f:
        json.dump(
            {
                "label": label,
                "model_name": model_name,
                "config_path": config_path,
                "vllm_base_url": vllm_base_url,
            },
            f,
            indent=2,
        )

    logger.info(
        "[%s] Complete: pass@1=%.3f, pass@4=%.3f, %.1fs",
        label,
        metrics.pass_at_1,
        metrics.pass_at_k,
        wall_clock,
    )

    return metrics


def _get_peak_vram() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Post-training evaluation")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base config")
    parser.add_argument("--output-dir", default="results/eval", help="Output directory")
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1", help="vLLM server URL"
    )
    parser.add_argument("--model-name", default=None, help="Override model name")
    parser.add_argument("--label", default="post-training", help="Evaluation label")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        evaluate_model(
            config_path=args.config,
            output_dir=args.output_dir,
            vllm_base_url=args.vllm_url,
            model_name=args.model_name,
            label=args.label,
        )
    )


if __name__ == "__main__":
    main()
