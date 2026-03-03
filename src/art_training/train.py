"""ART training entry point — GRPO training with tau-bench rollouts."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.art_training.config import ARTTrainingConfig, create_art_model
from src.art_training.rollout import art_rollout

logger = logging.getLogger(__name__)


async def train_art(
    config: ARTTrainingConfig | None = None,
    output_dir: str = "results/art",
    hardware_config: str | None = None,
) -> dict[str, Any]:
    """Run ART GRPO training on tau-bench.

    Args:
        config: Training configuration. If None, loads from YAML.
        output_dir: Directory to save results.
        hardware_config: Path to hardware config YAML.

    Returns:
        Dict with training results and metrics.
    """
    import art

    if config is None:
        config = ARTTrainingConfig.from_yaml(hardware_path=hardware_config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create and register the model
    model = await create_art_model(config)

    # Check for existing checkpoint to resume from
    current_step = await model.get_step()
    start_iteration = current_step if current_step > 0 else 0
    if start_iteration > 0:
        logger.info("Resuming from step %d", start_iteration)

    # Get training task IDs
    task_ids = _get_train_task_ids(config.domain, config.train_split)
    logger.info(
        "Starting ART training: %d tasks, %d iterations, group_size=%d",
        len(task_ids),
        config.num_iterations,
        config.group_size,
    )

    # Training metrics
    reward_history: list[float] = []
    timing_history: list[float] = []
    start_time = time.time()

    for iteration in range(start_iteration, config.num_iterations):
        iter_start = time.time()

        # Gather trajectory groups: one group per scenario (task)
        # Each group has `group_size` rollouts for GRPO
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    art_rollout(model, task_id, config)
                    for _ in range(config.group_size)
                )
                for task_id in task_ids
            ),
            pbar_desc=f"Iteration {iteration + 1}/{config.num_iterations}",
        )

        # Compute mean reward for this iteration
        rewards = []
        for group in train_groups:
            for traj in group.trajectories:
                rewards.append(traj.reward)
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        reward_history.append(mean_reward)

        # Train one GRPO step
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=config.learning_rate),
        )

        iter_time = time.time() - iter_start
        timing_history.append(iter_time)

        logger.info(
            "Iteration %d/%d: mean_reward=%.3f, time=%.1fs, rollouts=%d",
            iteration + 1,
            config.num_iterations,
            mean_reward,
            iter_time,
            len(rewards),
        )

        # Save periodic checkpoint info
        if (iteration + 1) % 10 == 0:
            _save_checkpoint_info(output_path, iteration + 1, reward_history, timing_history)

    total_time = time.time() - start_time

    # Collect results
    results = {
        "framework": "art",
        "base_model": config.base_model,
        "num_iterations": config.num_iterations,
        "group_size": config.group_size,
        "learning_rate": config.learning_rate,
        "reward_history": reward_history,
        "timing_history": timing_history,
        "total_time_seconds": total_time,
        "total_rollouts": sum(len(task_ids) * config.group_size for _ in range(config.num_iterations)),
        "final_mean_reward": reward_history[-1] if reward_history else 0.0,
        "peak_vram_mb": _get_peak_vram(),
    }

    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "ART training complete: %d iterations, final_reward=%.3f, total_time=%.1fs",
        config.num_iterations,
        results["final_mean_reward"],
        total_time,
    )

    return results


def _get_train_task_ids(domain: str, split: str) -> list[str]:
    """Get training task IDs from tau-bench."""
    from src.agent.tau_adapter import get_task_ids
    return get_task_ids(domain, split)


def _save_checkpoint_info(
    output_path: Path, iteration: int, rewards: list[float], timings: list[float]
):
    """Save intermediate checkpoint info."""
    info = {
        "iteration": iteration,
        "reward_history": rewards,
        "timing_history": timings,
    }
    with open(output_path / f"checkpoint_{iteration}.json", "w") as f:
        json.dump(info, f, indent=2)


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

    parser = argparse.ArgumentParser(description="ART GRPO training")
    parser.add_argument("--config", default=None, help="Path to art config")
    parser.add_argument("--output-dir", default="results/art", help="Output directory")
    parser.add_argument("--hardware", default=None, help="Hardware config path")
    parser.add_argument("--iterations", type=int, default=None, help="Override num iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ARTTrainingConfig.from_yaml(hardware_path=args.hardware)
    if args.iterations:
        config.num_iterations = args.iterations

    asyncio.run(train_art(config=config, output_dir=args.output_dir, hardware_config=args.hardware))


if __name__ == "__main__":
    main()
