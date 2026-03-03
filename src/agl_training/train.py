"""Agent Lightning training entry point — VERL-based GRPO training."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.agl_training.config import AGLTrainingConfig
from src.agl_training.lit_agent import create_lit_agent

logger = logging.getLogger(__name__)


def train_agl(
    config: AGLTrainingConfig | None = None,
    output_dir: str = "results/agl",
    hardware_config: str | None = None,
) -> dict[str, Any]:
    """Run Agent Lightning VERL training on tau-bench.

    Args:
        config: Training configuration. If None, loads from YAML.
        output_dir: Directory to save results.
        hardware_config: Path to hardware config YAML.

    Returns:
        Dict with training results and metrics.
    """
    import agentlightning as agl

    if config is None:
        config = AGLTrainingConfig.from_yaml(hardware_path=hardware_config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build VERL config
    verl_config = config.to_verl_config()

    # Save config for reproducibility
    with open(output_path / "verl_config.json", "w") as f:
        json.dump(verl_config, f, indent=2)

    # Create the LitAgent
    agent = create_lit_agent(
        domain=config.domain,
        user_model=config.user_model,
        reward_type=config.reward_type,
        max_turns=config.max_turns,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Build datasets
    train_dataset = _build_dataset(config.domain, config.train_split)
    val_dataset = _build_dataset(config.domain, config.test_split)

    # Create the trainer
    trainer = agl.Trainer(
        algorithm=agl.VERL(verl_config),
        n_runners=config.n_runners,
        adapter=agl.TraceToMessages(),
        strategy="cs",  # Client-server execution strategy for VERL
    )

    logger.info(
        "Starting AGL training: n_runners=%d, total_epochs=%d, batch_size=%d",
        config.n_runners,
        config.total_epochs,
        config.train_batch_size,
    )

    start_time = time.time()

    # Run training
    trainer.fit(
        agent=agent,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    total_time = time.time() - start_time

    # Collect results
    results = {
        "framework": "agl",
        "base_model": config.base_model,
        "total_epochs": config.total_epochs,
        "train_batch_size": config.train_batch_size,
        "n_runners": config.n_runners,
        "learning_rate": config.learning_rate,
        "total_time_seconds": total_time,
        "peak_vram_mb": _get_peak_vram(),
    }

    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "AGL training complete: %d epochs, total_time=%.1fs",
        config.total_epochs,
        total_time,
    )

    return results


def _build_dataset(domain: str, split: str) -> list[dict[str, Any]]:
    """Build a dataset of task dicts for Agent Lightning.

    Each entry is a dict with at minimum {"task_id": str} that gets
    passed to the LitAgent's rollout() method.
    """
    from src.agent.tau_adapter import get_task_ids
    task_ids = get_task_ids(domain, split)
    return [{"task_id": tid} for tid in task_ids]


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

    parser = argparse.ArgumentParser(description="Agent Lightning VERL training")
    parser.add_argument("--output-dir", default="results/agl", help="Output directory")
    parser.add_argument("--hardware", default=None, help="Hardware config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_agl(output_dir=args.output_dir, hardware_config=args.hardware)


if __name__ == "__main__":
    main()
