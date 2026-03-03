"""ART (OpenPipe) training configuration and model setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class ARTTrainingConfig:
    """Configuration for ART GRPO training."""

    # Model
    base_model: str = "Qwen/Qwen3-4B"
    model_name: str = "qwen3-4b-taubench"
    project: str = "agentic-rl-poc"

    # GRPO
    learning_rate: float = 1e-5
    group_size: int = 8
    num_iterations: int = 50
    max_grad_norm: float = 0.1

    # LoRA
    lora_alpha: int = 8

    # vLLM
    gpu_memory_utilization: float = 0.75

    # Backend
    backend_type: str = "local"
    backend_in_process: bool = True
    backend_path: str = "./.art"

    # Agent
    max_turns: int = 15
    temperature: float = 0.7
    max_tokens: int = 1024

    # tau-bench
    domain: str = "retail"
    user_model: str = "gpt-4"
    train_split: str = "train"
    test_split: str = "test"

    # Reward
    reward_type: str = "binary"

    # Hardware
    concurrent_rollouts: int = 64

    @classmethod
    def from_yaml(
        cls,
        base_path: str = "configs/base.yaml",
        art_path: str = "configs/art.yaml",
        hardware_path: str | None = None,
    ) -> ARTTrainingConfig:
        """Load config from YAML files."""
        with open(base_path) as f:
            base = yaml.safe_load(f)
        with open(art_path) as f:
            art = yaml.safe_load(f)

        hw = {}
        if hardware_path:
            with open(hardware_path) as f:
                hw = yaml.safe_load(f).get("art", {})

        return cls(
            base_model=base["model"]["name"],
            model_name=art["training"]["art_model"]["name"],
            project=art["training"]["art_model"]["project"],
            learning_rate=art["training"]["learning_rate"],
            group_size=art["training"]["group_size"],
            num_iterations=art["training"]["num_iterations"],
            max_grad_norm=art["training"]["max_grad_norm"],
            lora_alpha=art["training"]["lora"]["alpha"],
            gpu_memory_utilization=art["training"]["vllm"]["gpu_memory_utilization"],
            backend_type=art["training"]["backend"]["type"],
            backend_in_process=art["training"]["backend"]["in_process"],
            backend_path=art["training"]["backend"]["path"],
            max_turns=base["agent"]["max_turns"],
            temperature=base["model"]["temperature"],
            max_tokens=base["model"]["max_tokens"],
            domain=base["tau_bench"]["domain"],
            user_model=base["tau_bench"]["user_model"],
            train_split=base["tau_bench"]["splits"]["train"],
            test_split=base["tau_bench"]["splits"]["test"],
            reward_type=base["reward"]["type"],
            concurrent_rollouts=hw.get("concurrent_rollouts", 64),
        )


async def create_art_model(config: ARTTrainingConfig) -> Any:
    """Create and register an ART TrainableModel with LocalBackend.

    Returns:
        Registered art.TrainableModel instance.
    """
    import art
    from art.local.backend import LocalBackend

    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
        _internal_config=art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(gpu_memory_utilization=config.gpu_memory_utilization),
            peft_args=art.dev.PeftArgs(lora_alpha=config.lora_alpha),
            trainer_args=art.dev.TrainerArgs(max_grad_norm=config.max_grad_norm),
        ),
    )

    backend = LocalBackend(in_process=config.backend_in_process, path=config.backend_path)
    await model.register(backend)

    return model
