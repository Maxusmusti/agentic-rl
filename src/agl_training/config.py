"""Agent Lightning VERL configuration builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class AGLTrainingConfig:
    """Configuration for Agent Lightning VERL training."""

    # Model
    base_model: str = "Qwen/Qwen3-4B"

    # GRPO / Algorithm
    adv_estimator: str = "grpo"
    use_kl_in_reward: bool = False

    # Data
    train_batch_size: int = 32
    max_prompt_length: int = 4096
    max_response_length: int = 2048

    # Actor
    learning_rate: float = 1e-6
    ppo_mini_batch_size: int = 32
    ppo_micro_batch_size_per_gpu: int = 4
    use_kl_loss: bool = False
    entropy_coeff: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    param_offload: bool = False
    optimizer_offload: bool = False

    # Rollout
    tensor_model_parallel_size: int = 1
    rollout_n: int = 8
    rollout_log_prob_micro_batch: int = 4
    multi_turn_format: str = "hermes"
    gpu_memory_utilization: float = 0.6

    # Ref
    ref_log_prob_micro_batch: int = 8
    ref_param_offload: bool = True

    # Trainer
    n_gpus_per_node: int = 8
    nnodes: int = 1
    n_runners: int = 8
    val_before_train: bool = True
    critic_warmup: int = 0
    save_freq: int = 32
    test_freq: int = 16
    total_epochs: int = 2

    # Trace aggregator
    trace_level: str = "trajectory"
    trajectory_max_prompt_length: int = 4096
    trajectory_max_response_length: int = 8192

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

    @classmethod
    def from_yaml(
        cls,
        base_path: str = "configs/base.yaml",
        agl_path: str = "configs/agl.yaml",
        hardware_path: str | None = None,
    ) -> AGLTrainingConfig:
        """Load config from YAML files."""
        with open(base_path) as f:
            base = yaml.safe_load(f)
        with open(agl_path) as f:
            agl = yaml.safe_load(f)

        hw = {}
        if hardware_path:
            with open(hardware_path) as f:
                hw = yaml.safe_load(f).get("agl", {})

        t = agl["training"]
        return cls(
            base_model=base["model"]["name"],
            adv_estimator=t["algorithm"]["adv_estimator"],
            use_kl_in_reward=t["algorithm"]["use_kl_in_reward"],
            train_batch_size=hw.get("train_batch_size", t["data"]["train_batch_size"]),
            max_prompt_length=t["data"]["max_prompt_length"],
            max_response_length=t["data"]["max_response_length"],
            learning_rate=t["actor"]["learning_rate"],
            ppo_mini_batch_size=t["actor"]["ppo_mini_batch_size"],
            ppo_micro_batch_size_per_gpu=t["actor"]["ppo_micro_batch_size_per_gpu"],
            use_kl_loss=t["actor"]["use_kl_loss"],
            entropy_coeff=t["actor"]["entropy_coeff"],
            clip_ratio_low=t["actor"]["clip_ratio_low"],
            clip_ratio_high=t["actor"]["clip_ratio_high"],
            param_offload=hw.get("param_offload", t["actor"]["fsdp"]["param_offload"]),
            optimizer_offload=hw.get("optimizer_offload", t["actor"]["fsdp"]["optimizer_offload"]),
            tensor_model_parallel_size=t["rollout"]["tensor_model_parallel_size"],
            rollout_n=t["rollout"]["n"],
            rollout_log_prob_micro_batch=t["rollout"]["log_prob_micro_batch_size_per_gpu"],
            multi_turn_format=t["rollout"]["multi_turn_format"],
            gpu_memory_utilization=hw.get(
                "gpu_memory_utilization", t["rollout"]["gpu_memory_utilization"]
            ),
            ref_log_prob_micro_batch=t["ref"]["log_prob_micro_batch_size_per_gpu"],
            ref_param_offload=t["ref"]["fsdp"]["param_offload"],
            n_gpus_per_node=hw.get("n_gpus_per_node", t["trainer"]["n_runners"]),
            nnodes=hw.get("nnodes", 1),
            n_runners=hw.get("n_runners", t["trainer"]["n_runners"]),
            val_before_train=t["trainer"]["val_before_train"],
            critic_warmup=t["trainer"]["critic_warmup"],
            save_freq=t["trainer"]["save_freq"],
            test_freq=t["trainer"]["test_freq"],
            total_epochs=t["trainer"]["total_epochs"],
            trace_level=t["trace_aggregator"]["level"],
            trajectory_max_prompt_length=t["trace_aggregator"]["trajectory_max_prompt_length"],
            trajectory_max_response_length=t["trace_aggregator"]["trajectory_max_response_length"],
            max_turns=base["agent"]["max_turns"],
            temperature=base["model"]["temperature"],
            max_tokens=base["model"]["max_tokens"],
            domain=base["tau_bench"]["domain"],
            user_model=base["tau_bench"]["user_model"],
            train_split=base["tau_bench"]["splits"]["train"],
            test_split=base["tau_bench"]["splits"]["test"],
            reward_type=base["reward"]["type"],
        )

    def to_verl_config(self) -> dict[str, Any]:
        """Build the VERL config dictionary for Agent Lightning."""
        return {
            "algorithm": {
                "adv_estimator": self.adv_estimator,
                "use_kl_in_reward": self.use_kl_in_reward,
            },
            "data": {
                "train_batch_size": self.train_batch_size,
                "max_prompt_length": self.max_prompt_length,
                "max_response_length": self.max_response_length,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": self.base_model,
                    "use_remove_padding": True,
                    "enable_gradient_checkpointing": True,
                },
                "actor": {
                    "ppo_mini_batch_size": self.ppo_mini_batch_size,
                    "ppo_micro_batch_size_per_gpu": self.ppo_micro_batch_size_per_gpu,
                    "optim": {"lr": self.learning_rate},
                    "use_kl_loss": self.use_kl_loss,
                    "entropy_coeff": self.entropy_coeff,
                    "clip_ratio_low": self.clip_ratio_low,
                    "clip_ratio_high": self.clip_ratio_high,
                    "fsdp_config": {
                        "param_offload": self.param_offload,
                        "optimizer_offload": self.optimizer_offload,
                    },
                },
                "rollout": {
                    "tensor_model_parallel_size": self.tensor_model_parallel_size,
                    "n": self.rollout_n,
                    "log_prob_micro_batch_size_per_gpu": self.rollout_log_prob_micro_batch,
                    "multi_turn": {"format": self.multi_turn_format},
                    "name": "vllm",
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                },
                "ref": {
                    "log_prob_micro_batch_size_per_gpu": self.ref_log_prob_micro_batch,
                    "fsdp_config": {"param_offload": self.ref_param_offload},
                },
            },
            "trainer": {
                "n_gpus_per_node": self.n_gpus_per_node,
                "nnodes": self.nnodes,
                "val_before_train": self.val_before_train,
                "critic_warmup": self.critic_warmup,
                "logger": ["console", "wandb"],
                "save_freq": self.save_freq,
                "test_freq": self.test_freq,
                "total_epochs": self.total_epochs,
            },
            "agentlightning": {
                "trace_aggregator": {
                    "level": self.trace_level,
                    "trajectory_max_prompt_length": self.trajectory_max_prompt_length,
                    "trajectory_max_response_length": self.trajectory_max_response_length,
                },
            },
        }
