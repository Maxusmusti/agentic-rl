"""Reward functions for tau-bench evaluation results.

Both ART and Agent Lightning use the identical reward function to ensure
fair comparison.

tau2 returns reward directly via SimulationRun.reward_info.reward (float in [0,1]).
The gym env also returns reward as the step() return value.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class RewardType(str, Enum):
    BINARY = "binary"
    GRADED = "graded"


def binary_reward(reward_value: float) -> float:
    """Binary reward: pass=1.0, fail=0.0.

    tau2 considers a task successful if reward >= 1.0 (within epsilon).
    """
    if reward_value >= 1.0 - 1e-6:
        return 1.0
    return 0.0


def graded_reward(reward_value: float) -> float:
    """Graded reward: use tau2's continuous reward directly.

    tau2's reward is already in [0, 1] and represents partial credit
    from multi-criteria evaluation (product of applicable components).
    """
    return float(max(0.0, min(1.0, reward_value)))


def reward_from_simulation(sim_run: Any, reward_type: str = "binary") -> float:
    """Extract reward from a tau2 SimulationRun object.

    Args:
        sim_run: tau2 SimulationRun with reward_info.
        reward_type: "binary" or "graded".

    Returns:
        Scalar reward.
    """
    if sim_run is None:
        return 0.0

    raw_reward = 0.0
    if hasattr(sim_run, "reward_info") and sim_run.reward_info is not None:
        raw_reward = sim_run.reward_info.reward
    elif isinstance(sim_run, dict) and "reward_info" in sim_run:
        raw_reward = sim_run["reward_info"].get("reward", 0.0)

    fn = get_reward_fn(reward_type)
    return fn(raw_reward)


def get_reward_fn(reward_type: str | RewardType) -> callable:
    """Get the reward function by type.

    Args:
        reward_type: "binary" or "graded"

    Returns:
        Reward function callable(float) -> float.
    """
    if isinstance(reward_type, str):
        reward_type = RewardType(reward_type)

    if reward_type == RewardType.BINARY:
        return binary_reward
    elif reward_type == RewardType.GRADED:
        return graded_reward
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
