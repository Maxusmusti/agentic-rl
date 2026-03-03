"""Metric computation for tau-bench evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskMetrics:
    """Metrics for a single task across multiple trials."""

    task_id: str
    trials: list[float] = field(default_factory=list)  # reward per trial
    num_turns: list[int] = field(default_factory=list)  # turns per trial

    @property
    def pass_at_1(self) -> float:
        """Probability of passing on a single attempt."""
        if not self.trials:
            return 0.0
        return sum(1 for r in self.trials if r >= 1.0) / len(self.trials)

    @property
    def pass_at_k(self) -> float:
        """pass@k using all available trials (unbiased estimator)."""
        return compute_pass_at_k(len(self.trials), sum(1 for r in self.trials if r >= 1.0))

    @property
    def mean_reward(self) -> float:
        if not self.trials:
            return 0.0
        return sum(self.trials) / len(self.trials)

    @property
    def avg_turns(self) -> float:
        if not self.num_turns:
            return 0.0
        return sum(self.num_turns) / len(self.num_turns)


@dataclass
class EvalMetrics:
    """Aggregated metrics across all tasks."""

    task_metrics: list[TaskMetrics] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    rollouts_per_second: float = 0.0

    @property
    def pass_at_1(self) -> float:
        """Macro-averaged pass@1 across all tasks."""
        if not self.task_metrics:
            return 0.0
        return sum(t.pass_at_1 for t in self.task_metrics) / len(self.task_metrics)

    @property
    def pass_at_k(self) -> float:
        """Macro-averaged pass@k across all tasks."""
        if not self.task_metrics:
            return 0.0
        return sum(t.pass_at_k for t in self.task_metrics) / len(self.task_metrics)

    @property
    def mean_reward(self) -> float:
        if not self.task_metrics:
            return 0.0
        return sum(t.mean_reward for t in self.task_metrics) / len(self.task_metrics)

    @property
    def avg_turns(self) -> float:
        if not self.task_metrics:
            return 0.0
        turns = [t.avg_turns for t in self.task_metrics if t.avg_turns > 0]
        if not turns:
            return 0.0
        return sum(turns) / len(turns)

    @property
    def num_tasks(self) -> int:
        return len(self.task_metrics)

    @property
    def total_trials(self) -> int:
        return sum(len(t.trials) for t in self.task_metrics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a dictionary."""
        return {
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "mean_reward": self.mean_reward,
            "avg_turns": self.avg_turns,
            "num_tasks": self.num_tasks,
            "total_trials": self.total_trials,
            "wall_clock_seconds": self.wall_clock_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "rollouts_per_second": self.rollouts_per_second,
            "per_task": [
                {
                    "task_id": t.task_id,
                    "pass_at_1": t.pass_at_1,
                    "pass_at_k": t.pass_at_k,
                    "mean_reward": t.mean_reward,
                    "avg_turns": t.avg_turns,
                    "trials": t.trials,
                }
                for t in self.task_metrics
            ],
        }


def compute_pass_at_k(n: int, c: int, k: int | None = None) -> float:
    """Compute the unbiased pass@k estimator.

    Args:
        n: Total number of trials.
        c: Number of successful trials.
        k: k value. If None, uses k=n (pass@all).

    Returns:
        Estimated pass@k probability.
    """
    if k is None:
        k = n
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # 1 - C(n-c, k) / C(n, k)
    # Use log to avoid overflow
    log_numerator = sum(math.log(n - c - i) for i in range(min(k, n - c)))
    log_denominator = sum(math.log(n - i) for i in range(k))

    return 1.0 - math.exp(log_numerator - log_denominator)


def compute_pass_at_1(n: int, c: int) -> float:
    """Compute pass@1 = c/n."""
    if n == 0:
        return 0.0
    return c / n


def compute_pass_at_4(n: int, c: int) -> float:
    """Compute pass@4 using the unbiased estimator."""
    return compute_pass_at_k(n, c, k=4)
