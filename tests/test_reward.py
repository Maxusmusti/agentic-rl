"""Tests for reward functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.eval.reward import (
    binary_reward,
    graded_reward,
    reward_from_simulation,
    get_reward_fn,
    RewardType,
)


class TestBinaryReward:
    def test_pass(self):
        assert binary_reward(1.0) == 1.0

    def test_pass_near_one(self):
        assert binary_reward(0.9999999) == 1.0

    def test_fail(self):
        assert binary_reward(0.5) == 0.0

    def test_zero(self):
        assert binary_reward(0.0) == 0.0

    def test_fail_high(self):
        assert binary_reward(0.99) == 0.0


class TestGradedReward:
    def test_full_reward(self):
        assert graded_reward(1.0) == 1.0

    def test_partial_reward(self):
        assert graded_reward(0.5) == 0.5

    def test_zero_reward(self):
        assert graded_reward(0.0) == 0.0

    def test_clamps_negative(self):
        assert graded_reward(-0.5) == 0.0

    def test_clamps_above_one(self):
        assert graded_reward(1.5) == 1.0


class TestRewardFromSimulation:
    def test_with_reward_info(self):
        sim = MagicMock()
        sim.reward_info.reward = 1.0
        assert reward_from_simulation(sim, "binary") == 1.0

    def test_with_failed_reward_info(self):
        sim = MagicMock()
        sim.reward_info.reward = 0.5
        assert reward_from_simulation(sim, "binary") == 0.0

    def test_graded_from_simulation(self):
        sim = MagicMock()
        sim.reward_info.reward = 0.75
        assert reward_from_simulation(sim, "graded") == 0.75

    def test_none_simulation(self):
        assert reward_from_simulation(None, "binary") == 0.0

    def test_dict_format(self):
        sim = {"reward_info": {"reward": 1.0}}
        assert reward_from_simulation(sim, "binary") == 1.0


class TestGetRewardFn:
    def test_binary_string(self):
        fn = get_reward_fn("binary")
        assert fn is binary_reward

    def test_graded_string(self):
        fn = get_reward_fn("graded")
        assert fn is graded_reward

    def test_enum_type(self):
        fn = get_reward_fn(RewardType.BINARY)
        assert fn is binary_reward

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            get_reward_fn("invalid")
