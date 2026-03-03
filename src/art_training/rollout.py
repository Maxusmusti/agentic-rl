"""ART rollout function — runs tau-bench episodes via the shared ReAct agent."""

from __future__ import annotations

import logging
from typing import Any

from src.agent.react_agent import ReActConfig
from src.agent.tau_adapter import TauBenchRolloutEnv
from src.art_training.config import ARTTrainingConfig
from src.eval.reward import binary_reward, graded_reward

logger = logging.getLogger(__name__)


async def art_rollout(
    model: Any,  # art.TrainableModel
    task_id: str,
    config: ARTTrainingConfig,
) -> Any:
    """Run a single tau-bench episode and return an ART Trajectory.

    Uses model.openai_client() for inference so ART can track the completions
    and do proper token attribution for GRPO training.

    The episode runs via AgentGymEnv which handles:
    - User simulator (GPT-4)
    - Environment tool execution
    - Evaluation and reward computation

    Args:
        model: ART TrainableModel instance.
        task_id: tau-bench task ID to run.
        config: Training configuration.

    Returns:
        art.Trajectory with reward assigned.
    """
    import art

    # Get the OpenAI-compatible client from ART
    openai_client = model.openai_client()
    model_name = model.get_inference_name()

    react_config = ReActConfig(
        model=model_name,
        max_turns=config.max_turns,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    reward_fn = binary_reward if config.reward_type == "binary" else graded_reward

    rollout_env = TauBenchRolloutEnv(
        client=openai_client,
        domain=config.domain,
        model=model_name,
        user_model=config.user_model,
        config=react_config,
    )

    # Run the episode
    episode = await rollout_env.run_episode(task_id=task_id)

    # Apply reward function
    reward = reward_fn(episode.reward)

    # Build the ART Trajectory from the collected messages and choices
    messages_and_choices = _build_messages_and_choices(
        episode.react_result.messages,
        episode.react_result.choices,
    )

    trajectory = art.Trajectory(messages_and_choices=messages_and_choices)
    trajectory.reward = reward

    logger.info(
        "Rollout task=%s reward=%.1f turns=%d",
        task_id,
        reward,
        episode.react_result.num_turns,
    )

    return trajectory


def _build_messages_and_choices(
    messages: list[dict[str, Any]],
    choices: list[Any],
) -> list[Any]:
    """Build the messages_and_choices list for ART Trajectory.

    ART expects an interleaved list of message dicts and Choice objects,
    where each Choice follows the messages that preceded it (context).
    """
    result: list[Any] = []
    choice_idx = 0

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            # Replace assistant message dicts with the raw Choice objects
            if choice_idx < len(choices):
                result.append(choices[choice_idx])
                choice_idx += 1
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result
