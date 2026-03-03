"""Agent Lightning LitAgent subclass wrapping our shared ReAct agent."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from src.agent.react_agent import ReActConfig
from src.agent.tau_adapter import TauBenchRolloutEnv
from src.eval.reward import get_reward_fn

logger = logging.getLogger(__name__)


def create_lit_agent(
    domain: str = "retail",
    user_model: str = "gpt-4",
    reward_type: str = "binary",
    max_turns: int = 15,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Any:
    """Create a TauBenchLitAgent instance.

    Deferred import of agentlightning to avoid import errors
    when the package isn't installed.
    """
    import agentlightning as agl

    class TauBenchLitAgent(agl.LitAgent):
        """LitAgent that runs tau-bench episodes using the shared ReAct agent.

        Agent Lightning calls `rollout()` during training. We use the LLM
        endpoint provided by AGL's resource system to create an OpenAI client,
        then run a full tau-bench episode through our shared agent layer.
        """

        def __init__(self):
            super().__init__()
            self._domain = domain
            self._user_model = user_model
            self._reward_type = reward_type
            self._max_turns = max_turns
            self._temperature = temperature
            self._max_tokens = max_tokens

        def rollout(
            self,
            task: dict[str, Any],
            resources: agl.NamedResources,
            rollout: agl.Rollout,
        ) -> float:
            """Run a single tau-bench episode and return the reward.

            Args:
                task: Task dict containing at minimum {"task_id": str}.
                resources: AGL resources including "main_llm" endpoint.
                rollout: AGL Rollout object for trace recording.

            Returns:
                Reward float (0.0 or 1.0 for binary).
            """
            import asyncio

            # Get the LLM endpoint from AGL resources
            llm: agl.LLM = resources["main_llm"]

            # Create an async OpenAI client pointing to AGL's LLM proxy
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                base_url=llm.endpoint,
                api_key=llm.api_key or "dummy-key",
            )

            model_name = llm.model_name if hasattr(llm, "model_name") else "default"
            task_id = task.get("task_id", task.get("id", "unknown"))

            react_config = ReActConfig(
                model=model_name,
                max_turns=self._max_turns,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            reward_fn = get_reward_fn(self._reward_type)

            rollout_env = TauBenchRolloutEnv(
                client=client,
                domain=self._domain,
                model=model_name,
                user_model=self._user_model,
                config=react_config,
            )

            # Run the episode (bridge sync AGL interface to our async agent)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        episode = pool.submit(
                            asyncio.run,
                            rollout_env.run_episode(task_id=task_id, reward_fn=reward_fn),
                        ).result()
                else:
                    episode = asyncio.run(
                        rollout_env.run_episode(task_id=task_id, reward_fn=reward_fn)
                    )
            except RuntimeError:
                episode = asyncio.run(
                    rollout_env.run_episode(task_id=task_id, reward_fn=reward_fn)
                )

            logger.info(
                "AGL rollout task=%s reward=%.1f turns=%d",
                task_id,
                episode.reward,
                episode.react_result.num_turns,
            )

            return episode.reward

    return TauBenchLitAgent()
