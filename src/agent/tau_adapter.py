"""Bridge between tau-bench environment and the ReAct agent.

Provides two key components:
- TauReActAgent: implements tau2's BaseAgent interface using our ReAct agent
- TauBenchRolloutEnv: wraps a complete tau-bench episode for RL training
  using tau2's AgentGymEnv for step-by-step control
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from src.agent.react_agent import ReActAgent, ReActConfig, ReActResult

logger = logging.getLogger(__name__)

# Ensure tau2 data directory is set
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
if not os.environ.get("TAU2_DATA_DIR") and os.path.exists(_DATA_DIR):
    os.environ["TAU2_DATA_DIR"] = _DATA_DIR


@dataclass
class EpisodeResult:
    """Result from a complete tau-bench episode."""

    react_result: ReActResult
    reward: float
    task_id: str
    domain: str
    simulation_run: Any = None  # tau2 SimulationRun
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)


class TauBenchRolloutEnv:
    """Wraps tau-bench episodes for RL rollout collection.

    Uses tau2's AgentGymEnv for step-by-step control, which lets us
    capture raw Choice objects for ART token attribution while running
    through the full tau-bench orchestration (user simulator, environment,
    evaluation).

    Used by both ART and Agent Lightning.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        domain: str = "retail",
        model: str = "Qwen/Qwen3-4B",
        user_model: str = "gpt-4",
        config: ReActConfig | None = None,
        max_steps: int = 200,
    ):
        self.client = client
        self.domain = domain
        self.model = model
        self.user_model = user_model
        self.config = config or ReActConfig(model=model)
        self.max_steps = max_steps

    async def run_episode(
        self,
        task_id: str,
    ) -> EpisodeResult:
        """Run a single tau-bench episode using AgentGymEnv + our ReAct agent.

        Uses the Gym interface for step-by-step control, letting us
        capture all LLM completions (Choice objects) for RL training.

        Args:
            task_id: The tau-bench task ID to run.

        Returns:
            EpisodeResult with full trajectory, reward, and metadata.
        """
        from tau2.gym.gym_agent import AgentGymEnv
        from tau2.data_model.message import ToolCall as Tau2ToolCall

        # Create the gym environment
        gym_env = AgentGymEnv(
            domain=self.domain,
            task_id=str(task_id),
            max_steps=self.max_steps,
            user_llm=self.user_model,
        )

        obs, info = gym_env.reset()
        tools_openai = _tau2_tools_to_openai(info.get("tools", []))
        policy = info.get("policy", "")

        # Build system prompt from policy
        system_prompt = f"You are a helpful {self.domain} customer service agent.\n\n<policy>\n{policy}\n</policy>"

        # Message history for the LLM
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs},
        ]
        all_choices: list[Any] = []
        num_turns = 0
        terminated = False
        truncated = False
        reward = 0.0
        step_info: dict = {}

        while not terminated and not truncated:
            num_turns += 1

            # Get LLM completion
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools_openai if tools_openai else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            choice = response.choices[0]
            all_choices.append(choice)
            assistant_msg = choice.message

            # Convert to action string for gym env
            if assistant_msg.tool_calls:
                # Multiple tool calls: send each one
                for tc in assistant_msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    # Format as JSON ToolCall for gym env
                    action = json.dumps({
                        "name": tc.function.name,
                        "arguments": args,
                    })

                    obs, reward, terminated, truncated, step_info = gym_env.step(action)

                    if terminated or truncated:
                        break

                # Add assistant message with tool calls to history
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if assistant_msg.content:
                    msg_dict["content"] = assistant_msg.content
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
                messages.append(msg_dict)

                # Add tool results from observation
                if obs and not terminated and not truncated:
                    # The gym env returns the observation as a string
                    # We need to add tool result messages
                    for tc in assistant_msg.tool_calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": obs,
                        })
                elif obs:
                    # Final observation after termination
                    for tc in assistant_msg.tool_calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": obs,
                        })

            else:
                # Text response — send to user simulator via gym env
                content = assistant_msg.content or ""
                messages.append({"role": "assistant", "content": content})

                obs, reward, terminated, truncated, step_info = gym_env.step(content)

                if not terminated and not truncated and obs:
                    messages.append({"role": "user", "content": obs})

            if num_turns >= self.config.max_turns:
                break

        react_result = ReActResult(
            messages=messages,
            choices=all_choices,
            num_turns=num_turns,
            finished=terminated,
            final_response=messages[-1].get("content") if messages else None,
        )

        return EpisodeResult(
            react_result=react_result,
            reward=reward,
            task_id=str(task_id),
            domain=self.domain,
            simulation_run=step_info.get("simulation_run"),
            terminated=terminated,
            truncated=truncated,
            info=step_info,
        )


def _tau2_tools_to_openai(tools: list) -> list[dict[str, Any]]:
    """Convert tau2 Tool objects to OpenAI function calling format.

    tau2 Tool objects have an .openai_schema property that already returns
    the full {"type": "function", "function": {...}} dict.
    """
    result = []
    for tool in tools:
        if hasattr(tool, "openai_schema"):
            schema = tool.openai_schema
            # tau2's openai_schema already includes {"type": "function", "function": {...}}
            if "type" in schema and "function" in schema:
                result.append(schema)
            else:
                result.append({"type": "function", "function": schema})
        elif isinstance(tool, dict):
            if "type" in tool and "function" in tool:
                result.append(tool)
            else:
                result.append({"type": "function", "function": tool})
    return result


def get_task_ids(domain: str, split: str | None = None) -> list[str]:
    """Get task IDs from tau-bench for the given domain and split.

    Args:
        domain: Domain name (e.g., "retail", "airline").
        split: Split name (e.g., "train", "test", "base"). None for all tasks.

    Returns:
        List of task ID strings.
    """
    try:
        from tau2.run import load_tasks
        tasks = load_tasks(domain, task_split_name=split)
        return [str(t.id) for t in tasks]
    except Exception as e:
        logger.error("Failed to load tasks for %s/%s: %s", domain, split, e)
        return []
