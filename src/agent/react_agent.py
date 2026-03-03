"""Core ReAct agent — framework-agnostic, uses any OpenAI-compatible client."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

logger = logging.getLogger(__name__)


@dataclass
class ReActConfig:
    """Configuration for the ReAct agent."""

    max_turns: int = 15
    temperature: float = 0.7
    max_tokens: int = 1024
    model: str = "Qwen/Qwen3-4B"


@dataclass
class ReActResult:
    """Result from a complete ReAct episode."""

    messages: list[dict[str, Any]]
    choices: list[Any] = field(default_factory=list)
    num_turns: int = 0
    finished: bool = False
    final_response: str | None = None


# Type alias: given a tool call dict, return the tool result string
EnvStepFn = Callable[[dict[str, Any]], Awaitable[str]]


class ReActAgent:
    """ReAct tool-calling agent using any OpenAI-compatible chat completions client.

    Framework-agnostic: works with ART's model.openai_client(), Agent Lightning's
    LLM proxy, or a raw vLLM OpenAI-compatible server.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        config: ReActConfig | None = None,
    ):
        self.client = client
        self.config = config or ReActConfig()

    async def run(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        env_step_fn: EnvStepFn,
    ) -> ReActResult:
        """Run a complete ReAct episode.

        Args:
            system_prompt: System prompt for the agent.
            user_message: Initial user message (the customer's request).
            tools: OpenAI function calling tool definitions.
            env_step_fn: Async function that executes a tool call in the environment
                         and returns the result string.

        Returns:
            ReActResult with full message trajectory and metadata.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        all_choices: list[Any] = []
        num_turns = 0

        for turn in range(self.config.max_turns):
            num_turns = turn + 1

            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            choice = response.choices[0]
            all_choices.append(choice)
            assistant_msg = choice.message

            # Build the assistant message dict for the trajectory
            msg_dict = _message_to_dict(assistant_msg)
            messages.append(msg_dict)

            # If the model didn't call any tools, the episode is done
            if not assistant_msg.tool_calls:
                return ReActResult(
                    messages=messages,
                    choices=all_choices,
                    num_turns=num_turns,
                    finished=True,
                    final_response=assistant_msg.content,
                )

            # Execute each tool call and add results
            for tool_call in assistant_msg.tool_calls:
                action = _tool_call_to_action(tool_call)
                logger.debug(
                    "Tool call: %s(%s)", action["name"], json.dumps(action["arguments"])[:200]
                )

                try:
                    result = await env_step_fn(action)
                except Exception as e:
                    logger.warning("Tool call failed: %s", e)
                    result = f"Error: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        # Hit max turns without finishing
        return ReActResult(
            messages=messages,
            choices=all_choices,
            num_turns=num_turns,
            finished=False,
            final_response=None,
        )


def _message_to_dict(msg: ChatCompletionMessage) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage to a plain dict."""
    d: dict[str, Any] = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


def _tool_call_to_action(tool_call: ChatCompletionMessageToolCall) -> dict[str, Any]:
    """Convert an OpenAI tool call to an action dict."""
    args = tool_call.function.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"raw": args}
    return {"name": tool_call.function.name, "arguments": args}
