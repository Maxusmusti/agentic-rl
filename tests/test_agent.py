"""Tests for the shared ReAct agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from src.agent.react_agent import ReActAgent, ReActConfig, ReActResult


def _make_mock_choice(content: str | None = None, tool_calls: list | None = None):
    """Create a mock Choice object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    return choice


def _make_mock_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture
def mock_client():
    """Create a mock AsyncOpenAI client."""
    client = AsyncMock()
    return client


@pytest.fixture
def agent(mock_client):
    """Create a ReActAgent with mock client."""
    config = ReActConfig(max_turns=5, temperature=0.0, model="test-model")
    return ReActAgent(client=mock_client, config=config)


@pytest.mark.asyncio
async def test_simple_response(agent, mock_client):
    """Agent returns a simple text response with no tool calls."""
    choice = _make_mock_choice(content="Hello, how can I help you?")
    response = MagicMock()
    response.choices = [choice]
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    result = await agent.run(
        system_prompt="You are a helpful agent.",
        user_message="Hi there!",
        tools=[],
        env_step_fn=AsyncMock(),
    )

    assert isinstance(result, ReActResult)
    assert result.finished is True
    assert result.num_turns == 1
    assert result.final_response == "Hello, how can I help you?"
    assert len(result.messages) == 3  # system + user + assistant


@pytest.mark.asyncio
async def test_tool_call_then_response(agent, mock_client):
    """Agent makes a tool call, gets result, then responds."""
    # First call: tool call
    tool_call = _make_mock_tool_call("lookup_order", {"order_id": "123"})
    choice1 = _make_mock_choice(content=None, tool_calls=[tool_call])
    response1 = MagicMock()
    response1.choices = [choice1]

    # Second call: final response
    choice2 = _make_mock_choice(content="Your order 123 is on the way!")
    response2 = MagicMock()
    response2.choices = [choice2]

    mock_client.chat.completions.create = AsyncMock(side_effect=[response1, response2])

    env_step = AsyncMock(return_value='{"status": "shipped"}')

    result = await agent.run(
        system_prompt="You are a helpful agent.",
        user_message="Where is my order 123?",
        tools=[{"type": "function", "function": {"name": "lookup_order"}}],
        env_step_fn=env_step,
    )

    assert result.finished is True
    assert result.num_turns == 2
    assert result.final_response == "Your order 123 is on the way!"
    env_step.assert_called_once()
    call_args = env_step.call_args[0][0]
    assert call_args["name"] == "lookup_order"
    assert call_args["arguments"] == {"order_id": "123"}


@pytest.mark.asyncio
async def test_max_turns_reached(agent, mock_client):
    """Agent hits max_turns without finishing."""
    # Always return a tool call
    tool_call = _make_mock_tool_call("some_tool", {"x": 1})
    choice = _make_mock_choice(content=None, tool_calls=[tool_call])
    response = MagicMock()
    response.choices = [choice]
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    env_step = AsyncMock(return_value="ok")

    result = await agent.run(
        system_prompt="System",
        user_message="Do something",
        tools=[{"type": "function", "function": {"name": "some_tool"}}],
        env_step_fn=env_step,
    )

    assert result.finished is False
    assert result.num_turns == 5  # max_turns from config
    assert env_step.call_count == 5


@pytest.mark.asyncio
async def test_tool_call_error_handling(agent, mock_client):
    """Agent handles tool call errors gracefully."""
    tool_call = _make_mock_tool_call("failing_tool", {})
    choice1 = _make_mock_choice(content=None, tool_calls=[tool_call])
    response1 = MagicMock()
    response1.choices = [choice1]

    choice2 = _make_mock_choice(content="Sorry, I encountered an error.")
    response2 = MagicMock()
    response2.choices = [choice2]

    mock_client.chat.completions.create = AsyncMock(side_effect=[response1, response2])

    env_step = AsyncMock(side_effect=RuntimeError("Tool failed"))

    result = await agent.run(
        system_prompt="System",
        user_message="Do something",
        tools=[{"type": "function", "function": {"name": "failing_tool"}}],
        env_step_fn=env_step,
    )

    assert result.finished is True
    # The error message should be in the tool result
    tool_msg = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert "Error:" in tool_msg[0]["content"]


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_turn(agent, mock_client):
    """Agent makes multiple tool calls in a single turn."""
    tc1 = _make_mock_tool_call("tool_a", {"x": 1}, call_id="call_1")
    tc2 = _make_mock_tool_call("tool_b", {"y": 2}, call_id="call_2")
    choice1 = _make_mock_choice(content=None, tool_calls=[tc1, tc2])
    response1 = MagicMock()
    response1.choices = [choice1]

    choice2 = _make_mock_choice(content="Done!")
    response2 = MagicMock()
    response2.choices = [choice2]

    mock_client.chat.completions.create = AsyncMock(side_effect=[response1, response2])

    env_step = AsyncMock(return_value="result")

    result = await agent.run(
        system_prompt="System",
        user_message="Do things",
        tools=[],
        env_step_fn=env_step,
    )

    assert result.finished is True
    assert env_step.call_count == 2
    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
