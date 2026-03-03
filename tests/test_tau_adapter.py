"""Tests for tau-bench adapter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.agent.tau_adapter import _tau2_tools_to_openai
from src.agent.tool_formatter import tau_tool_to_openai, tau_tools_to_openai


class TestTau2ToolsToOpenAI:
    def test_tool_with_openai_schema(self):
        """tau2 Tool objects have .openai_schema property."""
        tool = MagicMock()
        tool.openai_schema = {
            "name": "get_order",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
            },
        }
        result = _tau2_tools_to_openai([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_order"

    def test_dict_tool_already_formatted(self):
        """Dict tools that already have type+function structure."""
        tool = {
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": "Cancel an order",
            },
        }
        result = _tau2_tools_to_openai([tool])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "cancel_order"

    def test_dict_tool_raw(self):
        """Dict tools without wrapper structure."""
        tool = {"name": "lookup", "description": "Look up info"}
        result = _tau2_tools_to_openai([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "lookup"

    def test_multiple_tools(self):
        t1 = MagicMock()
        t1.openai_schema = {"name": "a", "description": "A"}
        t2 = MagicMock()
        t2.openai_schema = {"name": "b", "description": "B"}
        result = _tau2_tools_to_openai([t1, t2])
        assert len(result) == 2


class TestToolFormatter:
    def test_basic_tool(self):
        tool = {
            "name": "get_order",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID"},
                },
                "required": ["order_id"],
            },
        }
        result = tau_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_order"
        assert result["function"]["description"] == "Get order details"
        assert result["function"]["parameters"]["properties"]["order_id"]["type"] == "string"

    def test_tool_without_params(self):
        tool = {"name": "get_status", "description": "Get status"}
        result = tau_tool_to_openai(tool)
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_batch_conversion(self):
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
        ]
        result = tau_tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"
