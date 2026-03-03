"""Convert tau-bench tool schemas to OpenAI function calling format."""

from __future__ import annotations

from typing import Any


def tau_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a single tau-bench tool definition to OpenAI function calling format.

    tau-bench tools typically have:
        - name: str
        - description: str
        - parameters: dict (JSON Schema)

    OpenAI function calling format:
        - type: "function"
        - function:
            - name: str
            - description: str
            - parameters: dict (JSON Schema)
    """
    func_def: dict[str, Any] = {
        "name": tool["name"],
        "description": tool.get("description", ""),
    }

    params = tool.get("parameters", {})
    if params:
        func_def["parameters"] = _normalize_parameters(params)
    else:
        func_def["parameters"] = {"type": "object", "properties": {}}

    return {"type": "function", "function": func_def}


def tau_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a list of tau-bench tool definitions to OpenAI format."""
    return [tau_tool_to_openai(t) for t in tools]


def _normalize_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Ensure parameters follow JSON Schema for OpenAI function calling."""
    normalized = dict(params)
    if "type" not in normalized:
        normalized["type"] = "object"
    if "properties" not in normalized:
        normalized["properties"] = {}
    return normalized


def openai_tool_call_to_tau(tool_call: Any) -> dict[str, Any]:
    """Convert an OpenAI tool call response to tau-bench action format.

    Args:
        tool_call: OpenAI ChatCompletionMessageToolCall object

    Returns:
        Dict with 'name' and 'arguments' suitable for tau-bench env.step()
    """
    import json

    args = tool_call.function.arguments
    if isinstance(args, str):
        args = json.loads(args)

    return {
        "name": tool_call.function.name,
        "arguments": args,
    }
