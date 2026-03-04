"""Tool call correctness reward for Toucan single-turn data."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def tool_call_reward(
    model_response: Any,
    expected_name: str,
    expected_args: dict,
) -> float:
    """Compute reward based on tool call correctness.

    Args:
        model_response: The OpenAI ChatCompletion Choice message object.
        expected_name: The expected tool/function name.
        expected_args: The expected arguments dict.

    Returns:
        1.0 — correct tool name AND correct arguments
        0.5 — correct tool name, wrong/partial arguments
        0.0 — wrong tool name or no tool call
    """
    # Extract the model's tool call
    predicted_name, predicted_args = _extract_tool_call(model_response)

    if predicted_name is None:
        return 0.0

    # Normalize names (strip prefixes like "weather-" for comparison)
    if not _names_match(predicted_name, expected_name):
        return 0.0

    # Name matches — check arguments
    if _args_match(predicted_args, expected_args):
        return 1.0

    return 0.5


def _extract_tool_call(message: Any) -> tuple[str | None, dict]:
    """Extract tool call name and arguments from an OpenAI message."""
    # Handle Choice object
    if hasattr(message, "message"):
        message = message.message

    # Check tool_calls (modern format)
    if hasattr(message, "tool_calls") and message.tool_calls:
        tc = message.tool_calls[0]
        name = tc.function.name
        args_str = tc.function.arguments
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_str or {}
        return name, args

    # Check function_call (legacy format)
    if hasattr(message, "function_call") and message.function_call:
        name = message.function_call.name
        args_str = message.function_call.arguments
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_str or {}
        return name, args

    # Dict format
    if isinstance(message, dict):
        if "tool_calls" in message and message["tool_calls"]:
            tc = message["tool_calls"][0]
            name = tc.get("function", {}).get("name")
            args = tc.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return name, args
        if "function_call" in message and message["function_call"]:
            name = message["function_call"].get("name")
            args = message["function_call"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return name, args

    return None, {}


def _names_match(predicted: str, expected: str) -> bool:
    """Check if tool names match, handling common variations."""
    if predicted == expected:
        return True
    # Toucan uses "server-tool_name" format, model might output just "tool_name"
    if predicted.endswith(expected) or expected.endswith(predicted):
        return True
    # Normalize: strip common prefixes, lowercase
    p = predicted.lower().replace("-", "_").replace(".", "_")
    e = expected.lower().replace("-", "_").replace(".", "_")
    if p == e:
        return True
    # Check if one is a suffix of the other (e.g., "weather-get_alerts" vs "get_alerts")
    if p.endswith(e) or e.endswith(p):
        return True
    return False


def _args_match(predicted: dict, expected: dict) -> bool:
    """Check if arguments match, with type coercion and normalization."""
    if not predicted and not expected:
        return True
    if not predicted or not expected:
        return False

    # Normalize both to sorted JSON strings for comparison
    try:
        pred_normalized = json.dumps(_normalize_args(predicted), sort_keys=True)
        exp_normalized = json.dumps(_normalize_args(expected), sort_keys=True)
        return pred_normalized == exp_normalized
    except (TypeError, ValueError):
        return False


def _normalize_args(args: dict) -> dict:
    """Normalize argument values for comparison."""
    normalized = {}
    for k, v in args.items():
        if isinstance(v, str):
            # Try to parse numeric strings
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = v.strip()
        normalized[k] = v
    return normalized
