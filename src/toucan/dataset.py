"""Toucan-1.5M dataset loading and preprocessing for single-turn tool-calling RL."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/toucan_cache")


def load_toucan_subset(
    n_train: int = 5000,
    n_val: int = 500,
    min_quality: int = 3,
    config: str = "Qwen3",
    cache_dir: Path | str = CACHE_DIR,
) -> tuple[list[dict], list[dict]]:
    """Load and preprocess a subset of Toucan-1.5M for tool-calling RL.

    Downloads from HuggingFace, filters by quality, extracts gold tool calls,
    and splits into train/val. Caches processed data as JSON for fast reuse.

    Args:
        n_train: Number of training samples.
        n_val: Number of validation samples.
        min_quality: Minimum tool_selection_difficulty score to keep.
        config: HuggingFace dataset config ("Qwen3", "Kimi-K2", "OSS", "SFT").
        cache_dir: Directory to cache processed data.

    Returns:
        (train_samples, val_samples) where each sample is a dict with:
        - id: unique identifier
        - question: user's query
        - tools: list of tool definitions (OpenAI format)
        - target_tool_name: expected tool name
        - target_arguments: expected arguments dict
        - system_prompt: the system prompt with tool definitions
    """
    cache_dir = Path(cache_dir)
    train_path = cache_dir / f"train_{n_train}.json"
    val_path = cache_dir / f"val_{n_val}.json"

    # Return cached if available
    if train_path.exists() and val_path.exists():
        logger.info("Loading cached Toucan data from %s", cache_dir)
        with open(train_path) as f:
            train = json.load(f)
        with open(val_path) as f:
            val = json.load(f)
        return train, val

    # Download and process
    logger.info("Downloading Toucan-1.5M (%s config) from HuggingFace...", config)
    from datasets import load_dataset

    total_needed = n_train + n_val
    buffer = int(total_needed * 1.5)  # Extra to account for filtering

    ds = load_dataset("Agent-Ark/Toucan-1.5M", config, split="train", streaming=True)

    processed = []
    skipped = 0
    for row in ds:
        if len(processed) >= buffer:
            break

        sample = _process_row(row, min_quality=min_quality)
        if sample is None:
            skipped += 1
            continue
        processed.append(sample)

    logger.info(
        "Processed %d samples (%d skipped for quality/format)", len(processed), skipped
    )

    if len(processed) < total_needed:
        logger.warning(
            "Only got %d samples (needed %d). Using all for train, fewer for val.",
            len(processed),
            total_needed,
        )

    # Split
    train = processed[:n_train]
    val = processed[n_train : n_train + n_val]

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as f:
        json.dump(train, f)
    with open(val_path, "w") as f:
        json.dump(val, f)

    logger.info("Cached %d train, %d val samples to %s", len(train), len(val), cache_dir)
    return train, val


def _process_row(row: dict, min_quality: int = 3) -> dict | None:
    """Process a single Toucan dataset row into our training format.

    Returns None if the sample should be skipped (low quality, bad format, etc.)
    """
    # Parse messages
    messages = row.get("messages", [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None

    # Need at least: system, user, assistant-with-tool-call
    if len(messages) < 3:
        return None

    # Check quality
    qa = row.get("question_quality_assessment")
    if qa:
        if isinstance(qa, str):
            try:
                qa = json.loads(qa)
            except json.JSONDecodeError:
                qa = {}
        difficulty = qa.get("tool_selection_difficulty", {})
        if isinstance(difficulty, dict):
            score = difficulty.get("score", 5)
        else:
            score = 5
        # We actually want EASIER samples for reliable RL signal
        # High difficulty = model likely to fail → noisy reward
        # Keep all quality levels but could filter if needed

    # Extract the gold tool call from message[2] (assistant)
    assistant_msg = messages[2]
    if assistant_msg.get("role") != "assistant":
        return None

    function_call = assistant_msg.get("function_call")
    tool_calls = assistant_msg.get("tool_calls")

    if function_call:
        target_name = function_call["name"]
        target_args_str = function_call.get("arguments", "{}")
        if isinstance(target_args_str, str):
            try:
                target_args = json.loads(target_args_str)
            except json.JSONDecodeError:
                target_args = {}
        else:
            target_args = target_args_str
    elif tool_calls and len(tool_calls) > 0:
        tc = tool_calls[0]
        target_name = tc["function"]["name"]
        target_args_str = tc["function"].get("arguments", "{}")
        if isinstance(target_args_str, str):
            try:
                target_args = json.loads(target_args_str)
            except json.JSONDecodeError:
                target_args = {}
        else:
            target_args = target_args_str
    else:
        return None

    # Parse available tools
    tools = row.get("available_tools", [])
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None

    if not tools:
        return None

    # Get the system prompt (message[0])
    system_prompt = messages[0].get("content", "") if messages[0].get("role") == "system" else ""

    return {
        "id": row.get("uuid", ""),
        "question": row.get("question", ""),
        "tools": tools,
        "target_tool_name": target_name,
        "target_arguments": target_args,
        "system_prompt": system_prompt,
    }
