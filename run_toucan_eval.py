"""Evaluate tool-calling accuracy on Toucan validation set."""
import asyncio
import json
import os
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
for name in ["LiteLLM", "litellm", "httpx", "openai"]:
    logging.getLogger(name).setLevel(logging.WARNING)

from openai import AsyncOpenAI
from src.toucan.dataset import load_toucan_subset
from src.toucan.reward import tool_call_reward

logger = logging.getLogger(__name__)


async def evaluate(
    vllm_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/toucan/eval",
    label: str = "baseline",
    max_samples: int | None = None,
):
    client = AsyncOpenAI(base_url=vllm_url, api_key="dummy")
    _, val_data = load_toucan_subset()

    if max_samples:
        val_data = val_data[:max_samples]

    print(f"[{label}] Evaluating {len(val_data)} samples with model={model_name}")

    results = []
    full_match = 0
    name_match = 0
    no_tool_call = 0
    start = time.time()

    sem = asyncio.Semaphore(32)

    async def eval_one(sample):
        async with sem:
            messages = [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["question"]},
            ]
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=sample["tools"],
                    temperature=0.0,  # Greedy for eval
                    max_tokens=512,
                )
                choice = response.choices[0]
                reward = tool_call_reward(
                    choice, sample["target_tool_name"], sample["target_arguments"]
                )
                return reward
            except Exception as e:
                logger.warning("Eval failed for %s: %s", sample["id"], e)
                return 0.0

    tasks = [eval_one(s) for s in val_data]
    rewards = await asyncio.gather(*tasks)

    for r in rewards:
        if r >= 1.0:
            full_match += 1
        if r >= 0.5:
            name_match += 1
        if r == 0.0:
            no_tool_call += 1
        results.append(r)

    elapsed = time.time() - start
    n = len(results)

    metrics = {
        "label": label,
        "model_name": model_name,
        "num_samples": n,
        "full_match_accuracy": full_match / n if n else 0,
        "name_match_accuracy": name_match / n if n else 0,
        "no_tool_call_rate": no_tool_call / n if n else 0,
        "mean_reward": sum(results) / n if n else 0,
        "wall_clock_seconds": elapsed,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{label}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{label}] Results:")
    print(f"  Full match (name+args): {full_match}/{n} ({metrics['full_match_accuracy']:.1%})")
    print(f"  Name match:             {name_match}/{n} ({metrics['name_match_accuracy']:.1%})")
    print(f"  No tool call:           {no_tool_call}/{n} ({metrics['no_tool_call_rate']:.1%})")
    print(f"  Mean reward:            {metrics['mean_reward']:.3f}")
    print(f"  Time:                   {elapsed:.1f}s")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="results/toucan/eval")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(evaluate(
        vllm_url=args.vllm_url,
        model_name=args.model_name,
        output_dir=args.output_dir,
        label=args.label,
        max_samples=args.max_samples,
    ))
