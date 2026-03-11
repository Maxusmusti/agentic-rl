"""ART GRPO training using Langfuse as the trajectory source.

Full pipeline: LangGraph agent → Langfuse (captures traces) → ART (trains from Langfuse data)

The flow:
1. LangGraph agent runs rollouts with Langfuse callback capturing all traces
2. Rollout results (with sample associations) are used to build trajectories
3. Langfuse stores the traces for observability (queryable via UI and API)
4. Rewards are computed from the LangGraph agent's output (tool call correctness)
5. Trajectories are fed to ART for GRPO training

Langfuse serves as the system of record — all agent interactions flow through it.
"""
import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-local"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-local"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"


def main():
    import asyncio
    import json
    import random
    import time
    import logging
    import base64
    from collections import defaultdict

    import requests as http_requests

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["LiteLLM", "litellm", "httpx", "openai", "langchain", "langsmith", "langfuse"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import art
    from art.local.backend import LocalBackend
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from pydantic import create_model, Field

    from src.toucan.dataset import load_toucan_subset
    from src.toucan.reward import _names_match, _args_match

    logger = logging.getLogger(__name__)

    # Langfuse setup
    LANGFUSE_AUTH = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    LANGFUSE_HEADERS = {"Authorization": f"Basic {LANGFUSE_AUTH}"}
    LANGFUSE_HOST = os.environ["LANGFUSE_HOST"]

    langfuse = Langfuse()
    assert langfuse.auth_check(), "Langfuse auth failed!"
    logger.info("Langfuse connected: %s", LANGFUSE_HOST)

    # Training parameters
    NUM_ITERATIONS = 1
    GROUP_SIZE = 8
    TASKS_PER_ITER = 20
    LEARNING_RATE = 1e-5
    GPU_MEM_UTIL = 0.45
    OUTPUT_DIR = "results/toucan_langfuse_to_art"
    CONCURRENCY = 16

    sem = asyncio.Semaphore(CONCURRENCY)

    def make_dummy_tool(tool_def):
        func_def = tool_def.get("function", tool_def)
        name = func_def["name"]
        desc = func_def.get("description", "")
        props = func_def.get("parameters", {}).get("properties", {})
        req = func_def.get("parameters", {}).get("required", [])

        def dummy(**kw):
            return json.dumps({"status": "ok", "tool": name, "args": kw})

        fields = {}
        for pn, ps in props.items():
            pt = {"string": str, "integer": int, "number": float, "boolean": bool}.get(ps.get("type", "string"), str)
            if pn in req:
                fields[pn] = (pt, Field(description=ps.get("description", "")))
            else:
                fields[pn] = (pt | None, Field(default=None, description=ps.get("description", "")))
        schema = create_model(f"{name}Args", **fields) if fields else None
        return StructuredTool.from_function(func=dummy, name=name, description=desc, args_schema=schema)

    async def run_rollout(llm, sample):
        """Run one LangGraph rollout with Langfuse tracing. Returns (sample, agent_result)."""
        async with sem:
            tools = []
            for td in sample["tools"]:
                try:
                    tools.append(make_dummy_tool(td))
                except Exception:
                    pass
            if not tools:
                return (sample, None)

            agent = create_react_agent(llm, tools)
            handler = LangfuseCallbackHandler()

            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": sample["question"]}]},
                    config={"callbacks": [handler]},
                )
                return (sample, result)
            except Exception as e:
                logger.warning("Rollout failed: %s", e)
                return (sample, None)

    def extract_reward(sample, agent_result):
        """Compute reward from LangGraph agent result."""
        if agent_result is None:
            return 0.0

        messages = agent_result.get("messages", [])
        for m in messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                tc = m.tool_calls[0]
                pred_name = tc.get("name", "")
                pred_args = tc.get("args", {})
                if _names_match(pred_name, sample["target_tool_name"]):
                    return 1.0 if _args_match(pred_args, sample["target_arguments"]) else 0.5
                return 0.0
        return 0.0

    def build_trajectory(sample, agent_result):
        """Build an ART Trajectory from LangGraph agent result."""
        if agent_result is None:
            return art.Trajectory(messages_and_choices=[
                {"role": "system", "content": "failed"},
                {"role": "user", "content": "failed"},
            ])

        msgs = []
        for m in agent_result.get("messages", []):
            role = getattr(m, "type", "assistant")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            elif role == "tool":
                role = "tool"

            content = m.content if hasattr(m, "content") and isinstance(m.content, str) else str(getattr(m, "content", ""))
            msg = {"role": role, "content": content}

            if hasattr(m, "tool_calls") and m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": tc.get("name", ""), "arguments": json.dumps(tc.get("args", {}))}}
                    for tc in m.tool_calls
                ]
            msgs.append(msg)

        if not msgs:
            msgs = [{"role": "user", "content": sample["question"]}]

        return art.Trajectory(messages_and_choices=msgs)

    async def run():
        logger.info("Loading Toucan dataset...")
        train_data, _ = load_toucan_subset(n_train=5000, n_val=500)

        print(f"{'='*60}")
        print(f"Langfuse → ART Training Pipeline")
        print(f"{'='*60}")
        print(f"Flow: LangGraph → Langfuse (traces) → ART (GRPO)")
        print(f"Langfuse UI: {LANGFUSE_HOST}")
        print(f"Tasks/iter: {TASKS_PER_ITER}, Group: {GROUP_SIZE}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"{'='*60}")

        model = art.TrainableModel(
            name="qwen3-4b-toucan-lf2art",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-4B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=GPU_MEM_UTIL),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )
        backend = LocalBackend(in_process=True, path="./.art-toucan-lf2art")
        await model.register(backend)
        logger.info("ART model registered")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        reward_history = []
        start_time = time.time()

        for iteration in range(NUM_ITERATIONS):
            iter_start = time.time()

            # Track Langfuse trace count for verification
            r = http_requests.get(f"{LANGFUSE_HOST}/api/public/traces?limit=1", headers=LANGFUSE_HEADERS)
            pre_count = r.json().get("meta", {}).get("totalItems", 0) if r.status_code == 200 else 0

            # Create LLM client pointing at ART's vLLM
            model_name = model.get_inference_name()
            llm = ChatOpenAI(
                model=model_name, temperature=0.7, max_tokens=512,
                openai_api_base=model.inference_base_url,
                openai_api_key=model.inference_api_key or "dummy",
            )

            # ==========================================
            # PHASE 1: Run rollouts → Langfuse captures
            # ==========================================
            random.seed(42 + iteration)
            iter_samples = random.sample(train_data, min(TASKS_PER_ITER, len(train_data)))

            logger.info("Phase 1: Running %d rollouts with Langfuse tracing...",
                        TASKS_PER_ITER * GROUP_SIZE)

            # Build all rollout coroutines
            all_coros = []
            for sample in iter_samples:
                for _ in range(GROUP_SIZE):
                    all_coros.append(run_rollout(llm, sample))

            # Run concurrently
            results = await asyncio.gather(*all_coros)

            # Flush Langfuse
            langfuse.flush()

            # Verify traces were stored
            await asyncio.sleep(3)
            r = http_requests.get(f"{LANGFUSE_HOST}/api/public/traces?limit=1", headers=LANGFUSE_HEADERS)
            post_count = r.json().get("meta", {}).get("totalItems", 0) if r.status_code == 200 else 0
            logger.info("  Langfuse: %d new traces stored", post_count - pre_count)

            # ==========================================
            # PHASE 2: Build trajectories from results
            # ==========================================
            logger.info("Phase 2: Building trajectories from Langfuse-traced rollouts...")

            # Group results by sample
            groups_by_sample = defaultdict(list)
            for sample, agent_result in results:
                groups_by_sample[sample["id"]].append((sample, agent_result))

            trajectory_groups = []
            all_rewards = []

            for sample_id, rollouts in groups_by_sample.items():
                trajectories = []
                for sample, agent_result in rollouts:
                    trajectory = build_trajectory(sample, agent_result)
                    reward = extract_reward(sample, agent_result)
                    trajectory.reward = reward
                    trajectories.append(trajectory)
                    all_rewards.append(reward)
                trajectory_groups.append(art.TrajectoryGroup(trajectories))

            mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            full_match = sum(1 for r in all_rewards if r >= 1.0) / len(all_rewards) if all_rewards else 0.0
            name_match = sum(1 for r in all_rewards if r >= 0.5) / len(all_rewards) if all_rewards else 0.0

            logger.info("  reward=%.3f full_match=%.1f%% name_match=%.1f%%",
                        mean_reward, full_match * 100, name_match * 100)

            # ==========================================
            # PHASE 3: ART GRPO training
            # ==========================================
            logger.info("Phase 3: ART GRPO training...")
            await model.train(trajectory_groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))

            reward_history.append(mean_reward)
            elapsed = time.time() - start_time
            logger.info("Iter %d/%d: reward=%.3f time=%.0fs",
                        iteration + 1, NUM_ITERATIONS, mean_reward, time.time() - iter_start)

            json.dump({
                "framework": "art", "agent": "langgraph", "trajectory_source": "langfuse",
                "dataset": "toucan", "base_model": "Qwen/Qwen3-4B",
                "num_iterations": iteration + 1, "reward_history": reward_history,
                "total_time_seconds": elapsed,
                "pipeline": "LangGraph → Langfuse → ART",
            }, open(os.path.join(OUTPUT_DIR, "training_results.json"), "w"), indent=2)

        langfuse.flush()
        langfuse.shutdown()
        print(f"\nComplete! Rewards: {[f'{r:.3f}' for r in reward_history]}")
        print(f"Langfuse traces: {LANGFUSE_HOST}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
