"""ART GRPO training on Toucan data using LangGraph + Langfuse observability.

Full stack: LangGraph (agent) + Langfuse (observability) + ART (RL training)
- LangGraph ReAct agent makes tool-calling decisions
- Langfuse captures traces, spans, and metrics for monitoring
- ART's capture_auto_trajectory captures trajectories for GRPO training
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

    from src.toucan.dataset import load_toucan_subset
    from src.toucan.reward import tool_call_reward

    logger = logging.getLogger(__name__)

    # Initialize Langfuse client
    langfuse = Langfuse()
    assert langfuse.auth_check(), "Langfuse auth failed!"
    logger.info("Langfuse connected: %s", os.environ["LANGFUSE_HOST"])

    # Training parameters
    NUM_ITERATIONS = 1
    GROUP_SIZE = 8
    TASKS_PER_ITER = 5  # Debug: small batch
    LEARNING_RATE = 1e-5
    GPU_MEM_UTIL = 0.45
    OUTPUT_DIR = "results/toucan_langgraph_langfuse"
    CONCURRENCY = 32

    sem = asyncio.Semaphore(CONCURRENCY)

    def make_dummy_tool(tool_def: dict) -> StructuredTool:
        """Create a LangChain tool from an OpenAI tool definition."""
        func_def = tool_def.get("function", tool_def)
        name = func_def["name"]
        description = func_def.get("description", "")
        params = func_def.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        def dummy_func(**kwargs):
            return json.dumps({"status": "ok", "tool": name, "args": kwargs})

        from pydantic import create_model, Field

        fields = {}
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            prop_desc = prop_schema.get("description", "")
            py_type = {"string": str, "integer": int, "number": float, "boolean": bool}.get(prop_type, str)
            if prop_name in required:
                fields[prop_name] = (py_type, Field(description=prop_desc))
            else:
                fields[prop_name] = (py_type | None, Field(default=None, description=prop_desc))

        ArgsModel = create_model(f"{name}Args", **fields) if fields else None

        return StructuredTool.from_function(
            func=dummy_func, name=name, description=description, args_schema=ArgsModel,
        )

    def _compute_reward(trajectory, expected_name, expected_args):
        """Extract tool call from trajectory and compute reward."""
        for msg in trajectory.messages():
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    func = tc.get("function", {})
                    pred_name = func.get("name", "")
                    pred_args_str = func.get("arguments", "{}")
                    pred_args = json.loads(pred_args_str) if isinstance(pred_args_str, str) else (pred_args_str or {})

                    from src.toucan.reward import _names_match, _args_match
                    if not _names_match(pred_name, expected_name):
                        return 0.0
                    return 1.0 if _args_match(pred_args, expected_args) else 0.5
        return 0.0

    async def toucan_rollout(model, sample, iteration):
        """Run LangGraph agent with Langfuse tracing + ART trajectory capture."""
        async with sem:
            model_name = model.get_inference_name()

            llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                max_tokens=512,
                openai_api_base=model.inference_base_url,
                openai_api_key=model.inference_api_key or "dummy",
            )

            tools = []
            for tool_def in sample["tools"]:
                try:
                    tools.append(make_dummy_tool(tool_def))
                except Exception:
                    pass

            if not tools:
                t = art.Trajectory(messages_and_choices=[
                    {"role": "system", "content": "f"},
                    {"role": "user", "content": "f"},
                ])
                t.reward = 0.0
                return t

            agent = create_react_agent(llm, tools)

            # Create Langfuse callback — traces agent runs for observability
            langfuse_handler = LangfuseCallbackHandler()
            try:
                # ART captures trajectory (for training)
                # Langfuse captures trace (for observability)
                trajectory = await art.capture_auto_trajectory(
                    agent.ainvoke(
                        {"messages": [{"role": "user", "content": sample["question"]}]},
                        config={"callbacks": [langfuse_handler]},
                    )
                )

                reward = _compute_reward(
                    trajectory, sample["target_tool_name"], sample["target_arguments"]
                )
                trajectory.reward = reward

                # Also score the trace in Langfuse
                # Langfuse traces captured via callback

                return trajectory

            except Exception as e:
                logger.error("Rollout failed for %s: %s", sample["id"], e, exc_info=True)
                t = art.Trajectory(messages_and_choices=[
                    {"role": "system", "content": "failed"},
                    {"role": "user", "content": "failed"},
                ])
                t.reward = 0.0
                return t

    async def run():
        logger.info("Loading Toucan dataset...")
        train_data, val_data = load_toucan_subset(n_train=5000, n_val=500)
        rollouts_per_iter = TASKS_PER_ITER * GROUP_SIZE

        print(f"{'='*60}")
        print(f"Toucan Training: LangGraph + Langfuse + ART")
        print(f"{'='*60}")
        print(f"Stack: LangGraph (agent) → Langfuse (observability) → ART (GRPO)")
        print(f"Langfuse UI: http://localhost:3000")
        print(f"Tasks/iter: {TASKS_PER_ITER}, Group: {GROUP_SIZE}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"{'='*60}")

        model = art.TrainableModel(
            name="qwen3-4b-toucan-lglf",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-4B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=GPU_MEM_UTIL),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )

        backend = LocalBackend(in_process=True, path="./.art-toucan-lglf")
        await model.register(backend)
        logger.info("ART model registered")

        current_step = await model.get_step()
        start_iteration = current_step if current_step > 0 else 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        reward_history = []
        full_match_history = []
        timing_history = []
        start_time = time.time()

        for iteration in range(start_iteration, NUM_ITERATIONS):
            iter_start = time.time()

            random.seed(42 + iteration)
            iter_samples = random.sample(train_data, min(TASKS_PER_ITER, len(train_data)))

            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        toucan_rollout(model, sample, iteration + 1)
                        for _ in range(GROUP_SIZE)
                    )
                    for sample in iter_samples
                ),
                pbar_desc=f"Iter {iteration+1}/{NUM_ITERATIONS}",
            )

            # Flush Langfuse traces for this iteration
            langfuse.flush()

            rewards = [traj.reward for group in train_groups for traj in group.trajectories]
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            full_match = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0.0
            name_match = sum(1 for r in rewards if r >= 0.5) / len(rewards) if rewards else 0.0
            reward_history.append(mean_reward)
            full_match_history.append(full_match)

            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=LEARNING_RATE),
            )

            iter_time = time.time() - iter_start
            timing_history.append(iter_time)
            elapsed = time.time() - start_time

            logger.info(
                "Iter %d/%d: reward=%.3f full=%.1f%% name=%.1f%% "
                "rollouts=%d time=%.0fs (%.1fh total)",
                iteration + 1, NUM_ITERATIONS,
                mean_reward, full_match * 100, name_match * 100,
                len(rewards), iter_time, elapsed / 3600,
            )

            results = {
                "framework": "art",
                "agent": "langgraph",
                "observability": "langfuse",
                "dataset": "toucan",
                "base_model": "Qwen/Qwen3-4B",
                "num_iterations": iteration + 1,
                "total_planned_iterations": NUM_ITERATIONS,
                "group_size": GROUP_SIZE,
                "tasks_per_iter": TASKS_PER_ITER,
                "learning_rate": LEARNING_RATE,
                "reward_history": reward_history,
                "full_match_history": full_match_history,
                "timing_history": timing_history,
                "total_time_seconds": elapsed,
                "total_rollouts": (iteration + 1) * rollouts_per_iter,
                "final_mean_reward": mean_reward,
                "langfuse_host": os.environ["LANGFUSE_HOST"],
            }
            with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
                json.dump(results, f, indent=2)

        langfuse.flush()
        langfuse.shutdown()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Rewards: {[f'{r:.3f}' for r in reward_history]}")
        print(f"Time: {total_time/60:.0f}min")
        print(f"Langfuse traces: http://localhost:3000")
        print(f"{'='*60}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
