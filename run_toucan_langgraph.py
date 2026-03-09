"""ART GRPO training on Toucan data using a LangGraph ReAct agent.

Uses ART's capture_auto_trajectory to automatically capture all LLM calls
made by the LangGraph agent — no manual trajectory building needed.
"""
import os

os.environ["VLLM_USE_V1"] = "0"


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
    for name in ["LiteLLM", "litellm", "httpx", "openai", "langchain", "langsmith"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import art
    from art.local.backend import LocalBackend

    from langchain_openai import ChatOpenAI
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    from src.toucan.dataset import load_toucan_subset
    from src.toucan.reward import tool_call_reward

    logger = logging.getLogger(__name__)

    # Training parameters
    NUM_ITERATIONS = 15
    GROUP_SIZE = 8
    TASKS_PER_ITER = 100
    LEARNING_RATE = 1e-5
    GPU_MEM_UTIL = 0.45
    OUTPUT_DIR = "results/toucan_langgraph"
    CONCURRENCY = 32

    sem = asyncio.Semaphore(CONCURRENCY)

    def make_dummy_tool(tool_def: dict) -> StructuredTool:
        """Create a LangChain tool from an OpenAI tool definition.

        The tool doesn't actually execute — we just need it for the agent
        to generate the right tool call. The reward comes from comparing
        the generated call against the expected one.
        """
        func_def = tool_def.get("function", tool_def)
        name = func_def["name"]
        description = func_def.get("description", "")
        params = func_def.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        # Build a simple function that returns a placeholder
        def dummy_func(**kwargs):
            return json.dumps({"status": "ok", "tool": name, "args": kwargs})

        # Build the args schema dynamically
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

        if fields:
            ArgsModel = create_model(f"{name}Args", **fields)
        else:
            ArgsModel = None

        return StructuredTool.from_function(
            func=dummy_func,
            name=name,
            description=description,
            args_schema=ArgsModel,
        )

    async def toucan_rollout_langgraph(
        model: "art.TrainableModel",
        sample: dict,
    ) -> "art.Trajectory":
        """Run a single-turn LangGraph agent and capture the trajectory."""
        async with sem:
            client_kwargs = {
                "base_url": model.inference_base_url,
                "api_key": model.inference_api_key or "dummy",
            }
            model_name = model.get_inference_name()

            # Create LangChain ChatOpenAI pointing at ART's vLLM
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                max_tokens=512,
                openai_api_base=model.inference_base_url,
                openai_api_key=model.inference_api_key or "dummy",
            )

            # Build tools from the sample's tool definitions
            tools = []
            for tool_def in sample["tools"]:
                try:
                    tools.append(make_dummy_tool(tool_def))
                except Exception:
                    pass

            if not tools:
                # No valid tools, return zero reward
                t = art.Trajectory(messages_and_choices=[
                    {"role": "system", "content": "f"},
                    {"role": "user", "content": "f"},
                ])
                t.reward = 0.0
                return t

            # Create a ReAct agent
            agent = create_react_agent(llm, tools)

            try:
                # capture_auto_trajectory intercepts all httpx responses
                # and automatically builds the Trajectory from OpenAI calls
                trajectory = await art.capture_auto_trajectory(
                    agent.ainvoke(
                        {"messages": [{"role": "user", "content": sample["question"]}]}
                    )
                )

                # Compute reward from the captured trajectory
                # Check if the agent made the right tool call
                reward = _compute_reward_from_trajectory(
                    trajectory, sample["target_tool_name"], sample["target_arguments"]
                )
                trajectory.reward = reward
                return trajectory

            except Exception as e:
                logger.warning("LangGraph rollout failed for %s: %s", sample["id"], e)
                t = art.Trajectory(messages_and_choices=[
                    {"role": "system", "content": "failed"},
                    {"role": "user", "content": "failed"},
                ])
                t.reward = 0.0
                return t

    def _compute_reward_from_trajectory(
        trajectory: "art.Trajectory",
        expected_name: str,
        expected_args: dict,
    ) -> float:
        """Extract the tool call from a captured trajectory and compute reward."""
        # The trajectory's messages contain the full conversation
        # Look for assistant messages with tool_calls
        for msg in trajectory.messages():
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    func = tc.get("function", {})
                    predicted_name = func.get("name", "")
                    predicted_args_str = func.get("arguments", "{}")
                    if isinstance(predicted_args_str, str):
                        try:
                            predicted_args = json.loads(predicted_args_str)
                        except json.JSONDecodeError:
                            predicted_args = {}
                    else:
                        predicted_args = predicted_args_str or {}

                    # Use our existing reward function logic
                    from src.toucan.reward import _names_match, _args_match
                    if not _names_match(predicted_name, expected_name):
                        return 0.0
                    if _args_match(predicted_args, expected_args):
                        return 1.0
                    return 0.5

        # No tool call found
        return 0.0

    async def run():
        logger.info("Loading Toucan dataset...")
        train_data, val_data = load_toucan_subset(n_train=5000, n_val=500)
        logger.info("Loaded %d train, %d val samples", len(train_data), len(val_data))

        rollouts_per_iter = TASKS_PER_ITER * GROUP_SIZE

        print(f"{'='*60}")
        print(f"Toucan ART Training — LangGraph Agent")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_data)}")
        print(f"Tasks/iter: {TASKS_PER_ITER}, Group: {GROUP_SIZE}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Rollouts/iter: {rollouts_per_iter}")
        print(f"Total rollouts: {rollouts_per_iter * NUM_ITERATIONS}")
        print(f"Using: LangGraph ReAct agent + ART capture_auto_trajectory")
        print(f"{'='*60}")

        model = art.TrainableModel(
            name="qwen3-4b-toucan-langgraph",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-4B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=GPU_MEM_UTIL),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )

        backend = LocalBackend(in_process=True, path="./.art-toucan-langgraph")
        await model.register(backend)
        logger.info("Model registered")

        current_step = await model.get_step()
        start_iteration = current_step if current_step > 0 else 0
        if start_iteration > 0:
            logger.info("Resuming from step %d", start_iteration)

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
                        toucan_rollout_langgraph(model, sample)
                        for _ in range(GROUP_SIZE)
                    )
                    for sample in iter_samples
                ),
                pbar_desc=f"Iter {iteration+1}/{NUM_ITERATIONS}",
            )

            rewards = []
            for group in train_groups:
                for traj in group.trajectories:
                    rewards.append(traj.reward)

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
                "Iter %d/%d: mean_reward=%.3f full_match=%.1f%% name_match=%.1f%% "
                "rollouts=%d time=%.0fs elapsed=%.0fs (%.1fh)",
                iteration + 1, NUM_ITERATIONS,
                mean_reward, full_match * 100, name_match * 100,
                len(rewards), iter_time, elapsed, elapsed / 3600,
            )

            results = {
                "framework": "art",
                "agent": "langgraph",
                "dataset": "toucan",
                "base_model": "Qwen/Qwen3-4B",
                "num_iterations": iteration + 1,
                "total_planned_iterations": NUM_ITERATIONS,
                "group_size": GROUP_SIZE,
                "tasks_per_iter": TASKS_PER_ITER,
                "learning_rate": LEARNING_RATE,
                "concurrency": CONCURRENCY,
                "reward_history": reward_history,
                "full_match_history": full_match_history,
                "timing_history": timing_history,
                "total_time_seconds": elapsed,
                "total_rollouts": (iteration + 1) * rollouts_per_iter,
                "final_mean_reward": mean_reward,
                "final_full_match": full_match,
            }
            with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
                json.dump(results, f, indent=2)

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Toucan LangGraph Training Complete")
        print(f"{'='*60}")
        print(f"Reward curve: {[f'{r:.3f}' for r in reward_history]}")
        print(f"Full match:   {[f'{r:.1%}' for r in full_match_history]}")
        print(f"Total time:   {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"{'='*60}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
