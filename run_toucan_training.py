"""ART GRPO training on Toucan single-turn tool-calling data.

Much faster than tau-bench: each rollout is a single LLM call (no user simulator).
Reward is deterministic tool call correctness (no LLM judge).
"""
import os
import sys

os.environ.setdefault("TAU2_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
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
    for name in ["LiteLLM", "litellm", "httpx", "openai"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import art
    from art.local.backend import LocalBackend

    from src.toucan.dataset import load_toucan_subset
    from src.toucan.reward import tool_call_reward

    logger = logging.getLogger(__name__)

    # Training parameters
    NUM_ITERATIONS = 15
    GROUP_SIZE = 8
    TASKS_PER_ITER = 100
    LEARNING_RATE = 1e-5
    GPU_MEM_UTIL = 0.45
    OUTPUT_DIR = "results/toucan"
    CONCURRENCY = 32  # Higher concurrency — all local, no API rate limits

    sem = asyncio.Semaphore(CONCURRENCY)

    async def toucan_rollout(
        model: "art.TrainableModel",
        sample: dict,
    ) -> "art.Trajectory":
        """Single-turn rollout: one LLM call, check tool correctness."""
        async with sem:
            client = model.openai_client()
            model_name = model.get_inference_name()

            # Build messages — system prompt with tools + user question
            messages = [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["question"]},
            ]

            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=sample["tools"],
                    temperature=0.7,
                    max_tokens=512,
                )

                choice = response.choices[0]

                # Compute reward: does the tool call match the expected one?
                reward = tool_call_reward(
                    choice,
                    expected_name=sample["target_tool_name"],
                    expected_args=sample["target_arguments"],
                )

                # Build trajectory: [system_msg, user_msg, model_choice]
                trajectory = art.Trajectory(
                    messages_and_choices=[
                        messages[0],  # system (not trainable)
                        messages[1],  # user (not trainable)
                        choice,       # model output (trainable, has logprobs)
                    ]
                )
                trajectory.reward = reward
                return trajectory

            except Exception as e:
                logger.warning("Rollout failed for %s: %s", sample["id"], e)
                trajectory = art.Trajectory(
                    messages_and_choices=[
                        {"role": "system", "content": "failed"},
                        {"role": "user", "content": "failed"},
                    ]
                )
                trajectory.reward = 0.0
                return trajectory

    async def run():
        # Load data
        logger.info("Loading Toucan dataset...")
        train_data, val_data = load_toucan_subset(n_train=5000, n_val=500)
        logger.info("Loaded %d train, %d val samples", len(train_data), len(val_data))

        rollouts_per_iter = TASKS_PER_ITER * GROUP_SIZE

        print(f"{'='*60}")
        print(f"Toucan ART Training")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        print(f"Tasks per iteration: {TASKS_PER_ITER}")
        print(f"Group size: {GROUP_SIZE}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Rollouts/iteration: {rollouts_per_iter}")
        print(f"Total rollouts: {rollouts_per_iter * NUM_ITERATIONS}")
        print(f"Concurrency: {CONCURRENCY}")
        print(f"{'='*60}")

        # Create model
        model = art.TrainableModel(
            name="qwen3-4b-toucan",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-4B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=GPU_MEM_UTIL),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )

        backend = LocalBackend(in_process=True, path="./.art-toucan")
        await model.register(backend)
        logger.info("Model registered")

        # Resume check
        current_step = await model.get_step()
        start_iteration = current_step if current_step > 0 else 0
        if start_iteration > 0:
            logger.info("Resuming from step %d", start_iteration)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        reward_history = []
        timing_history = []
        full_match_history = []
        start_time = time.time()

        for iteration in range(start_iteration, NUM_ITERATIONS):
            iter_start = time.time()

            # Sample tasks for this iteration
            random.seed(42 + iteration)
            iter_samples = random.sample(train_data, min(TASKS_PER_ITER, len(train_data)))

            # Gather trajectory groups
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        toucan_rollout(model, sample) for _ in range(GROUP_SIZE)
                    )
                    for sample in iter_samples
                ),
                pbar_desc=f"Iter {iteration+1}/{NUM_ITERATIONS}",
            )

            # Compute metrics
            rewards = []
            for group in train_groups:
                for traj in group.trajectories:
                    rewards.append(traj.reward)

            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            full_match = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0.0
            name_match = sum(1 for r in rewards if r >= 0.5) / len(rewards) if rewards else 0.0
            reward_history.append(mean_reward)
            full_match_history.append(full_match)

            # Train
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

            # Save results every iteration
            results = {
                "framework": "art",
                "dataset": "toucan",
                "base_model": "Qwen/Qwen3-4B",
                "num_iterations": iteration + 1,
                "total_planned_iterations": NUM_ITERATIONS,
                "group_size": GROUP_SIZE,
                "tasks_per_iter": TASKS_PER_ITER,
                "total_train_samples": len(train_data),
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
        print(f"Toucan Training Complete")
        print(f"{'='*60}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Total rollouts: {NUM_ITERATIONS * rollouts_per_iter}")
        print(f"Reward curve: {[f'{r:.3f}' for r in reward_history]}")
        print(f"Full match curve: {[f'{r:.1%}' for r in full_match_history]}")
        print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"{'='*60}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
