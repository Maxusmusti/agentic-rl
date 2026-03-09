"""ART GRPO training on Toucan data with Qwen3-32B (4-bit quantized)."""
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
    for name in ["LiteLLM", "litellm", "httpx", "openai"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import art
    from art.local.backend import LocalBackend

    from src.toucan.dataset import load_toucan_subset
    from src.toucan.reward import tool_call_reward

    logger = logging.getLogger(__name__)

    # Training parameters — conservative for 32B on single GPU
    NUM_ITERATIONS = 5
    GROUP_SIZE = 8
    TASKS_PER_ITER = 50       # Fewer tasks since inference is slower with 32B
    LEARNING_RATE = 5e-6      # Lower LR for larger model
    GPU_MEM_UTIL = 0.85       # Offload fix frees GPU before vLLM
    OUTPUT_DIR = "results/toucan_32b"
    CONCURRENCY = 8           # Lower concurrency — 32B inference is heavier

    sem = asyncio.Semaphore(CONCURRENCY)

    async def toucan_rollout(model, sample):
        async with sem:
            client = model.openai_client()
            model_name = model.get_inference_name()

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
                reward = tool_call_reward(
                    choice, sample["target_tool_name"], sample["target_arguments"]
                )
                trajectory = art.Trajectory(
                    messages_and_choices=[messages[0], messages[1], choice]
                )
                trajectory.reward = reward
                return trajectory
            except Exception as e:
                logger.warning("Rollout failed: %s", e)
                t = art.Trajectory(
                    messages_and_choices=[
                        {"role": "system", "content": "f"},
                        {"role": "user", "content": "f"},
                    ]
                )
                t.reward = 0.0
                return t

    async def run():
        logger.info("Loading Toucan dataset...")
        train_data, val_data = load_toucan_subset(n_train=5000, n_val=500)
        logger.info("Loaded %d train, %d val samples", len(train_data), len(val_data))

        rollouts_per_iter = TASKS_PER_ITER * GROUP_SIZE

        print(f"{'='*60}")
        print(f"Toucan ART Training — Qwen3-32B (4-bit)")
        print(f"{'='*60}")
        print(f"Model: Qwen/Qwen3-32B (bf16)")
        print(f"Tasks/iter: {TASKS_PER_ITER}, Group: {GROUP_SIZE}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Rollouts/iter: {rollouts_per_iter}")
        print(f"Total rollouts: {rollouts_per_iter * NUM_ITERATIONS}")
        print(f"{'='*60}")

        model = art.TrainableModel(
            name="qwen3-32b-toucan",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-32B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(
                    max_seq_length=4096,
                    gpu_memory_utilization=GPU_MEM_UTIL,
                    load_in_4bit=False,
                ),
                engine_args=art.dev.EngineArgs(
                    max_model_len=4096,
                    gpu_memory_utilization=GPU_MEM_UTIL,
                ),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )

        backend = LocalBackend(in_process=True, path="./.art-toucan-32b")
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
                        toucan_rollout(model, sample) for _ in range(GROUP_SIZE)
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
                "dataset": "toucan",
                "base_model": "Qwen/Qwen3-32B",
                "quantization": "4bit",
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
        print(f"Toucan 32B Training Complete")
        print(f"{'='*60}")
        print(f"Reward curve: {[f'{r:.3f}' for r in reward_history]}")
        print(f"Full match:   {[f'{r:.1%}' for r in full_match_history]}")
        print(f"Total time:   {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"{'='*60}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
