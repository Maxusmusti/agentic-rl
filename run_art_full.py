"""Full ART GRPO training run on tau-bench retail domain.

Uses task batching and concurrency limiting to avoid API timeouts.
"""
import os
import sys

os.environ.setdefault("TAU2_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")
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
    for name in ["LiteLLM", "litellm", "httpx", "openai", "tau2"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import art
    from art.local.backend import LocalBackend

    from src.agent.react_agent import ReActConfig
    from src.agent.tau_adapter import TauBenchRolloutEnv, get_task_ids
    from src.eval.reward import binary_reward

    logger = logging.getLogger(__name__)

    # Full training parameters
    NUM_ITERATIONS = 15
    GROUP_SIZE = 8
    TASKS_PER_ITER = 20      # Sample 20 tasks per iteration (avoids API overload)
    LEARNING_RATE = 1e-5
    GPU_MEM_UTIL = 0.45      # Keep low — ART needs headroom for training + vLLM on same GPU
    OUTPUT_DIR = "results/art_full"
    MAX_RETRIES = 3

    # Semaphore to limit concurrent rollouts (avoid GPT-4 rate limit)
    CONCURRENCY = 16
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_art_rollout(model, task_id: str) -> "art.Trajectory":
        """Run a single rollout with concurrency limiting and retry."""
        async with sem:
            openai_client = model.openai_client()
            model_name = model.get_inference_name()

            config = ReActConfig(
                model=model_name, max_turns=15, temperature=0.7, max_tokens=1024
            )
            rollout_env = TauBenchRolloutEnv(
                client=openai_client,
                domain="retail",
                model=model_name,
                user_model="gpt-4",
                config=config,
            )

            for attempt in range(MAX_RETRIES):
                try:
                    episode = await rollout_env.run_episode(task_id=task_id)
                    reward = binary_reward(episode.reward)

                    messages_and_choices = []
                    choice_idx = 0
                    for msg in episode.react_result.messages:
                        if msg.get("role") == "assistant":
                            if choice_idx < len(episode.react_result.choices):
                                messages_and_choices.append(
                                    episode.react_result.choices[choice_idx]
                                )
                                choice_idx += 1
                            else:
                                messages_and_choices.append(msg)
                        else:
                            messages_and_choices.append(msg)

                    trajectory = art.Trajectory(messages_and_choices=messages_and_choices)
                    trajectory.reward = reward
                    return trajectory

                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait = 2 ** attempt * 5
                        logger.warning(
                            f"Rollout task={task_id} attempt {attempt+1} failed: {e}. "
                            f"Retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"Rollout task={task_id} failed after {MAX_RETRIES} attempts: {e}")
                        # Return a zero-reward trajectory
                        trajectory = art.Trajectory(
                            messages_and_choices=[
                                {"role": "system", "content": "failed"},
                                {"role": "user", "content": "failed"},
                            ]
                        )
                        trajectory.reward = 0.0
                        return trajectory

    async def run():
        all_train_ids = get_task_ids("retail", "train")
        rollouts_per_iter = TASKS_PER_ITER * GROUP_SIZE

        print(f"{'='*60}")
        print(f"ART Full Training Run")
        print(f"{'='*60}")
        print(f"Total tasks available: {len(all_train_ids)}")
        print(f"Tasks per iteration: {TASKS_PER_ITER} (sampled)")
        print(f"Group size: {GROUP_SIZE}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Rollouts/iteration: {rollouts_per_iter}")
        print(f"Total rollouts: {rollouts_per_iter * NUM_ITERATIONS}")
        print(f"Concurrency limit: {CONCURRENCY}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"{'='*60}")

        model = art.TrainableModel(
            name="qwen3-4b-taubench-full",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-4B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=GPU_MEM_UTIL),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )

        backend = LocalBackend(in_process=True, path="./.art-full")
        await model.register(backend)
        logger.info("Model registered")

        current_step = await model.get_step()
        start_iteration = current_step if current_step > 0 else 0
        if start_iteration > 0:
            logger.info(f"Resuming from step {start_iteration}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        reward_history = []
        timing_history = []
        start_time = time.time()

        for iteration in range(start_iteration, NUM_ITERATIONS):
            iter_start = time.time()

            # Sample a random subset of tasks each iteration
            random.seed(42 + iteration)
            iter_tasks = random.sample(all_train_ids, min(TASKS_PER_ITER, len(all_train_ids)))

            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        run_art_rollout(model, task_id) for _ in range(GROUP_SIZE)
                    )
                    for task_id in iter_tasks
                ),
                pbar_desc=f"Iter {iteration+1}/{NUM_ITERATIONS}",
            )

            rewards = []
            for group in train_groups:
                for traj in group.trajectories:
                    rewards.append(traj.reward)
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            pass_rate = sum(1 for r in rewards if r >= 1.0) / len(rewards) if rewards else 0.0
            reward_history.append(mean_reward)

            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=LEARNING_RATE),
            )

            iter_time = time.time() - iter_start
            timing_history.append(iter_time)
            elapsed = time.time() - start_time

            logger.info(
                f"Iter {iteration+1}/{NUM_ITERATIONS}: "
                f"mean_reward={mean_reward:.3f} pass_rate={pass_rate:.3f} "
                f"rollouts={len(rewards)} iter_time={iter_time:.0f}s "
                f"elapsed={elapsed:.0f}s ({elapsed/3600:.1f}h)"
            )

            # Save results every iteration
            results = {
                "framework": "art",
                "base_model": "Qwen/Qwen3-4B",
                "num_iterations": iteration + 1,
                "total_planned_iterations": NUM_ITERATIONS,
                "group_size": GROUP_SIZE,
                "tasks_per_iter": TASKS_PER_ITER,
                "total_train_tasks": len(all_train_ids),
                "learning_rate": LEARNING_RATE,
                "concurrency": CONCURRENCY,
                "reward_history": reward_history,
                "timing_history": timing_history,
                "total_time_seconds": elapsed,
                "total_rollouts": (iteration + 1) * rollouts_per_iter,
                "final_mean_reward": mean_reward,
            }
            with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
                json.dump(results, f, indent=2)

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"ART Full Training Complete")
        print(f"{'='*60}")
        print(f"Iterations: {NUM_ITERATIONS}")
        print(f"Total rollouts: {NUM_ITERATIONS * rollouts_per_iter}")
        print(f"Final mean reward: {reward_history[-1]:.3f}")
        print(f"Reward curve: {[f'{r:.3f}' for r in reward_history]}")
        print(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
        print(f"{'='*60}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
