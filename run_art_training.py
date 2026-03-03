"""Run ART GRPO training on tau-bench (mini version for PoC)."""
import os
import sys

os.environ.setdefault("TAU2_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")
# Force vLLM V0 engine to avoid multiprocessing issues
os.environ["VLLM_USE_V1"] = "0"


def main():
    import asyncio
    import json
    import time
    import logging

    logging.basicConfig(level=logging.INFO)
    for name in ["LiteLLM", "litellm", "httpx"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import art
    from art.local.backend import LocalBackend

    from src.agent.react_agent import ReActConfig
    from src.agent.tau_adapter import TauBenchRolloutEnv, get_task_ids
    from src.eval.reward import binary_reward

    logger = logging.getLogger(__name__)

    # Mini training parameters
    NUM_ITERATIONS = 3
    NUM_TASKS = 5
    GROUP_SIZE = 4
    LEARNING_RATE = 1e-5

    async def run_art_rollout(model, task_id: str) -> "art.Trajectory":
        """Run a single rollout and return an ART Trajectory."""
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

        episode = await rollout_env.run_episode(task_id=task_id)
        reward = binary_reward(episode.reward)

        # Build ART trajectory from messages and choices
        messages_and_choices = []
        choice_idx = 0
        for msg in episode.react_result.messages:
            if msg.get("role") == "assistant":
                if choice_idx < len(episode.react_result.choices):
                    messages_and_choices.append(episode.react_result.choices[choice_idx])
                    choice_idx += 1
                else:
                    messages_and_choices.append(msg)
            else:
                messages_and_choices.append(msg)

        trajectory = art.Trajectory(messages_and_choices=messages_and_choices)
        trajectory.reward = reward
        return trajectory

    async def run():
        print(f"=== ART Training ===")
        print(f"Iterations: {NUM_ITERATIONS}, Tasks: {NUM_TASKS}, Group size: {GROUP_SIZE}")

        # Create and register model
        model = art.TrainableModel(
            name="qwen3-4b-taubench",
            project="agentic-rl-poc",
            base_model="Qwen/Qwen3-4B",
            _internal_config=art.dev.InternalModelConfig(
                init_args=art.dev.InitArgs(gpu_memory_utilization=0.5),
                peft_args=art.dev.PeftArgs(lora_alpha=8),
                trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
            ),
        )

        backend = LocalBackend(in_process=True, path="./.art")
        await model.register(backend)
        print("Model registered with LocalBackend")

        # Get training task IDs
        train_ids = get_task_ids("retail", "train")[:NUM_TASKS]
        print(f"Training on {len(train_ids)} tasks: {train_ids}")

        reward_history = []
        timing_history = []
        start_time = time.time()

        for iteration in range(NUM_ITERATIONS):
            iter_start = time.time()
            print(f"\n--- Iteration {iteration+1}/{NUM_ITERATIONS} ---")

            # Gather trajectory groups
            train_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        run_art_rollout(model, task_id) for _ in range(GROUP_SIZE)
                    )
                    for task_id in train_ids
                ),
                pbar_desc=f"Iteration {iteration+1}",
            )

            # Compute mean reward
            rewards = []
            for group in train_groups:
                for traj in group.trajectories:
                    rewards.append(traj.reward)
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            reward_history.append(mean_reward)

            # Train
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=LEARNING_RATE),
            )

            iter_time = time.time() - iter_start
            timing_history.append(iter_time)
            print(
                f"  mean_reward={mean_reward:.3f}, "
                f"rollouts={len(rewards)}, time={iter_time:.1f}s"
            )

        total_time = time.time() - start_time

        # Save results
        results = {
            "framework": "art",
            "base_model": "Qwen/Qwen3-4B",
            "num_iterations": NUM_ITERATIONS,
            "group_size": GROUP_SIZE,
            "learning_rate": LEARNING_RATE,
            "reward_history": reward_history,
            "timing_history": timing_history,
            "total_time_seconds": total_time,
            "total_rollouts": NUM_ITERATIONS * NUM_TASKS * GROUP_SIZE,
            "final_mean_reward": reward_history[-1] if reward_history else 0.0,
        }

        os.makedirs("results/art", exist_ok=True)
        with open("results/art/training_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n=== ART Training Complete ===")
        print(f"  Iterations: {NUM_ITERATIONS}")
        print(f"  Final mean reward: {results['final_mean_reward']:.3f}")
        print(f"  Reward history: {reward_history}")
        print(f"  Total time: {total_time:.1f}s")

    asyncio.run(run())


if __name__ == "__main__":
    main()
