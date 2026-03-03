"""Run Agent Lightning VERL training on tau-bench (mini version for PoC)."""
import os
import sys

os.environ.setdefault("TAU2_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY environment variable must be set")
# Force vLLM V0 engine to avoid compatibility issues with VERL's worker integration
os.environ["VLLM_USE_V1"] = "0"


def main():
    import json
    import time
    import logging

    logging.basicConfig(level=logging.INFO)
    for name in ["LiteLLM", "litellm", "httpx"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    import agentlightning as agl
    from agentlightning.adapter.triplet import LlmProxyTraceToTriplet

    from src.agent.react_agent import ReActConfig
    from src.agent.tau_adapter import TauBenchRolloutEnv, get_task_ids
    from src.eval.reward import binary_reward

    logger = logging.getLogger(__name__)

    # Training parameters — batch_size must be <= NUM_TASKS
    # ppo_mini_batch_size / n_gpus must be divisible by ppo_micro_batch_size_per_gpu
    NUM_TASKS = 32
    N_RUNNERS = 4
    BATCH_SIZE = 32

    print("=== Agent Lightning Training ===")

    # Build VERL config
    verl_config = {
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {
            "train_batch_size": BATCH_SIZE,
            "max_prompt_length": 4096,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen3-4B",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "ppo_mini_batch_size": BATCH_SIZE,
                "ppo_micro_batch_size_per_gpu": BATCH_SIZE // 8,
                "optim": {"lr": 1e-6},
                "use_kl_loss": False,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                },
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": 0.5,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 8,
                "fsdp_config": {"param_offload": True},
            },
        },
        "trainer": {
            "n_gpus_per_node": 8,
            "nnodes": 1,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console"],
            "save_freq": 16,
            "test_freq": 8,
            "total_epochs": 1,
        },
        "agentlightning": {
            "trace_aggregator": {
                "level": "trajectory",
                "trajectory_max_prompt_length": 4096,
                "trajectory_max_response_length": 8192,
            },
        },
    }

    # Create the LitAgent with proper proxy URL routing
    # The rollout runs synchronously using the OpenAI sync client and
    # tau2's AgentGymEnv (which is also sync). This avoids the
    # "asyncio.run() cannot be called from a running event loop" error
    # since AGL runners may already be in an async context.
    class TauBenchLitAgent(agl.LitAgent):
        def rollout(self, task, resources, rollout):
            import json as _json

            llm = resources["main_llm"]

            # Use the properly routed proxy URL with rollout/attempt context
            rollout_id = rollout.rollout_id
            attempt_id = getattr(rollout, "attempt", None)
            if attempt_id is not None:
                attempt_id = attempt_id.attempt_id if hasattr(attempt_id, "attempt_id") else str(attempt_id)

            if hasattr(llm, "get_base_url") and attempt_id:
                base_url = llm.get_base_url(rollout_id, attempt_id)
            elif hasattr(llm, "get_base_url"):
                base_url = llm.get_base_url()
            else:
                base_url = llm.endpoint

            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=getattr(llm, "api_key", None) or "dummy-key",
            )
            model_name = getattr(llm, "model", None) or "default"
            task_id = task.get("task_id", "0")

            try:
                import re
                from tau2.gym.gym_agent import AgentGymEnv
                from src.agent.tau_adapter import _tau2_tools_to_openai

                gym_env = AgentGymEnv(
                    domain="retail",
                    task_id=str(task_id),
                    max_steps=200,
                    user_llm="gpt-4",
                )
                obs, info = gym_env.reset()
                tools_openai = _tau2_tools_to_openai(info.get("tools", []))
                policy = info.get("policy", "")

                # Build tool descriptions for the system prompt
                # (VERL's internal vLLM doesn't support --enable-auto-tool-choice)
                tool_desc = ""
                for t in tools_openai:
                    fn = t.get("function", t)
                    tool_desc += f"\n- {fn['name']}: {fn.get('description', '')}"
                    if fn.get("parameters", {}).get("properties"):
                        params = fn["parameters"]["properties"]
                        param_str = ", ".join(f"{k}: {v.get('type','')}" for k, v in params.items())
                        tool_desc += f" ({param_str})"

                system_prompt = (
                    f"You are a helpful retail customer service agent.\n\n"
                    f"<policy>\n{policy}\n</policy>\n\n"
                    f"Available tools:{tool_desc}\n\n"
                    f"To call a tool, output EXACTLY this format on its own line:\n"
                    f'<tool_call>{{"name": "tool_name", "arguments": {{"key": "value"}}}}</tool_call>\n'
                    f"After the tool result, continue the conversation.\n"
                    f"When done helping the customer, just respond normally without tool calls."
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs},
                ]

                terminated = False
                truncated = False
                reward = 0.0
                num_turns = 0

                tool_call_pattern = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

                while not terminated and not truncated and num_turns < 15:
                    num_turns += 1

                    # No tools= param — VERL's vLLM doesn't support it
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1024,
                    )

                    choice = response.choices[0]
                    content = choice.message.content or ""
                    messages.append({"role": "assistant", "content": content})

                    # Parse tool calls from the text output
                    tool_matches = tool_call_pattern.findall(content)
                    if tool_matches:
                        for match in tool_matches:
                            try:
                                tc = _json.loads(match)
                                action = _json.dumps({"name": tc["name"], "arguments": tc.get("arguments", {})})
                                obs, reward, terminated, truncated, _ = gym_env.step(action)
                                if obs:
                                    messages.append({"role": "user", "content": f"Tool result: {obs}"})
                                if terminated or truncated:
                                    break
                            except (_json.JSONDecodeError, KeyError):
                                pass
                    else:
                        # Plain text response to user
                        # Strip any <think> tags from the response for the gym env
                        clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        if clean:
                            obs, reward, terminated, truncated, _ = gym_env.step(clean)
                            if not terminated and not truncated and obs:
                                messages.append({"role": "user", "content": obs})

                final_reward = binary_reward(reward)
                logger.info(f"AGL rollout task={task_id} reward={final_reward} turns={num_turns}")
                return final_reward

            except Exception as e:
                logger.error(f"AGL rollout failed for task={task_id}: {e}", exc_info=True)
                return 0.0

    agent = TauBenchLitAgent()

    # Build dataset
    train_ids = get_task_ids("retail", "train")[:NUM_TASKS]
    train_dataset = [{"task_id": tid} for tid in train_ids]
    val_dataset = [{"task_id": tid} for tid in get_task_ids("retail", "test")[:3]]

    print(f"Training tasks: {len(train_dataset)}, Val tasks: {len(val_dataset)}")
    print(f"N runners: {N_RUNNERS}, Batch size: {BATCH_SIZE}")

    # Create trainer with LlmProxyTraceToTriplet adapter
    # The proxy generates 'litellm_request'/'raw_gen_ai_request' spans,
    # NOT 'openai.chat.completion', so we need the matching adapter
    trainer = agl.Trainer(
        algorithm=agl.VERL(verl_config),
        n_runners=N_RUNNERS,
        adapter=LlmProxyTraceToTriplet(),
        strategy="cs",
    )

    start_time = time.time()

    # Run training
    trainer.fit(
        agent=agent,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    total_time = time.time() - start_time

    # Save results
    results = {
        "framework": "agl",
        "base_model": "Qwen/Qwen3-4B",
        "total_epochs": 1,
        "train_batch_size": BATCH_SIZE,
        "n_runners": N_RUNNERS,
        "learning_rate": 1e-6,
        "total_time_seconds": total_time,
        "num_train_tasks": len(train_dataset),
        "status": "completed",
    }

    os.makedirs("results/agl", exist_ok=True)
    with open("results/agl/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== AGL Training Complete ===")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
