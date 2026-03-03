# Our ART Training Loop: Step-by-Step Breakdown

This document walks through exactly what happens when we run `run_art_full.py` — from process startup through every rollout, training step, and checkpoint.

## High-Level Loop

```
for each iteration (15 total):
    1. Sample 20 random tasks from the 74-task retail train split
    2. For each task, run 8 rollouts concurrently (160 rollouts per iteration)
       - Each rollout: Qwen3-4B plays a customer service agent on tau-bench
       - GPT-4 plays the customer (user simulator)
       - tau-bench evaluates: pass (reward=1.0) or fail (reward=0.0)
    3. Feed all 160 trajectories to ART for one GRPO gradient step
    4. Save checkpoint, update vLLM with new LoRA weights
    5. Repeat with the updated model
```

## Detailed Walkthrough

### Phase 0: Process Startup (lines 1-49)

```python
os.environ["VLLM_USE_V1"] = "0"  # Force vLLM V0 engine (V1 has multiprocessing issues)
```

Before anything runs, we:
- Set `TAU2_DATA_DIR` so tau-bench can find its task definitions and databases
- Require `OPENAI_API_KEY` (GPT-4 is used for the user simulator)
- Force vLLM to use the V0 engine to avoid spawn-related crashes
- Create an `asyncio.Semaphore(16)` to cap concurrent rollouts at 16 (prevents GPT-4 API rate limit timeouts)

### Phase 1: Model Registration (lines 128-146)

```python
model = art.TrainableModel(
    name="qwen3-4b-taubench-full",
    project="agentic-rl-poc",
    base_model="Qwen/Qwen3-4B",
    _internal_config=art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(gpu_memory_utilization=0.45),
        peft_args=art.dev.PeftArgs(lora_alpha=8),
        trainer_args=art.dev.TrainerArgs(max_grad_norm=0.1),
    ),
)
backend = LocalBackend(in_process=True, path="./.art-full")
await model.register(backend)
```

**What happens under the hood:**

```
model.register(backend)
│
├─ 1. backend.register(model)
│     └─ Creates .art-full/agentic-rl-poc/models/qwen3-4b-taubench-full/
│        Writes model.json with metadata
│
├─ 2. backend._prepare_backend_for_training(model)
│     │
│     ├─ 2a. Creates UnslothService (first time only)
│     │      ├─ unsloth.FastLanguageModel.from_pretrained("Qwen/Qwen3-4B")
│     │      │  → Downloads model from HuggingFace (or loads from cache)
│     │      │  → Loads 4B parameter model onto GPU (~7.6 GiB)
│     │      │
│     │      ├─ unsloth.FastLanguageModel.get_peft_model(model,
│     │      │      r=16, lora_alpha=8, target_modules=[...])
│     │      │  → Wraps model with LoRA adapters
│     │      │  → Only LoRA matrices are trainable (~0.1% of total params)
│     │      │
│     │      ├─ GRPOTrainer(model, dataset=dummy_10M_rows)
│     │      │  → Creates HuggingFace Trainer with a fake dataset
│     │      │  → Monkey-patches _prepare_inputs() to read from an async queue
│     │      │     (we'll feed real data through this queue during training)
│     │      │
│     │      └─ trainer.create_optimizer() → AdamW(weight_decay=0.1)
│     │
│     ├─ 2b. service.start_openai_server()
│     │      ├─ Finds or creates initial LoRA checkpoint (step 0000)
│     │      ├─ offload_to_cpu()
│     │      │  → Copies all LoRA weights + AdamW optimizer states
│     │      │    to pinned CPU memory using non_blocking transfers
│     │      │  → Frees GPU memory
│     │      │
│     │      └─ Starts vLLM AsyncLLM server:
│     │         - Loads base Qwen3-4B weights onto GPU
│     │         - enable_lora=True, enable_sleep_mode=True
│     │         - Registers LoRA adapter from checkpoint
│     │         - Starts OpenAI-compatible HTTP server on a random port
│     │         - enable_auto_tool_choice + hermes tool parser
│     │
│     └─ Returns (base_url="http://localhost:PORT/v1", api_key)
│
└─ 3. Sets model.inference_base_url, model.inference_model_name
       → model.openai_client() will now point at local vLLM
       → model.get_inference_name() returns "qwen3-4b-taubench-full@0"
```

**State after registration:**
- GPU: vLLM serving Qwen3-4B + LoRA adapter (inference mode)
- CPU: LoRA weights + AdamW optimizer states in pinned memory (ready for training)

### Phase 2: Resume Check (lines 143-146)

```python
current_step = await model.get_step()
start_iteration = current_step if current_step > 0 else 0
```

ART scans `.art-full/.../checkpoints/` for the highest-numbered subdirectory. If we crashed at iteration 5, it finds checkpoint `0005` and starts from iteration 5. The LoRA weights from that checkpoint are already loaded.

### Phase 3: The Training Loop (lines 153-212)

Each iteration has 4 sub-phases:

#### 3A. Task Sampling (lines 156-158)

```python
random.seed(42 + iteration)
iter_tasks = random.sample(all_train_ids, min(TASKS_PER_ITER, len(all_train_ids)))
```

Randomly selects 20 tasks from the 74 available retail training tasks. The seed is deterministic per iteration so results are reproducible. Each iteration sees a different subset, ensuring the model trains on diverse scenarios over time.

#### 3B. Rollout Collection (lines 160-168)

```python
train_groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(
            run_art_rollout(model, task_id) for _ in range(GROUP_SIZE)
        )
        for task_id in iter_tasks
    ),
    pbar_desc=f"Iter {iteration+1}/{NUM_ITERATIONS}",
)
```

This creates 20 TrajectoryGroups (one per task), each with 8 rollouts = **160 concurrent rollout coroutines**. The `asyncio.Semaphore(16)` ensures at most 16 run simultaneously.

**Inside each `run_art_rollout(model, task_id)` (lines 51-109):**

```
run_art_rollout(model, task_id="42")
│
├─ 1. Acquire semaphore (max 16 concurrent)
│
├─ 2. Get inference client
│     client = model.openai_client()     → AsyncOpenAI pointed at local vLLM
│     model_name = model.get_inference_name()  → "qwen3-4b-taubench-full@3"
│
├─ 3. Create TauBenchRolloutEnv
│     └─ Wraps tau-bench's AgentGymEnv for step-by-step control
│
├─ 4. Run episode (tau_adapter.py lines 72-221)
│     │
│     ├─ 4a. gym_env = AgentGymEnv(domain="retail", task_id="42", user_llm="gpt-4")
│     │       └─ tau-bench creates:
│     │          - A retail Environment with 17 tools (get_order, cancel_order, etc.)
│     │          - A retail database (users, orders, products)
│     │          - A GPT-4 user simulator with the customer's scenario
│     │          - An evaluation criteria (expected actions, DB state, info to communicate)
│     │
│     ├─ 4b. obs, info = gym_env.reset()
│     │       └─ tau-bench orchestrator:
│     │          - Agent sends greeting: "Hi! How can I help you today?"
│     │          - GPT-4 user simulator responds with the customer's request
│     │          - Returns observation = the customer's first message
│     │          - info contains: 17 tool schemas, the retail policy text
│     │
│     ├─ 4c. Build system prompt from retail policy (~6700 chars)
│     │       messages = [
│     │           {"role": "system", "content": "You are a helpful retail agent...<policy>...</policy>"},
│     │           {"role": "user", "content": "Hi, I want to return my order #W2378156..."}
│     │       ]
│     │
│     ├─ 4d. AGENT LOOP (up to 15 turns):
│     │       │
│     │       ├─ Call Qwen3-4B via vLLM:
│     │       │   response = await client.chat.completions.create(
│     │       │       model="qwen3-4b-taubench-full@3",
│     │       │       messages=messages,
│     │       │       tools=tools_openai,          ← 17 tau-bench tools in OpenAI format
│     │       │       temperature=0.7,
│     │       │       max_tokens=1024,
│     │       │   )
│     │       │   → vLLM runs Qwen3-4B with LoRA adapter
│     │       │   → Hermes tool parser extracts tool calls from model output
│     │       │   → Returns Choice object with per-token logprobs
│     │       │
│     │       ├─ SAVE the raw Choice object (carries logprobs for training)
│     │       │   all_choices.append(response.choices[0])
│     │       │
│     │       ├─ If model made a TOOL CALL:
│     │       │   │ e.g., get_order_details(order_id="W2378156")
│     │       │   │
│     │       │   ├─ Format as JSON and send to gym_env.step(action)
│     │       │   │   → tau-bench executes the tool against the retail DB
│     │       │   │   → Returns tool result as observation string
│     │       │   │
│     │       │   ├─ Append assistant message (with tool_calls) to messages
│     │       │   └─ Append tool result message to messages
│     │       │
│     │       └─ If model sent a TEXT RESPONSE:
│     │           │ e.g., "Your order has been cancelled. Is there anything else?"
│     │           │
│     │           ├─ Send to gym_env.step(content)
│     │           │   → tau-bench forwards to GPT-4 user simulator
│     │           │   → User either responds or ends conversation
│     │           │
│     │           ├─ Append assistant message to messages
│     │           └─ Append user response to messages (if conversation continues)
│     │
│     │       Loop continues until:
│     │       - terminated=True (agent or user ended normally)
│     │       - truncated=True (hit max steps)
│     │       - num_turns >= 15 (our max)
│     │
│     ├─ 4e. tau-bench EVALUATES the episode:
│     │       Checks against EvaluationCriteria:
│     │       - Did the agent make the correct tool calls? (action matching)
│     │       - Is the database in the expected state? (DB hash comparison)
│     │       - Did the agent communicate required info? (LLM-judged)
│     │       → reward = product of applicable criteria (0.0 or 1.0 for binary)
│     │
│     └─ 4f. Returns EpisodeResult:
│             - messages: full conversation history (list of dicts)
│             - choices: raw Choice objects from every Qwen3-4B call (with logprobs)
│             - reward: 0.0 (fail) or 1.0 (pass)
│             - num_turns: how many LLM calls were made
│
├─ 5. Apply binary reward function
│     reward = binary_reward(episode.reward)  → 0.0 or 1.0
│
├─ 6. BUILD ART TRAJECTORY (lines 73-88)
│     │
│     │  Interleave messages (context) with Choice objects (trainable):
│     │
│     │  messages_and_choices = [
│     │      {"role": "system", "content": "..."},        ← dict (not trainable)
│     │      {"role": "user", "content": "Hi, I want..."}, ← dict (not trainable)
│     │      choices[0],                                    ← Choice (TRAINABLE, has logprobs)
│     │      {"role": "tool", "content": "{order...}"},    ← dict (not trainable)
│     │      choices[1],                                    ← Choice (TRAINABLE, has logprobs)
│     │      {"role": "user", "content": "Yes, please"},   ← dict (not trainable)
│     │      choices[2],                                    ← Choice (TRAINABLE, has logprobs)
│     │  ]
│     │
│     │  The Choice objects are the model's actual outputs with token-level logprobs.
│     │  ART will compute gradients only on these turns.
│     │  Everything else (system prompt, user messages, tool results) is context.
│     │
│     trajectory = art.Trajectory(messages_and_choices=messages_and_choices)
│     trajectory.reward = reward  # 0.0 or 1.0
│
└─ 7. Release semaphore, return trajectory
```

**Example of a single rollout's conversation:**

```
Turn 1: [Qwen3-4B] → "I need to verify your identity. What's your email?"          (Choice, logprobs saved)
Turn 2: [GPT-4 user] → "Sure, it's john@example.com"
Turn 3: [Qwen3-4B] → tool_call: find_user_id_by_email("john@example.com")          (Choice, logprobs saved)
         [tau-bench] → tool_result: "john_doe_123"
Turn 4: [Qwen3-4B] → tool_call: get_order_details("W2378156")                      (Choice, logprobs saved)
         [tau-bench] → tool_result: {"status": "delivered", "items": [...]}
Turn 5: [Qwen3-4B] → tool_call: return_delivered_order_items("W2378156", [...])     (Choice, logprobs saved)
         [tau-bench] → tool_result: {"status": "return_initiated"}
Turn 6: [Qwen3-4B] → "Your return has been initiated! Anything else?"               (Choice, logprobs saved)
Turn 7: [GPT-4 user] → stops conversation

tau-bench evaluation: DB state matches expected → reward = 1.0
```

#### 3C. GRPO Training Step (lines 178-181)

```python
await model.train(
    train_groups,
    config=art.TrainConfig(learning_rate=LEARNING_RATE),
)
```

**What happens under the hood:**

```
model.train(train_groups, config)
│
├─ 1. TOKENIZATION & PACKING (on CPU)
│     │
│     │  For each of the 20 TrajectoryGroups:
│     │    For each of the 8 Trajectories in the group:
│     │      ├─ tokenizer.apply_chat_template(messages) → token IDs
│     │      ├─ Mark assistant turns with assistant_mask=True
│     │      ├─ Extract old_logprobs from Choice.logprobs
│     │      └─ Compute GRPO advantage:
│     │           advantage = reward - mean(group_rewards)
│     │           e.g., if group rewards are [1,0,0,1,0,0,1,0]:
│     │             mean = 0.375
│     │             pass trajectories get advantage = +0.625
│     │             fail trajectories get advantage = -0.375
│     │           If all rewards are the same (all pass or all fail):
│     │             advantages are all 0 → group is skipped (nothing to learn)
│     │
│     │  Pack trajectories into fixed-length sequences:
│     │    Multiple conversations concatenated into [B, S] tensors
│     │    with group_ids preventing cross-attention between different conversations
│     │
│     └─ Write packed tensors to disk (memory-mapped)
│
├─ 2. GPU SWAP: vLLM → TRAINING
│     ├─ llm.pause_generation()        → stop accepting inference requests
│     ├─ vLLM workers do_sleep()       → back up base model weights to CPU
│     │                                   unmap GPU memory allocations
│     └─ training_state.reload_to_gpu() → copy LoRA + optimizer from CPU to GPU
│
├─ 3. GRADIENT STEP
│     │
│     │  For each packed batch:
│     │    ├─ Forward pass through Qwen3-4B + LoRA:
│     │    │   hidden_states = model(input_ids, attention_bias)
│     │    │
│     │    ├─ Compute new logprobs (chunked, 1024 tokens at a time):
│     │    │   logits = hidden_states @ lm_head_weights.T
│     │    │   new_logprobs = log_softmax(logits)
│     │    │
│     │    ├─ Importance sampling ratio:
│     │    │   ratio = exp(new_logprobs - old_logprobs)
│     │    │   (old_logprobs from when the rollout was generated)
│     │    │
│     │    ├─ CISPO loss (default, not PPO):
│     │    │   clipped_ratio = clip(ratio.detach(), 0.0, 5.0)
│     │    │   loss = -(clipped_ratio * advantages * new_logprobs)
│     │    │   │
│     │    │   │ Only applied where assistant_mask=True (model's own outputs)
│     │    │   │ Gradient flows through new_logprobs only, not ratio
│     │    │   │
│     │    │   │ Effect: trajectories with reward=1 (advantage > 0)
│     │    │   │   → loss pushes model to INCREASE logprobs of these tokens
│     │    │   │ Trajectories with reward=0 (advantage < 0)
│     │    │   │   → loss pushes model to DECREASE logprobs of these tokens
│     │    │
│     │    ├─ loss.backward()
│     │    ├─ clip_grad_norm_(max_norm=0.1)
│     │    └─ AdamW.step(lr=1e-5)
│     │         → Updates only the LoRA weight matrices
│     │         → Base Qwen3-4B weights remain frozen
│     │
│     └─ Logs metrics: policy_loss, entropy, learning_rate
│
├─ 4. SAVE CHECKPOINT
│     └─ trainer.save_model(checkpoints/0004/)
│        → adapter_config.json + adapter_model.safetensors (~50MB)
│
├─ 5. GPU SWAP: TRAINING → vLLM
│     ├─ training_state.offload_to_cpu() → LoRA + optimizer back to pinned CPU memory
│     ├─ vLLM workers do_wake_up()       → restore base model weights to GPU
│     ├─ llm.add_lora(LoRARequest(       → load new checkpoint as LoRA adapter
│     │      name="qwen3-4b-taubench-full@4",
│     │      path="checkpoints/0004/"))
│     └─ llm.resume_generation()         → vLLM ready for next iteration's rollouts
│
└─ Returns LocalTrainResult(step=4, metrics={...}, checkpoint_path="checkpoints/0004/")
```

#### 3D. Logging & Checkpointing (lines 183-212)

After each iteration:
- Compute mean reward and pass rate across all 160 rollouts
- Append to reward_history
- Save full results JSON to `results/art_full/training_results.json`
- This enables monitoring progress while training runs, and resuming if the process crashes

### Phase 4: Post-Training Evaluation

After all iterations complete (or we stop early), we run `run_post_eval.py`:

```
1. Start vLLM with: --model Qwen/Qwen3-4B --enable-lora
     --lora-modules art-full=.art-full/.../checkpoints/0006
   → Serves the base model with the best LoRA checkpoint

2. Run 10 test tasks × 4 trials each (same as baseline)
   → Uses the SAME tau-bench test split (disjoint from training)
   → Same GPT-4 user simulator, same evaluation criteria

3. Compare pass@1 and pass@4 against baseline
```

## Concrete Numbers From Our Run

```
Iteration 1:  mean_reward=0.200  (32/160 rollouts passed)  — 0.8h
Iteration 2:  mean_reward=0.069  (11/160 passed)           — new task sample, model adjusting
Iteration 3:  mean_reward=0.175  (28/160 passed)           — recovering
Iteration 4:  mean_reward=0.188  (30/160 passed)           — climbing
Iteration 5:  mean_reward=0.206  (33/160 passed)           — climbing
Iteration 6:  mean_reward=0.294  (47/160 passed)           — peak training reward
Iteration 7:  mean_reward=0.138  (22/160 passed)           — variance from new task sample
```

**Post-training eval (test split, 10 tasks × 4 trials):**
- Baseline:  pass@1=0.175, pass@4=0.279
- After 7 iterations: pass@1=0.325, pass@4=0.529

The pass@4 improvement from 0.279 to 0.529 (+90%) means the model went from solving ~28% of test tasks (at least once in 4 tries) to solving ~53%.

## What Gets Learned

The LoRA adapter learns to:
1. **Authenticate users correctly** — ask for email or name+zip before doing anything
2. **Use the right tools** — call `find_user_id_by_email` not `get_user_details` first
3. **Follow the policy** — check order status before attempting modifications
4. **Handle edge cases** — ask for confirmation before irreversible actions
5. **Communicate required information** — tell the user what happened after each action

All of this is learned from binary reward signals (pass/fail on tau-bench evaluation criteria) — no human demonstrations, no prompt engineering of the RL training itself. The model discovers the correct behavior through trial and error, reinforced by GRPO.
