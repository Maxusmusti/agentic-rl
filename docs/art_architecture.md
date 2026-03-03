# ART Architecture: Complete Breakdown

## Overview

ART (by OpenPipe) runs everything on a **single GPU** by time-sharing between inference (vLLM) and training (Unsloth/PEFT). Only one occupies GPU memory at a time — they swap via a sleep/wake mechanism using pinned CPU memory.

## Components

```
┌──────────────────────────────────────────────────────────┐
│                     User Code                            │
│  model = TrainableModel(base_model="Qwen/Qwen3-4B")     │
│  await model.register(LocalBackend())                    │
│  client = model.openai_client()                          │
│  ... rollouts via client ...                             │
│  await backend.train(model, groups)                      │
└──────────────┬───────────────────────┬───────────────────┘
               │                       │
       ┌───────▼───────┐       ┌───────▼───────┐
       │  LocalBackend  │       │ UnslothService │
       │  Tokenization  │◄─────│  GPU Scheduler  │
       │  Packing       │       │  Sleep/Wake     │
       │  Checkpoints   │       └───┬───────┬─────┘
       └───────────────┘           │       │
                              ┌────▼──┐ ┌──▼────────┐
                              │ vLLM  │ │  Unsloth   │
                              │Server │ │ GRPOTrainer│
                              │(infer)│ │  (train)   │
                              └───────┘ └────────────┘
                                 ▲           ▲
                                 └─── GPU ───┘
                              (only one at a time)
```

## Step-by-Step Flow

### 1. `model.register(backend)` — Initialization

```
model.register(backend)
  └─> backend._prepare_backend_for_training(model)
      ├─ UnslothService created (lazily):
      │   ├─ unsloth.FastLanguageModel.from_pretrained("Qwen/Qwen3-4B")
      │   │   → loads base model weights to GPU (~7.6 GiB)
      │   ├─ unsloth.FastLanguageModel.get_peft_model(model, r=16, lora_alpha=8, ...)
      │   │   → wraps with LoRA adapters (only ~0.1% of params trainable)
      │   ├─ GRPOTrainer(model, dummy_dataset_of_10M_rows, ...)
      │   │   → monkey-patches _prepare_inputs() to read from async queue
      │   └─ trainer.create_optimizer()  → AdamW (weight_decay=0.1)
      │
      ├─ service.start_openai_server():
      │   ├─ Saves initial LoRA checkpoint to disk (step 0000)
      │   ├─ offload_to_cpu() → copies all PEFT params + optimizer states
      │   │   to pinned CPU memory, frees GPU
      │   └─ Starts vLLM AsyncLLM with enable_lora=True, enable_sleep_mode=True
      │       → loads base model weights on GPU, registers LoRA from checkpoint
      │
      └─ Returns (base_url="http://localhost:{port}/v1", api_key)
```

After register: vLLM is serving on GPU, training model is parked on CPU.

### 2. `model.openai_client()` — Getting an Inference Client

Returns an `AsyncOpenAI` client pointed at the local vLLM server. The model name is `"{name}@{step}"` (e.g., `"qwen3-4b-taubench@3"`), which routes to the latest LoRA adapter in vLLM.

Config: 1200s request timeout, 100K max connections.

### 3. Rollouts — Running Episodes

User code calls `client.chat.completions.create(model=model.get_inference_name(), ...)`. This hits vLLM which serves the base model + LoRA adapter. The returned `Choice` objects carry **per-token logprobs** — these become the "old logprobs" for importance sampling during training.

### 4. `Trajectory` Construction

```python
Trajectory(
    messages_and_choices=[
        {"role": "system", "content": "..."},     # dict → not trainable
        {"role": "user", "content": "..."},        # dict → not trainable
        response.choices[0],                        # Choice → trainable (has logprobs)
        {"role": "tool", "content": "..."},        # dict → not trainable
        response2.choices[0],                       # Choice → trainable
    ],
    reward=1.0,  # scalar from environment
)
```

The key distinction: **dicts are context, Choice objects are trainable outputs**. The Choice objects carry the logprobs from when they were generated, enabling importance sampling ratio computation.

### 5. `TrajectoryGroup` — GRPO Grouping

```python
group = await TrajectoryGroup([
    rollout(prompt) for _ in range(8)  # 8 rollouts of same scenario
])
```

The constructor detects awaitables and resolves them concurrently via `asyncio.as_completed`. GRPO needs multiple rollouts of the same prompt to compute relative advantages (rewards normalized within the group).

### 6. `gather_trajectory_groups()` — Concurrent Collection

Runs all groups concurrently with a tqdm progress bar showing running reward average, completion tokens, and exception count. Configurable `max_exceptions` for fault tolerance.

### 7. `backend.train(model, groups)` — The Training Step

This is the most complex part. Four phases:

#### Phase A: Tokenization & Packing (on CPU)

```
For each TrajectoryGroup:
  For each Trajectory:
    tokenizer.apply_chat_template(messages) → token IDs
    Mark assistant turns (from Choice objects) with assistant_mask=True
    Record old_logprobs from Choice.logprobs
    Compute advantage = reward - group_mean_reward  (GRPO)

Pack multiple trajectories into fixed-length sequences [B, S]:
  - tokens:         [B, S]  input IDs
  - group_ids:      [B, S]  which GRPO group each token belongs to
  - parent_ids:     [B, S]  shared prefix identification
  - assistant_mask: [B, S]  which tokens are trainable (1=assistant, 0=context)
  - logprobs:       [B, S]  old logprobs from sampling time
  - advantages:     [B, S]  per-token advantages (same for all tokens in a turn)
  - weights:        [B, S]  per-token weights
```

#### Phase B: GPU Swap — vLLM → Training (the sleep/wake)

```
1. llm.pause_generation()          → stop accepting new requests
2. run_on_workers(llm, do_sleep)   → on each vLLM worker:
   - Back up base model weights to CPU pinned memory
   - Back up KV cache to CPU (level 1) or discard (level 2)
   - Unmap GPU memory allocations via CuMemAllocator
3. state.reload_to_gpu()           → copy PEFT params + optimizer from
                                     pinned CPU memory back to GPU
```

#### Phase C: GRPO Gradient Step (on GPU)

```
For each packed sequence batch:
  1. Feed batch to GRPOTrainer via async queue
  2. Trainer calls patched compute_loss():
     a. Forward pass: hidden_states = model(input_ids, attention_bias)
        - attention_bias is a custom causal mask that prevents
          cross-group attention within packed sequences
     b. Compute new_logprobs via chunked matmul:
        for chunk in chunks(hidden_states, size=1024):
            logits = chunk @ lm_head_weights.T
            new_logprobs = log_softmax(logits)
     c. Compute importance sampling ratio:
        ratio = exp(new_logprobs - old_logprobs)
     d. Compute loss (CISPO by default):
        loss = -(clip(ratio.detach(), 0.0, 5.0) * advantages * new_logprobs)
        # Note: gradient flows only through new_logprobs, not ratio
     e. If beta > 0: disable LoRA, forward pass again for ref_logprobs,
        kl = exp(ref - new) - (ref - new) - 1
        loss += beta * kl
  3. loss.backward()
  4. clip_grad_norm_(max_norm from TrainerArgs, default 0.1)
  5. AdamW.step()  (lr set dynamically per batch)
```

#### Phase D: Checkpoint & GPU Swap — Training → vLLM

```
1. trainer.save_model(checkpoint_dir)  → saves adapter_model.safetensors
2. state.offload_to_cpu()             → copies PEFT params + optimizer
                                        to pinned CPU, frees GPU
3. run_on_workers(llm, do_wake_up)    → restores base model weights
                                        from CPU pinned memory to GPU
4. llm.add_lora(LoRARequest(          → registers new checkpoint as a
       name="{model}@{step}",            LoRA adapter in vLLM
       path=checkpoint_dir))
5. llm.resume_generation()            → vLLM ready for inference again
```

## Loss Function Details

### Default: CISPO (Clipped IS-weight Policy Optimization)

```
loss = -(clip(ratio.detach(), 1-eps, 1+eps_high) * advantages * new_logprobs)
```

Default clipping: `epsilon=1.0`, `epsilon_high=4.0` (asymmetric — allows the ratio to go up to 5x but not below 0x).

Gradients flow **only through `new_logprobs`**, not through the ratio. This makes optimization more stable for multi-turn agent training where logprob ratios can be noisy.

### Optional: PPO Mode (`ppo=True`)

```
loss = -min(ratio * adv, clip(ratio, 1-0.2, 1+0.2) * adv)
```

Standard PPO clipped objective where gradients flow through the ratio.

### KL Penalty (`beta > 0`)

When enabled, computes reference logprobs by disabling the LoRA adapter (exposing frozen base model weights) and doing a second forward pass:

```
kl = exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1
final_loss = policy_loss + beta * kl
```

### Optimizer

AdamW from HuggingFace Transformers with:
- `weight_decay = 0.1` (RL) or `0.01` (SFT)
- Learning rate set dynamically per batch on the optimizer param groups
- Gradient clipping: `max_grad_norm = 0.1` (configurable via TrainerArgs)

## Unsloth Standby: The Sleep/Wake Mechanism

The system uses **GPU time-sharing**, not weight sharing. Only one of vLLM or the trainer occupies GPU memory at a time.

### Sleep (`do_sleep`)

Runs on vLLM workers via `run_on_workers()`:
- Accesses `CuMemAllocator` to iterate over all GPU memory allocations tagged "weights" or "kv_cache"
- **Level 1**: Backs up both weights and KV cache to CPU pinned memory, then unmaps GPU memory
- **Level 2**: Backs up weights only, discards KV cache, saves model buffers

### Wake (`do_wake_up`)

- Re-maps GPU memory allocations
- Copies backed-up data back from CPU pinned memory
- Restores buffers saved during level 2 sleep

### Why Pinned Memory?

`torch.cuda.pin_memory()` enables asynchronous CPU↔GPU transfers with `non_blocking=True`, making the swap take ~seconds instead of tens of seconds for a 4B model.

## Trajectory Packing

Multiple short trajectories are packed into single sequences (rounded to 2048-token multiples) to maximize GPU utilization:

```
Sequence: [SYS1 USR1 AST1 TOOL1 AST1 | SYS2 USR2 AST2 | PAD PAD PAD]
group_ids: [  1    1    1    1     1   |  2    2    2    |  0   0   0 ]
parent_ids:[  1    1    1    1     1   |  2    2    2    |  0   0   0 ]
asst_mask: [  0    0    1    0     1   |  0    0    1    |  0   0   0 ]
```

The attention bias uses `group_ids` and `parent_ids` to construct custom causal masks that prevent cross-group attention leakage while allowing shared prefix computation within a GRPO group.

## Checkpoint Format

```
.art/project/models/name/checkpoints/
├── 0000/
│   ├── adapter_config.json       # LoRA hyperparams (r, alpha, target_modules)
│   └── adapter_model.safetensors # LoRA weight matrices (~50MB for r=16)
├── 0001/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── ...
```

Standard PEFT/HuggingFace format — compatible with `transformers.AutoModel.from_pretrained()` and vLLM's `--enable-lora`.

### Resumption

On restart, `UnslothService` checks for the latest checkpoint directory. If found, `unsloth.FastLanguageModel.from_pretrained(checkpoint_dir)` loads the model with LoRA weights already applied, skipping `get_peft_model()`.

`model.get_step()` scans the `checkpoints/` directory for the highest-numbered subdirectory.

## Dynamic LoRA Updates in vLLM

After each training step:
1. New LoRA weights saved to `checkpoints/{step:04d}/`
2. `llm.add_lora(LoRARequest(name="{model}@{step}", path=checkpoint_dir))` registers the new adapter
3. A patched `add_lora` also updates `_openai_serving_models.lora_requests` so the new adapter appears in the `/v1/models` endpoint
4. Old LoRA adapters remain loaded (vLLM pages them out as memory pressure increases)
5. `model.get_inference_name()` returns `"{name}@{latest_step}"` so subsequent rollouts use the updated model

## Key Design Decisions

**Why single-GPU?** ART's sleep/wake mechanism operates on `CuMemAllocator` which manages per-GPU memory. Multi-GPU support is planned (Issue #210) but not yet implemented.

**Why CISPO over PPO?** The default loss clips the **detached** ratio (as a weight) rather than the ratio itself. Gradients only flow through `new_logprobs`, making optimization more stable for multi-turn agent training where logprob ratios can be noisy across many turns.

**Why pack trajectories?** Packing multiple short trajectories into one sequence (with attention masking to prevent cross-contamination) maximizes GPU utilization. Group IDs and parent IDs enable shared prefix computation within a GRPO group.

**Why pinned memory?** Enables asynchronous CPU↔GPU transfers, making the sleep/wake swap fast enough (~seconds) to be practical in a training loop.

## ServerlessBackend (Alternative)

ART also offers a `ServerlessBackend` that sends trajectories to OpenPipe's cloud API for training:
- No local GPU needed for training
- Inference via OpenPipe's managed endpoints
- Checkpoints stored as W&B artifacts
- ~40% lower cost and ~28% faster than local H100 (per OpenPipe's benchmarks)

## Complete Data Flow Diagram

```
User code                    ART Framework                          GPU
─────────                    ─────────────                          ───

1. model = TrainableModel(name=..., base_model=...)
2. await model.register(backend)
   ├─> backend.register(model)        ─> writes model.json
   └─> backend._prepare_backend_for_training(model)
       ├─> UnslothService created
       │   ├─> from_pretrained()                              ─> loads to GPU
       │   ├─> get_peft_model()                               ─> LoRA adapter
       │   ├─> GRPOTrainer() + create_optimizer()             ─> AdamW
       │   └─> patches trainer._prepare_inputs()
       ├─> service.start_openai_server()
       │   ├─> offload training model to CPU (pinned memory)
       │   └─> start vLLM AsyncLLM + OpenAI server            ─> base model on GPU
       └─> returns (base_url, api_key)

3. client = model.openai_client()   ─> AsyncOpenAI(base_url, api_key)

4. response = await client.chat.completions.create(
       model=model.get_inference_name(),
       messages=[...], logprobs=True
   )                                ─> vLLM serves base + LoRA

5. trajectory = Trajectory(messages_and_choices=[...], reward=...)

6. group = await TrajectoryGroup([rollout() for _ in range(N)])

7. groups = await gather_trajectory_groups([group1, group2, ...])

8. result = await backend.train(model, groups, learning_rate=5e-6)
   ├─> tokenize + pack into PackedTensors [B, S]
   ├─> write packed tensors to disk
   └─> service.train():
       ├─> vLLM pause + sleep                                 ─> GPU free
       ├─> reload training model                              ─> PEFT on GPU
       ├─> for each batch:
       │   ├─> forward → logprobs → loss → backward → step
       ├─> save LoRA checkpoint
       ├─> offload training model                             ─> GPU free
       ├─> wake vLLM + register new LoRA                      ─> vLLM on GPU
       └─> resume generation

9. model.log(groups, metrics=result.metrics, step=result.step)
```
