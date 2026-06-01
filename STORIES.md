# Implementation Stories — mlxforge

> Project: **mlxforge** — a from-scratch LLaMA inference engine in C++ on Apple MLX (the C++ core library, not mlx-lm), served behind an OpenAI-compatible HTTP API with continuous batching.
> Target model: `mlx-community/Llama-3.2-1B-Instruct`, fp16 first (16 layers, hidden 2048, 32 query / 8 KV heads GQA, head_dim 64, RMSNorm, RoPE, SwiGLU, tied embeddings).
> Last updated: 2026-06-01

## Goal

Build the entire path from raw safetensors weights to a production-shaped server: weight loading, the transformer graph, KV cache, sampling, the generation loop, a continuous-batching scheduler, and an OpenAI-compatible HTTP layer (cpp-httplib). Every numerically-sensitive phase is validated against an `mlx-lm` golden reference because the failure mode here is **silent garbage, not a crash**.

## How stories map to the spec's phases

The spec defines 11 phases (0–10). Stories preserve that dependency ordering but are grouped into six sprints. Each story carries a stable ID (`MLXFORGE-NNN`), maps to one or more modules from the spec's decomposition table, and — where the spec defines a per-phase golden-reference assertion — embeds that exact check as a required acceptance criterion.

The golden-reference discipline is threaded throughout: **MLXFORGE-002** stands up the Python reference dump early, and every numerically-relevant story afterward asserts tensor-closeness (fp16 rel ~1e-2) or exact greedy-token equality against it.

Per the spec, the **tokenizer (Phase 2) is intentionally sequenced late** — until it lands, all engine work feeds on pre-tokenized IDs dumped from Python.

---

## Summary

- **Total Stories**: 25
- **Total Sprints**: 6
- **Story Size Legend**: S (~0.5–1 day), M (~1–2 days), L (~3–4 days), XL (~5+ days, high-risk)
- **Highest-risk stories**: MLXFORGE-010 (single→batch KV cache), MLXFORGE-016/017/018 (continuous-batching scheduler), MLXFORGE-006 (RoPE conventions), MLXFORGE-013 (eval placement), MLXFORGE-019 (batch-shape bucketing/recompilation), MLXFORGE-021 (tokenizer streaming).

---

## Testing standard (cross-cutting)

Applies to every story unless noted otherwise:

- **Framework**: unit tests use a single C++ framework (recommended **doctest** or **Catch2** — header-only, trivial CMake integration), registered with **`ctest`**. MLXFORGE-001 wires the framework into the build; MLXFORGE-005 adds the `.npy` reference-compare helper on top.
- **CI gate**: `ctest` must be green before a story is considered done. New tests live under `tests/` mirroring the module path (e.g. `tests/core/config_test.cpp`).
- **Two test tiers**:
  - **Unit tests** — pure-logic pieces tested in isolation, no model weights / no GPU eval required (config parsing, weight-key sanitize, KV-cache index bookkeeping, sampler math, JSON (de)serialization, SSE framing). These stories carry explicit unit-test acceptance criteria below.
  - **Golden-reference / integration checks** — numerically-sensitive stories (forward pass, KV equivalence, scheduler) are validated against the `mlx-lm` reference dump (tensor-closeness / exact-token equality), as already specified per story. These are the primary correctness gate for those stories; discrete unit tests are not duplicated for them.

---

## Sprint 1 — Build foundation, weights & golden-reference harness

**Focus**: Stand up the CMake/MLX build, load and validate weights as fp16, and establish the Python golden-reference dump that gates every later phase. (Spec Phases 0, 1, plus the cross-cutting reference discipline.)

### MLXFORGE-001: CMake project links MLX and runs "hello array"

- **Size**: M
- **Dependencies**: none
- **Files/Modules**: `CMakeLists.txt`, `apps/mlxforge_cli.cpp` (stub), top-level build scaffolding
- **Description**: Create the greenfield CMake project. Pull MLX C++ core (FetchContent or submodule, **pinned to a recent commit**) and cpp-httplib + nlohmann/json. Write a minimal program that creates two arrays, adds them, calls `mx::eval`, and prints the result. This is the lazy-eval lesson and the build smoke test.
- **Acceptance Criteria**:
  - [ ] `cmake` configure + build succeeds on Apple Silicon producing a runnable binary.
  - [ ] Program creates two arrays, adds them, calls `mx::eval`, and prints the correct sum.
  - [ ] `mx::metal::is_available()` returns true and is asserted/printed at startup.
  - [ ] MLX is pinned to a specific commit/tag recorded in the build config.
  - [ ] Unit-test framework (doctest/Catch2) is wired into CMake and registered with `ctest`; a trivial passing test runs via `ctest` (establishes the per-story testing gate).

### MLXFORGE-002: Python golden-reference dump (`dump_ref.py`)

- **Size**: M
- **Dependencies**: none (can proceed in parallel with MLXFORGE-001)
- **Files/Modules**: `reference/dump_ref.py`, `reference/` fixtures
- **Description**: Stand up the non-negotiable golden-reference tooling early. Using `mlx-lm` on the same model + a small fixed prompt set, emit `.npy` tensors: token IDs for fixed prompts, `embeddings`, block-0 output, final logits, and the greedy token stream. These artifacts gate every numerically-relevant story. **This story must land before MLXFORGE-003 onward can be validated.**
- **Acceptance Criteria**:
  - [ ] Running `python reference/dump_ref.py` produces `.npy` files for: tokenized IDs of each fixed prompt, embedding output, block-0 output, final logits, and a greedy token stream.
  - [ ] A fixed prompt set and the model revision are committed/recorded so dumps are reproducible.
  - [ ] Output `.npy` shapes and dtypes are documented for the C++ side to load.
  - [ ] Pre-tokenized prompt IDs are exported so engine phases can run before the C++ tokenizer exists.

### MLXFORGE-003: `ModelConfig` parsed from `config.json`

- **Size**: S
- **Dependencies**: MLXFORGE-001
- **Files/Modules**: `core/config.{h,cpp}`
- **Description**: Parse `config.json` into a `ModelConfig` struct: `n_layers`, `hidden`, `n_heads`, `n_kv_heads`, `head_dim`, `vocab`, `rope_theta`, `rms_eps`, `max_position_embeddings`.
- **Acceptance Criteria**:
  - [ ] Loads Llama-3.2-1B config and exposes all listed fields with correct values (16 layers, hidden 2048, 32 q / 8 kv heads, head_dim 64).
  - [ ] `rope_theta` and `rms_eps` are read from config (not hard-coded).
  - [ ] Missing/extra keys handled gracefully (clear error on missing required field).
  - [ ] **Unit tests** (no GPU): parse a known-good `config.json` fixture and assert every field; parse a config with a missing required key and assert a clear error; parse one with extra/unknown keys and assert it is ignored. `ctest` green.

### MLXFORGE-004: Load, sanitize & fp16-cast weights from safetensors

- **Size**: L
- **Dependencies**: MLXFORGE-001, MLXFORGE-003
- **Files/Modules**: `core/weights.{h,cpp}` (`mx::load`)
- **Description**: Load safetensors via `mx::load(path)` (string path only). Iterate the returned map, print every key + shape. Handle sharded `model-0000N-of-*.safetensors` with the index JSON, and apply any key-name `sanitize` remap. **Cast all weights to fp16 immediately after load** (unified memory; no host/device copy).
- **Acceptance Criteria**:
  - [ ] Every expected weight tensor loads with correct shape (e.g. `model.layers.0.self_attn.q_proj.weight`).
  - [ ] Sharded models load correctly via the index JSON; any required `sanitize` remap is applied.
  - [ ] All weights are fp16 after load.
  - [ ] Peak resident memory after load is ≈ 2.5 GiB (Phase 1 done criterion).
  - [ ] A key→shape dump can be printed for inspection.
  - [ ] **Unit tests** (no GPU/weights): the `sanitize` key-remap function maps a table of raw→canonical key names correctly (incl. a no-op passthrough case); the shard-index parser resolves which file each tensor key lives in from a fixture index JSON. `ctest` green.

---

## Sprint 2 — Numerically-correct forward pass

**Focus**: Build the transformer one block at a time, then stack to full logits, validating each step against the golden reference. Ends at the project's pivotal milestone: a numerically-correct model. (Spec Phases 3, 4.)

### MLXFORGE-005: C++ tensor-compare test harness

- **Size**: S
- **Dependencies**: MLXFORGE-002
- **Files/Modules**: `tests/` (npy loader + closeness asserts)
- **Description**: A C++ test utility that loads the `.npy` reference tensors and asserts closeness (fp16 rel ~1e-2) and, for token streams, exact equality. This is the assertion machinery every forward-pass story uses.
- **Acceptance Criteria**:
  - [ ] Can load `.npy` files produced by `dump_ref.py` into MLX arrays.
  - [ ] Provides a `assert_close(actual, expected, rel_tol=1e-2)` helper and an exact-token-equality helper.
  - [ ] On mismatch, prints the first divergent index and magnitude for debugging.

### MLXFORGE-006: Embedding + RMSNorm + Q/K/V projections + RoPE (HIGH RISK)

- **Size**: L
- **Dependencies**: MLXFORGE-004, MLXFORGE-005
- **Files/Modules**: `model/llama.{h,cpp}` (`fast::rms_norm`, `fast::rope`)
- **Description**: Wire the front half of a single decoder layer: embedding lookup → `fast::rms_norm` → Q/K/V projections → `fast::rope`. **RISK: RoPE conventions are a classic silent bug** — `rope_theta` base and the `traditional`/interleaved flag must match the model. Mitigation: assert RoPE'd Q/K against the reference dump before proceeding; verify the `fast::rope` `const array& offset` overload exists on the pinned MLX build (linchpin of later batching) and record the assertion.
- **Acceptance Criteria**:
  - [ ] Embedding-lookup output matches the reference `embeddings.npy` within tolerance.
  - [ ] Post-RMSNorm and post-RoPE Q/K tensors match reference within fp16 rel ~1e-2.
  - [ ] RoPE uses the config `rope_theta` and the correct `traditional`/interleaved setting (verified against reference, not assumed).
  - [ ] A build-time/startup assertion confirms `fast::rope(const array& offset, ...)` overload exists on the pinned MLX.

### MLXFORGE-007: Single decoder block — attention + SwiGLU MLP + residuals

- **Size**: L
- **Dependencies**: MLXFORGE-006
- **Files/Modules**: `model/llama.{h,cpp}` (`fast::scaled_dot_product_attention`)
- **Description**: Complete one decoder layer (prefill only): `fast::scaled_dot_product_attention` with `mask_mode="causal"` → o_proj → residual → `rms_norm` → SwiGLU MLP (`silu(gate)*up` then `down`) → residual. GQA: 8 KV heads repeat across 32 query heads — SDPA handles this natively when q_heads > kv_heads. **Watch the GQA repeat as another silent-bug candidate.**
- **Acceptance Criteria**:
  - [ ] Single block output matches the reference block-0 tensor within fp16 rel ~1e-2 (Phase 3 done criterion).
  - [ ] SDPA invoked with `mask_mode="causal"`; scale = 1/sqrt(head_dim).
  - [ ] GQA works with 32 q / 8 kv heads with no manual head replication beyond what SDPA requires.
  - [ ] SwiGLU computes `down(silu(gate(x)) * up(x))`.

### MLXFORGE-008: Full forward pass to logits (MILESTONE: numerically correct)

- **Size**: M
- **Dependencies**: MLXFORGE-007
- **Files/Modules**: `model/llama.{h,cpp}`
- **Description**: Stack all 16 layers, apply the final `rms_norm`, then the LM head (tied embedding) to produce logits for the last position. This is the pivotal milestone — once argmax matches, the hard numerical part is done.
- **Acceptance Criteria**:
  - [ ] Final logits for a fixed prompt match the reference logits within fp16 rel ~1e-2.
  - [ ] **Argmax of first-token logits for a fixed prompt matches mlx-lm** (Phase 4 done criterion — exact).
  - [ ] LM head correctly uses the tied embedding weights.
  - [ ] Forward pass runs over the full 16-layer stack with final RMSNorm.

---

## Sprint 3 — KV cache & single-stream generation

**Focus**: Prove the prefill/decode split with a single-sequence cache, generalize to the batched cache the server needs, then add sampling and the CLI generation loop. (Spec Phases 5, 6.)

### MLXFORGE-009: Single-sequence `KVCache` (prefill + decode append)

- **Size**: M
- **Dependencies**: MLXFORGE-008
- **Files/Modules**: `cache/batch_kv_cache.{h,cpp}` (single-seq path)
- **Description**: Build a single-sequence KV cache to prove the prefill/decode split: prefill fills it once; each decode step appends one token's K/V and attends over cached history. Lays the groundwork before batching.
- **Acceptance Criteria**:
  - [ ] Prefill populates the cache; subsequent single-token decode appends K/V correctly.
  - [ ] Decode-with-cache produces the same logits as a full recompute over the same sequence (within tolerance).
  - [ ] Greedy token stream with cache matches the reference greedy stream exactly.

### MLXFORGE-010: `BatchKVCache` — layout, `update_and_fetch`, offset/left-padding (HIGH RISK)

- **Size**: XL
- **Dependencies**: MLXFORGE-009
- **Files/Modules**: `cache/batch_kv_cache.{h,cpp}` (`zeros`, `concatenate`, `take`)
- **Description**: Port `mlx_lm/models/cache.py::BatchKVCache`. Layout `(B, n_kv_heads=8, S_cap, head_dim=64)` per layer, **contiguous, left-padded**, grown in blocks of `step=256` via `zeros` + `concatenate(axis=2)`. Per-row `offset` array (`= -left_padding` at admission, `+= written_len` after each write) and `left_padding` array. `update_and_fetch(k, v)` writes the populated slice and returns `[..., :idx, :]`. **RISK: the offset/left-padding bookkeeping is the foundation of correct batched attention — a single-stream regression test (MLXFORGE-013) gates it.**
- **Acceptance Criteria**:
  - [ ] Cache stores K/V as `(B, 8, S_cap, 64)` per layer, contiguous and left-padded.
  - [ ] `S_cap` grows in blocks of 256 via `zeros` + `concatenate(axis=2)`.
  - [ ] `offset` initializes to `-left_padding` and increments by written length per write.
  - [ ] `update_and_fetch` returns the correct populated slice `[..., :idx, :]`.
  - [ ] **Unit tests** (tiny tensors, single eval): after N writes, `offset`/`left_padding`/`_idx` hold the expected per-row values; crossing a 256 boundary triggers exactly one growth and preserves prior contents; `update_and_fetch` returns a slice of the expected shape. `ctest` green.

### MLXFORGE-011: `BatchKVCache` — `filter` (eviction) and `merge` (admission)

- **Size**: M
- **Dependencies**: MLXFORGE-010
- **Files/Modules**: `cache/batch_kv_cache.{h,cpp}` (`take`, `concatenate`)
- **Description**: Add `filter(keep_idx)` = `take(axis=0)` on every layer's K/V plus offset/padding (used for eviction), and `merge(other)` = pad both to common `S_cap` then `concatenate(axis=0)` (used for admitting a prefilled batch). These are the batch-axis surgery operations the scheduler depends on.
- **Acceptance Criteria**:
  - [ ] `filter(keep_idx)` correctly drops rows from K/V, offset, and left_padding on every layer.
  - [ ] `merge(other)` pads both caches to a common `S_cap` and concatenates on the batch axis.
  - [ ] After filter/merge, per-row offsets remain correct (post-op decode still matches solo runs).
  - [ ] **Unit tests** (tiny tensors): `filter([keep])` drops the right rows and leaves K/V/offset/left_padding consistent for a known keep-set (incl. dropping the first and last row); `merge` of two caches with different `S_cap` yields the expected combined batch size and offset array. `ctest` green.

### MLXFORGE-012: KV admission gate (OOM guard)

- **Size**: S
- **Dependencies**: MLXFORGE-011
- **Files/Modules**: `cache/batch_kv_cache.{h,cpp}`, `scheduler/scheduler.{h,cpp}` (gate hook)
- **Description**: Implement the KV memory projection guard: 32 KiB/token (fp16, all layers). Project `(max_len + max_new) × 32 KiB × B` and refuse/queue admission if it exceeds a configured KV budget. MLX allocates until Metal fails, so this gate is the real OOM guard.
- **Acceptance Criteria**:
  - [ ] A function projects KV memory for a candidate batch using the 32 KiB/token figure.
  - [ ] Admission is refused/queued when projected usage exceeds the configured KV budget.
  - [ ] Sanity check matches the spec's memory math (1 seq @2048 = 64 MiB; batch 32 @2048 ≈ 2 GiB).

### MLXFORGE-013: Batched decode equivalence test (eval-placement guard) (HIGH RISK)

- **Size**: M
- **Dependencies**: MLXFORGE-011
- **Files/Modules**: `tests/`, `model/llama.{h,cpp}`
- **Description**: Validate the batched cache end-to-end: batched decode of 2–3 fixed prompts must produce identical tokens to single-sequence runs (Phase 5 done criterion). **RISK: eval placement** — establish here that a batched step uses a single eval over the whole batch (never per-row/per-layer), setting the invariant the scheduler will enforce.
- **Acceptance Criteria**:
  - [ ] Batched decode of 2–3 fixed prompts produces **identical tokens** to single-sequence runs (Phase 5 done — exact).
  - [ ] The batched forward issues one `eval`/`async_eval` per step over the whole batch (asserted/instrumented).
  - [ ] Ragged batches (different prompt lengths via left-padding) still match solo runs.

### MLXFORGE-014: Sampling ops — greedy, temperature, top-k, top-p (as graph ops)

- **Size**: M
- **Dependencies**: MLXFORGE-008
- **Files/Modules**: `sample/sampler.{h,cpp}` (MLX ops)
- **Description**: Implement sampling entirely as MLX graph ops (do **not** pull logits to CPU — that breaks the pipeline). Greedy (argmax) first for deterministic reference matching, then temperature, top-k, top-p. Support a seed for reproducibility.
- **Acceptance Criteria**:
  - [ ] Greedy = argmax over logits, computed as a graph op (no host readback of logits).
  - [ ] Temperature, top-k, and top-p are implemented as MLX ops folded into the graph.
  - [ ] Sampling accepts a batched logits tensor `(B, vocab)` and returns `(B,)` next tokens + logprobs.
  - [ ] Greedy is deterministic; seeded sampling is reproducible.
  - [ ] **Unit tests** (synthetic logits, no model): greedy returns the argmax index on a hand-built logits vector; `temperature→0` collapses to greedy; top-k keeps exactly k candidates; top-p keeps the smallest prefix whose mass ≥ p; same seed → identical draws, different seed → (statistically) different. `ctest` green.

### MLXFORGE-015: `mlxforge-cli` generation loop (MILESTONE: it generates text)

- **Size**: M
- **Dependencies**: MLXFORGE-013, MLXFORGE-014
- **Files/Modules**: `apps/mlxforge_cli.cpp`
- **Description**: Wire the single-stream generation loop in the CLI: sample → append → (placeholder) detokenize → stream to stdout → until EOS or `max_tokens`. Until the C++ tokenizer lands (MLXFORGE-021), feed pre-tokenized IDs and dump token IDs / use a Python decode step.
- **Acceptance Criteria**:
  - [ ] **Greedy CLI output matches mlx-lm token-for-token** for a fixed prompt (Phase 6 done — exact).
  - [ ] Loop terminates on EOS or `max_tokens`.
  - [ ] `mlxforge-cli` streams tokens incrementally to stdout (IDs acceptable until tokenizer lands).

---

## Sprint 4 — Continuous-batching scheduler (the core systems piece)

**Focus**: The vLLM-style three-queue scheduler on a single GPU worker thread. This sprint carries the highest concentration of systems risk. (Spec Phase 8 — note Phase 7 quantization is deferred to Sprint 6.)

### MLXFORGE-016: Single GPU worker thread + three-queue skeleton (HIGH RISK)

- **Size**: L
- **Dependencies**: MLXFORGE-013, MLXFORGE-014
- **Files/Modules**: `runtime/worker.{h,cpp}`, `scheduler/scheduler.{h,cpp}`
- **Description**: Establish the single GPU worker thread that **owns all MLX state** (weights, per-layer `BatchKVCache`, sampler) and is the **only** caller of `mx::eval`/`async_eval`. Build the three-queue state machine skeleton: `waiting` → `prefill_batch` → `decode_batch`. HTTP/test threads only touch their own `Request` struct (MLX is not thread-safe for concurrent eval). **RISK: thread-safety** — enforce the single-thread invariant from the start.
- **Acceptance Criteria**:
  - [ ] A single worker thread owns model weights, per-layer cache, and sampler; no other thread calls any MLX op.
  - [ ] Three queues (`waiting`, `prefill_batch`, `decode_batch`) exist with mutex + condition_variable handoff.
  - [ ] A `Request` struct carries `token_ids`, sampling params, a bounded token queue, and a `cancelled` atomic.
  - [ ] Requests can be enqueued from another thread and picked up by the worker.

### MLXFORGE-017: Prefill pass — left-pad, chunked prefill, merge into decode cache (HIGH RISK)

- **Size**: L
- **Dependencies**: MLXFORGE-016, MLXFORGE-012
- **Files/Modules**: `scheduler/scheduler.{h,cpp}`, `cache/batch_kv_cache.{h,cpp}`
- **Description**: Implement prefill as a **separate pass, then joined** (do NOT chunk-interleave prefill into the decode batch). Left-pad waiting requests to a common `P_max`, run a dedicated prefill forward (chunked by `prefill_step_size=2048` for long prompts, with `eval(cache.state)` at each chunk boundary to bound graph growth), then `merge` into the decode cache. **RISK: graph/memory growth during long prefill** — the per-chunk eval is the mitigation; monitor `mx::metal::get_active_memory()` and `clear_cache()` if the high-watermark creeps.
- **Acceptance Criteria**:
  - [ ] Waiting requests are left-padded to a common `P_max` and prefilled in a dedicated forward pass.
  - [ ] Long prompts are chunked at `prefill_step_size=2048` with `eval(cache.state)` at each chunk boundary.
  - [ ] Prefilled batch is `merge`d into the decode cache with correct per-row offsets.
  - [ ] `prefill_batch_size` (8) is respected; prefill shape `(B_p, P_max, …)` stays distinct from decode shape.

### MLXFORGE-018: Steady-state decode step — single async_eval + 1-step lookahead (HIGH RISK)

- **Size**: XL
- **Dependencies**: MLXFORGE-017
- **Files/Modules**: `scheduler/scheduler.{h,cpp}`, `runtime/worker.{h,cpp}` (`mx::async_eval`)
- **Description**: Implement the steady-state decode loop exactly per spec: (1) `inputs = decode_batch.last_tokens (B,)`, `offsets = cache.offset (B,)`; (2) `logits = model(inputs[:,None], cache)[:, -1, :]`; (3) `next, logprobs = sampler(logits)`; (4) **`mx::async_eval(next, logprobs)` — the ONLY eval per step**, reading the previous step's result (1-step lookahead so graph build for t+1 overlaps compute of t); (5) push each row's token to its `Request` queue + notify, update finish flags; (6) evict finished/cancelled rows via `filter(keep)`, admit from `waiting`. **RISK (highest severity): eval placement** — treat "evals per decode step" as a guarded invariant (exactly one, on the whole batch).
- **Acceptance Criteria**:
  - [ ] Exactly **one `async_eval` per decode step** over the whole batch (instrumented/asserted; never per-row/per-layer).
  - [ ] 1-step lookahead pipeline: results read one step late so graph build overlaps compute.
  - [ ] Each produced token is pushed to its `Request` token queue with notify.
  - [ ] Finished (EOS/max) and cancelled rows are evicted via `filter`; freed slots admit from `waiting`.

### MLXFORGE-019: Ragged-batch additive fp16 mask + batch-size bucketing (HIGH RISK)

- **Size**: L
- **Dependencies**: MLXFORGE-018
- **Files/Modules**: `scheduler/scheduler.{h,cpp}`, `model/llama.{h,cpp}`
- **Description**: Build the per-step attention mask `[B, 1, T_q, T_kv]` as **additive fp16** (avoid the bool >2³¹ bug #2894): `causal = kv_pos <= offset[:,row]`, `valid = kv_pos >= left_padding[:,row]`, `mask = where(causal & valid, 0, -inf)` — per-row offset makes causal + variable context length fall out of one comparison. Then **batch-size bucketing**: pad active `B` up to fixed buckets (1,2,4,8,16,32) with masked dummy rows so the forward graph shape recurs. **RISK: regraph/recompile when batch shape changes every step** — bucketing is the mitigation; `S_cap` is already on the 256 step. Validate uncompiled before considering `mx::compile`.
- **Acceptance Criteria**:
  - [ ] Mask is additive fp16 of shape `[B, 1, T_q, T_kv]` (no boolean mask).
  - [ ] Mask combines causal (`kv_pos <= offset`) and validity (`kv_pos >= left_padding`) per row.
  - [ ] Active batch is padded to the nearest bucket in {1,2,4,8,16,32} with masked dummy rows.
  - [ ] Dummy rows provably do not affect real rows' outputs.
  - [ ] Decode-step graph shape recurs across steps (no per-step regraph for a fixed bucket).

### MLXFORGE-020: Scheduler correctness & throughput validation (MILESTONE: concurrent batching works)

- **Size**: M
- **Dependencies**: MLXFORGE-019
- **Files/Modules**: `tests/`, `scheduler/scheduler.{h,cpp}`
- **Description**: Validate the scheduler: submit N identical + distinct fixed prompts concurrently; each request's tokens must equal its solo greedy run (continuous batching must not change outputs). Measure throughput at concurrency 1, 4, 8.
- **Acceptance Criteria**:
  - [ ] **N concurrent fixed prompts each produce the same tokens as if run alone** (Phase 8 done — exact).
  - [ ] Mixed-length prompts admitted/evicted at different times still match solo runs.
  - [ ] Aggregate throughput rises with concurrency; target ~4× at 8× concurrency (documented sanity target).
  - [ ] The "one eval per decode step" invariant holds under concurrent load.

---

## Sprint 5 — Tokenizer & OpenAI server

**Focus**: Land the C++ tokenizer (deferred Phase 2) just in time for the server, then the cpp-httplib OpenAI-compatible HTTP layer with SSE streaming and cancellation. (Spec Phases 2, 9.)

### MLXFORGE-021: C++ tokenizer — encode/decode + chat template (HIGH RISK: streaming)

- **Size**: L
- **Dependencies**: MLXFORGE-002 (for round-trip fixtures)
- **Files/Modules**: `tokenizer/tokenizer.{h,cpp}` (tokenizers-cpp)
- **Description**: Link a C++ tokenizer that reads the HF `tokenizer.json` (recommended `mlc-ai/tokenizers-cpp`, or the HF `tokenizers` Rust crate via C binding) — do not hand-roll BPE. Support encode, decode, and **incremental/streaming decode of one new token** (handle multi-byte UTF-8 and byte-level BPE so partial tokens never emit broken characters). Load (or hard-code) the Llama-3.2 chat template to format `messages[]` → prompt string. **RISK: tokenizer streaming** — the incremental decoder is what keeps SSE from emitting garbled UTF-8.
- **Acceptance Criteria**:
  - [ ] Encode/decode round-trips match `mlx-lm`'s tokenizer on the fixed string set (Phase 2 done).
  - [ ] The Llama-3.2 chat template produces the **exact** prompt mlx-lm produces for the same `messages`.
  - [ ] Incremental decode of one new token at a time never emits broken multi-byte UTF-8 / partial byte-BPE characters.
  - [ ] `mlxforge-cli` (MLXFORGE-015) can now stream real text instead of token IDs.

### MLXFORGE-022: cpp-httplib OpenAI server — routes, JSON parse, enqueue

- **Size**: L
- **Dependencies**: MLXFORGE-020, MLXFORGE-021
- **Files/Modules**: `server/http_server.{h,cpp}` (cpp-httplib, nlohmann/json), `apps/mlxforge.cpp`
- **Description**: Build the server binary with routes `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`. Parse OpenAI requests (`model`, `messages`/`prompt`, `max_tokens`, `temperature`, `top_p`, `stream`, `stop`, `n`, `seed`) with nlohmann/json; apply the chat template; tokenize; build a `Request` and enqueue to the scheduler's `waiting` queue. Implement **non-streaming** response assembly first (block until sentinel, assemble full `chat.completion` with `usage`).
- **Acceptance Criteria**:
  - [ ] All four routes exist and return correctly shaped responses; `GET /health` and `GET /v1/models` are correct.
  - [ ] OpenAI request fields are parsed and mapped into a `Request` with sampling params.
  - [ ] Non-streaming `chat.completions.create(...)` via the official `openai` Python client returns a well-formed `chat.completion` with correct `usage` token counts.
  - [ ] Response shape parity: `id`, `object`, `created`, `model`, `choices[].message`, `finish_reason` (`stop`/`length`), `usage`.
  - [ ] **Unit tests** (no server/GPU): request-parsing maps a table of OpenAI request JSON fixtures → `Request` (incl. defaults for omitted fields and an error on malformed JSON / out-of-range params); response serializer emits the exact `chat.completion` JSON shape for a fixed completion. `ctest` green.

### MLXFORGE-023: SSE streaming + cancellation (MILESTONE: OpenAI server live)

- **Size**: L
- **Dependencies**: MLXFORGE-022
- **Files/Modules**: `server/http_server.{h,cpp}` (cpp-httplib SSE)
- **Description**: Implement streaming via `set_chunked_content_provider`: the sink blocks on the per-request token queue + cv (single-producer worker / single-consumer request thread), formats each token as `data: {chat.completion.chunk ...}\n\n`, and writes it; on EOS push a sentinel → `data: [DONE]\n\n` and return false to close. Bound the per-request queue (backpressure for slow consumers). **Cancellation:** content provider returning false (client disconnect) sets the per-request `cancelled` atomic; the worker treats it as eviction at the next iteration boundary.
- **Acceptance Criteria**:
  - [ ] `stream=True` via the `openai` Python client yields incremental `chat.completion.chunk` deltas then `[DONE]`.
  - [ ] ~16 concurrent client requests all complete, interleaved, no crash, no garbled UTF-8 (Phase 9 done).
  - [ ] Disconnecting a streaming client mid-generation frees its batch slot (eviction) while others continue.
  - [ ] The per-request token queue is bounded (backpressure applies for slow consumers).
  - [ ] `delta` streaming and `message` non-streaming both report correct `finish_reason`.
  - [ ] **Unit tests** (no server/GPU): the chunk formatter produces a byte-exact `data: {chat.completion.chunk ...}\n\n` frame for a given delta and a final `data: [DONE]\n\n`; the bounded per-request queue blocks/yields at capacity rather than growing unbounded. `ctest` green.

---

## Sprint 6 — Hardening, ops & optional quantization

**Focus**: Production hardening (config, logging, metrics, graceful shutdown, OpenAI error shapes) and the optional 4-bit quantization path. (Spec Phases 10, 7.)

### MLXFORGE-024: Hardening & ops — config, metrics, graceful shutdown, error shapes

- **Size**: L
- **Dependencies**: MLXFORGE-023
- **Files/Modules**: `server/http_server.{h,cpp}`, `runtime/worker.{h,cpp}`, `apps/mlxforge.cpp`, `scheduler/scheduler.{h,cpp}`
- **Description**: Add CLI-flag/env config (model path, port, max batch, KV budget, ctx length, thread count). Structured logging + per-request metrics (TTFT, tokens/s, queue depth, batch occupancy). Graceful shutdown (drain in-flight). OpenAI-shaped error JSON: 400 (bad params), 429 (queue full), 503 (model loading). Bounded `waiting` queue with 429 on overflow. Optional `/metrics` (Prometheus).
- **Acceptance Criteria**:
  - [ ] All listed config knobs are settable via CLI flags / env and take effect.
  - [ ] Oversized prompt → 400; queue saturation → 429; model-loading → 503, all in OpenAI error JSON shape.
  - [ ] Per-request metrics (TTFT, tokens/s, queue depth, batch occupancy) are logged.
  - [ ] Graceful shutdown drains in-flight requests before exit.
  - [ ] Bounded `waiting` queue returns 429 on overflow.

### MLXFORGE-025: Optional 4-bit quantization via `quantized_matmul`

- **Size**: M
- **Dependencies**: MLXFORGE-008 (numerically-correct fp16 baseline); slots in any time after, independent of server work
- **Files/Modules**: `core/weights.{h,cpp}`, `model/llama.{h,cpp}` (`ops::quantized_matmul`)
- **Description**: (Spec Phase 7, optional.) Swap fp16 linears for `ops::quantized_matmul` (4-bit, group_size 64, from `mlx/ops.h`). Sanitize/load the quantized weight + scale + bias tensors. Reduces footprint ~2.5 GiB → ~0.7 GiB for the 1B model.
- **Acceptance Criteria**:
  - [ ] Quantized weight + scale + bias tensors load and sanitize correctly.
  - [ ] `ops::quantized_matmul` used with group_size 64, 4-bit.
  - [ ] Quantized greedy output stays coherent and close to the fp16 reference (Phase 7 done).
  - [ ] Resident memory drops to ~0.7 GiB for the 1B model.

---

## Dependency-ordered build sequence

```
MLXFORGE-001 ─┬─> MLXFORGE-003 ─> MLXFORGE-004 ─┬─> MLXFORGE-006 ─> MLXFORGE-007 ─> MLXFORGE-008 ─┬─> MLXFORGE-009 ─> MLXFORGE-010 ─> MLXFORGE-011 ─┬─> MLXFORGE-012
MLXFORGE-002 ─┴─> MLXFORGE-005 ──────────────┘                                     │                                   ├─> MLXFORGE-013
                                                                           │                                   └─> (MLXFORGE-025 optional, any time after 008)
                                       MLXFORGE-008 ─> MLXFORGE-014 ───────────────┘
MLXFORGE-013 + MLXFORGE-014 ─> MLXFORGE-015  (CLI generates text)
MLXFORGE-013 + MLXFORGE-014 ─> MLXFORGE-016 ─> MLXFORGE-017 (+MLXFORGE-012) ─> MLXFORGE-018 ─> MLXFORGE-019 ─> MLXFORGE-020  (concurrent batching)
MLXFORGE-002 ─> MLXFORGE-021
MLXFORGE-020 + MLXFORGE-021 ─> MLXFORGE-022 ─> MLXFORGE-023  (OpenAI server live)
MLXFORGE-023 ─> MLXFORGE-024
```

**Critical path**: 001/002 → 003 → 004 → 006 → 007 → 008 → 009 → 010 → 011 → (012) → 013 → 016 → 017 → 018 → 019 → 020 → 022 → 023 → 024.

**Key milestones (from spec)**: MLXFORGE-008 (numerically-correct model), MLXFORGE-015 (it generates text), MLXFORGE-020 (concurrent batching works), MLXFORGE-023 (OpenAI server live).

**Parallelizable**: MLXFORGE-002 / MLXFORGE-005 (reference harness) run alongside MLXFORGE-001/003/004. MLXFORGE-014 (sampler) can start as soon as MLXFORGE-008 lands. MLXFORGE-021 (tokenizer) can be built any time after MLXFORGE-002 and is only *required* by MLXFORGE-022. MLXFORGE-025 (quantization) is independent of all server work.

---

## Risk register

| # | Risk (spec severity) | Stories | Mitigation (built into the story) |
|---|---|---|---|
| 1 | **Eval placement / serialization (highest)** | MLXFORGE-013, MLXFORGE-018 | Exactly **one `async_eval` per decode step** over the whole batch; treat "evals per step" as a guarded, instrumented invariant; never per-row/per-layer. |
| 2 | **Regraph/recompile on changing batch shape** | MLXFORGE-019 | Bucket `B` to {1,2,4,8,16,32} with masked dummy rows; keep `S_cap` on the 256 step; validate uncompiled before adopting `mx::compile`. |
| 3 | **RoPE per-row offset overload missing** | MLXFORGE-006 | Assert the `fast::rope(const array& offset, …)` overload exists on the pinned MLX; fallback is `freqs` materialization. |
| 4 | **Silent numerical bugs (RoPE base/interleave, weight transpose, GQA repeat)** | MLXFORGE-006, MLXFORGE-007, MLXFORGE-008 | Golden-reference comparison at every phase (MLXFORGE-002 + MLXFORGE-005), exact argmax/token checks. |
| 5 | **Graph/memory growth during long prefill** | MLXFORGE-017 | `eval(cache.state)` at each prefill chunk boundary; monitor `mx::metal::get_active_memory()`; `clear_cache()` if high-watermark creeps. |
| 6 | **OOM (Metal allocates until failure)** | MLXFORGE-012, MLXFORGE-024 | KV admission gate (32 KiB/token projection vs budget) + bounded `waiting` queue (429 on overflow). |
| 7 | **Tokenizer streaming emits broken UTF-8** | MLXFORGE-021, MLXFORGE-023 | Incremental UTF-8 / byte-BPE decode so SSE never emits partial characters. |
| 8 | **MLX thread-safety** | MLXFORGE-016 | All MLX work on the single worker thread; HTTP/test threads touch only their own `Request`. |
| 9 | **Boolean mask >2³¹ bug (#2894)** | MLXFORGE-019 | Use additive fp16 masks, never boolean. |

---

## Recommended review points

- **After MLXFORGE-008**: Confirm the model is numerically correct (argmax matches mlx-lm) before building any caching/serving on top of it. Stop-the-line if reference checks fail.
- **After MLXFORGE-013**: Confirm batched cache equivalence and the single-eval invariant before committing to the scheduler.
- **After MLXFORGE-020**: Confirm continuous batching preserves outputs and scales throughput before wiring the HTTP layer.
- **After MLXFORGE-023**: Confirm OpenAI client parity (streaming + non-streaming + cancellation) before hardening.
