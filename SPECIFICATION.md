# mlxforge — Technical Specification

> A from-scratch LLaMA inference engine in C++ on Apple **MLX** (the C++ core library, not mlx-lm),
> served behind an **OpenAI-compatible HTTP API** with **continuous batching** for concurrent users.
>
> - **Status:** design / pre-implementation
> - **Last updated:** 2026-06-01
> - **Companion docs:** [`STORIES.md`](./STORIES.md) — the story-by-story execution breakdown (25 stories, 6 sprints).

---

## 1. Context & motivation

We are building a complete LLM inference application from raw weights to a production-shaped server.
MLX provides the fused kernels (`rms_norm`, `rope`, `scaled_dot_product_attention`, `quantized_matmul`);
we build everything between them — weight loading, tokenizer, transformer graph, KV cache, sampling,
the generation loop — and wrap it in an OpenAI-compatible server with a request queue and continuous
batching.

The design is both pedagogical and practical: the batched KV cache the engine needs *is* the same data
structure the concurrent server needs, so the "learn the internals" path and the "serve many users"
path are one and the same.

**Primary failure mode to design against:** numerical bugs here are **silent** — a wrong RoPE
convention, a transposed weight, or a bad GQA head-repeat produces subtly garbage tokens, not a crash.
The entire engine is therefore validated against an `mlx-lm` golden reference at every numerically
sensitive step.

---

## 2. Goals & non-goals

### Goals
- Load Llama-3.2-family safetensors and run a numerically correct forward pass on Apple Silicon.
- Generate text with greedy + temperature/top-k/top-p sampling and a KV cache.
- Serve an OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`) with **SSE streaming**.
- Handle **concurrent users** via a request queue + **continuous batching** (vLLM-style).
- Be observable and operable: metrics, config, graceful shutdown, OpenAI-shaped errors.

### Non-goals (v1)
- Paged/block KV cache and prefix sharing (contiguous padded cache is sufficient at this scale).
- HTTP/2 / HTTP/3 (SSE over HTTP/1.1 is sufficient).
- Multi-model hosting, LoRA/adapters, speculative decoding, multi-GPU.
- Tool/function calling, vision/multimodal, embeddings endpoints.
- Quantization is **optional** (Phase 7 / MLXFORGE-025), not required for v1.

---

## 3. Target model (frozen)

`mlx-community/Llama-3.2-1B-Instruct` (or an fp16-converted equivalent), **fp16 first**.

| Property | Value |
|---|---|
| Layers | 16 |
| Hidden size | 2048 |
| Query heads | 32 |
| KV heads (GQA) | 8 |
| Head dim | 64 |
| Norm | RMSNorm |
| Positional | RoPE (`rope_theta` from config) |
| MLP | SwiGLU (`down(silu(gate)·up)`) |
| Embeddings | tied (LM head shares embedding weights) |

The architecture generalizes to other LLaMA-family decoders (e.g. Qwen2.5-0.5B), but v1 freezes the
above to keep the golden-reference comparison unambiguous.

---

## 4. Verified MLX C++ API surface

The design references real primitives, confirmed against current `mlx/fast.h`, `mlx/ops.h`, MLX docs,
and mlx-lm source:

- `fast::rms_norm(x, weight, eps, s)`.
- `fast::rope(x, dims, traditional, base, scale, offset, freqs={}, s)` — **two overloads**: `int offset`
  *and* `const array& offset`. The **array-offset overload is the linchpin of batching**: each batch row
  is at a different absolute position, and this applies the correct position per row in one call.
  **The build pins a recent MLX and asserts this overload exists** (older releases only had `int`).
- `fast::scaled_dot_product_attention(q, k, v, scale, mask_mode="", mask_arr={}, sinks={}, s)` — mask
  array may be additive or boolean, ≤4-D, broadcast-compatible with `[B, n_heads, T_q, T_kv]`; the only
  supported `mask_mode` string is `"causal"`; softmax runs fp32 internally; **GQA handled natively**
  (q_heads > kv_heads).
- `ops::quantized_matmul(...)` (in `mlx/ops.h`): `group_size ∈ {32,64,128}`, `bits ∈ {2..8}` — used only
  in the optional quantization phase.
- `mx::load(path)` lazy-loads safetensors **from a string path only**; large models shard with an index
  JSON; some models need a key-name `sanitize` remap. Weights are cast to fp16 right after load.
- **Lazy evaluation:** nothing computes until `mx::eval` / `mx::async_eval`. Unified memory → no
  host/device copy. **Where `eval` is placed determines latency and memory.**

### Reference implementations ported (not reinvented)
- `mlx_lm/models/cache.py::BatchKVCache` — KV layout `(B, n_kv_heads, S_cap, head_dim)`, contiguous,
  **left-padded**, grown in `step=256` blocks; per-row `offset` array; `filter`/`merge`.
- `mlx_lm/models/base.py::create_causal_mask` — additive/bool mask from `arange` comparisons.
- `mlx_lm/generate.py::BatchGenerator` — three-queue scheduler; `prefill_batch_size` /
  `completion_batch_size` / `prefill_step_size`; **one `async_eval` per step** with 1-step lookahead.

---

## 5. Architecture overview

```
            ┌──────────────────────── cpp-httplib (thread pool) ─────────────────────────┐
 clients →  │  /v1/chat/completions  /v1/completions  /v1/models  /health  (SSE or JSON) │
            │     each request thread: build Request, enqueue, then block on its         │
            │     own token queue + condition_variable, write SSE chunks as tokens land  │
            └───────────────┬───────────────────────────────────────────▲───────────────┘
                            │ enqueue Request{token_ids, params,          │ push token + notify
                            │   token_queue, cancelled flag}              │
                            ▼                                             │
                   [ mutex-guarded waiting queue, condition_variable ]    │
                            │                                             │
        ┌═══════════════════▼═════════════ single GPU worker thread ══════╧══════════════┐
        ║  Scheduler loop (the ONLY thread that calls mx::eval / mx::async_eval):         ║
        ║    admit → prefill pass → batched decode step → sample → evict(EOS/max/cancel)  ║
        ║  Owns ALL MLX state: model weights, BatchKVCache (per layer), sampler.          ║
        ╚════════════════════════════════════════════════════════════════════════════════╝
```

**Core invariant:** MLX is **not thread-safe for concurrent eval**. Every MLX op (forward, sampling,
eval, cache mutation) runs on a single worker thread. HTTP threads only touch their own `Request`.

---

## 6. Technology stack & build

- **Language:** C++17/20. **Build:** CMake (FetchContent or submodule for deps).
- **Dependencies:** MLX C++ core (pinned commit), cpp-httplib (header-only HTTP/SSE), `nlohmann/json`,
  a C++ tokenizer (`mlc-ai/tokenizers-cpp` reading HF `tokenizer.json`), and doctest/Catch2 for tests.
- **Artifacts:** `mlxforge` (server binary) and `mlxforge-cli` (local greedy generation, used during bring-up).
- **Platform:** Apple Silicon (Metal, unified memory); startup asserts `mx::metal::is_available()`.

---

## 7. Module decomposition

| Module | Responsibility | Key MLX/lib calls |
|---|---|---|
| `core/config.{h,cpp}` | `ModelConfig` from `config.json` | — |
| `core/weights.{h,cpp}` | load + sanitize + fp16 cast safetensors | `mx::load` |
| `model/llama.{h,cpp}` | embedding, decoder blocks, final norm, LM head | `fast::rms_norm`, `fast::rope` (array offset), `fast::scaled_dot_product_attention` |
| `cache/batch_kv_cache.{h,cpp}` | contiguous left-padded `(B,8,S_cap,64)` KV; `update_and_fetch`/`filter`/`merge`; `offset`/`left_padding`; step=256 growth | `zeros`,`concatenate`,`take` |
| `sample/sampler.{h,cpp}` | greedy/temp/top-k/top-p as graph ops | MLX ops |
| `tokenizer/tokenizer.{h,cpp}` | encode/decode/incremental decode + chat template | tokenizers-cpp |
| `scheduler/scheduler.{h,cpp}` | three-queue state machine; prefill/decode/evict; bucketing; KV gate | — |
| `runtime/worker.{h,cpp}` | single GPU thread; **only** caller of `mx::eval`/`async_eval` | `mx::async_eval` |
| `server/http_server.{h,cpp}` | cpp-httplib routes, SSE provider, JSON (de)serialization, cancellation | cpp-httplib, nlohmann/json |
| `apps/mlxforge_cli.cpp`, `apps/mlxforge.cpp` | CLI and server entry points | — |
| `reference/dump_ref.py`, `tests/` | golden-reference dumps + tensor-compare tests | mlx-lm |

---

## 8. Key data structures & algorithms

### 8.1 `BatchKVCache`
- **Layout:** per layer, K and V as `(B, n_kv_heads=8, S_cap, head_dim=64)`, **contiguous, left-padded**.
- **Growth:** `S_cap` extended in blocks of `step=256` via `zeros` + `concatenate(axis=2)` (amortized).
- **Per-row bookkeeping:** `offset` array (`= -left_padding` at admission, `+= written_len` per write) and
  `left_padding` array. `update_and_fetch(k,v)` writes the populated slice and returns `[..., :idx, :]`.
- **Batch surgery:** `filter(keep_idx)` = `take(axis=0)` on K/V + offset/padding (eviction);
  `merge(other)` = pad both to common `S_cap` then `concatenate(axis=0)` (admission of a prefilled batch).
- **Why contiguous (not paged):** MLX C++ exposes no paged-attention primitive; SDPA wants contiguous
  K/V; at 1B scale, padding waste is bounded and acceptable. Trade-off: no prefix sharing in v1.

### 8.2 Memory model
- **KV per token:** `n_kv_heads × head_dim × 2 (K+V) × 16 layers × 2 bytes (fp16) = 32 KiB/token`.
- 1 seq @2048 = 64 MiB; batch 32 @2048 ≈ 2 GiB; weights (fp16) ≈ 2.5 GiB.
- **Admission gate:** project `(max_len + max_new) × 32 KiB × B` and refuse/queue if it exceeds a
  configured KV budget (MLX allocates until Metal fails — this is the OOM guard).

### 8.3 Ragged-batch attention mask
- Built per step as **additive fp16** of shape `[B, 1, T_q, T_kv]` (avoids the boolean-mask >2³¹ bug):
  `causal = kv_pos <= offset[row]`, `valid = kv_pos >= left_padding[row]`,
  `mask = where(causal & valid, 0, -inf)`.
- Per-row `offset` makes **causal + per-row variable context length** fall out of one comparison.
  **Left-padding is required** so every active row appends its new token at the same physical column.

### 8.4 Sampling
- All sampling is MLX graph ops folded into the forward graph (no host logit readback — that breaks the
  pipeline). Greedy = argmax; temperature/top-k/top-p layered on; seedable for reproducibility.

---

## 9. Continuous-batching scheduler

A single GPU worker thread runs a **three-queue state machine**: `waiting → prefill_batch → decode_batch`.

**Knobs:** `prefill_batch_size` (8), `decode_batch_size` (32, ≥ prefill), `prefill_step_size` (2048,
chunked prefill).

**Prefill is a separate pass, then joined** (not chunk-interleaved into decode): left-pad waiting
requests to common `P_max`, run a dedicated prefill forward (chunked for long prompts, `eval(cache.state)`
at each chunk boundary to bound graph growth), then `merge` into the decode cache. Rationale: prefill
`(B_p, P_max, …)` and decode `(B_d, 1, …)` are different shapes; keeping them as two regular-shaped
passes keeps masks simple.

**Steady-state decode step:**
1. `inputs = decode_batch.last_tokens` `(B,)`; `offsets = cache.offset` `(B,)`.
2. `logits = model(inputs[:,None], cache)[:, -1, :]`.
3. `next, logprobs = sampler(logits)`.
4. **`mx::async_eval(next, logprobs)` — the only eval per step**; read the previous step's result
   (1-step lookahead so graph build for step *t+1* overlaps compute of *t*).
5. push each row's token to its `Request` queue + notify; update finish flags.
6. evict finished/cancelled rows (`filter(keep)`); admit from `waiting`.

**Batch-size bucketing:** pad active `B` to fixed buckets {1,2,4,8,16,32} with masked dummy rows so the
forward graph shape recurs across steps — avoids per-step regraph/recompile cost. `S_cap` is already on
the 256 step.

---

## 10. HTTP API contract (OpenAI-compatible)

### 10.1 Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | chat completion (streaming or full) |
| POST | `/v1/completions` | text completion (streaming or full) |
| GET | `/v1/models` | list served model(s) |
| GET | `/health` | liveness/readiness |
| GET | `/metrics` *(optional)* | Prometheus metrics |

### 10.2 Request (supported fields)
`model`, `messages[]` (chat) or `prompt` (completions), `max_tokens`, `temperature`, `top_p`,
`stream`, `stop`, `n`, `seed`. Chat requests are rendered through the Llama-3.2 chat template before
tokenization.

### 10.3 Responses
- **Non-streaming:** standard `chat.completion` object — `id`, `object`, `created`, `model`,
  `choices[].message`, `finish_reason` (`stop` | `length`), and `usage` (prompt/completion/total tokens).
- **Streaming (`stream: true`):** `Content-Type: text/event-stream`; a sequence of
  `data: {chat.completion.chunk ...}\n\n` frames carrying incremental `choices[].delta`, terminated by
  `data: [DONE]\n\n`.

### 10.4 Streaming mechanics
cpp-httplib `set_chunked_content_provider`: the worker pushes each generated token to the request's
bounded token queue and notifies; the request thread's sink wakes, formats the OpenAI chunk, and writes
it. On EOS → `[DONE]` and close. The per-request queue is bounded (backpressure for slow consumers).
Incremental UTF-8 / byte-BPE detokenization ensures no broken characters are emitted mid-stream.

### 10.5 Cancellation
If the content provider returns false (client disconnect), a per-request `cancelled` atomic is set; the
worker treats it as an eviction at the next iteration boundary, freeing the batch slot while other
streams continue.

### 10.6 Errors (OpenAI error JSON shape)
- `400` invalid/out-of-range parameters or malformed JSON.
- `429` queue full (bounded `waiting` queue overflow).
- `503` model still loading.

---

## 11. Implementation phases (summary)

The work is sequenced in 11 phases (0–10), broken into 25 stories across 6 sprints in
[`STORIES.md`](./STORIES.md). Milestones:

- **Phase 4 / MLXFORGE-008** — numerically correct model (argmax matches mlx-lm).
- **Phase 6 / MLXFORGE-015** — it generates text (greedy matches mlx-lm token-for-token).
- **Phase 8 / MLXFORGE-020** — concurrent batching works (batched output == solo output; throughput scales).
- **Phase 9 / MLXFORGE-023** — OpenAI server live (streaming + non-streaming verified with the `openai` client).
- **Phase 7 / MLXFORGE-025** — optional 4-bit quantization (~2.5 GiB → ~0.7 GiB).

---

## 12. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | Eval placement / serialization (**highest**) | Exactly **one `async_eval` per decode step** over the whole batch; guarded, instrumented invariant. |
| 2 | Regraph/recompile on changing batch shape | Bucket `B` to {1,2,4,8,16,32} with masked dummies; keep `S_cap` on the 256 step; validate uncompiled before `mx::compile`. |
| 3 | RoPE per-row offset overload missing | Assert `fast::rope(const array& offset, …)` exists on pinned MLX; fallback is `freqs` materialization. |
| 4 | Silent numerical bugs (RoPE base/interleave, weight transpose, GQA repeat) | Golden-reference comparison at every phase; exact argmax/token checks. |
| 5 | Graph/memory growth during long prefill | `eval(cache.state)` at each prefill chunk boundary; monitor `mx::metal::get_active_memory()`. |
| 6 | OOM (Metal allocates until failure) | KV admission gate + bounded `waiting` queue (429 on overflow). |
| 7 | Tokenizer streaming emits broken UTF-8 | Incremental UTF-8 / byte-BPE decode so SSE never emits partial characters. |
| 8 | MLX thread-safety | All MLX work on the single worker thread; HTTP threads touch only their `Request`. |
| 9 | Boolean mask >2³¹ bug (#2894) | Use additive fp16 masks, never boolean. |

---

## 13. Testing & verification

### 13.1 Golden-reference discipline
`reference/dump_ref.py` emits `.npy` tensors (tokenized IDs, embeddings, block-0 output, final logits,
greedy token stream) from mlx-lm on the same model + fixed prompts. A C++ harness loads them and asserts
tensor-closeness (fp16 rel ~1e-2) and exact greedy-token equality. This gates every numerically sensitive
story.

### 13.2 Unit tests (`ctest` gate)
A C++ framework (doctest/Catch2) is wired into CMake and registered with `ctest`; green `ctest` is a done
condition. Unit tests cover the pure-logic pieces with no GPU/weights: config parsing, weight-key
`sanitize` + shard index, KV-cache index/offset bookkeeping and `filter`/`merge`, sampler math, OpenAI
request/response (de)serialization, and SSE chunk framing.

### 13.3 End-to-end
- **Scheduler correctness:** N concurrent (identical + distinct) fixed prompts each produce the same
  tokens as a solo run; mixed-length admit/evict still matches.
- **Throughput:** tokens/s measured at concurrency 1/4/8; aggregate throughput rises with concurrency.
- **API parity (official `openai` Python client at `base_url=http://localhost:PORT/v1`):**
  non-streaming returns well-formed `chat.completion` + `usage`; `stream=True` yields incremental
  `chat.completion.chunk` deltas then `[DONE]`; ~16 concurrent clients complete interleaved with no
  garbled UTF-8; a mid-stream disconnect frees that slot while others continue.
- **Robustness:** oversized prompt → 400; queue saturation → 429; `/health` and `/v1/models` correct.

---

## 14. Operational concerns

- **Config** (CLI flags / env): model path, port, max batch, KV budget, context length, thread count.
- **Observability:** structured logging + per-request metrics (TTFT, tokens/s, queue depth, batch
  occupancy); optional `/metrics`.
- **Lifecycle:** graceful shutdown drains in-flight requests before exit.
- **Backpressure:** bounded `waiting` queue (429) and bounded per-request token queues.
