# Architecture

This document describes how the engine is put together: the threading model, the
request lifecycle, the continuous-batching scheduler, and the module layout. For
the math inside the transformer see [llm-architecture.md](./llm-architecture.md).

## Design goals

- **Numerical correctness first.** Match `mlx-lm` token-for-token on greedy
  decoding. Everything else is subordinate to that.
- **Serve concurrent users.** A vLLM-style continuous-batching scheduler so many
  in-flight requests share each GPU step instead of running one at a time.
- **One reusable data structure.** The batched KV cache the engine needs for
  efficiency *is* the same structure the concurrent server needs. The "learn the
  internals" path and the "serve many users" path are the same path.
- **Operable.** OpenAI-compatible API, SSE streaming, cancellation, per-request
  metrics, graceful shutdown, OpenAI-shaped errors.

## The central invariant: one GPU thread owns all MLX state

MLX is **not safe for concurrent evaluation**, and MLX arrays are **thread-bound**
— GPU work must run on the thread that created the arrays. The entire design
falls out of this:

- A single **worker thread** (`runtime/worker`) loads the model *on itself* and
  is the **only** thread that ever calls `mx::eval` / `mx::async_eval` or touches
  the weights, the `BatchKVCache`, or the sampler.
- Every other thread (the cpp-httplib request threads, test threads) touches only
  its own `Request` struct and the thread-safe `Scheduler` queue. They never call
  an MLX op.

This is enforced by structure, not by locking the GPU: there is exactly one place
MLX runs, so there is nothing to contend over.

```
            ┌──────────────── cpp-httplib (its own thread pool) ─────────────────┐
 clients →  │  /v1/chat/completions  /v1/completions  /v1/models  /health        │
            │  each request thread: parse → chat template → tokenize → build a    │
            │  Request, submit() it, then block on its own TokenQueue and stream  │
            └───────────────┬──────────────────────────────────────▲────────────┘
                            │ submit(Request)                       │ push token + notify
                            ▼                                        │
                   [ Scheduler: waiting deque, mutex + cv ]          │
                            │ next_waiting / take_waiting            │
        ┌═══════════════════▼═════════ single GPU worker thread ═════╧══════════════┐
        ║  run() loop — the ONLY caller of mx::eval / mx::async_eval:                ║
        ║    admit (prefill → merge) → decode step (1 async_eval) → evict (filter)   ║
        ║  Owns: model weights, the persistent decode BatchKVCache, the sampler.     ║
        ╚═══════════════════════════════════════════════════════════════════════════┝
```

## Model source resolution

Before anything is loaded, both binaries turn the user-supplied model **spec**
into a concrete local directory via `mlxforge::resolve_model_dir`
(`src/core/model_source.{h,cpp}`). A spec is either a local directory or a
HuggingFace repo id (`org/name`), giving llama.cpp-style ergonomics. Resolution
is tiered, cheapest first:

1. an existing dir containing `config.json` → used as-is;
2. an existing HF cache parent (`…/models--org--name`) → its `snapshots/<rev>/`
   is read from `refs/main` (this is why passing the cache parent now works
   instead of failing on a missing `config.json`);
3. a repo id already in the HF hub cache (`HF_HUB_CACHE`/`HF_HOME`/`~/.cache/huggingface/hub`)
   → that snapshot is reused;
4. a repo id already downloaded by mlxforge → that download is reused;
5. otherwise the repo is **downloaded** into `$MLXFORGE_CACHE`
   (default `~/.cache/mlxforge`).

Downloading lives in `src/core/hf_download.{h,cpp}` — the only HTTP **client** in
the tree (cpp-httplib is server-only). It lists the repo's files via the HF
model-info API, filters to the inference set (config/tokenizer JSON +
`*.safetensors`, excluding PyTorch/GGUF/ONNX weights), and fetches each with
system libcurl, following the cross-host 302 redirect to the LFS CDN. Files are
staged in a sibling `.incomplete-<pid>` directory and the whole directory is
renamed into place only after every file succeeds, so a half-finished pull never
looks like a usable model and a re-run resumes cleanly. The server resolves
**once** on the main thread before constructing the worker, so the
network/cache lookup never touches the GPU thread.

## The request lifecycle

A request moves through three logical states: **waiting → prefill → decode**, and
finally **eviction**.

1. **Submit (any thread).** The HTTP layer parses the OpenAI body, renders the
   chat template, tokenizes the prompt, and builds a `Request` carrying the prompt
   ids, sampling params, `max_tokens`, the EOS ids, a bounded `TokenQueue`, and a
   `cancelled` atomic. `Scheduler::submit()` pushes it onto the `waiting` deque and
   notifies the worker. If the waiting queue is full it returns `false` (the
   server replies `429`).

2. **Admit / prefill (worker).** When the worker has spare capacity it drains up
   to `kPrefillBatchSize` (8) waiting requests, left-pads their prompts to a
   common length `P_max`, and runs a **dedicated prefill forward** (see below).
   The prefilled rows are `merge`d into the persistent decode `BatchKVCache`, and
   each row's first token is sampled.

3. **Decode (worker).** Each loop iteration runs exactly **one** batched decode
   step over all active rows: one `model.forward(inputs, cache)`, one sample, one
   `mx::async_eval`. Each row's new token is pushed to that request's `TokenQueue`
   (which wakes its HTTP thread to stream it).

4. **Evict (worker).** A row finishes when it emits an EOS token (`finish_reason =
   "stop"`), reaches `max_tokens` (`"length"`), or is cancelled by client
   disconnect (`"cancel"`). Finished rows are dropped from the cache via
   `filter(keep)` and their `TokenQueue` is `close()`d, ending the response. The
   freed slots are filled by admitting more waiting requests.

The producer/consumer handoff per request is a bounded, blocking `TokenQueue`
(single-producer worker, single-consumer request thread). Bounding it gives
backpressure: a slow SSE client cannot make the worker accumulate unbounded
tokens.

## Continuous batching in detail

The scheduler keeps a single persistent decode batch and continually admits new
work into it and evicts finished work from it — rather than running fixed batches
to completion. The three moving parts:

### Prefill is a separate pass, then joined — and interleaved with decode

Prefill shape `(B_prefill, P_max, …)` and decode shape `(B_decode, 1, …)` are
different, so they are kept as two regular-shaped passes (never mixed into one
ragged batch). Prefill:

- Left-pads all prompts in the batch to a common `P_max` (left-padding so every
  row's last real token sits at the same physical column, `P_max - 1`).
- Runs the forward in chunks, calling `cache.eval_state()` at each chunk
  boundary to bound graph/memory growth.
- Produces a populated `BatchKVCache` and the last-position logits per row. The
  worker `merge`s this cache into the live decode cache.

**Chunked-prefill interleaving** (`prefill_chunk`, default 256; `0` =
monolithic): rather than running an admission's whole prefill before the next
decode step, the worker queues it as a *pending unit* and advances it one
`prefill_chunk`-token chunk per loop iteration, with a decode step in between —
so in-flight rows keep streaming while new prompts prefill (+25–35% batched
throughput, up to 60% lower TTFT under load; chunking also pipelines better
through Metal than one monolithic forward). Cold admissions share one batched
unit; a prefix-cache hit becomes its own single-row unit (seeded cache +
uncached suffix). Units complete FIFO; chunk boundaries change only *when*
`eval_state()` runs, not what is computed, so the greedy stream is gated
token-identical across chunk sizes (`tests/scheduler/worker_test.cpp`). At
shutdown the loop drains pending units before exiting. The monolithic path
(`prefill(...)` in `runtime/batching`, internal chunking at `kPrefillStepSize`
2048) remains for `prefill_chunk = 0`.

### The steady-state decode step

In `Worker::decode_step()`:

1. Gather the next input token per row (host side) into `inputs` `(B, 1)`.
2. `logits = model.forward(inputs, cache)` → `(B, 1, vocab)`. The batched forward
   reads each row's RoPE position from `cache.offset()` and builds a ragged
   additive mask, so left-padded rows attend over exactly their own real history.
3. `next = Sampler::greedy(logits)` (or full sampling).
4. **`mx::async_eval(next)` — the one and only eval for the whole batch this
   step.** Never per-row, never per-layer. `Worker::decode_steps()` counts these
   calls; under load it is far below the number of tokens produced, which is the
   proof that one eval covers the whole batch.
5. Read the chosen ids back to the host, push each row's token, mark finished rows.

**Multi-row GEMV kernels** (`skinny_mm`, default on): a decode step is
weight-bandwidth-bound, and MLX's tiled GEMM drops to ~1/3 of GEMV bandwidth
the moment the batch reaches 2 rows (ml-explore/mlx#3661) — historically a
~2.6× per-row decode cliff between B=1 and B=2. `model/skinny_matmul` provides
custom `fast::metal_kernel` kernels for the dense-fp16 decode shape, picked by
(B, weight size): a one-column-per-simdgroup scalar variant for B≤4 (~GEMV
bandwidth), a two-column scalar variant for B∈[5,16] on the small per-layer
weights (barrier-free simdgroups tolerate the latency of short back-to-back
ops best), and a simdgroup-matrix MMA variant for B∈[5,32] on big weights —
in practice the vocab head — using hardware `simdgroup_half8x8`
multiply-accumulates over 8-output-column tiles, with the activation chunk
staged once per threadgroup in threadgroup memory and each weight element
streamed from device exactly once grid-wide (~134 GB/s at B=8, still ~68 at
B=32, vs the GEMM's flat ~55). Anywhere else the tiled GEMM wins and
`linear()` falls back. Accumulation is fp32 in a different order than `mx::matmul`, so
logits differ at fp16-noise scale; the gate is row-for-row token equality of a
kernel-on batch against the stock-matmul batch
(`tests/scheduler/worker_test.cpp`), plus a pure-kernel `allclose` grid
(`tests/model/skinny_matmul_test.cpp`).

### Batch-size bucketing

If the active batch shape changed every step, MLX would re-trace/re-compile the
graph each time. To keep the graph shape stable, the active batch size `B` is
rounded up to a fixed **bucket** in `{1, 2, 4, 8, 16, 32}` (and multiples of 32
beyond that) by appending **masked dummy rows** (`BatchKVCache::pad_dummies`).
Dummy rows attend only to their own position, so they can never produce NaNs or
affect the real rows; they are trimmed back with `filter()` after the step.

## Memory and the OOM guard

MLX allocates Metal memory until allocation fails — there is no soft limit. The
KV cache is the dominant growing allocation, so `cache/kv_budget` projects a
batch's peak KV footprint before admission:

```
bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × sizeof(fp16)
```

For Llama-3.2-1B that is `2 × 16 × 8 × 64 × 2 = 32 KiB/token`. A candidate batch
of `B` sequences each reaching `max_len + max_new` tokens is refused/queued if it
would exceed the configured `--kv-budget`. Combined with the bounded waiting
queue (which returns `429` on overflow), this is the real OOM defence.

## KV-cache quantization

`--kv-bits 8|4` (engine option `kv_bits`; default 0 = dense fp16) stores the KV
cache quantized, cutting its memory ~1.9× (8-bit, near-lossless) or ~3.6×
(4-bit). The port mirrors mlx-lm exactly:

- **Storage** (`cache/kv_quant`): each cached K/V tensor is the `mx::quantize`
  triplet — packed uint32 words plus per-group fp16 scales and biases
  (group size 64) — quantized **at write time**, per position, so prefill
  chunking cannot change stored values. Both `KVCache` and `BatchKVCache` hold
  per-layer component vectors (1 array dense, 3 quantized); all batch surgery
  (`filter`/`merge`/`pad_dummies`, block growth) runs per component unchanged.
- **Attention** (`model/sdpa`): MLX has no fused quantized SDPA kernel, so
  `quantized_sdpa` ports `mlx_lm/models/base.py` op-for-op — `quantized_matmul`
  for the scores and the output, GQA via a `(B, n_kv, n_rep, L, D)` reshape,
  precise softmax. `sdpa_with_cache` is the dispatch seam every model attention
  call site uses (dense fast-kernel vs quantized path, by the cache's config).
- **Setting scope**: engine-wide, never per-request — the batched cache's
  storage is physically shared across rows. Unsupported setups (vision-language
  and hybrid Qwen3.5 models, which have no quantized golden reference yet;
  group sizes that don't divide `head_dim`) **fail engine creation**; there is
  no silent fp16 fallback.
- **Gating**: teacher-forced greedy walks against mlx-lm `QuantizedKVCache`
  streams (Llama + Qwen3, 8- and 4-bit), asserting token equality at every step
  whose reference top-2 margin clears the fusion-context noise (quantized
  matmuls shift ~1 logit between lazy and materialized inputs, so bit-exact
  cross-implementation gating is unsound); plus an exact batched-vs-single-
  stream coherence gate.

The per-token budget figure adjusts accordingly: a K-or-V head row is
`head_dim × bits/8` packed bytes plus a fp16 scale and bias per group (D=64/g=64:
68 B at 8-bit, 36 B at 4-bit, vs 128 B fp16).

## Prefix cache (block-pool KV storage, optional SSD tier)

`--prefix-cache 1` (engine option `prefix_cache`; default off) reuses KV across
requests that share a token prefix — the shared-system-prompt and multi-turn
shapes. On a 2048-token shared prefix the warm TTFT drops ~20× (see
`mlxforge-cli bench-prefix`); decode throughput is unchanged.

The design is **gather-on-admit, not paged attention**: MLX has no paged SDPA
kernel and mlx-lm has no paged reference to gate one against, so the decode
batch stays the contiguous left-padded `BatchKVCache`. The *pages* live in a
pool instead:

- **BlockPool** (`cache/block_pool`): immutable `kv_block_size`-token blocks
  (default 256; all layers, dense or quantized component vectors), keyed by a
  salted **chain hash** of the token-id prefix — a key identifies the entire
  prefix up to the block's end. LRU-evicted under `--kv-pool` bytes.
- **PrefixCache** (`cache/prefix_cache`): longest-chain match (clamped to
  `prompt_len - 1`: the last prompt token is always recomputed so admission
  still yields next-token logits) + harvest policy. On admission, matched
  blocks seed a batch-1 cache (`BatchKVCache::from_prefix`, written through
  the standard block-grow writer so buffer layout matches a cold prefill) and
  only the suffix is prefilled (`prefill_with_prefix`); the row then merges
  into the decode batch like any single-row admission.
- **Harvest is prompt-only.** When a row finishes, only its *prompt* span is
  sealed into the pool. Decode-produced K/V differs from a recompute by fp16
  accumulation order (the decode-vs-recompute gap) and demonstrably flips
  later greedy choices; prefill-produced K/V is the proven exact-stable class,
  so warm == cold stays token-exact. Multi-turn reuse still converges: the
  next turn's prompt contains the prior answer as text, so its (seeded)
  prefill recomputes that span once and pools it. Multimodal rows are never
  harvested (a token-id hash cannot identify image content / 3D positions).
- **SSD tier** (`cache/block_store`, `--kv-spill-dir`): RAM-evicted blocks are
  serialized on the worker thread and written by a byte-only IO thread (one
  0600 file per block, salted-hash name, tmp+rename); a pool miss revives the
  file synchronously. The directory is rescanned at startup, so the cache
  survives restarts; the salt (model fingerprint + storage config + block
  size) is verified on load, so a block can never be revived for a different
  model or quantization setting.

Engine-wide, like `kv_bits` (the pool stores one storage layout); hybrid
(Qwen3.5) and vision-language models reject the option at engine creation. The
gate is **warm == cold**: reuse may change speed, never tokens — no new
mlx-lm fixtures are needed because the cold path is already golden-gated and
prefix reuse is an engine-internal equivalence property
(`tests/scheduler/prefix_reuse_test.cpp`, `tests/scheduler/prefix_spill_test.cpp`).

## Module map

Source lives under `src/`, grouped by responsibility. Tests mirror the module
path under `tests/`.

| Module | Responsibility |
| --- | --- |
| `core/config` | Parse `config.json` into a `ModelConfig` (hyperparameters, `rope_scaling`, quantization, EOS/BOS ids). |
| `core/weights` | Load safetensors (single-file or sharded + index JSON), sanitize key names, cast every tensor to fp16. |
| `core/model_source` | Resolve a model spec (local dir or HF repo id) to a local snapshot dir; HF-cache reuse + download fallback. |
| `core/hf_download` | Download HF repos via libcurl (the only HTTP client): list files, filter, fetch through the CDN redirect, atomic rename. |
| `core/env` | Tiny `env_or`/`env_long` helpers for environment-variable overrides. |
| `model/` | The transformer: `DecoderModel` base (embedding, RMSNorm, RoPE, GQA SDPA, SwiGLU, LM head; fp16 and quantized paths; single-stream and batched forward) with `LlamaModel`/`Qwen3Model`/`Qwen3MoeModel` subclasses and a `create_model` factory. |
| `cache/kv_cache` | Single-sequence KV cache (the simplest prefill/decode split). |
| `cache/batch_kv_cache` | Batched, left-padded KV cache: `update_and_fetch`, `filter` (evict), `merge` (admit), `pad_dummies` (bucketing). |
| `cache/kv_quant` | Quantized-KV shared types (`KVQuantConfig`, triplets) + the block-grow component writer both caches use. |
| `cache/kv_budget` | KV memory projection / admission gate (fp16 and quantized accounting). |
| `cache/block_pool` | Prefix-cache pages: immutable KV blocks keyed by a salted chain hash, LRU under a byte budget. |
| `cache/prefix_cache` | Longest-prefix block matching + prompt-only harvest over the pool. |
| `cache/block_store` | SSD spill tier: byte-only writer thread, salted/versioned block files, restart rescan. |
| `model/sdpa` | Cache-aware SDPA dispatch: dense fast kernel vs the hand-rolled quantized path (mlx-lm port). |
| `sample/sampler` | greedy / temperature / top-k / top-p, all as MLX graph ops. |
| `scheduler/request` | The `Request` struct and the bounded, blocking `TokenQueue`. |
| `scheduler/scheduler` | The waiting queue + worker handoff (mutex + condition variable). |
| `runtime/worker` | The single GPU worker thread: the admit / decode / evict loop. |
| `runtime/batching` | The prefill pass and the batch-size bucketing helper. |
| `runtime/single_stream` | The CLI's greedy single-stream generation loop. |
| `tokenizer/tokenizer` | tokenizer facade: chat templates + the UTF-8-safe streaming detokenizer. |
| `tokenizer/bpe` | from-scratch byte-level BPE (`BpeTokenizer`): encode/decode over `tokenizer.json`. |
| `server/openai` | OpenAI request parsing + response serialization (pure functions, unit-tested). |
| `server/http_server` | cpp-httplib routes, the blocking and SSE handlers, OpenAI error shapes. |
| `server/config` | CLI-flag / environment server configuration. |
| `apps/mlxforge` | The server binary entry point. |
| `apps/mlxforge-cli` | The CLI binary (smoke test, weight dump, generation). |

## Why these structural choices

- **Contiguous, left-padded KV cache (not paged).** MLX C++ exposes no
  paged-attention primitive and SDPA wants contiguous K/V. At the ~1B scale the
  padding waste is bounded and acceptable. The trade-off is no prefix sharing.
- **Additive fp16 masks, never boolean.** A known MLX boolean-mask bug (#2894)
  forces additive masks; the mask is built per step in `DecoderModel::batch_mask`.
- **Sampling stays on the GPU.** Pulling logits back to the host to sample would
  serialize the decode pipeline; sampling is expressed as graph ops folded into
  the forward graph so the single `async_eval` realizes both.
