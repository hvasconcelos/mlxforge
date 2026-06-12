# Prefill & decode performance research (June 2026)

A survey of inference-performance techniques (2024–2026) applicable to mlxforge,
prioritized by measured Apple-Silicon evidence, implementation effort, and
compatibility with the engine's invariants (one `async_eval` per decode step,
token-exact golden gates, thread-bound arrays, additive fp16 masks). Each item
notes whether it is **exact** (token-preserving, gateable against the existing
golden reference) or **approximate** (needs an opt-in tier with its own quality
gate).

Baseline for this survey (what the engine already does): continuous batching
with chunked-prefill interleaving (`prefill_chunk` = 256 default,
`kPrefillStepSize` = 2048, decode step between chunks), batched cold prefill up
to 8 prompts, GPU-side sampling inside the single per-step `async_eval`,
batch-size bucketing, quantized KV (8/4-bit) with hand-rolled `quantized_sdpa`,
exact-prefix block pool + SSD spill, optional skinny-GEMV decode matmuls. No
`mx::compile`, no multi-step `async_eval` pipelining, no speculative decoding,
no wired-memory management.

## Framework status (decides what is even possible)

- **MLX v0.31.2 (the current pin) is the latest tagged release** as of
  2026-06-12 — there is no upgrade waiting. Post-release `main` is bugfix-only;
  two items matter: a `fast::rope` fix for **single-token, multi-sequence
  batches** ([mlx#3498](https://github.com/ml-explore/mlx/pull/3498)) — exactly
  the batched-decode call shape we use, worth verifying whether v0.31.2 has the
  bug for our usage — and the `MLX_SDPA_BLOCKS` env knob
  ([mlx#3455](https://github.com/ml-explore/mlx/pull/3455)) for tuning the
  two-pass vector (decode) SDPA kernel on long contexts.
- **No fused quantized-KV SDPA is coming upstream.** The feature request
  ([mlx#3404](https://github.com/ml-explore/mlx/issues/3404)) was closed
  without action; mlx-lm still hand-rolls it too. Our `quantized_sdpa` remains
  the right architecture. Community kernels exist (
  [mlx-qsdpa](https://github.com/Thump604/mlx-qsdpa): fused 4/8-bit decode
  attention, claims 1.7× over the two-`quantized_matmul` pattern at 128k
  context) but are unaudited — golden-gate before considering.
- **MLX has no varlen/ragged attention API** and a packing-utilities request
  was closed wontfix ([mlx#1248](https://github.com/ml-explore/mlx/issues/1248)).
  Ragged batching on MLX means padding + additive masks (what we do) or packing
  with block-diagonal masks (§ Tier 2). Paged attention stays out-of-tree
  ([mlx#2955](https://github.com/ml-explore/mlx/issues/2955)).
- **`fast::metal_kernel` is first-class from C++** (template args, explicit
  grid mapping, JIT-compiled once and cached) and is what the community
  quantized-SDPA kernels are built on — a viable escape hatch when composed MLX
  ops show up in profiles.
- **MLX fused-SDPA dispatch is conditional**: the fast decode (vector) kernels
  require head dims 64/96/128/256 and `T_q·gqa_factor ≤ 32`; the fused prefill
  ("steel", flash-style) kernel requires head dims 64/80/128. Anything else
  silently falls back to the slow composite path. Cheap hardening: assert the
  dispatch conditions per model at load.
- **M5 GPU neural accelerators** give ~4× prefill (TTFT) vs M4 for free through
  `mx::fast::scaled_dot_product_attention` (the NAx kernel variant); decode is
  only ~1.2× (bandwidth-bound). No code change, but it shifts the
  prefill/decode balance on new hardware.
  ([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5))

## Tier 1 — low effort, exact, proven on Apple Silicon

### 1. Wired memory limit + allocator-cache hygiene
`mx::set_wired_limit` (Metal residency sets, macOS ≥ 15,
[mlx#1510](https://github.com/ml-explore/mlx/pull/1510)) pins weights so the OS
can't compress/swap idle pages between decode steps. Measured on M2 Ultra,
Llama-3-70B fp16: generation **0.23 → 4.7 tok/s (~20×)**, prompt ~12×
([mlx-examples#1069](https://github.com/ml-explore/mlx-examples/pull/1069)).
The effect is binary — huge near the memory ceiling, ~nil when comfortably
resident — which is precisely the regime a batched server with growing KV
occupies. Set once on the worker thread after weight load (adding buffers to
the residency set is expensive; never per-step). Pair with periodic
`mx::clear_cache()` in the decode loop (mlx-lm does it every 256 tokens;
unbounded allocator-cache growth has caused kernel panics in `mlx_lm.server`,
[mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)). Token-exact.
**Probably the highest ROI/effort item in this document.**

### 2. n-gram / prompt-lookup speculative decoding
Draft tokens by matching the tail of the generated text against the
prompt/context and copying the continuation — no draft model, no extra forward
pass for drafting; the verify pass is one batched forward, so the
one-`async_eval`-per-step invariant holds. **2–4× on input-grounded tasks**
(code editing, RAG, summarization;
[prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding));
shipped in vLLM (`method: "ngram"`), transformers, llama.cpp. Greedy
verification is token-exact (it emits exactly the target's greedy sequence);
rejection sampling provably preserves the sampled distribution
([Leviathan et al.](https://arxiv.org/abs/2211.17192),
[Chen et al.](https://arxiv.org/abs/2302.01318)). It is the only speculative
method with no Apple-Silicon downside case: drafting cost is ~zero, so the
worst case is wasted verify width (cf. MoE draft-model regressions in Tier 2).
Batched integration still needs the ragged-acceptance handling of § Tier 2.4 —
start single-rows-in-batch (speculate only when the batch is small).

### 3. Compile the sampler chain
mlx-lm's proven pattern: do **not** compile the model forward (KV growth
changes shapes every step → recompile churn); compile only the fixed-shape
sampling ops (`top_k`/`top_p`/`min_p`/`categorical`), threading
`mx::random` state through compiled inputs/outputs
([sample_utils.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py)).
Our logits are `(B, vocab)` with B already bucketed, so compiled shapes recur.
Related dispatch-count win: build per-row sampler subgraphs as one batched op
rather than B per-row subgraphs concatenated (each per-row op is an extra
Metal dispatch per step). Fusion can shift fp16 accumulation order — gate at
token level, as usual.

### 4. Prefill chunk-size and scheduling tuning
On Apple GPUs the per-chunk efficiency optimum is **large**: raising mlx-lm's
chunk 512 → 8192 gave 1.22–1.56× prefill on M1 Pro (16384 regressed under
memory pressure; [lmstudio-js#507](https://github.com/lmstudio-ai/lmstudio-js/issues/507)).
That is in tension with small chunks protecting inter-token latency — the
Sarathi-Serve answer ([OSDI '24](https://arxiv.org/abs/2403.02310), up to 2.6×
serving capacity; vLLM V1's default scheduling mode) is a per-iteration
**token budget**: admit all decode rows first, fill the remainder of the
budget with prefill chunk(s), one mixed batch, one `async_eval`. Our
interleaved mode (one 256-token chunk, then a separate decode step) is the
right skeleton; the upgrades are (a) a budget-based chunk size in the 1–4k
range instead of fixed 256, ideally adaptive to in-flight decode count, and
(b) optionally fusing the chunk and the decode rows into one step. Chunked
attention is the same math as monolithic prefill modulo fp16 accumulation
order (same class as the documented decode-vs-recompute gap) — gate
exact-token, not raw-logit.

### 5. Deepen `async_eval` pipelining
mlx-lm keeps one step in flight: submit step N (`async_eval`), build and
submit step N+1's graph, *then* synchronize on step N's token readback —
hiding graph-construction cost behind GPU execution
([generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py),
[Writing Fast MLX](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)).
Our loop is strictly sequential (build → `async_eval` → read tokens → repeat).
The C++ graph-build cost is smaller than Python's but nonzero at large B;
the complication is that step N+1's input *is* step N's sampled token, so true
lookahead needs speculative graph reuse or readback restructuring — measure
the per-step CPU gap first (GPU utilization via `mactop`; any hidden sync
point — `.item()`, `data<T>()`, sync `eval` on the same stream — destroys the
overlap, cf. [mlx-examples#1040](https://github.com/ml-explore/mlx-examples/pull/1040)).
Token-exact (pure scheduling).

## Tier 2 — medium effort, high value

### 1. Native MTP-head speculative decoding (Qwen3.5)
Qwen3.5 checkpoints **ship** a one-layer multi-token-prediction head
(`mtp_num_hidden_layers: 1`) — a free, perfectly matched EAGLE-style draft, no
training. Measured on MLX: **15.7 → 24.6 tok/s (1.57×, 88% acceptance)** for
Qwen3.5-27B-4bit on M4 Pro ([mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990));
[mtplx](https://github.com/youssofal/mtplx) reports 2.24× on Qwen3.6-27B with
lossless rejection sampling at any temperature. Caveats: MoE acceptance is poor
(9–11% — one MTP layer can't predict expert routing), and the Qwen3.5 hybrids
need **SSM/conv state rollback** on rejection (mlx-lm's PR implements it; real
engineering for our cache layer). Contrast llama.cpp's Metal MTP, which is a
net **loss** at every config on M1 Max
([llama.cpp#23752](https://github.com/ggml-org/llama.cpp/issues/23752)) — the
win is implementation-sensitive: per-step draft overhead must be tiny and the
draft depth auto-tuned. SGLang/vLLM treat MTP as the production spec-decode
path (up to +60% lossless on DeepSeek-class models).

### 2. Packed multi-prompt prefill (Prepacking)
Bin-pack several variable-length prompts into one sequence with a
block-diagonal mask and per-segment RoPE offsets — one prefill pass computes
several prompts' KV with no padding waste. **Up to 6× prefill** vs padded
batching, growing with length variance ([AISTATS '25](https://arxiv.org/abs/2404.09529)).
**Exact** (masking/position trick only) and a natural fit: our masks are
already additive fp16 and the fused steel kernel accepts array masks. Costs:
the mask is materialized O(T²), and fused-kernel head-dim constraints apply.
Replaces the left-pad-to-`P_max` waste in the batched cold-prefill path when
prompt lengths diverge.

### 3. Fused Metal kernels for profiled hot spots
Two candidates with precedent, both via `fast::metal_kernel`:
(a) the **kv-quant write path** — a composed quantize pipeline generated
thousands of tiny dispatches per token on a 32B model; fusing it into one
kernel was **2.7× for that stage**
([TurboQuant-on-MLX](https://medium.com/@antonrozanov/turboquant-on-mlx-4-6x-kv-cache-compression-with-custom-metal-kernels-9cdee3f7d2a2));
(b) a **fused quantized-KV decode SDPA** (mlx-qsdpa precedent, ~1.7× at long
context). Both replace exact-gated numerics with new accumulation orders —
budget for fixture work, and keep the margin-gated (not raw-exact) comparison
style the kv-quant gates already use.

### 4. Batched speculative decoding, done right
Per-row acceptance lengths make ragged batches, and an audit found essentially
every batch spec-decode implementation **violated output equivalence**
(outputs from subtle drift to gibberish — vLLM/SGLang included on Qwen3):
["Batch Speculative Decoding Done Right"](https://arxiv.org/abs/2510.22876)
(EqSpec/EXSpec: formal sync invariants + grouping same-acceptance-length rows;
up to 3× at batch 8). Speedup decays with batch size (EAGLE-2 goes negative by
batch ~24 on H100; expect an earlier crossover on M-series) — so speculation
depth must shrink with load, down to off (vLLM's dynamic speculative decoding,
[TurboSpec](https://arxiv.org/pdf/2406.14066)). **Nobody in the MLX ecosystem
ships batched spec decode** (LM Studio's engine explicitly errors on it) —
this is a gap mlxforge could own, and it is exactly the feature class the
golden-gate discipline exists for (mlx-lm shipped a silent token-dropping
spec-decode bug, [mlx-lm#846](https://github.com/ml-explore/mlx-lm/issues/846),
and an output-corruption bug fixed in v0.30.6). Gate every variant token-exact
greedy against the non-speculative stream.

## Tier 3 — high effort or approximate (opt-in only)

- **Paged / varlen Metal attention.** Where the frontier moved in 2026:
  [vllm-metal](https://github.com/vllm-project/vllm-metal) v0.2.0 made a
  unified paged-varlen Metal attention kernel its default (project-reported
  83× TTFT / 3.6× throughput vs its v0.1.0 contiguous path under load) and is
  adding occupancy-gated split-KV flash-decoding; mistral.rs's Metal
  PagedAttention measured **+77–131%** under batching on M3 Max
  ([mlx#2228](https://github.com/ml-explore/mlx/issues/2228), never landed in
  MLX). Removes left-padding waste and grow-and-`slice_update` copies — but it
  means hand-written kernels outside MLX ops, block tables through the whole
  attention path, and a full numerics re-validation (vllm-metal hit silent MoE
  routing divergence from paged attention,
  [vllm-metal#281](https://github.com/vllm-project/vllm-metal/issues/281)).
  Revisit if/when batch sizes and context lengths make padding waste dominant.
- **Sliding window + attention sinks (StreamingLLM / RotatingKVCache).**
  Production-adopted (HF, TensorRT-LLM; mlx-lm ships `RotatingKVCache(keep=4)`
  as a reference implementation), unbounded context at fixed KV cost — but
  **approximate** (evicted tokens are gone). If added: opt-in engine setting,
  excluded from prefix-pool harvest, own quality gate.
- **Draft-model speculative decoding.** mlx-lm's gives ~1.8× on 32B-class
  dense targets (M3 Max) but regresses on 14B and is a **35–45% slowdown on
  MoE targets** whose active params ≈ draft size
  ([mlx-lm#1132](https://github.com/ml-explore/mlx-lm/issues/1132)) — wrong
  default for our Qwen3-MoE support; MTP heads and n-gram dominate it here.
- **Not recommended:** SnapKV/H2O/PyramidKV eviction (research-grade,
  query-dependent accuracy cliffs, breaks warm==cold reuse); CacheBlend/EPIC
  non-prefix cache fusion (2.2–3.3× TTFT on RAG but inherently approximate —
  violates the exact-prefix gate; only ever as a quarantined opt-in tier);
  prompt compression (compressor cost lands on the same shared GPU);
  vllm-mlx's sparse prefill / MoE expert reduction (approximate); YOCO/CLA
  shared-KV (requires retrained checkpoints).

## Where mlxforge stands vs peers

Feature parity with vllm-mlx and oMLX is already there or better — continuous
batching, exact-prefix cache, SSD tier, quantized KV — with a stricter
exactness discipline than any of them document. The two real gaps vs the 2026
ecosystem are **speculative decoding** (mlx-lm, vllm-mlx `--mtp`, mtplx,
dflash all ship 1.4–2.3× single-stream wins on M-series) and, longer-term,
**paged/varlen attention kernels**. The recommended sequence: Tier 1 items
1–3 first (days each, token-exact, measurable), then n-gram spec decode with
load-adaptive depth, then Qwen3.5 MTP, with packed prefill and fused kernels
driven by profiling.
