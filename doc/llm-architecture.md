# LLM architecture: the forward pass

This document describes the transformer the engine actually runs — a LLaMA-family
decoder-only model — and how each stage maps onto MLX primitives. This is the
numerically-sensitive part of the codebase: every stage below is gated against an
`mlx-lm` golden reference (see [supported-models.md](./supported-models.md) and
[contributing.md](./contributing.md)).

All of this lives in `model/llama.{h,cpp}` in a single `LlamaModel` class. The
architecture is **shared across the LLaMA family** — onboarding a new family
generally needs no forward-pass changes, only tokenizer/chat-format work.

## The decoder, end to end

For input token ids of shape `(B, L)`:

```
tokens (B, L)
   │  embedding lookup
   ▼
h (B, L, hidden)
   │
   ├── for each of n_layers decoder blocks:
   │      x_norm   = RMSNorm(h, input_layernorm)
   │      q,k,v    = project(x_norm)              # GQA: n_heads q, n_kv_heads k/v
   │      q,k      = RoPE(q,k, position)          # llama3-scaled
   │      a        = SDPA(q,k,v, causal/mask)     # scaled dot-product attention
   │      h        = h + o_proj(a)                # residual
   │      h_norm   = RMSNorm(h, post_attention_layernorm)
   │      h        = h + down(silu(gate(h_norm)) * up(h_norm))   # SwiGLU, residual
   │
   ▼
h = RMSNorm(h, model.norm)
   │  LM head  (tied embedding, or a separate lm_head.weight)
   ▼
logits (B, L, vocab)
```

### The reference model

The primary target is `Llama-3.2-1B-Instruct`:

| Property | Value |
| --- | --- |
| Layers | 16 |
| Hidden size | 2048 |
| Query heads | 32 |
| KV heads (GQA) | 8 |
| Head dim | 64 |
| Norm | RMSNorm |
| Positional | RoPE, `rope_theta` from config, llama3 frequency scaling |
| MLP | SwiGLU (`down(silu(gate)·up)`) |
| Embeddings | tied (the LM head reuses the embedding weights) |

These values are read from `config.json` into `ModelConfig`, never hard-coded, so
the same code runs other LLaMA-family decoders (e.g. variants with a separate
`lm_head.weight` or a larger intermediate size).

## Stage by stage

### Embedding

`embed(tokens)` is a row gather (`mx::take`) from `model.embed_tokens.weight`. For
quantized models the gathered rows are de-quantized after the gather, so only the
selected vocabulary rows are materialized in fp16.

### RMSNorm

`fast::rms_norm(x, weight, eps)` with `eps` from the config (`rms_eps`). Used as
the input norm and post-attention norm in every block, plus the final norm before
the LM head.

### Q/K/V projection and GQA

`project_qkv` applies the input RMSNorm, then three linear projections, reshaping
each to `(B, heads, L, head_dim)`. The model is **grouped-query attention (GQA)**:
`n_heads` query heads (32) but only `n_kv_heads` key/value heads (8). The
head-repeat is handled **natively by MLX SDPA** when `q_heads > kv_heads` — there
is no manual `repeat` of K/V, which is a classic source of silent bugs.

### RoPE (rotary position embeddings, llama3-scaled)

RoPE rotates Q and K by a position-dependent angle. Llama-3.2 uses `rope_type
"llama3"`, which **rescales the rotation frequencies** to extend the usable
context. `compute_rope_freqs` (in `model/llama.cpp`) mirrors `mlx_lm`'s
`Llama3RoPE` exactly:

- Start from the standard schedule `freqs = base ** (arange(0, head_dim, 2) /
  head_dim)`.
- Apply the low/high-frequency rescaling driven by `rope_scaling`'s `factor`,
  `low_freq_factor`, `high_freq_factor`, and `original_max_position_embeddings`.

The precomputed `freqs` (head_dim/2 float32 values) are handed to
`fast::rope(..., freqs)` with the analytic `base` **disabled** — exactly as
`mlx_lm` does. A plain (non-llama3) model falls back to the standard `base`
schedule. RoPE is `traditional=false` (interleaved/GPT-NeoX style), matching the
model.

`fast::rope` has two overloads that matter here:

- A **uniform** `int offset` — one position for the whole batch (single-stream or
  prefill).
- A **per-row** `const array& offset` — each batch row at a different absolute
  position. **This array-offset overload is the linchpin of batched decode**: a
  left-padded ragged batch has every row at a different real-token count, and this
  applies the correct position per row in one call. The build asserts this
  overload exists on the pinned MLX (`rope_array_offset_overload_available`).

### Attention (SDPA)

`fast::scaled_dot_product_attention(q, k, v, scale, mask_mode, mask)` with `scale
= 1/sqrt(head_dim)`. Softmax runs in fp32 internally. Two masking regimes:

- **Single-stream / prefill** (`model/llama.cpp::attention`): a multi-token chunk
  uses `mask_mode="causal"`; a single decode token over the full cached history is
  unmasked (it may attend to everything already cached).
- **Batched decode** (`attention_batched`): the mask must encode both causality
  *and* the per-row left-padding, so it is supplied explicitly as an **additive
  fp16 array** rather than the `"causal"` string mode.

### The ragged batched mask

`LlamaModel::batch_mask` builds a `[B, 1, N, T_kv]` additive fp16 mask for a
batched step. With key positions `kpos` and query positions `qpos`:

- `causal[q, k] = qpos[q] >= kpos[k]`
- `valid[b, k] = left_padding[b] <= kpos[k]` (drop each row's left-pad region)
- `mask = where(causal AND valid, 0, -inf)` in fp16.

Per-row `offset` makes "causal + per-row variable context length" fall out of a
single comparison, and **left-padding is what lets every active row append its new
token at the same physical column**. The mask is **additive fp16, never boolean**
— a boolean mask hits MLX bug #2894.

### SwiGLU MLP

`down(silu(gate(x)) * up(x))`, where `silu(z) = z * sigmoid(z)`. Three linear
projections (`gate_proj`, `up_proj`, `down_proj`) with the intermediate width from
`intermediate_size`.

### LM head

The final RMSNorm output is projected to vocabulary logits. The model uses the
**tied embedding** (`model.embed_tokens.weight`) unless a separate
`lm_head.weight` is present in the checkpoint, in which case that is used.

## Linear layers and quantization

Every projection goes through `LlamaModel::linear`. HF stores a `Linear` weight as
`(out, in)`, so the fp16 path is `x @ Wᵀ` (`mx::matmul(x, transpose(W))`).

For a quantized checkpoint (detected from `config.json`'s `quantization` block),
`linear` instead calls `ops::quantized_matmul(x, W, scales, biases,
transpose=true, group_size, bits)`. The default is **4-bit, group_size 64**, which
drops the 1B model's resident footprint from ~2.3 GiB (fp16) to ~0.65 GiB. The
quantized weight, `scales`, and `biases` tensors share a key base (`<name>.weight`
/ `.scales` / `.biases`). The embedding is de-quantized per-gathered-row as noted
above. fp16 and quantized paths produce coherent output against their respective
references; the quantized path is expected to be *close to* — not bit-identical
to — the fp16 reference.

## The KV cache

Decoding reuses the K/V of all previous positions instead of recomputing them. Two
implementations:

### Single-sequence (`cache/kv_cache`)

The simplest form: prefill writes the prompt's K/V once; each decode step appends
one token's K/V along the sequence axis and returns the full history to attend
over. `offset()` (tokens written so far) is the RoPE position for the next chunk
and is bumped once per token sweep via `advance()`.

### Batched, left-padded (`cache/batch_kv_cache`)

The layout the server needs, ported from `mlx_lm`'s `BatchKVCache`. Per layer, K
and V are stored as `(B, n_kv_heads, S_cap, head_dim)` — **contiguous and
left-padded**. Key points:

- **Capacity grows in blocks of `kStep` (256)** via `zeros` + `concatenate(axis=2)`
  (amortized growth), not per token.
- Two per-row int32 arrays drive correct attention, shared across all layers (they
  process the same tokens):
  - `offset` — initialized to `-left_padding`, incremented by the written length
    each token sweep, so each row's offset equals its real-token count (its RoPE
    position).
  - `left_padding` — how many pad tokens were prepended to each row.
- `update_and_fetch(layer, k, v)` writes one layer's slice at the current position
  and returns the populated slice `[..., :idx, :]` to attend over.
- **Batch-axis surgery** — the operations the scheduler depends on:
  - `filter(keep)` = `take(axis=0)` across every layer's K/V plus
    offset/left_padding (eviction), then shift off any common left-padding.
  - `merge(other)` = pad both caches to a common `S_cap` and `concatenate(axis=0)`
    (admitting a freshly prefilled batch into the decode cache).
  - `pad_dummies(extra)` = append masked dummy rows for batch-size bucketing.

Why contiguous rather than paged: MLX C++ has no paged-attention primitive and
SDPA wants contiguous K/V; at this scale the padding cost is acceptable.

## Sampling

`sample/sampler` turns logits `(B, vocab)` into a next token per row `(B,)` plus
log-probs — entirely as **MLX graph ops**, so the values never leave the GPU and
the decode pipeline is not serialized by a host readback.

- **Greedy** = `argmax` over the vocab axis. Deterministic; the baseline that
  matches the golden reference token-for-token.
- **Temperature** scales logits (`temperature <= 0` collapses to greedy).
- **top-k** keeps the `k` largest logits per row, the rest set to `-inf`.
- **top-p (nucleus)** keeps the smallest prefix (by descending probability) whose
  cumulative mass reaches `p`, the rest set to `-inf`.
- Sampling takes an MLX random key (`random::key(seed)`) so a fixed seed is
  reproducible.

## Where numerical bugs hide (and how they're caught)

The classic silent-failure spots, all gated by golden-reference fixtures:

- **RoPE** — wrong base, wrong `traditional`/interleave flag, or missing the
  llama3 frequency rescaling. Caught by asserting RoPE'd Q/K against the reference.
- **Weight transpose** — HF `Linear` is `(out, in)`; forgetting the transpose is
  silent. Caught by the embedding/block/logits fixtures.
- **GQA head-repeat** — letting SDPA do it vs. doing it manually. Caught by the
  block-0 output fixture.
- **Decode-with-cache vs full recompute** — these differ by fp16 accumulation
  order, so they are compared by **argmax / exact tokens**, not raw logits at a
  tight tolerance.

See [contributing.md](./contributing.md#hard-won-numerical-gotchas) for the full
list and the debugging workflow.
