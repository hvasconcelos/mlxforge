# Supported models

mlxforge runs LLaMA-family decoder-only transformers, including Qwen3 dense, Qwen3
MoE, and Qwen3.5 hybrid (Gated-DeltaNet) models. The forward pass (the
`DecoderModel` base in `model/`) is shared across families; what differs per family
is a small set of deltas — the chat template and special-token handling, plus any
per-family forward-pass tweak (Qwen3's QK-Norm, Qwen3 MoE's sparse MLP, Qwen3.5's
gated attention + linear-attention layers) expressed as a subclass hook override —
all selected automatically by `create_model` from `config.json` (`model_type`,
`num_experts`, `full_attention_interval`) and the presence of the distinguishing
weights.

## Families that run today

| Family | Example repo | Precision | Chat format | End-to-end |
| --- | --- | --- | --- | --- |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-bf16` | fp16 (cast on load) | `<\|start_header_id\|>…` | yes |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 4-bit (mixed bits ok) | `<\|start_header_id\|>…` | yes |
| Llama-3.2 (GGUF) | `bartowski/Llama-3.2-1B-Instruct-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K | `<\|start_header_id\|>…` | yes |
| Qwen3 (dense) | `mlx-community/Qwen3-0.6B-bf16` | fp16 (cast on load) | ChatML (`<\|im_start\|>…`) | yes |
| Qwen3 (dense, 4-bit) | `mlx-community/Qwen3-4B-4bit` | 4-bit (mixed bits ok) | ChatML (`<\|im_start\|>…`) | yes |
| Qwen3 (GGUF) | `Qwen/Qwen3-0.6B-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K | ChatML (`<\|im_start\|>…`) | yes |
| Qwen3 (MoE) | `mlx-community/Qwen3-30B-A3B-4bit` | 4-bit / fp16 (mixed bits ok) | ChatML (`<\|im_start\|>…`) | yes |
| Qwen3 (MoE, GGUF) | `Qwen/Qwen3-30B-A3B-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K | ChatML (`<\|im_start\|>…`) | yes |
| Qwen3.5 (hybrid) | `mlx-community/Qwen3.5-0.8B-4bit` | 4-bit (mixed bits ok) | ChatML (`<\|im_start\|>…`) | yes (text only) |

**Qwen3 dense** models (0.6B/1.7B/4B/8B/14B/32B) run end-to-end. They add three
deltas over Llama-3.2, all handled automatically: per-head **QK-Norm** (an
RMSNorm on each Q/K head before RoPE, gated on the `q_norm`/`k_norm` weights),
the **ChatML** chat template (with an `enable_thinking` toggle for Qwen3's
reasoning mode), and single-digit number pre-tokenization in the byte-level BPE.
Qwen3 has **no BOS token**.

**Qwen3 MoE** models (e.g. 30B-A3B, 235B-A22B) run end-to-end too. They share the
dense Qwen3 attention (QK-Norm) and ChatML tokenizer; the only delta is the
feed-forward block. On the MoE layers (selected by `config.json`'s `num_experts`,
`decoder_sparse_step`, and `mlp_only_layers`), the dense SwiGLU MLP is replaced by a
**sparse mixture-of-experts** block: a router (`mlp.gate`) softmaxes over the experts,
the top `num_experts_per_tok` are selected, each runs its own SwiGLU, and their outputs
are summed weighted by the routing scores (optionally renormalized via
`norm_topk_prob`). The per-expert weights are stored stacked as
`model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight` of shape
`(num_experts, out, in)` — raw per-expert HF checkpoints (`…mlp.experts.{e}.…`) are
stacked into that form on load. The experts run via MLX's gather matmul
(`gather_qmm` when quantized, `gather_mm` otherwise), so 4-bit experts with an 8-bit
router (the common mixed-precision layout) work transparently through the per-weight
quantization detection.

**Qwen3.5 (hybrid)** models (e.g. 0.8B) run their **text tower** end-to-end. Qwen3.5 is
a multimodal checkpoint over a Qwen3-Next-style *hybrid* decoder; the vision tower
(`vision_tower.*`) is dropped on load and the language model runs. It is selected by
`config.json`'s `full_attention_interval`, and interleaves two token-mixing families
(`config.is_linear_layer`): every `full_attention_interval`-th layer is **gated
full attention**, the rest are **Gated-DeltaNet linear attention**. The full layers
extend Qwen3 attention (QK-Norm) with a sigmoid **output gate** (`q_proj` is 2× width:
`queries‖gate`) and **partial RoPE** (only the leading `head_dim · partial_rotary_factor`
dims are rotated; the MRoPE config is a no-op for text). The linear layers replace
attention entirely with Gated-DeltaNet: a causal depthwise `Conv1d` → SiLU, L2/RMS-normed
Q/K, a delta-rule recurrence with exponential gating (`g = exp(-exp(A_log)·softplus(a +
dt_bias))`, `beta = sigmoid(b)`), a gated RMSNorm, and `out_proj`. `A_log` is kept fp32.
For continuous batching the cache is **hybrid**: the full layers grow a KV cache, the
linear layers carry a fixed-size conv buffer + recurrent state per sequence (so KV memory
scales with only the full-attention layers). The chat template is ChatML; with thinking
enabled (the default) it opens the reasoning block (`<think>`) in the generation prompt.

Other LLaMA-family models will be re-onboarded as needed; because the forward
pass is shared, that work is mostly tokenizer/chat-format plus any small
attention delta (see [Adding a new model family](#adding-a-new-model-family)).

### Compatible repos

Loading is **org-agnostic** — the engine accepts any HuggingFace repo id and only
checks *format*, not provenance: safetensors weights (PyTorch `.bin` is rejected),
the canonical HF weight-key layout, and a byte-level BPE `tokenizer.json`. Any
Llama-3.2 **text** repo (1B/3B) that meets those — across orgs and quant formats —
runs. The 11B/90B variants are vision/multimodal and are **not** supported.

| Repo | Precision | Notes |
| --- | --- | --- |
| `mlx-community/Llama-3.2-3B-Instruct-bf16` | fp16 (cast on load) | 3B sibling of the reference model |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4-bit (mixed bits ok) | MLX affine quant |
| `meta-llama/Llama-3.2-1B-Instruct` | bf16 → fp16 | official repo; **gated** (Llama license + HF token) |
| `meta-llama/Llama-3.2-3B-Instruct` | bf16 → fp16 | official repo; **gated** |
| `meta-llama/Llama-3.2-1B` / `…-3B` | bf16 → fp16 | base (non-instruct); same loader path |
| `bartowski/Llama-3.2-3B-Instruct-GGUF` | Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K | single-file GGUF (pass the `.gguf` path) |

mlx-community is just the convenient default: its repos are **public** (the official
fp16 repos are gated) and ship pre-converted/pre-quantized safetensors. A dense
`meta-llama` or `bf16` repo runs in fp16 — on-the-fly quantization of a dense
checkpoint is not implemented, so use a pre-quantized (`-4bit`) or GGUF repo for
quantized inference.

The primary, frozen reference target is **Llama-3.2-1B-Instruct** (16 layers,
hidden 2048, 32 query / 8 KV heads GQA, head_dim 64, RMSNorm, llama3-scaled RoPE,
SwiGLU, tied embeddings). It is the model the golden reference is dumped from.

## Getting the weights

```sh
# Llama-3.2-1B, fp16 (the -bf16 repo is public; cast to fp16 on load)
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-bf16

# Llama-3.2-1B, 4-bit
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit
```

A "model directory" is any folder containing `config.json`, `tokenizer.json`, and
the safetensors weights (single `model.safetensors`, or sharded
`model-0000N-of-*.safetensors` plus the index JSON). The HF cache snapshot dir
(`~/.cache/huggingface/hub/.../snapshots/<rev>`) works directly.

### GGUF (single-file, self-contained)

A `.gguf` file bundles the config, tokenizer, and weights in one file (no
`config.json` / `tokenizer.json` on disk). Point either binary at a local file
path, or at a GGUF repo id with a `:VARIANT` suffix (the variant, default `Q4_0`,
selects which quant the engine downloads — a single `.gguf`, cached per variant):

```sh
# Local file path:
./build/mlxforge-cli generate /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf "What is the capital of France?" 64
# Or auto-download one .gguf from a GGUF repo (org/name:VARIANT):
./build/mlxforge-cli generate bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_0 "What is the capital of France?" 64
./build/mlxforge -m bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M --port 8080
```

`core/gguf` parses the file itself (it does **not** use `mx::load_gguf`, whose
bundled gguflib mis-handles several quant types — see below): it reads the
metadata to build the `ModelConfig` and BPE tokenizer, then loads every tensor,
remapping the ggml names (`blk.N.attn_q.weight` →
`model.layers.N.self_attn.q_proj.weight`) and un-permuting the q/k projections
that llama.cpp stores in its interleaved RoPE layout. The llama3 RoPE rescaling
baked into `rope_freqs.weight` is lifted into the config.

Quantization (all read straight from the file and validated against the bf16
golden weights):

- `Q4_0`, `Q4_1`, `Q8_0` — kept quantized (group_size 32). MLX v0.31.2 mis-unpacks
  `Q4_1`, so these legacy types are extracted by `core/gguf` directly.
- `Q4_K`, `Q5_K`, `Q6_K` — dequantized to fp16 by our own super-block dequant
  (MLX v0.31.2 mis-dequantizes `Q4_K`/`Q5_K` and fails outright on some K-quants).
  This covers the popular `Q4_K_M` / `Q5_K_M` / `Q6_K` files. No memory savings —
  K-quants become dense fp16.
- `F16` / `F32` — read directly.
- Anything else (`Q2_K`, `Q3_K`, `Q5_0/Q5_1`, `IQ*`, …) is **rejected with a clear
  error** rather than risk silent garbage.

The `llama`, `qwen2`, `qwen3`, and `qwen3moe` architectures are accepted. A
`qwen3moe` file maps the router (`ffn_gate_inp`) and the stacked experts
(`ffn_{gate,up,down}_exps`, one 3-D tensor whose reversed dims land as
`(num_experts, out, in)`) onto the same `mlp.switch_mlp.*` form the safetensors
path uses, so the MoE forward pass is shared. Qwen3.5 / `qwen3next` (hybrid
gated-DeltaNet) is **rejected with a clear error**: its linear-attention tensors
have no settled, validated GGUF layout yet. A GGUF *repo id* auto-downloads via the
`org/name:VARIANT` spec above; a bare repo id still resolves to safetensors.

> **Note on the gated fp16 repo.** The official fp16 Llama-3.2 repo is gated
> behind a Llama license + HF token. The public `-bf16` repo is the *same
> architecture and weights*; the engine casts to fp16 on load, and the Python
> golden reference loads the same `-bf16` weights, so the comparison stays
> self-consistent.

Point either binary at the directory:

```sh
./build/mlxforge-cli generate "$MODEL_DIR" "What is the capital of France?" 64
./build/mlxforge -m "$MODEL_DIR" --port 8080
```

## How a model is configured

Everything the engine needs comes from the checkpoint, not from hard-coded
constants:

- **`config.json` → `ModelConfig`** (`core/config`): layer/head/dim counts,
  `rope_theta`, `rms_eps`, `rope_scaling`, `tie_word_embeddings`, the quantization
  block (`quantized`, `quant_group_size`, `quant_bits`), and the EOS/BOS token ids.
- **`model_type`** selects the chat format (currently always the Llama-3.2 header
  format) via `chat_format_from_model_type`.
- **`tokenizer.json`** drives encode/decode and supplies the special-token ids
  (`added_tokens[*].special`) that are skipped on decode — there are no hard-coded
  token ids. Two backends sit behind one `EncoderBackend` interface, picked by
  inspecting the blob: **byte-level BPE** (`ByteLevel` decoder; Llama-3.2 / Qwen)
  and **SentencePiece-BPE** (`byte_fallback` + metaspace; Gemma). BOS is resolved
  from the fast tokenizer's post-processor first (Gemma prepends `<bos>` despite
  `add_bos_token: false`), then the `tokenizer_config.json` flag.
- A separate `lm_head.weight` is used if present; otherwise the embedding is tied.

## What is and isn't implemented

**Supported:** GQA, RMSNorm, llama3-scaled and plain RoPE, SwiGLU, **sparse
mixture-of-experts** (Qwen3 MoE: routed top-k experts via gather matmul, dense and
quantized), tied and untied LM heads, greedy / temperature / top-k / top-p sampling,
single-stream and continuous-batched decode, and both safetensors and GGUF checkpoints.
Quantization is detected **per-weight** (a `<base>.scales` sibling), so a
checkpoint may mix quantized and dense tensors or vary the bit-width per layer:
fp16, MLX affine quants (any bits/group_size, incl. mixed-precision repos), and
GGUF `Q4_0`/`Q4_1`/`Q8_0` (group_size 32) all run; GGUF `Q4_K`/`Q5_K`/`Q6_K`
(`Q4_K_M`/`Q5_K_M`/`Q6_K` files) run dequantized to fp16 via our own dequant.

**Not implemented:**

- **Sliding-window attention.** Only plain causal attention is supported, so
  models that rely on a sliding window are not.
- **Other tokenizer families / chat templates.** Byte-level BPE (Llama-3.2 /
  Qwen) and SentencePiece-BPE (Gemma) are implemented; other families (e.g.
  Unigram/WordPiece) make `Tokenizer::from_file` throw. Chat formats are limited
  to the Llama-3.2 header format and Qwen ChatML.
- **Tool / function-calling tokens**, vision/multimodal, LoRA/adapters,
  speculative decoding, multi-model hosting, prefix sharing.

**Embeddings** *are* implemented: any LLaMA/Qwen checkpoint embeds (mean or last-token
pooling + L2-norm via `DecoderModel::forward_hidden`), and **Qwen3-Embedding** is
first-class — the canonical backbone-root checkpoint loads, the engine self-selects
last-token pooling + a trailing EOS, retrieval queries take an `Instruct:`/`Query:`
prefix, and the pooled vector is golden-gated. Exposed through `engine.embed` /
`mlxforge_embed[_ex]`, the bindings, the CLI `embed` command, and the server's
`POST /v1/embeddings`. See [`embedding.md`](./embedding.md).

## Adding a new model family

Because the transformer is shared, the work is usually confined to tokenization
and configuration:

1. **Confirm the architecture matches.** It must be a LLaMA-family decoder:
   RMSNorm, RoPE, GQA (or MHA as a special case), SwiGLU, the standard HF weight
   key layout (`model.layers.N.self_attn.{q,k,v,o}_proj`,
   `model.layers.N.mlp.{gate,up,down}_proj`, `input_layernorm`,
   `post_attention_layernorm`, `model.norm`, `model.embed_tokens` / `lm_head`).
   Sliding-window, attention-bias, or non-RoPE position schemes need new
   forward-pass code.
2. **Map the config.** Make sure `ModelConfig::from_json` reads every field the
   family needs; add fields if necessary. Verify `rope_scaling` is parsed (or
   absent → plain RoPE).
3. **Add a chat format.** Extend `ChatFormat` and `render_chat_template` in
   `tokenizer/tokenizer.cpp`, and map the new `model_type` in
   `chat_format_from_model_type`.
4. **Sanitize weight keys** if the checkpoint uses non-standard names — add the
   alias to `sanitize_key` in `core/weights.cpp`.
5. **Add a golden reference** (see below) and gate the new family's forward pass
   and chat template against it.

## The golden-reference discipline

Each model is validated against its own `mlx-lm` golden reference — `.npy`
tensors dumped from `mlx-lm` running the *same* weights the C++ engine loads. The
fixtures are committed (they are tiny); the model weights and the throwaway venv
are not.

```sh
python3.12 -m venv reference/.venv
reference/.venv/bin/pip install mlx-lm numpy
reference/.venv/bin/python reference/dump_ref.py --model llama     # -> reference/fixtures/
```

The fixtures gate the embedding output, a single decoder block, the final logits,
the first-token argmax (exact), the greedy token stream (exact), and chat-template
parity. The C++ side compares with `tests/support/reference.h` (`assert_close`,
fp16 rel ~1e-2; and exact token equality). Integration tests **self-skip** if the
model isn't present locally, so a green `ctest` without weights has only exercised
the pure-logic unit tests. See `reference/README.md` for the fixture catalogue and
[contributing.md](./contributing.md) for the workflow.
