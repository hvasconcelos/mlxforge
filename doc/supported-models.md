# Supported models

mlxforge runs LLaMA-family decoder-only transformers. The forward pass
(`model/llama`) is shared across families; what differs per family is the chat
template and special-token handling, which are selected automatically from
`config.json`'s `model_type`.

## Families that run today

| Family | Example repo | Precision | Chat format | End-to-end |
| --- | --- | --- | --- | --- |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-bf16` | fp16 (cast on load) | `<\|start_header_id\|>…` | yes |
| Llama-3.2 | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 4-bit | `<\|start_header_id\|>…` | yes |

Support is currently limited to **Llama-3.2** while the engine stabilizes. Other
LLaMA-family models will be re-onboarded later; because the forward pass is
shared, that work is mostly tokenizer/chat-format (see [Adding a new model
family](#adding-a-new-model-family)).

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

> **Note on the gated fp16 repo.** The official fp16 Llama-3.2 repo is gated
> behind a Llama license + HF token. The public `-bf16` repo is the *same
> architecture and weights*; the engine casts to fp16 on load, and the Python
> golden reference loads the same `-bf16` weights, so the comparison stays
> self-consistent.

Point either binary at the directory:

```sh
./build/mlxforge-cli generate "$MODEL_DIR" "What is the capital of France?" 64
./build/mlxforge "$MODEL_DIR" --port 8080
```

## How a model is configured

Everything the engine needs comes from the checkpoint, not from hard-coded
constants:

- **`config.json` → `ModelConfig`** (`core/config`): layer/head/dim counts,
  `rope_theta`, `rms_eps`, `rope_scaling`, `tie_word_embeddings`, the quantization
  block (`quantized`, `quant_group_size`, `quant_bits`), and the EOS/BOS token ids.
- **`model_type`** selects the chat format (currently always the Llama-3.2 header
  format) via `chat_format_from_model_type`.
- **`tokenizer.json`** drives BPE encode/decode and supplies the special-token ids
  (`added_tokens[*].special`) that are skipped on decode — there are no hard-coded
  token ids.
- A separate `lm_head.weight` is used if present; otherwise the embedding is tied.

## What is and isn't implemented

**Supported:** GQA, RMSNorm, llama3-scaled and plain RoPE, SwiGLU, tied and
untied LM heads, fp16 and 4-bit (`quantized_matmul`, group_size 64) weights,
greedy / temperature / top-k / top-p sampling, single-stream and
continuous-batched decode.

**Not implemented:**

- **Sliding-window attention.** Only plain causal attention is supported, so
  models that rely on a sliding window are not.
- **Non-Llama tokenizers / chat templates.** Only Llama-3.2-style byte-level BPE
  and its header chat format are implemented (`Tokenizer::from_file` throws on
  others, e.g. SentencePiece).
- **Tool / function-calling tokens**, vision/multimodal, embeddings endpoints,
  LoRA/adapters, speculative decoding, multi-model hosting, prefix sharing.

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
