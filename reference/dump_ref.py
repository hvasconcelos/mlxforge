#!/usr/bin/env python3
"""MLXFORGE-002: golden-reference dump.

Runs mlx-lm on the *exact* model the C++ engine will load and emits .npy tensors
that gate every numerically-sensitive C++ story (embeddings, block-0 output,
final logits, greedy token stream) plus pre-tokenized prompt IDs so engine
phases can run before the C++ tokenizer exists (MLXFORGE-021).

The C++ side loads the SAME repo and casts to fp16, so this reference is
self-consistent: any divergence in C++ is a real bug, not a model mismatch.

Usage:
    reference/.venv/bin/python reference/dump_ref.py

Outputs land in reference/fixtures/ (committed; tiny for a 1B model). A
manifest.json records every array's shape + dtype for the C++ loader, and the
resolved model revision for reproducibility.
"""

import json
import os

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

# Public bf16 repo (the gated fp16 repo needs a Llama license + HF token; this
# is the same architecture and we cast to fp16 below, matching the C++ engine).
MODEL_REPO = "mlx-community/Llama-3.2-1B-Instruct-bf16"
# Exact snapshot the fixtures were dumped from (recorded for reproducibility).
MODEL_REVISION = "863c846a9ac6fad4e49e1743d52984dff262e953"
COMPUTE_DTYPE = mx.float16

# Fixed prompt set — committed so dumps are reproducible. Index 0 is the primary
# prompt used for the forward-pass intermediates; the rest exercise tokenization
# and (later) batched decode with ragged lengths.
PROMPTS = [
    "The capital of France is",
    "Hello, world!",
    "Once upon a time, in a land far away,",
]
# A chat-templated prompt (for MLXFORGE-021/022 template parity checks).
CHAT_MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]

GREEDY_MAX_NEW = 20  # tokens of greedy continuation to dump for the primary prompt

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    print(f"loading {MODEL_REPO} ...")
    model, tok = load(MODEL_REPO)
    model.set_dtype(COMPUTE_DTYPE)
    mx.eval(model.parameters())

    manifest = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "compute_dtype": "float16",
        "prompts": PROMPTS,
        "chat_messages": CHAT_MESSAGES,
        "greedy_max_new": GREEDY_MAX_NEW,
        "arrays": {},
    }

    def save(name, arr):
        """Eval an MLX array (or accept a numpy array) and write it as .npy."""
        if isinstance(arr, mx.array):
            mx.eval(arr)
            np_arr = np.array(arr)
        else:
            np_arr = np.asarray(arr)
        path = os.path.join(FIXTURES_DIR, name + ".npy")
        np.save(path, np_arr)
        manifest["arrays"][name] = {"shape": list(np_arr.shape), "dtype": str(np_arr.dtype)}
        print(f"  wrote {name}.npy  shape={np_arr.shape} dtype={np_arr.dtype}")

    # --- Pre-tokenized prompt IDs (one file per prompt) ----------------------
    prompt_id_lists = []
    for i, p in enumerate(PROMPTS):
        ids = tok.encode(p)
        prompt_id_lists.append(ids)
        save(f"prompt_{i}_ids", np.array(ids, dtype=np.int32))

    # Chat-templated prompt IDs (template applied by mlx-lm's tokenizer).
    chat_ids = tok.apply_chat_template(CHAT_MESSAGES, add_generation_prompt=True)
    save("chat_ids", np.array(chat_ids, dtype=np.int32))

    # --- Forward-pass intermediates for the primary prompt -------------------
    primary_ids = mx.array(prompt_id_lists[0], dtype=mx.int32)[None]  # (1, T)

    # Embedding lookup (MLXFORGE-006 gate).
    embeddings = model.model.embed_tokens(primary_ids)  # (1, T, hidden)
    save("embeddings", embeddings)

    # Front-half of layer 0 (MLXFORGE-006 gate): input RMSNorm, then RoPE'd Q/K and
    # the (un-roped) V, each (1, n_heads, T, head_dim). Mirrors Attention.__call__.
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    x_normed = layer0.input_layernorm(embeddings)  # (1, T, hidden)
    save("attn_norm0", x_normed)
    B, T, _ = embeddings.shape
    q = attn.q_proj(x_normed).reshape(B, T, attn.n_heads, -1).transpose(0, 2, 1, 3)
    k = attn.k_proj(x_normed).reshape(B, T, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = attn.v_proj(x_normed).reshape(B, T, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    save("rope_freqs", attn.rope._freqs)  # (head_dim/2,) llama3-rescaled freqs
    save("q_pre0", q)  # pre-RoPE queries (1, 32, T, 64)
    save("q_rope0", attn.rope(q))  # (1, 32, T, 64)
    save("k_rope0", attn.rope(k))  # (1, 8, T, 64)
    save("v0", v)  # (1, 8, T, 64)

    # Block-0 output: exactly what LlamaModel.__call__ computes for layer 0
    # (MLXFORGE-007 gate).
    mask = create_attention_mask(embeddings, cache=None)
    block0 = model.model.layers[0](embeddings, mask, None)  # (1, T, hidden)
    save("block0", block0)

    # Full forward to logits; dump the last position + its argmax (MLXFORGE-008 gate).
    logits = model(primary_ids)  # (1, T, vocab)
    logits_last = logits[:, -1, :]  # (1, vocab)
    save("logits_last", logits_last)
    argmax = mx.argmax(logits_last, axis=-1)  # (1,)
    save("argmax", np.array(argmax, dtype=np.int32))

    # --- Greedy token stream (full-recompute reference; MLXFORGE-009/015 gate) ---
    # Pure argmax loop, no cache, so it is an independent oracle the cached C++
    # path must reproduce exactly.
    ids = list(prompt_id_lists[0])
    greedy = []
    for _ in range(GREEDY_MAX_NEW):
        cur = mx.array(ids, dtype=mx.int32)[None]
        next_logits = model(cur)[:, -1, :]
        nxt = int(mx.argmax(next_logits, axis=-1).item())
        greedy.append(nxt)
        ids.append(nxt)
    save("greedy_tokens", np.array(greedy, dtype=np.int32))

    manifest["eos_token_ids"] = sorted(int(x) for x in tok.eos_token_ids)
    with open(os.path.join(FIXTURES_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest.json ({len(manifest['arrays'])} arrays)")


if __name__ == "__main__":
    main()
