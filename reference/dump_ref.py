#!/usr/bin/env python3
"""Golden-reference dump.

Runs mlx-lm on the *exact* model the C++ engine will load and emits .npy tensors
that gate every numerically-sensitive C++ stage (embeddings, block-0 output,
final logits, greedy token stream) plus pre-tokenized prompt IDs so engine
phases can run before the C++ tokenizer exists.

The C++ side loads the SAME repo and casts to fp16, so this reference is
self-consistent: any divergence in C++ is a real bug, not a model mismatch.

Usage:
    reference/.venv/bin/python reference/dump_ref.py [--model llama]

Outputs land in reference/fixtures/ (committed; tiny). A manifest.json records
every array's shape + dtype for the C++ loader, and the resolved model revision
for reproducibility.
"""

import argparse
import json
import os

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

# Per-model dump settings. The C++ engine loads the SAME repo, so each reference
# is self-consistent: any divergence in C++ is a real bug, not a model mismatch.
#   - llama: public bf16 repo (the gated fp16 repo needs a license); cast to fp16
#     to match the C++ engine, which loads bf16 and casts on load.
MODELS = {
    "llama": {
        "repo": "mlx-community/Llama-3.2-1B-Instruct-bf16",
        "fixtures": "fixtures",
        # Cast bf16 -> fp16 to mirror the C++ engine's load.
        "compute_dtype": mx.float16,
        # The Llama-3.2 chat template has a system role with a date preamble.
        "chat_messages": [{"role": "user", "content": "What is the capital of France?"}],
        # Qwen3-only: also dump a thinking-disabled chat prompt.
        "thinking": False,
    },
    "qwen3": {
        "repo": "mlx-community/Qwen3-0.6B-bf16",
        "fixtures": "fixtures_qwen3",
        "compute_dtype": mx.float16,
        # Qwen3 ChatML: no default system message.
        "chat_messages": [{"role": "user", "content": "What is the capital of France?"}],
        "thinking": True,
    },
}

# Fixed prompt set — committed so dumps are reproducible. Index 0 is the primary
# prompt used for the forward-pass intermediates; the rest exercise tokenization
# and (later) batched decode with ragged lengths.
PROMPTS = [
    "The capital of France is",
    "Hello, world!",
    "Once upon a time, in a land far away,",
]

GREEDY_MAX_NEW = 20  # tokens of greedy continuation to dump for the primary prompt

# Diverse strings exercising the byte-level BPE pre-tokenizer's edge cases
# (whitespace runs, newlines, contractions, digits, punctuation, CJK, accented
# Latin, emoji/ZWJ, code, and inline special tokens). Their mlx-lm token ids are
# committed as golden fixtures so the from-scratch tokenizer can be validated
# without a live HF-tokenizer oracle. Llama-only (byte-level BPE).
TOKENIZER_CORPUS = [
    "",
    "The capital of France is Paris.",
    "Hello, world!",
    "don't I'll we've they're it's can't",
    "DON'T SHOUT",
    "spaces    here     and       more",
    "  leading and trailing  ",
    "tabs\tand\tmore\ttabs",
    "newlines\n\nand\r\nwindows\r\nendings",
    "mixed \n  \t whitespace \n\n",
    "1 12 123 1234 100000 3.14159",
    "snake_case camelCase kebab-case",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    "café naïve résumé Zürich",
    "Ünïcödé ßharp",
    "你好世界，今天天气很好。",
    "こんにちは世界",
    "Привет мир",
    "emoji 😀 and 👨‍👩‍👧‍👦 family",
    "math ∑∫√≠≤ symbols",
    "<|begin_of_text|>hi<|eot_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\nWhat?<|eot_id|>",
    "a<|eot_id|><|eot_id|>b",
    "URL: https://example.com/path?q=1&x=2#frag",
    "@user #hashtag $100 50% (parens) [brackets] {braces}",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="llama")
    args = ap.parse_args()
    spec = MODELS[args.model]

    MODEL_REPO = spec["repo"]
    COMPUTE_DTYPE = spec["compute_dtype"]
    CHAT_MESSAGES = spec["chat_messages"]
    FIXTURES_DIR = os.path.join(os.path.dirname(__file__), spec["fixtures"])

    os.makedirs(FIXTURES_DIR, exist_ok=True)
    print(f"loading {MODEL_REPO} ...")
    model, tok = load(MODEL_REPO)
    if COMPUTE_DTYPE is not None:
        model.set_dtype(COMPUTE_DTYPE)
    mx.eval(model.parameters())

    manifest = {
        "model_repo": MODEL_REPO,
        "compute_dtype": "float16" if COMPUTE_DTYPE is mx.float16 else "quantized",
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

    # Qwen3-style reasoning toggle: the thinking-disabled prompt appends an empty
    # <think></think> block. (Templates that ignore the kwarg dump an identical
    # array, which the C++ side simply doesn't exercise.)
    if spec.get("thinking"):
        chat_ids_nothink = tok.apply_chat_template(
            CHAT_MESSAGES, add_generation_prompt=True, enable_thinking=False
        )
        save("chat_ids_nothink", np.array(chat_ids_nothink, dtype=np.int32))

    # Diverse tokenizer corpus -> committed golden ids (validates the from-scratch
    # byte-level BPE; tok.encode matches mlxforge::Tokenizer::encode for the family).
    # Dumped per-model so each tokenizer (Llama digit runs vs Qwen single digits) is
    # validated against its own oracle.
    corpus = [{"text": s, "ids": [int(x) for x in tok.encode(s)]} for s in TOKENIZER_CORPUS]
    with open(os.path.join(FIXTURES_DIR, "tokenizer_corpus.json"), "w") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"  wrote tokenizer_corpus.json ({len(corpus)} strings)")

    # --- Forward-pass intermediates for the primary prompt -------------------
    primary_ids = mx.array(prompt_id_lists[0], dtype=mx.int32)[None]  # (1, T)

    # Embedding lookup (gates the embedding stage).
    embeddings = model.model.embed_tokens(primary_ids)  # (1, T, hidden)
    save("embeddings", embeddings)

    # Front-half of layer 0 (gates the embedding/attention front-half): input RMSNorm, then RoPE'd Q/K and
    # the (un-roped) V, each (1, n_heads, T, head_dim). Mirrors Attention.__call__.
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    x_normed = layer0.input_layernorm(embeddings)  # (1, T, hidden)
    save("attn_norm0", x_normed)
    B, T, _ = embeddings.shape
    q = attn.q_proj(x_normed).reshape(B, T, attn.n_heads, -1)
    k = attn.k_proj(x_normed).reshape(B, T, attn.n_kv_heads, -1)
    v = attn.v_proj(x_normed).reshape(B, T, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    # Qwen3 normalizes each Q/K head (RMSNorm over head_dim) before RoPE.
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q = q.transpose(0, 2, 1, 3)  # (1, n_heads, T, head_dim)
    k = k.transpose(0, 2, 1, 3)  # (1, n_kv_heads, T, head_dim)
    # The precomputed `_freqs` is specific to the llama3-rescaled RoPE; plain-RoPE
    # models (Qwen3) lack it. The front-half intermediates are dumped for both.
    if hasattr(attn.rope, "_freqs"):
        save("rope_freqs", attn.rope._freqs)  # (head_dim/2,) llama3-rescaled freqs
    save("q_pre0", q)  # pre-RoPE queries (post q_norm for Qwen3)
    save("q_rope0", attn.rope(q))
    save("k_rope0", attn.rope(k))
    save("v0", v)

    # Block-0 output: exactly what LlamaModel.__call__ computes for layer 0
    # (gates the single decoder block).
    mask = create_attention_mask(embeddings, cache=None)
    block0 = model.model.layers[0](embeddings, mask, None)  # (1, T, hidden)
    save("block0", block0)

    # Full forward to logits; dump the last position + its argmax (gates the full forward pass).
    logits = model(primary_ids)  # (1, T, vocab)
    logits_last = logits[:, -1, :]  # (1, vocab)
    save("logits_last", logits_last)
    argmax = mx.argmax(logits_last, axis=-1)  # (1,)
    save("argmax", np.array(argmax, dtype=np.int32))

    # --- Greedy token stream (full-recompute reference; gates the greedy decode stream) ---
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
