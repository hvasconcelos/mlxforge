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
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

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
    # Qwen3 MoE shares the dense Qwen3 attention (QK-Norm) and ChatML tokenizer;
    # only the feed-forward block differs (sparse expert routing). The smallest
    # MoE checkpoint is 30B-A3B; the 4bit repo keeps the download manageable.
    # set_dtype(fp16) casts scales/norms to fp16 (packed 4-bit weights stay
    # uint32), mirroring the C++ engine's load so the reference stays self-consistent.
    "qwen3_moe": {
        "repo": "mlx-community/Qwen3-30B-A3B-4bit",
        "fixtures": "fixtures_qwen3_moe",
        "compute_dtype": mx.float16,
        "chat_messages": [{"role": "user", "content": "What is the capital of France?"}],
        "thinking": True,
    },
    # Qwen3.5 is a multimodal wrapper over a *hybrid* text decoder: 3 Gated-DeltaNet
    # linear-attention layers then 1 gated full-attention layer, repeating. The
    # engine loads only the text tower (the ViT is dropped). The 4bit repo keeps the
    # download small; set_dtype(fp16) casts norms/scales to fp16 (packed 4-bit weights
    # stay uint32), mirroring the C++ engine's load. ChatML with a thinking toggle.
    "qwen3_5": {
        "repo": "mlx-community/Qwen3.5-0.8B-4bit",
        "fixtures": "fixtures_qwen3_5",
        "compute_dtype": mx.float16,
        "chat_messages": [{"role": "user", "content": "What is the capital of France?"}],
        "thinking": True,
    },
    # Qwen3-Embedding is a Qwen3 dense decoder used as an embedding model: the
    # sentence embedding is the LAST token's final hidden state (the appended
    # <|endoftext|>=151643), L2-normalized. Retrieval queries are wrapped with an
    # instruction ("Instruct: {task}\nQuery:{q}"); documents are embedded raw.
    # We dump the pooled+normalized query and document vectors (and their exact
    # token ids) so the C++ embed path is gated, not just eyeballed. The base Qwen
    # repo is public and loads as the same fp16 weights the C++ engine uses.
    "qwen3_embedding": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "fixtures": "fixtures_qwen3_embedding",
        "compute_dtype": mx.float16,
        "embedding": True,
        "embed_task": "Given a web search query, retrieve relevant passages that answer the query",
        "embed_query": "What is the capital of China?",
        "embed_doc": "The capital of China is Beijing.",
    },
    # Gemma-2 validates the from-scratch SentencePiece-BPE backend (Metaspace
    # space->U+2581 normalization + byte_fallback), a different family from the
    # byte-level Llama/Qwen BPE. The C++ engine has no Gemma model class, so this
    # is tokenizer-only: an ungated mlx-community mirror (the official repo is
    # gated) supplies the identical Gemma tokenizer, and only its tokenizer files
    # are downloaded — no weights. The golden ids include the post-processor's
    # leading <bos> (id 2), matching how the wrapper prepends BOS on encode.
    "gemma": {
        "repo": "mlx-community/gemma-2-2b-it-4bit",
        "fixtures": "fixtures_gemma",
        "tokenizer_only": True,
        # Extra corpus lines exercising Gemma's own special tokens (split out
        # before BPE) and the metaspace/byte-fallback paths.
        "extra_corpus": [
            "<start_of_turn>user\nHello<end_of_turn>",
            "<bos>hi<eos>",
            "café ▁ leading metaspace",
        ],
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


def _dump_greedy(model, save, prompt_ids):
    """Greedy continuation as a full-recompute (no cache) argmax loop. An
    independent oracle the cached C++ decode path must reproduce token-for-token."""
    ids = list(prompt_ids)
    greedy = []
    for _ in range(GREEDY_MAX_NEW):
        cur = mx.array(ids, dtype=mx.int32)[None]
        next_logits = model(cur)[:, -1, :]
        nxt = int(mx.argmax(next_logits, axis=-1).item())
        greedy.append(nxt)
        ids.append(nxt)
    save("greedy_tokens", np.array(greedy, dtype=np.int32))


def _write_manifest(fixtures_dir, manifest, tok):
    manifest["eos_token_ids"] = sorted(int(x) for x in tok.eos_token_ids)
    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest.json ({len(manifest['arrays'])} arrays)")


def dump_hybrid_intermediates(model, save, primary_ids):
    """Forward-pass intermediates for a Qwen3.5 hybrid text decoder.

    The decoder nests under model.language_model.model and interleaves two
    attention families, so it needs its own intermediate dump (the dense path
    assumes model.model.* with a uniform self_attn). We probe the first layer of
    each family — layer 0 (Gated-DeltaNet linear attention) and layer 3 (gated
    full attention) — each run standalone on the embeddings, exactly the input a
    C++ unit test feeds the corresponding stage. Finer per-stage tensors (conv
    output, recurrent state, RoPE'd Q/K) are added as those C++ stages are built.
    """
    backbone = model.language_model.model  # Qwen3_5TextModel
    embeddings = backbone.embed_tokens(primary_ids)  # (1, T, hidden)
    save("embeddings", embeddings)

    fa_mask = create_attention_mask(embeddings, cache=None)
    ssm_mask = create_ssm_mask(embeddings, cache=None)
    for idx in (0, 3):
        layer = backbone.layers[idx]
        mask = ssm_mask if layer.is_linear else fa_mask
        sub = layer.linear_attn if layer.is_linear else layer.self_attn
        x = layer.input_layernorm(embeddings)
        save(f"attn_in{idx}", x)  # post input-RMSNorm (the sublayer input)
        save(f"attn_out{idx}", sub(x, mask, None))  # sublayer output (pre-residual)
        save(f"block{idx}", layer(embeddings, mask, None))  # full decoder block


def dump_embedding_model(spec):
    """Pooled sentence-embedding golden dump for a Qwen3-Embedding-style model.

    The canonical Qwen3-Embedding checkpoint stores the decoder backbone at the
    root ("layers.N.*", "embed_tokens.weight", "norm.weight" — no "model." prefix,
    no lm_head), so mlx-lm's load() rejects it. We build the mlx-lm Qwen3 model
    directly and remap the keys to "model.*" — exactly the normalization the C++
    loader (normalize_backbone_root_keys) does — keeping the reference and the
    engine self-consistent on the SAME fp16 weights.

    Then we mirror the canonical recipe: tokenize with the HF tokenizer (which
    appends <|endoftext|>=151643 and prepends no BOS), run the backbone to its
    final-norm hidden states, take the LAST token, cast to fp32, and L2-normalize
    — exactly what C++ embed_pooled(Pooling::Last) computes. A retrieval query is
    instruction-wrapped; the document is raw. We save both vectors and their token
    ids so the C++ side gates the pooled vector AND its tokenization."""
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from mlx_lm.models.qwen3 import Model, ModelArgs

    fixtures_dir = os.path.join(os.path.dirname(__file__), spec["fixtures"])
    os.makedirs(fixtures_dir, exist_ok=True)
    print(f"loading {spec['repo']} (embedding; backbone-root layout) ...")
    path = snapshot_download(spec["repo"])
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    model = Model(ModelArgs.from_dict(cfg))
    w = mx.load(os.path.join(path, "model.safetensors"))
    w = {(k if k.startswith(("model.", "lm_head")) else "model." + k): v for k, v in w.items()}
    model.load_weights(list(w.items()), strict=False)  # lm_head is tied -> absent is OK
    model.set_dtype(mx.float16)  # mirror the C++ engine's fp16 load
    tok = AutoTokenizer.from_pretrained(path)

    manifest = {"model_repo": spec["repo"], "compute_dtype": "float16", "arrays": {}}

    def save(name, arr):
        np_arr = np.array(arr) if isinstance(arr, mx.array) else np.asarray(arr)
        np.save(os.path.join(fixtures_dir, name + ".npy"), np_arr)
        manifest["arrays"][name] = {"shape": list(np_arr.shape), "dtype": str(np_arr.dtype)}
        print(f"  wrote {name}.npy  shape={np_arr.shape} dtype={np_arr.dtype}")

    task = spec["embed_task"]
    # The wrap must match Engine::embed exactly — "Instruct: {t}\nQuery:{q}" (no
    # space after "Query:"), per Qwen's get_detailed_instruct.
    query_text = f"Instruct: {task}\nQuery:{spec['embed_query']}"

    def pooled(text):
        ids = list(tok(text)["input_ids"])  # HF tokenizer: no BOS, appends EOS 151643
        tokens = mx.array(ids, dtype=mx.int32)[None]  # (1, T)
        hidden = model.model(tokens)  # (1, T, hidden), post final-norm
        last = hidden[:, -1, :].astype(mx.float32)  # (1, hidden) last-token pool
        norm = mx.sqrt(mx.sum(last * last))
        mx.eval(last)
        return ids, mx.reshape(last / mx.maximum(norm, mx.array(1e-12)), (-1,))

    q_ids, q_vec = pooled(query_text)
    d_ids, d_vec = pooled(spec["embed_doc"])
    save("embed_query", q_vec)
    save("embed_query_ids", np.array(q_ids, dtype=np.int32))
    save("embed_doc", d_vec)
    save("embed_doc_ids", np.array(d_ids, dtype=np.int32))

    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote manifest.json ({len(manifest['arrays'])} arrays)")


def dump_tokenizer_only(spec):
    """Tokenizer-only dump for a family the C++ engine tokenizes but does not run
    (e.g. Gemma). Downloads just the tokenizer files (no weights) from an ungated
    mirror and emits the golden corpus the C++ backend is validated against."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    fixtures_dir = os.path.join(os.path.dirname(__file__), spec["fixtures"])
    os.makedirs(fixtures_dir, exist_ok=True)
    print(f"downloading {spec['repo']} tokenizer files (no weights) ...")
    # mlx-lm builds its tokenizer from transformers' AutoTokenizer, so the fast
    # tokenizer's ids here match mlx-lm's tok.encode exactly.
    path = snapshot_download(spec["repo"], allow_patterns=["*.json", "*.model", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(path)

    corpus_texts = TOKENIZER_CORPUS + spec.get("extra_corpus", [])
    corpus = [{"text": s, "ids": [int(x) for x in tok.encode(s)]} for s in corpus_texts]
    with open(os.path.join(fixtures_dir, "tokenizer_corpus.json"), "w") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"  wrote tokenizer_corpus.json ({len(corpus)} strings)")
    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump({"model_repo": spec["repo"], "tokenizer_only": True}, f, indent=2)
    print(f"\ntokenizer-only dump complete -> {spec['fixtures']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="llama")
    args = ap.parse_args()
    spec = MODELS[args.model]

    if spec.get("tokenizer_only"):
        dump_tokenizer_only(spec)
        return

    if spec.get("embedding"):
        dump_embedding_model(spec)
        return

    MODEL_REPO = spec["repo"]
    COMPUTE_DTYPE = spec["compute_dtype"]
    CHAT_MESSAGES = spec.get("chat_messages", [])  # unused by embedding-only dumps
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

    # Qwen3.5's hybrid decoder nests under model.language_model and mixes two
    # attention families, so it dumps its own intermediates; the dense path below
    # assumes a uniform model.model.* / self_attn layout.
    if hasattr(model, "language_model"):
        dump_hybrid_intermediates(model, save, primary_ids)
        logits = model(primary_ids)  # (1, T, vocab)
        logits_last = logits[:, -1, :]
        save("logits_last", logits_last)
        save("argmax", np.array(mx.argmax(logits_last, axis=-1), dtype=np.int32))
        _dump_greedy(model, save, prompt_id_lists[0])
        _write_manifest(FIXTURES_DIR, manifest, tok)
        return

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

    # MoE-specific intermediates for layer 0 (gates the sparse expert block in
    # isolation, the same discipline used for the dense front-half). Only dumped
    # when layer 0 is a sparse-MoE block; dense models skip these. The MoE input is
    # post_attention_layernorm(h), where h = embeddings + attention (i.e. the
    # post-attention residual), exactly the tensor the C++ moe_mlp() receives.
    mlp0 = model.model.layers[0].mlp
    if hasattr(mlp0, "gate") and hasattr(mlp0, "switch_mlp"):
        r = layer0.self_attn(layer0.input_layernorm(embeddings), mask, None)
        h = embeddings + r
        moe_in = layer0.post_attention_layernorm(h)  # MoE input (1, T, hidden)
        gates = mx.softmax(mlp0.gate(moe_in), axis=-1, precise=True)  # router probs
        save("moe_gates0", gates)  # (1, T, n_experts)
        save("moe_out0", mlp0(moe_in))  # sparse MoE block output (1, T, hidden)

    # Full forward to logits; dump the last position + its argmax (gates the full forward pass).
    logits = model(primary_ids)  # (1, T, vocab)
    logits_last = logits[:, -1, :]  # (1, vocab)
    save("logits_last", logits_last)
    argmax = mx.argmax(logits_last, axis=-1)  # (1,)
    save("argmax", np.array(argmax, dtype=np.int32))

    # --- Greedy token stream (full-recompute reference; gates the greedy decode stream) ---
    _dump_greedy(model, save, prompt_id_lists[0])
    _write_manifest(FIXTURES_DIR, manifest, tok)


if __name__ == "__main__":
    main()
