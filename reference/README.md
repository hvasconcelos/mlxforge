# Golden-reference fixtures

The C++ engine's failure mode is **silent numerical garbage, not a crash**. Every
numerically-sensitive stage asserts its output against a trusted oracle: `mlx-lm`
run on the *same* weights the C++ engine loads. `dump_ref.py` produces that oracle
once; the `.npy` files in `fixtures/` are committed and the C++ tests compare
against them (fp16 rel ~1e-2, or exact equality for token streams).

## Model

- **Repo**: `mlx-community/Llama-3.2-1B-Instruct-bf16` (public; the gated fp16 repo
  needs a Llama license + HF token — this is the same architecture).
- **Revision**: `863c846a9ac6fad4e49e1743d52984dff262e953`
- **Compute dtype**: fp16 — weights cast to `float16` after load, matching the C++
  engine (`Cast all weights to fp16 immediately after load`).
- Architecture: 16 layers, hidden 2048, 32 query / 8 KV heads, head_dim 64,
  vocab 128256, rope_theta 500000, rms_eps 1e-5, tied embeddings.
- EOS token ids: 128001, 128008, 128009.

## Regenerating

```sh
python3.12 -m venv reference/.venv
reference/.venv/bin/pip install mlx-lm numpy
reference/.venv/bin/python reference/dump_ref.py
```

## Fixtures (`fixtures/`, shapes/dtypes also in `manifest.json`)

| File | Shape | Dtype | Gates |
|---|---|---|---|
| `prompt_{0,1,2}_ids.npy` | `(T_i,)` | int32 | pre-tokenized prompt IDs (engine before tokenizer) |
| `chat_ids.npy` | `(42,)` | int32 | chat-template parity |
| `embeddings.npy` | `(1, 6, 2048)` | float16 | embedding lookup |
| `block0.npy` | `(1, 6, 2048)` | float16 | single decoder block |
| `logits_last.npy` | `(1, 128256)` | float16 | final logits |
| `argmax.npy` | `(1,)` | int32 | first-token argmax (exact) |
| `greedy_tokens.npy` | `(20,)` | int32 | greedy stream (exact) |

The primary prompt (index 0) `"The capital of France is"` is used for all forward
intermediates; its greedy continuation is `" Paris. The Eiffel Tower is located in
Paris. The Louvre Museum is also located in"`.
