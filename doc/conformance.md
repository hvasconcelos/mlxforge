# Conformance kit

mlxforge's defining constraint is that the failure mode is **silent numerical
garbage, not a crash** — so "it compiled and ran" is not enough. The conformance
kit is how you prove a build (a fork, a port, a packaging change, a new binding)
still produces output that matches the reference.

## Run it

```sh
scripts/conformance.sh
```

This configures + builds, runs the **ABI guard** (`check-abi.sh`), and runs the
full doctest suite — including the golden-reference tests that compare, against
committed `.npy` fixtures dumped from `mlx-lm`:

- the embedding output, a decoder block, and the final logits (tensor-closeness);
- the first-token **argmax** and the greedy **token stream** (exact);
- the **tokenizer** byte-matching the HF tokenizer on a golden id corpus;
- the chat-template rendering;
- the **C-ABI** behavior (`tests/capi`): batched-greedy == single-stream,
  constrained output is valid JSON, embeddings are unit-norm + semantically
  ordered.

## Important: weights

The numerically-sensitive integration tests **self-skip** when the reference
model isn't cached locally — a green run without weights has only exercised the
pure-logic units. For full conformance, cache the model first:

```sh
huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-bf16
scripts/conformance.sh   # now the golden-reference + scheduler paths run
```

The script prints a note when it detects the model is absent.

## Regenerating the reference fixtures

The fixtures are committed (tiny `.npy` files under `reference/fixtures*`). They
rarely need regenerating — only when intentionally changing what is validated:

```sh
python3.12 -m venv reference/.venv
reference/.venv/bin/pip install mlx-lm numpy
reference/.venv/bin/python reference/dump_ref.py --model llama     # llama / qwen3 / qwen3_5 / ...
```

See [`supported-models.md`](./supported-models.md) for the golden-reference
discipline and how each model family is gated.
