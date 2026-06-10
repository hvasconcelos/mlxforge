# mlxforge white paper

`mlxforge-whitepaper.tex` is the consolidated technical white paper for the
engine: product thesis, system architecture, threading and continuous-batching
model, the mathematics of the transformer forward pass and every model-family
variant (RoPE / llama3 rescaling, GQA, SwiGLU, MoE, Gated-DeltaNet, Qwen3-VL
M-RoPE / DeepStack), sampling, the stable C ABI and bindings, software patterns,
and the golden-reference testing methodology.

It is a single monolithic LaTeX source (document class `report`) with
self-contained TikZ diagrams (no external images). `references.bib` is the
bibliography — the canonical place to add references for this work.

## Build

Requires a TeX distribution (TeX Live / MacTeX) with `latexmk`.

```sh
make            # -> mlxforge-whitepaper.pdf
make clean      # remove build artifacts (keep the PDF)
make distclean  # remove artifacts and the PDF
```

Or directly:

```sh
latexmk -pdf mlxforge-whitepaper.tex
```

## Relation to `doc/*.md`

This paper consolidates and deepens the Markdown design docs (`doc/embedding.md`,
`doc/architecture.md`, `doc/llm-architecture.md`, etc.); it does not replace them.
When the engine changes in a numerically or architecturally significant way, update
the relevant chapter here and add any new citations to `references.bib`.
