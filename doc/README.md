# mlxforge documentation

`mlxforge` is a from-scratch LLaMA-family inference engine written in **C++ on
Apple MLX** (the C++ core library, not `mlx-lm`), served behind an
**OpenAI-compatible HTTP API** with **continuous batching**. It targets Apple
Silicon (the Metal backend) and is written in C++17.

This folder is the design reference for maintainers and contributors. Start with
whichever document matches what you're trying to do:

| Document | Read it when you want to understand… |
| --- | --- |
| [architecture.md](./architecture.md) | How the whole engine fits together: the single GPU-worker thread model, the three-queue scheduler, the request lifecycle, continuous batching, and where every module lives. |
| [llm-architecture.md](./llm-architecture.md) | The transformer itself: embedding, RMSNorm, RoPE (llama3 scaling), GQA attention, SwiGLU, the KV cache layout, sampling, and 4-bit quantization — i.e. the numerically-sensitive forward pass. |
| [supported-models.md](./supported-models.md) | Which model families run today (Llama-3.2), exactly which weights to download, what is and isn't implemented, and how to add a new family. |
| [applications.md](./applications.md) | The two binaries — the `mlxforge` server and the `mlxforge-cli` tool — their flags, the OpenAI API surface, and example clients. |
| [contributing.md](./contributing.md) | The maintainer guide: build/test workflow, the golden-reference discipline, coding conventions, and the hard-won numerical gotchas you must respect when touching the forward pass. |

## The one constraint that shapes everything

The defining property of this codebase is that **its failure mode is silent
numerical garbage, not a crash**. A wrong RoPE convention, a transposed weight,
a bad GQA head-repeat, or an off-by-one in the KV cache produces subtly wrong
tokens — never an exception. Every numerically-sensitive stage is therefore
validated against an `mlx-lm` *golden reference* (committed `.npy` fixtures),
and the design favours exact token/argmax comparisons over eyeballing output.

If you take away one thing before editing the forward pass, KV cache, or
sampler: see the gotchas in [contributing.md](./contributing.md) and run the
golden-reference tests.

## A 30-second tour

```
HTTP request ─▶ server/http_server (cpp-httplib)
                │  parse OpenAI JSON, apply chat template, tokenize
                ▼
            scheduler/  ── waiting queue (mutex + cv) ──▶ runtime/worker
                                                          (the ONE GPU thread)
                                                          │ owns weights,
                                                          │ BatchKVCache, sampler
                                                          ▼
   admit (prefill ─▶ merge) ─▶ decode step (1 async_eval/batch) ─▶ evict (filter)
                                                          │
   each row's tokens ─▶ per-request bounded TokenQueue ─▶ SSE / blocking response
```

Read [architecture.md](./architecture.md) for the full version.
