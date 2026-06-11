# bench — cross-engine benchmark harness

Measures mlxforge against the other local-inference engines on Apple Silicon
(**llama.cpp**'s `llama-server`, **vllm-mlx**, **omlx**) and sweeps mlxforge's
own configuration space (KV quantization, prefix cache). Every engine is
driven the same way — through its OpenAI-compatible `/v1/chat/completions`
SSE endpoint — so the comparison is apples-to-apples, and engines run
strictly one at a time (they all contend for the same Metal GPU).

This is a QA harness, like the server and CLI: it exists to characterize and
prove the engine, not as a product deliverable.

## Run

```sh
uv run bench/bench.py --quick --engines mlxforge --models qwen3-0.6b   # ~3 min smoke
uv run bench/bench.py                                                  # full run
uv run bench/bench.py --rerender bench/results/bench-<ts>.json         # re-render report
```

No `uv`? `python3 -m venv bench/.venv && bench/.venv/bin/pip install httpx`
then `bench/.venv/bin/python bench/bench.py ...` (needs Python ≥ 3.11).

`build/mlxforge` must exist (`cmake --build build --parallel`). Engines that
aren't installed are skipped with a note in the report:

- llama.cpp: `brew install llama.cpp` (provides `llama-server`)
- vllm-mlx: `pip install vllm-mlx`
- omlx: see its install docs; models must be present in its model dir
  (set `omlx_model_dir` in a TOML config if not using omlx's default)

Models download from HuggingFace on first use (each engine into its own
cache), so the first run per engine×model is slow — `--ready-timeout 900`
helps on slow links. Pre-pulling is recommended for clean timing.

## Scenarios

| name | what it measures |
|---|---|
| `concurrency` | aggregate tok/s + per-request TTFT/decode at 1/4/8/16 concurrent streams — the continuous-batching story |
| `prompt` | single-stream TTFT and prefill tok/s at 128/512/2k/8k prompt tokens |
| `config` | mlxforge only: kv_bits {0,8,4} × prefix_cache {off,on}, each a server relaunch, with `/health` counter deltas |
| `multiturn` | cross-engine: chat turns sharing a ~2k-token system prompt; cold vs warm TTFT (prompt/prefix caches) |

Prompts are deterministic synthesized word streams; target token counts are
hit by calibrating tokens-per-word against each server's reported
`usage.prompt_tokens` (no Python tokenizer dependency). Scenarios 1–2 give
every request a unique lead-in so prompt caches can't fake prefill numbers;
`multiturn` shares a prefix on purpose. Requests use `temperature 0`, fixed
seed, and an open-ended task (plus `ignore_eos` on llama.cpp) so decode
lengths are comparable; throughput always uses *actual* completion tokens.

## Output

Plain-text tables on stdout, a self-contained HTML report
(`bench/results/bench-<ts>.html`, no external assets — best value per
comparable group highlighted), and the raw results as JSON next to it;
`--rerender FILE.json` regenerates the HTML next to the JSON. Token counts are flagged per row: `server` =
engine-reported usage, `client` = SSE content-chunk count (mlxforge streams
no usage chunk, so its rows are client-counted; aggregates can be
cross-checked against `/health`).

## Configuration

Defaults are embedded in `config.py`; `--config my.toml` overrides them:

```toml
ready_timeout = 900
omlx_model_dir = "/Users/me/.omlx/models"

[models.llama1b-4bit]
gguf_path = "/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf"  # skip -hf download

# If an external engine's CLI drifts from the assumptions baked into
# engines.py, override the full argv ({model}/{port}/... placeholders):
[engines.vllm-mlx]
cmd = ["vllm-mlx", "serve", "{model}", "--port", "{port}"]
```

Assumed external launch commands (override as above if they drift):
`llama-server -hf <repo:quant> -c <ctx*np> -np <N> -ngl 99 --no-webui`,
`vllm-mlx serve <repo> --continuous-batching`, `omlx serve [--model-dir D]`.

## Caveats

- GGUF quants (Q4_K_M) are the *nearest* equivalents to MLX 4-bit group
  quants, not identical — cross-engine 4-bit rows are approximate by nature.
- Two discarded warmup generations follow every server launch (Metal JIT);
  the first concurrency level also runs a discarded extra pass.
- One engine at a time, always; never compare numbers from a run where
  anything else was using the GPU.
