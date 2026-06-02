# Applications

The build produces two binaries plus the test suite:

| Binary | Source | Purpose |
| --- | --- | --- |
| `build/mlxforge` | `apps/mlxforge.cpp` | The OpenAI-compatible HTTP server with continuous batching. |
| `build/mlxforge-cli` | `apps/mlxforge_cli.cpp` | A CLI for the build smoke test, weight inspection, and single-stream generation. |
| `build/tests/mlxforge_tests` | `tests/` | The doctest suite (see [contributing.md](./contributing.md)). |

`$MODEL_DIR` below is any directory containing `config.json`, `tokenizer.json`,
and the safetensors weights — see [supported-models.md](./supported-models.md).

## The server: `mlxforge`

```sh
./build/mlxforge "$MODEL_DIR" --port 8080 --max-ctx 8192 --max-waiting 256
```

On startup it loads the config and tokenizer, launches the GPU worker (which
loads the model on its own thread), and serves the HTTP API. `SIGINT`/`SIGTERM`
trigger a graceful shutdown that drains in-flight requests before exit.

### Configuration

Flags are positional `model_dir` followed by `--flag value` or `--flag=value`,
with environment-variable fallbacks (`server/config`):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `<model_dir>` | — | (required) | The model directory. |
| `--host` | `MLXFORGE_HOST` | `0.0.0.0` | Bind address. |
| `--port` | `MLXFORGE_PORT` | `8080` | Listen port. |
| `--max-ctx` | `MLXFORGE_MAX_CTX` | `8192` | Reject prompts longer than this → `400`. |
| `--max-waiting` | `MLXFORGE_MAX_WAITING` | `256` | Bounded waiting queue → `429` on overflow. |
| `--kv-budget` | `MLXFORGE_KV_BUDGET` | `0` (unbounded) | KV-memory admission budget in bytes. |

### Logging

Both binaries log through spdlog (`core/logging`) to **stderr**, leaving stdout
for program output (the CLI's streamed text / weight dump). The logger is
initialized once at the top of `main()` from environment variables; an unset
variable uses the default shown:

| Env | Default | Meaning |
| --- | --- | --- |
| `MLXFORGE_LOG_LEVEL` | `info` | `trace`/`debug`/`info`/`warn`/`error`/`critical`/`off`; an unrecognized value falls back to `info`. |
| `MLXFORGE_LOG_FILE` | _(unset)_ | If set, logs are **appended** to this file in addition to the console. |
| `MLXFORGE_LOG_PATTERN` | `[%H:%M:%S.%e] [%^%l%$] %v` | spdlog message pattern. |

Levels in use: `info` for lifecycle (model load, server listen/stop) and the
per-request metrics below; `debug` for hot-path detail (per decode step, request
admit, HTTP request lines, encode sizes); `warn` for rejections (queue full,
non-fp16 weights); `error` for caught exceptions in the worker loop.

### Endpoints (OpenAI-compatible)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/chat/completions` | Chat completion (streaming or full). |
| POST | `/v1/completions` | Text completion from a raw `prompt`. |
| GET | `/v1/models` | List the served model. |
| GET | `/health` | Liveness / readiness (`503` until the model has loaded). |

**Supported request fields:** `model`, `messages[]` (chat) or `prompt`
(completions), `max_tokens`, `temperature`, `top_p`, `stream`, `stop`, `n`,
`seed`. Chat requests are rendered through the model's chat template before
tokenization.

**Responses:**

- *Non-streaming* — a standard `chat.completion` object: `id`, `object`,
  `created`, `model`, `choices[].message`, `finish_reason` (`stop` | `length`),
  and `usage` (prompt/completion/total tokens).
- *Streaming* (`stream: true`) — `Content-Type: text/event-stream`; a sequence of
  `data: {chat.completion.chunk …}\n\n` frames carrying incremental
  `choices[].delta`, terminated by `data: [DONE]\n\n`. Detokenization is
  incremental and UTF-8-safe, so no frame ever carries a broken multi-byte
  character.

**Cancellation.** If a streaming client disconnects, the chunked content provider
fails to write; the server sets the request's `cancelled` atomic, and the worker
evicts that row at the next iteration boundary — freeing the batch slot while
other streams continue.

**Errors (OpenAI error JSON shape):** `400` for invalid/out-of-range params or
malformed JSON; `429` when the waiting queue is full; `503` while the model is
still loading.

**Metrics.** Each finished request logs per-request metrics at `info` level
(stderr by default): TTFT (time to first token), tokens/s, the active batch
size, and the waiting-queue depth. Raise `MLXFORGE_LOG_LEVEL` to `warn`/`error`
to suppress them (see [Logging](#logging)).

### Example client

Use the official `openai` Python client pointed at the local server:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="x")

# non-streaming
r = c.chat.completions.create(
    model="mlxforge",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=32)
print(r.choices[0].message.content)            # "The capital of France is Paris."

# streaming
for ev in c.chat.completions.create(
        model="mlxforge",
        messages=[{"role": "user", "content": "Tell me a joke."}],
        max_tokens=64, stream=True):
    print(ev.choices[0].delta.content or "", end="", flush=True)
```

## The CLI: `mlxforge-cli`

Three subcommands.

### `generate` — single-stream greedy generation

```sh
# from a raw chat prompt (rendered through the chat template, real text streamed)
./build/mlxforge-cli generate "$MODEL_DIR" "What is the capital of France?" 64

# from a pre-tokenized .npy prompt (ids loaded directly)
./build/mlxforge-cli generate "$MODEL_DIR" reference/fixtures/prompt_0_ids.npy 20
```

Prefills the prompt, then greedily samples and streams the detokenized text to
stdout until an EOS token or `max_tokens` (default 64). A `.npy` argument is
treated as pre-tokenized ids; anything else is raw text run through the chat
template. This is the path the golden reference compares against token-for-token.

### `dump-weights` — inspect a checkpoint

```sh
./build/mlxforge-cli dump-weights "$MODEL_DIR"
```

Loads the weights, prints `key → shape → dtype` for every tensor (sorted),
asserts every tensor is fp16, and reports peak resident memory. Exits non-zero if
any tensor is not fp16.

### no arguments — build smoke test

```sh
./build/mlxforge-cli
```

Confirms Metal is available, adds two small arrays, calls `mx::eval`, and prints
the sum — the minimal "MLX is working" check and a demonstration of MLX's lazy
evaluation (nothing computes until `eval`).
