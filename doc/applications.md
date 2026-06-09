# Applications (the engine harnesses)

> **These binaries are harnesses, not the product.** `libmlxforge` — the engine behind
> its C ABI — is the deliverable (see [`embedding.md`](./embedding.md)). The server and
> CLI here exist to **drive and validate** that engine: the server is the
> scheduler/batching concurrency & load harness, the CLI is the golden-reference and
> weight-inspection smoke test. They are how we prove stability and how you can try the
> engine without writing a binding — they are not what mlxforge ships as its product.
> Both are built only when `MLXFORGE_BUILD_SERVER` / `MLXFORGE_BUILD_CLI` are on (the
> default for development and CI); the released library is built without them.

The build produces two harness binaries plus the test suite:

| Binary | Source | Purpose |
| --- | --- | --- |
| `build/mlxforge` | `apps/mlxforge.cpp` | Server **harness**: drives the engine over the OpenAI HTTP API to load-test the scheduler and continuous batching. |
| `build/mlxforge-cli` | `apps/mlxforge_cli.cpp` | CLI **harness**: build smoke test, weight inspection, and golden-reference single-stream generation. |
| `build/tests/mlxforge_tests` | `tests/` | The doctest suite (see [contributing.md](./contributing.md)). |

### Build options

The harnesses are on by default for development; the released library is built
with them off. A lean library build —
`-DMLXFORGE_BUILD_SERVER=OFF -DMLXFORGE_BUILD_CLI=OFF -DMLXFORGE_ENABLE_HF_DOWNLOAD=OFF`
— produces a `libmlxforge.dylib` with neither `cpp-httplib` nor `libcurl` linked.

| Option | Default | Effect |
| --- | --- | --- |
| `MLXFORGE_BUILD_SHARED` | `ON` | build `libmlxforge.dylib` (the C-ABI product) |
| `MLXFORGE_BUILD_SERVER` | `ON` | build the HTTP server harness (pulls `cpp-httplib`) |
| `MLXFORGE_BUILD_CLI` | `ON` | build the CLI harness |
| `MLXFORGE_ENABLE_HF_DOWNLOAD` | `ON` | HuggingFace download (pulls `libcurl`) |

`$MODEL_DIR` below is any directory containing `config.json`, `tokenizer.json`,
and the safetensors weights — see [supported-models.md](./supported-models.md).

## The server: `mlxforge`

```sh
./build/mlxforge -m "$MODEL_DIR" --port 8080 --max-ctx 8192 --max-waiting 256
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
(completions), `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop`,
`n`, `seed`, plus `tools` / `tool_choice` for function calling (see
[Tool / function calling](#tool--function-calling)). Chat requests are rendered
through the model's chat template before tokenization. For Qwen3, `enable_thinking`
(or `chat_template_kwargs.enable_thinking`) toggles the reasoning block — `false`
emits an empty `<think></think>` to suppress it; the field is ignored by chat
formats that don't support it (e.g. Llama-3.2).

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

### Tool / function calling

The chat endpoint implements the OpenAI tool-calling surface, so the model can
ask the client to run a function and continue from its result — the basis for
agentic, multi-turn loops (e.g. *create a branch → implement → run tests →
commit*).

**Request fields.** `tools` is an array of function definitions; `tool_choice`
is `"auto"` (default), `"none"` (suppress tools entirely), or `"required"` (an
object naming a function is also accepted and collapsed to `"required"`).

```jsonc
{
  "model": "mlxforge",
  "messages": [{"role": "user", "content": "What's the weather in SF?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }]
}
```

**How it flows.** When `tools` are present (and `tool_choice != "none"`), each
function schema is injected into the first user turn of the chat template, ahead
of the question, with the Llama-3.2 tool preamble. If the model responds with a
call, the response carries `message.tool_calls` (and `content: null`) with
`finish_reason: "tool_calls"`:

```jsonc
"message": {
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_0", "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
  }]
}
```

The client executes the function, appends the assistant turn and a
`{"role": "tool", "content": "<result>"}` message, and calls the endpoint again;
the tool result is replayed under the model's `ipython` role so it can produce
the final answer. Parallel calls emitted as `;`-separated JSON objects are parsed
into multiple `tool_calls`. If the model replies with prose instead of a call,
the response is an ordinary `chat.completion` — there is no forced call.

**Streaming.** With tools enabled, output that begins with `{` is buffered until
it can be classified; a confirmed call is emitted as a single `tool_calls` delta
(arguments are not streamed incrementally), while plain text streams live as
usual.

**Caveat.** This is the protocol plumbing. How reliably a given model plans and
emits well-formed calls depends on the model — the small Llama-3.2 1B/3B
checkpoints are weak tool planners, so validate against your target model before
relying on it for an autonomous loop.

```python
# one round-trip of the tool loop with the openai client
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in SF?"}]

r = c.chat.completions.create(model="mlxforge", messages=msgs, tools=tools)
call = r.choices[0].message.tool_calls[0]                  # finish_reason == "tool_calls"
msgs.append(r.choices[0].message)
msgs.append({"role": "tool", "tool_call_id": call.id,
             "content": '{"temp_c": 21}'})                 # the function's result
final = c.chat.completions.create(model="mlxforge", messages=msgs, tools=tools)
print(final.choices[0].message.content)
```

## The CLI: `mlxforge-cli`

Subcommands (the main ones below; `embed` and `bench` also exist).

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

### `image` — vision-language (Qwen3-VL) image → text

```sh
./build/mlxforge-cli image "$MODEL_DIR" photo.jpg "What is in this image?" 128
# convenience wrapper that builds the CLI and resolves the cached VL model:
scripts/test-image.sh photo.jpg "What animals are here?" 200
```

Decodes the image (`stb_image`), smart-resizes + patchifies it, runs the ViT, and
streams a response — the whole vision pipeline behind one command. The model must
be a vision-language checkpoint (e.g. Qwen3-VL); otherwise it errors. Served
single-stream. The same path is reachable in-process via `mlxforge_submit_image`
(C ABI) and `engine.image(...)` in the Node/Swift/Rust bindings.

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
