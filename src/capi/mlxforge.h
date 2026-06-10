/*
 * libmlxforge — the stable C ABI for the mlxforge inference engine.
 *
 * This is the product surface: a single `extern "C"` header that wraps the
 * batched MLX engine (one GPU worker + continuous-batching scheduler) behind
 * opaque handles, so it can be bound from any language (Node first, then Swift,
 * Rust) and embedded in-process. The HTTP server and CLI in this repo are QA
 * harnesses on the same engine; this header is what mlxforge ships.
 *
 * Contract:
 *   - No C++ types cross this boundary. Strings are UTF-8, NUL-terminated, and
 *     ownership is documented per function. Strings returned by the library are
 *     freed by the caller with mlxforge_string_free().
 *   - Every fallible call takes a `char** err`: on failure it returns NULL / a
 *     negative code and (if err != NULL) sets *err to a newly-allocated message
 *     the caller frees with mlxforge_string_free(). A C++ exception is never
 *     allowed to cross the boundary.
 *   - The ABI is append-only and versioned (mlxforge_abi_version()).
 *
 * Threading: the engine owns a single GPU worker thread; all MLX work happens
 * there (MLX arrays are thread-bound). Handles are not thread-safe to use
 * concurrently with themselves, but distinct requests on one engine may be
 * driven from distinct threads — that is exactly how concurrent requests batch.
 */
#ifndef MLXFORGE_H
#define MLXFORGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any additive ABI change; never on a breaking one (those get a new
 * symbol). Query at runtime with mlxforge_abi_version().
 *   v2: mlxforge_embed_ex + mlxforge_embed_opts (Qwen3-Embedding conventions:
 *       last-token pooling, trailing EOS, instruction prefix).
 *   v3: mlxforge_submit_image (Qwen3-VL vision-language: a prompt + one image,
 *       served single-stream).
 *   v4: mlxforge_image + mlxforge_submit_images (a prompt + N images).
 *   v5: mlxforge_sampling.logprobs + mlxforge_request_logprobs (OpenAI per-token
 *       log-probabilities: the chosen token's logprob and its top-N alternatives,
 *       accumulated as the request is drained and returned as an OpenAI-shaped
 *       JSON array).
 *   v6: mlxforge_engine_create2 + mlxforge_engine_opts2 (KV-cache quantization;
 *       opts2 carries struct_size so future fields append without a create3). */
#define MLXFORGE_ABI_VERSION 6

typedef struct mlxforge_engine mlxforge_engine;
typedef struct mlxforge_request mlxforge_request;

/* Engine creation options. Zero-initialize (`{0}`) for defaults. */
typedef struct {
  int max_waiting; /* max queued requests; <= 0 => default (256) */
} mlxforge_engine_opts;

/* One chat message. role is "system" | "user" | "assistant" | "tool". */
typedef struct {
  const char* role;
  const char* content;
} mlxforge_msg;

/* Sampling parameters. Zero-initialize (`{0}`) for deterministic greedy decode
 * with the default token budget. Mirrors the engine's SamplingParams; the
 * library normalizes "disabled" sentinels so a zeroed struct is always valid. */
typedef struct {
  float temperature;          /* <= 0 => greedy (argmax) */
  int   top_k;                /* <= 0 => disabled */
  float top_p;                /* <= 0 or >= 1 => disabled */
  float min_p;                /* <= 0 => disabled */
  float repetition_penalty;   /* 0 or 1 => disabled */
  float frequency_penalty;    /* 0 => disabled */
  float presence_penalty;     /* 0 => disabled */
  unsigned long long seed;    /* RNG seed (used when temperature > 0) */
  int   max_tokens;           /* <= 0 => default (64) */
  /* Constrained decoding (optional; NULL/empty => off): "json" forces any valid
   * JSON value; otherwise a JSON-Schema string (supported subset: a top-level
   * object with ordered, required, scalar-typed properties). The output is
   * masked so it can only be well-formed JSON. */
  const char* json_schema;
  /* Per-token log-probabilities (OpenAI logprobs). 0 (the zero-init default) =>
   * off. N > 0 => report each emitted token's own log-prob plus its (N - 1) most
   * likely alternatives, so 1 = the chosen token's log-prob only. Retrieve them
   * with mlxforge_request_logprobs() once the request is drained. Not supported
   * on vision (mlxforge_submit_image) requests, which ignore it. */
  int logprobs;
} mlxforge_sampling;

/* ---- Library info ---------------------------------------------------------*/

/* Human-readable version string (owned by the library; do not free). */
const char* mlxforge_version(void);

/* The ABI version this build implements (== MLXFORGE_ABI_VERSION at build). */
int mlxforge_abi_version(void);

/* Free a string the library allocated (every `char** err` and text out-param).
 * NULL is ignored. */
void mlxforge_string_free(char* s);

/* Free a float array returned by the library (mlxforge_embed). NULL is ignored. */
void mlxforge_floats_free(float* p);

/* ---- Engine ---------------------------------------------------------------*/

/* Create an engine for a model spec: a local directory, a HuggingFace repo id,
 * or a `.gguf` file (or `org/name:VARIANT` for a GGUF download). Loads config +
 * tokenizer on the calling thread and spawns the GPU worker (which loads weights
 * on its own thread). `opts` may be NULL for defaults.
 *
 * Returns the engine, or NULL on failure (sets *err). Free with
 * mlxforge_engine_free(). The model may still be loading when this returns;
 * poll mlxforge_engine_ready(). */
mlxforge_engine* mlxforge_engine_create(const char* model_spec,
                                        const mlxforge_engine_opts* opts, char** err);

/* Extended engine creation options (v6+). Set struct_size = sizeof(...) and
 * zero-initialize the rest for defaults; the library reads only the fields
 * struct_size covers, so binaries built against this header stay correct when
 * later versions append fields.
 *
 * kv_bits enables KV-cache quantization (engine-wide; the batched cache's
 * storage is shared across rows, so it cannot be per-request): 0 = dense fp16
 * (the default), 8 or 4 store the cache as quantized triplets matching
 * mlx-lm's QuantizedKVCache (8 is near-lossless at ~1.9x less cache memory;
 * 4 is ~3.6x). Unsupported setups (vision-language or hybrid Qwen3.5 models,
 * invalid bits/group sizes) FAIL engine creation with a clear *err — there is
 * never a silent fp16 fallback. */
typedef struct {
  size_t struct_size;   /* caller sets sizeof(mlxforge_engine_opts2) */
  int max_waiting;      /* max queued requests; <= 0 => default (256) */
  int kv_bits;          /* 0 = fp16 KV cache (default); 8 or 4 = quantized */
  int kv_group_size;    /* quantization group size; <= 0 => default (64) */
} mlxforge_engine_opts2;

/* Create an engine with extended options (v6+). Identical contract to
 * mlxforge_engine_create; `opts` may be NULL for all defaults. */
mlxforge_engine* mlxforge_engine_create2(const char* model_spec,
                                         const mlxforge_engine_opts2* opts, char** err);

/* Non-zero once the model has finished loading on the worker thread. Requests
 * may be submitted before this returns true; they are served once ready. */
int mlxforge_engine_ready(mlxforge_engine* engine);

/* The served model name (the spec passed to create). Owned by the engine;
 * valid until mlxforge_engine_free(). */
const char* mlxforge_engine_model_name(mlxforge_engine* engine);

/* Drain in-flight work, stop the worker thread, and destroy the engine. Any
 * still-open requests must not be used afterward. NULL is ignored. */
void mlxforge_engine_free(mlxforge_engine* engine);

/* ---- Embeddings -----------------------------------------------------------*/

/* Embed `text` into a unit-normalized vector (synchronous: blocks until the
 * worker runs the forward pass). `pooling` is 0 = mean over the sequence
 * (default), 1 = last token. On success returns 0, sets *out to a newly
 * allocated float array of length *out_len (free with mlxforge_floats_free).
 * On failure returns non-zero and sets *err. Any LLaMA/Qwen checkpoint works; an
 * embedding-tuned checkpoint produces a higher-quality vector.
 *
 * This is the simple form: pooling is explicit and no EOS/instruction is added.
 * For Qwen3-Embedding conventions (last-token + EOS + instruction) or to let the
 * model pick its own defaults, use mlxforge_embed_ex. */
int mlxforge_embed(mlxforge_engine* engine, const char* text, int pooling, float** out,
                   size_t* out_len, char** err);

/* Options for mlxforge_embed_ex. `pooling`/`add_eos` are tri-state: use -1 to
 * defer to the model's detected default (a Qwen3-Embedding checkpoint ->
 * last-token pooling + trailing EOS; a plain LLM -> mean pooling, no EOS).
 * Note: a zero-initialized (`{0}`) struct therefore means *explicit* mean
 * pooling + no EOS + normalize-on; to get the model's defaults pass NULL, or set
 * pooling and add_eos to -1. */
typedef struct {
  int pooling;             /* -1 = model default; 0 = mean; 1 = last token */
  int add_eos;             /* -1 = model default; 0 = off; 1 = append EOS id */
  int skip_normalize;      /* 0 = L2-normalize (default); 1 = return raw pooled */
  const char* instruction; /* optional; wraps as "Instruct: {it}\nQuery: {text}" */
} mlxforge_embed_opts;

/* Embed `text` with explicit options (Qwen3-Embedding conventions). `opts` may
 * be NULL for all model-detected defaults. Output and error contract identical
 * to mlxforge_embed. */
int mlxforge_embed_ex(mlxforge_engine* engine, const char* text,
                      const mlxforge_embed_opts* opts, float** out, size_t* out_len,
                      char** err);

/* ---- Requests -------------------------------------------------------------*/

/* Submit a chat completion: renders the model's chat template over `messages`,
 * tokenizes, and enqueues onto the batching scheduler. Many requests may be in
 * flight at once on one engine — they share GPU steps (continuous batching).
 * `sampling` may be NULL for greedy defaults.
 *
 * Returns a request handle, or NULL on failure (sets *err). Free with
 * mlxforge_request_free(). */
mlxforge_request* mlxforge_submit_chat(mlxforge_engine* engine,
                                       const mlxforge_msg* messages, size_t n_messages,
                                       const mlxforge_sampling* sampling, char** err);

/* Submit a raw-text completion: encodes `prompt` directly (no chat template).
 * Otherwise identical to mlxforge_submit_chat. */
mlxforge_request* mlxforge_submit_text(mlxforge_engine* engine, const char* prompt,
                                       const mlxforge_sampling* sampling, char** err);

/* Submit a multimodal (vision-language) request: a text `prompt` plus one image
 * as raw encoded bytes (`image_data` of length `image_len`; JPEG/PNG/…). The
 * engine decodes the image, runs the ViT, renders the chat prompt with the image,
 * and streams tokens. The loaded model must be a vision-language checkpoint
 * (e.g. Qwen3-VL); otherwise the request finishes with an error. Served
 * single-stream (not merged into the continuous-decode batch). Drive and free it
 * exactly like a mlxforge_submit_chat handle (mlxforge_request_next, _free).
 *
 * Returns a request handle, or NULL on failure (sets *err). */
mlxforge_request* mlxforge_submit_image(mlxforge_engine* engine, const char* prompt,
                                        const unsigned char* image_data, size_t image_len,
                                        const mlxforge_sampling* sampling, char** err);

/* One image as raw encoded bytes, for mlxforge_submit_images. */
typedef struct {
  const unsigned char* data;
  size_t len;
} mlxforge_image;

/* Submit a multimodal request with N images: a text `prompt` plus `images[0..n-1]`
 * (each raw encoded bytes). The images are expanded into the prompt — and attended
 * over — in array order. Otherwise identical to mlxforge_submit_image (single-turn,
 * served single-stream; requires a vision-language model). `n_images` must be >= 1.
 *
 * Returns a request handle, or NULL on failure (sets *err). */
mlxforge_request* mlxforge_submit_images(mlxforge_engine* engine, const char* prompt,
                                         const mlxforge_image* images, size_t n_images,
                                         const mlxforge_sampling* sampling, char** err);

/* Pull the next chunk of generated text. Blocks until decoded text is available
 * or the request finishes. The detokenizer is UTF-8-safe: a chunk is always a
 * run of complete characters (never a split multi-byte sequence).
 *
 * Returns:
 *    0  => *text set to a newly-allocated, non-empty UTF-8 chunk
 *          (free it with mlxforge_string_free); call again for more.
 *    1  => done; *text set to NULL. Inspect mlxforge_request_finish_reason().
 *   -1  => error; *text set to NULL (and *err-style state is terminal). */
int mlxforge_request_next(mlxforge_request* req, char** text);

/* Request cancellation: the worker evicts this row at the next step boundary,
 * freeing its batch slot while other streams continue. Idempotent. After this,
 * mlxforge_request_next() drains and returns 1 with finish_reason "cancel". */
void mlxforge_request_cancel(mlxforge_request* req);

/* Why generation stopped: "stop" (EOS) | "length" (max_tokens) | "cancel".
 * Empty string while still running. Valid once mlxforge_request_next() returned
 * 1. Owned by the request. */
const char* mlxforge_request_finish_reason(mlxforge_request* req);

/* Per-token log-probabilities for the tokens emitted so far, as a newly-allocated
 * JSON array in the OpenAI logprobs `content` shape:
 *   [{ "token": "...", "logprob": <float>, "bytes": [<int>...],
 *      "top_logprobs": [{ "token", "logprob", "bytes" }, ...] }, ...]
 * Log-probs accumulate as mlxforge_request_next() drains the stream, so call this
 * once the request is done (next() returned 1). Returns NULL when logprobs were
 * not requested (sampling.logprobs == 0) or none were produced. The caller owns
 * the string and frees it with mlxforge_string_free(). */
char* mlxforge_request_logprobs(mlxforge_request* req);

/* Destroy a request. If it is still running it is cancelled and drained first
 * (so the worker never blocks on a full token queue). NULL is ignored. */
void mlxforge_request_free(mlxforge_request* req);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MLXFORGE_H */
