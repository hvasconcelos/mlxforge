// Implementation of the libmlxforge C ABI (capi/mlxforge.h).
//
// Each entry point is a thin, exception-safe wrapper over the C++ Engine: it
// translates flat C structs to/from the engine's types, and a try/catch around
// every body guarantees no C++ exception escapes across `extern "C"` (an
// exception unwinding into C is undefined behavior). Failures are reported as
// NULL / negative returns plus an allocated message in `*err`.
#include "capi/mlxforge.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/config.h"
#include "runtime/engine.h"
#include "scheduler/request.h"
#include "tokenizer/tokenizer.h"

namespace {

// Allocate a C string the caller frees with mlxforge_string_free (which uses
// std::free). Returns nullptr on allocation failure (callers treat that as "no
// message"); never throws.
char* dup_cstr(const std::string& s) noexcept {
  char* out = static_cast<char*>(std::malloc(s.size() + 1));
  if (out) std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

void set_err(char** err, const std::string& msg) noexcept {
  if (err) *err = dup_cstr(msg);
}

// Translate the flat C sampling struct into the engine's SamplingParams,
// normalizing the "disabled" sentinels so a zero-initialized struct is valid
// (zeroed => greedy). A null pointer means greedy defaults.
mlxforge::SamplingParams to_params(const mlxforge_sampling* s) {
  mlxforge::SamplingParams p;  // defaults: temperature 1, everything disabled
  if (!s) {
    p.temperature = 0.0f;  // null sampling => deterministic greedy
    return p;
  }
  // Sanitize every field so no hostile/garbage value reaches the sampler graph:
  // a non-finite (NaN/Inf) value would propagate through the math (a NaN
  // temperature divides the logits to all-NaN), and out-of-range knobs are
  // treated as their "disabled" sentinel. The comparisons below double as
  // non-finite guards: any comparison with NaN is false, so NaN falls through to
  // the safe default; std::isfinite handles the temperature, which is a divisor.
  p.temperature = std::isfinite(s->temperature) ? s->temperature : 0.0f;  // non-finite => greedy
  p.top_k = s->top_k > 0 ? s->top_k : 0;  // the sampler further clamps to vocab
  p.top_p = (s->top_p > 0.0f && s->top_p < 1.0f) ? s->top_p : 1.0f;
  p.min_p = (s->min_p > 0.0f && s->min_p <= 1.0f) ? s->min_p : 0.0f;  // a probability in (0,1]
  p.repetition_penalty =
      (std::isfinite(s->repetition_penalty) && s->repetition_penalty > 0.0f)
          ? s->repetition_penalty
          : 1.0f;
  p.frequency_penalty = std::isfinite(s->frequency_penalty) ? s->frequency_penalty : 0.0f;
  p.presence_penalty = std::isfinite(s->presence_penalty) ? s->presence_penalty : 0.0f;
  p.seed = s->seed;
  // logprobs: 0 => off (-1); N > 0 => the chosen token's logprob plus (N - 1)
  // alternatives, so the engine's top_logprobs (alternatives count) is N - 1.
  p.top_logprobs = s->logprobs > 0 ? s->logprobs - 1 : -1;
  return p;
}

int sampling_max_tokens(const mlxforge_sampling* s) {
  return (s && s->max_tokens > 0) ? s->max_tokens : 64;
}

// One {token, logprob, bytes} entry: decode `id` to its text and raw UTF-8 bytes.
nlohmann::json lp_entry(int id, float logprob, const mlxforge::Tokenizer& tok) {
  const std::string text = tok.decode({id});
  nlohmann::json bytes = nlohmann::json::array();
  for (unsigned char c : text) bytes.push_back(static_cast<int>(c));
  return {{"token", text}, {"logprob", logprob}, {"bytes", std::move(bytes)}};
}

}  // namespace

// Opaque handles. The engine wrapper owns the C++ Engine. The request wrapper
// holds a shared_ptr to the Request (so it stays alive while the worker still
// references it) plus its own streaming detokenizer over the engine's
// tokenizer; `done` latches once the stream is fully drained and flushed.
struct mlxforge_engine {
  std::unique_ptr<mlxforge::Engine> engine;
  std::string model_name;  // cached so model_name() can return a stable c_str
};

struct mlxforge_request {
  std::shared_ptr<mlxforge::Request> req;
  std::unique_ptr<mlxforge::StreamingDetokenizer> detok;
  bool done = false;
  // OpenAI logprobs: when `want_logprobs`, each token drained by
  // mlxforge_request_next pops its log-prob (in lockstep) into `logprobs`;
  // mlxforge_request_logprobs serializes the accumulated list using `tok`.
  bool want_logprobs = false;
  const mlxforge::Tokenizer* tok = nullptr;
  std::vector<mlxforge::TokenLogprob> logprobs;
};

extern "C" {

const char* mlxforge_version(void) { return "0.1.0"; }

int mlxforge_abi_version(void) { return MLXFORGE_ABI_VERSION; }

void mlxforge_string_free(char* s) { std::free(s); }

void mlxforge_floats_free(float* p) { std::free(p); }

mlxforge_engine* mlxforge_engine_create(const char* model_spec,
                                        const mlxforge_engine_opts* opts, char** err) {
  if (err) *err = nullptr;
  if (!model_spec || !*model_spec) {
    set_err(err, "model_spec is null or empty");
    return nullptr;
  }
  try {
    mlxforge::EngineConfig cfg;
    cfg.model_spec = model_spec;
    if (opts && opts->max_waiting > 0) cfg.max_waiting = opts->max_waiting;

    auto handle = std::make_unique<mlxforge_engine>();
    handle->model_name = model_spec;
    handle->engine = std::make_unique<mlxforge::Engine>(std::move(cfg));
    return handle.release();
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error creating engine");
  }
  return nullptr;
}

mlxforge_engine* mlxforge_engine_create2(const char* model_spec,
                                         const mlxforge_engine_opts2* opts, char** err) {
  if (err) *err = nullptr;
  if (!model_spec || !*model_spec) {
    set_err(err, "model_spec is null or empty");
    return nullptr;
  }
  try {
    mlxforge::EngineConfig cfg;
    cfg.model_spec = model_spec;
    // Read only the fields the caller's struct_size covers, so a binary built
    // against this header stays correct when later versions append fields.
    auto covered = [&](const void* field_end) {
      return opts && static_cast<size_t>(static_cast<const char*>(field_end) -
                                         reinterpret_cast<const char*>(opts)) <=
                         opts->struct_size;
    };
    if (covered(&opts->max_waiting + 1) && opts->max_waiting > 0)
      cfg.max_waiting = opts->max_waiting;
    if (covered(&opts->kv_bits + 1)) cfg.kv_bits = opts->kv_bits;
    if (covered(&opts->kv_group_size + 1) && opts->kv_group_size > 0)
      cfg.kv_group_size = opts->kv_group_size;

    auto handle = std::make_unique<mlxforge_engine>();
    handle->model_name = model_spec;
    handle->engine = std::make_unique<mlxforge::Engine>(std::move(cfg));
    return handle.release();
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error creating engine");
  }
  return nullptr;
}

int mlxforge_engine_ready(mlxforge_engine* engine) {
  if (!engine || !engine->engine) return 0;
  try {
    return engine->engine->ready() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

const char* mlxforge_engine_model_name(mlxforge_engine* engine) {
  return (engine) ? engine->model_name.c_str() : "";
}

namespace {

// Shared embed core: run the engine, then copy the vector into a malloc buffer
// the caller frees with mlxforge_floats_free. Returns 0 on success.
int embed_into(mlxforge_engine* engine, const char* text,
               const mlxforge::EmbedOptions& opts, float** out, size_t* out_len,
               char** err) {
  if (err) *err = nullptr;
  if (out) *out = nullptr;
  if (out_len) *out_len = 0;
  if (!engine || !engine->engine) {
    set_err(err, "engine is null");
    return 1;
  }
  if (!out || !out_len) {
    set_err(err, "out/out_len must not be null");
    return 1;
  }
  try {
    std::vector<float> v = engine->engine->embed(text ? text : "", opts);
    if (v.empty()) {
      set_err(err, "embedding failed (empty input or model error)");
      return 1;
    }
    float* buf = static_cast<float*>(std::malloc(v.size() * sizeof(float)));
    if (!buf) {
      set_err(err, "out of memory");
      return 1;
    }
    std::memcpy(buf, v.data(), v.size() * sizeof(float));
    *out = buf;
    *out_len = v.size();
    return 0;
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error computing embedding");
  }
  return 1;
}

}  // namespace

int mlxforge_embed(mlxforge_engine* engine, const char* text, int pooling, float** out,
                   size_t* out_len, char** err) {
  // Simple form: explicit pooling, no EOS/instruction (back-compat with v1).
  mlxforge::EmbedOptions opts;
  opts.pooling = pooling;
  opts.add_eos = 0;
  return embed_into(engine, text, opts, out, out_len, err);
}

int mlxforge_embed_ex(mlxforge_engine* engine, const char* text,
                      const mlxforge_embed_opts* o, float** out, size_t* out_len,
                      char** err) {
  mlxforge::EmbedOptions opts;  // defaults: pooling -1, add_eos -1, normalize
  if (o) {
    opts.pooling = o->pooling;
    opts.add_eos = o->add_eos;
    opts.normalize = (o->skip_normalize == 0);
    if (o->instruction && *o->instruction) opts.instruction = o->instruction;
  }
  return embed_into(engine, text, opts, out, out_len, err);
}

void mlxforge_engine_free(mlxforge_engine* engine) {
  if (!engine) return;
  try {
    if (engine->engine) engine->engine->stop();
  } catch (...) {
    // best-effort drain; never throw out of free
  }
  delete engine;
}

namespace {

// Shared submit path: build a Request from already-tokenized prompt ids, attach
// sampling/limits, wire its streaming detokenizer, and enqueue. Returns the
// request handle or nullptr (sets *err).
mlxforge_request* submit_ids(mlxforge_engine* engine, std::vector<int> prompt_ids,
                             const mlxforge_sampling* sampling, char** err) {
  auto req = std::make_shared<mlxforge::Request>();
  req->prompt_ids = std::move(prompt_ids);
  req->params = to_params(sampling);
  req->max_tokens = sampling_max_tokens(sampling);
  req->eos_ids = engine->engine->config().eos_token_ids;
  if (sampling && sampling->json_schema && *sampling->json_schema)
    req->json_schema = sampling->json_schema;

  if (!engine->engine->scheduler().submit(req)) {
    set_err(err, "request rejected: waiting queue is full");
    return nullptr;
  }

  auto handle = std::make_unique<mlxforge_request>();
  handle->req = req;
  handle->detok =
      std::make_unique<mlxforge::StreamingDetokenizer>(engine->engine->tokenizer());
  handle->tok = &engine->engine->tokenizer();
  handle->want_logprobs = req->params.top_logprobs >= 0;
  return handle.release();
}

}  // namespace

mlxforge_request* mlxforge_submit_chat(mlxforge_engine* engine,
                                       const mlxforge_msg* messages, size_t n_messages,
                                       const mlxforge_sampling* sampling, char** err) {
  if (err) *err = nullptr;
  if (!engine || !engine->engine) {
    set_err(err, "engine is null");
    return nullptr;
  }
  try {
    std::vector<mlxforge::Tokenizer::Message> msgs;
    msgs.reserve(n_messages);
    for (size_t i = 0; i < n_messages; ++i) {
      const char* role = messages[i].role ? messages[i].role : "user";
      const char* content = messages[i].content ? messages[i].content : "";
      msgs.push_back({role, content, /*tool_call=*/""});
    }
    std::vector<int> ids = engine->engine->tokenizer().apply_chat_template(msgs);
    return submit_ids(engine, std::move(ids), sampling, err);
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error submitting chat");
  }
  return nullptr;
}

namespace {
// Shared multimodal submit: build a single-turn Request from `prompt` + raw image
// byte-spans, attach sampling/limits, enqueue, and return the handle. The worker
// renders the prompt (one user turn with one block per image) and generates.
mlxforge_request* submit_mm(mlxforge_engine* engine, const char* prompt,
                            std::vector<std::vector<std::uint8_t>> images,
                            const mlxforge_sampling* sampling, char** err) {
  auto req = std::make_shared<mlxforge::Request>();
  req->mm_text = prompt ? prompt : "";
  req->mm_images = std::move(images);
  req->params = to_params(sampling);
  req->max_tokens = sampling_max_tokens(sampling);
  req->eos_ids = engine->engine->config().eos_token_ids;
  if (!engine->engine->scheduler().submit(req)) {
    set_err(err, "request rejected: waiting queue is full");
    return nullptr;
  }
  auto handle = std::make_unique<mlxforge_request>();
  handle->req = req;
  handle->detok = std::make_unique<mlxforge::StreamingDetokenizer>(engine->engine->tokenizer());
  return handle.release();
}
}  // namespace

mlxforge_request* mlxforge_submit_image(mlxforge_engine* engine, const char* prompt,
                                        const unsigned char* image_data, size_t image_len,
                                        const mlxforge_sampling* sampling, char** err) {
  if (err) *err = nullptr;
  if (!engine || !engine->engine) {
    set_err(err, "engine is null");
    return nullptr;
  }
  if (!image_data || image_len == 0) {
    set_err(err, "image_data is empty");
    return nullptr;
  }
  try {
    std::vector<std::vector<std::uint8_t>> images;
    images.emplace_back(image_data, image_data + image_len);
    return submit_mm(engine, prompt, std::move(images), sampling, err);
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error submitting image");
  }
  return nullptr;
}

mlxforge_request* mlxforge_submit_images(mlxforge_engine* engine, const char* prompt,
                                         const mlxforge_image* images, size_t n_images,
                                         const mlxforge_sampling* sampling, char** err) {
  if (err) *err = nullptr;
  if (!engine || !engine->engine) {
    set_err(err, "engine is null");
    return nullptr;
  }
  if (!images || n_images == 0) {
    set_err(err, "images is empty");
    return nullptr;
  }
  try {
    std::vector<std::vector<std::uint8_t>> imgs;
    imgs.reserve(n_images);
    for (size_t i = 0; i < n_images; ++i) {
      if (!images[i].data || images[i].len == 0) {
        set_err(err, "an image is empty");
        return nullptr;
      }
      imgs.emplace_back(images[i].data, images[i].data + images[i].len);
    }
    return submit_mm(engine, prompt, std::move(imgs), sampling, err);
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error submitting images");
  }
  return nullptr;
}

mlxforge_request* mlxforge_submit_text(mlxforge_engine* engine, const char* prompt,
                                       const mlxforge_sampling* sampling, char** err) {
  if (err) *err = nullptr;
  if (!engine || !engine->engine) {
    set_err(err, "engine is null");
    return nullptr;
  }
  try {
    std::vector<int> ids = engine->engine->tokenizer().encode(prompt ? prompt : "");
    return submit_ids(engine, std::move(ids), sampling, err);
  } catch (const std::exception& e) {
    set_err(err, e.what());
  } catch (...) {
    set_err(err, "unknown error submitting text");
  }
  return nullptr;
}

int mlxforge_request_next(mlxforge_request* req, char** text) {
  if (text) *text = nullptr;
  if (!req || !req->req) return -1;
  try {
    if (req->done) return 1;
    // Pull tokens until one completes some UTF-8 text, or the stream ends. The
    // detokenizer may return "" for a token that only advances a multi-byte
    // character; we keep going rather than emit empty chunks.
    for (;;) {
      int tok = 0;
      if (req->req->tokens.pop(tok)) {
        // Pop this token's log-prob in lockstep (the worker pushes one per emitted
        // token) and accumulate it for mlxforge_request_logprobs.
        if (req->want_logprobs) {
          mlxforge::TokenLogprob lp;
          if (req->req->logprobs.pop(lp)) req->logprobs.push_back(std::move(lp));
        }
        std::string piece = req->detok->add(tok);
        if (piece.empty()) continue;
        if (text) *text = dup_cstr(piece);
        return 0;
      }
      // Producer closed and queue drained: flush any trailing complete bytes
      // once, then we are done.
      std::string tail = req->detok->finish();
      req->done = true;
      if (!tail.empty()) {
        if (text) *text = dup_cstr(tail);
        return 0;
      }
      return 1;
    }
  } catch (...) {
    req->done = true;
    return -1;
  }
}

void mlxforge_request_cancel(mlxforge_request* req) {
  if (req && req->req) req->req->cancelled.store(true);
}

const char* mlxforge_request_finish_reason(mlxforge_request* req) {
  return (req && req->req) ? req->req->finish_reason.c_str() : "";
}

char* mlxforge_request_logprobs(mlxforge_request* req) {
  if (!req || !req->tok || req->logprobs.empty()) return nullptr;
  try {
    nlohmann::json content = nlohmann::json::array();
    for (const mlxforge::TokenLogprob& lp : req->logprobs) {
      nlohmann::json entry = lp_entry(lp.id, lp.logprob, *req->tok);
      nlohmann::json top = nlohmann::json::array();
      for (const auto& alt : lp.top) top.push_back(lp_entry(alt.first, alt.second, *req->tok));
      entry["top_logprobs"] = std::move(top);
      content.push_back(std::move(entry));
    }
    return dup_cstr(content.dump());
  } catch (...) {
    return nullptr;
  }
}

void mlxforge_request_free(mlxforge_request* req) {
  if (!req) return;
  try {
    // If still running, cancel and drain the token queue so the worker's
    // producer never blocks on a full, abandoned queue.
    if (!req->done && req->req) {
      req->req->cancelled.store(true);
      int tok = 0;
      while (req->req->tokens.pop(tok)) {
        // Drain log-probs in lockstep too, so the worker's producer never blocks
        // on a full, abandoned log-prob queue.
        if (req->want_logprobs) {
          mlxforge::TokenLogprob lp;
          req->req->logprobs.pop(lp);
        }
      }
    }
  } catch (...) {
    // never throw out of free
  }
  delete req;
}

}  // extern "C"
