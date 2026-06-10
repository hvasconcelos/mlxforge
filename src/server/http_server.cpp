#include "server/http_server.h"

#include <array>
#include <cstdint>
#include <ctime>
#include <stdexcept>
#include <vector>

#include "core/logging.h"
#include "server/anthropic.h"
#include "vision/image_decode.h"
#include "vision/preprocess.h"

namespace mlxforge {

namespace {
using nlohmann::json;

// OpenAI finish_reason is "stop" or "length"; map the worker's reasons onto it.
std::string finish_reason_of(const std::string& worker_reason) {
  return worker_reason == "length" ? "length" : "stop";
}

json error_body(const std::string& message, const std::string& type, const std::string& code) {
  return {{"error", {{"message", message}, {"type", type}, {"code", code}}}};
}
}  // namespace

HttpServer::HttpServer(Scheduler* scheduler, const Tokenizer* tokenizer, ModelConfig config,
                       std::string model_name, std::function<bool()> ready, int max_ctx,
                       std::function<WorkerMetrics()> metrics, EmbedFn embed)
    : sched_(scheduler),
      tok_(tokenizer),
      cfg_(std::move(config)),
      model_name_(std::move(model_name)),
      ready_(std::move(ready)),
      max_ctx_(max_ctx),
      metrics_(std::move(metrics)),
      embed_(std::move(embed)) {
  // Each streaming connection holds a worker thread for its whole lifetime, so
  // size the pool well above the expected concurrency (default is ~8).
  svr_.new_task_queue = [] { return new httplib::ThreadPool(64); };
  setup_routes();
}

std::string HttpServer::next_id(const char* prefix) {
  return std::string(prefix) + std::to_string(counter_.fetch_add(1));
}

std::shared_ptr<Request> HttpServer::make_request(const ChatRequest& cr) const {
  auto req = std::make_shared<Request>();
  req->params = cr.params;
  // OpenAI logprobs -> engine top_logprobs: off (-1) unless `logprobs` is set,
  // then the requested alternatives count (0 = chosen-token logprob only).
  // Unsupported on the single-stream vision path, so off when an image is present.
  req->params.top_logprobs = (cr.logprobs && !cr.has_images()) ? cr.top_logprobs : -1;
  req->max_tokens = cr.max_tokens;
  req->eos_ids = cfg_.eos_token_ids;

  // Multimodal (Qwen3-VL): attached image(s) make this a single-stream vision
  // turn. We render the FULL chat history (system + prior turns) here, sizing each
  // <|image_pad|> run from its image's dimensions (a CPU probe — no decode) and
  // attaching it to the turn the image belongs to, so images placed across
  // different turns land at the right positions. The worker decodes + ViT-encodes
  // the bytes (in order) and generates from these prompt_ids.
  if (cr.has_images()) {
    if (!cfg_.has_vision_tower())
      throw std::runtime_error("this model does not support image input");
    const PreprocessConfig pc = PreprocessConfig::from(*cfg_.vision);
    std::vector<Tokenizer::Message> msgs = cr.messages;  // copy: set placeholder counts

    // Mirror render_qwen3's vision-block emission order: a leading system turn and
    // any tool/assistant turns render no vision block, so we skip their images
    // (and never let them desync the placeholder count).
    const bool have_system = !msgs.empty() && msgs.front().role == "system";
    for (size_t i = (have_system ? 1 : 0); i < msgs.size(); ++i) {
      if (msgs[i].role == "tool" || msgs[i].role == "assistant") continue;
      for (const auto& img : cr.message_images[i]) {
        const std::array<int, 2> hw =
            image_info(reinterpret_cast<const uint8_t*>(img.data()), img.size());
        msgs[i].image_token_counts.push_back(image_token_count(hw[0], hw[1], pc));
        req->mm_images.emplace_back(img.begin(), img.end());
      }
    }
    req->prompt_ids = tok_->apply_chat_template(msgs, true, "", {}, cr.enable_thinking);
    return req;
  }

  // tool_choice "none" suppresses the schemas (and later, the output parsing).
  const std::vector<std::string> tools = cr.tools_enabled() ? cr.tools : std::vector<std::string>{};
  req->prompt_ids = cr.is_chat
                        ? tok_->apply_chat_template(cr.messages, true, "", tools, cr.enable_thinking)
                        : tok_->encode(cr.messages.front().content);
  return req;
}

nlohmann::json HttpServer::run_blocking(const std::shared_ptr<Request>& req,
                                       const ChatRequest& cr) {
  const int prompt_tokens = static_cast<int>(req->prompt_ids.size());

  // Drain tokens and (when enabled) their log-probs in lockstep.
  const bool want_lp = req->params.top_logprobs >= 0;
  std::vector<int> out;
  std::vector<TokenLogprob> out_lp;
  int tok = 0;
  while (req->tokens.pop(tok)) {
    out.push_back(tok);
    if (want_lp) {
      TokenLogprob lp;
      if (req->logprobs.pop(lp)) out_lp.push_back(std::move(lp));
    }
  }

  const std::string content = tok_->decode(out);
  const std::string finish = finish_reason_of(req->finish_reason);
  const long created = static_cast<long>(std::time(nullptr));
  const int completion_tokens = static_cast<int>(out.size());

  if (cr.is_chat) {
    if (cr.tools_enabled()) {
      auto calls = parse_tool_calls(content);
      if (!calls.empty())
        return make_chat_completion_tools(next_id("chatcmpl-"), created, model_name_, calls,
                                          prompt_tokens, completion_tokens);
    }
    // Pass an array (possibly empty) when logprobs were requested, else null.
    const json logprobs_content =
        want_lp ? make_logprobs_content(out_lp, *tok_) : json(nullptr);
    return make_chat_completion(next_id("chatcmpl-"), created, model_name_, content, finish,
                                prompt_tokens, completion_tokens, logprobs_content);
  }
  // Legacy text completion shape.
  return {{"id", next_id("cmpl-")},
          {"object", "text_completion"},
          {"created", created},
          {"model", model_name_},
          {"choices", json::array({{{"index", 0}, {"text", content}, {"finish_reason", finish}}})},
          {"usage", make_usage(prompt_tokens, completion_tokens)}};
}

void HttpServer::stream_chat(const std::shared_ptr<Request>& req, httplib::Response& res,
                             bool allow_tools) {
  const std::string id = next_id("chatcmpl-");
  const long created = static_cast<long>(std::time(nullptr));
  const std::string model = model_name_;
  const Tokenizer* tok = tok_;

  res.set_chunked_content_provider(
      "text/event-stream",
      [req, id, created, model, tok, allow_tools](size_t /*offset*/,
                                                  httplib::DataSink& sink) -> bool {
        auto send = [&](const std::string& frame) {
          if (sink.write(frame.data(), frame.size())) return true;
          req->cancelled.store(true);  // client disconnected -> worker evicts
          return false;
        };
        // Per-token log-probs, popped in lockstep with the token ids and attached
        // (as choices[0].logprobs.content) to the next content delta we emit.
        const bool want_lp = req->params.top_logprobs >= 0;
        std::vector<TokenLogprob> pending_lp;
        auto send_content = [&](const std::string& s) {
          if (s.empty()) return true;
          json lp = want_lp ? make_logprobs_content(pending_lp, *tok) : json(nullptr);
          pending_lp.clear();
          return send(sse_frame(make_chat_chunk(id, created, model, {{"content", s}}, nullptr, lp)));
        };

        // First chunk announces the assistant role.
        if (!send(sse_frame(make_chat_chunk(id, created, model, {{"role", "assistant"}}, nullptr))))
          return false;

        // With tools enabled we can't stream incrementally until we know the
        // output isn't a tool call, so buffer until the first non-space char
        // (a leading '{' => candidate call) decides; plain text streams live.
        StreamingDetokenizer detok(*tok);
        std::string buffered;            // only accumulated while `buffering`
        bool buffering = allow_tools;
        int t = 0;
        while (req->tokens.pop(t)) {
          // Pop this token's log-prob in lockstep (the worker pushes one per
          // emitted token) so it stays aligned with the detokenized text.
          if (want_lp) {
            TokenLogprob lp;
            if (req->logprobs.pop(lp)) pending_lp.push_back(std::move(lp));
          }
          std::string piece = detok.add(t);
          if (piece.empty()) continue;
          if (buffering) {
            buffered += piece;
            const size_t lead = buffered.find_first_not_of(" \t\r\n");
            if (lead == std::string::npos) continue;  // still only whitespace
            if (buffered[lead] == '{') continue;      // candidate tool call: keep buffering
            buffering = false;                         // plain text: flush what we held
            if (!send_content(buffered)) return false;
            buffered.clear();
          } else if (!send_content(piece)) {
            return false;
          }
        }
        const std::string tail = detok.finish();

        std::string finish = finish_reason_of(req->finish_reason);
        if (buffering) {
          // The whole output was held back as a tool-call candidate.
          buffered += tail;
          auto calls = parse_tool_calls(buffered);
          if (!calls.empty()) {
            json delta = {{"tool_calls", make_tool_calls(calls)}};
            if (!send(sse_frame(make_chat_chunk(id, created, model, delta, nullptr)))) return false;
            finish = "tool_calls";
          } else if (!send_content(buffered)) {
            return false;  // not a call after all: emit as content
          }
        } else if (!send_content(tail)) {
          return false;  // trailing text from detok.finish()
        }

        // Final chunk carries the finish_reason, then the [DONE] sentinel.
        send(sse_frame(make_chat_chunk(id, created, model, json::object(), finish)));
        send(kSseDone);
        return false;  // stream complete
      });
}

namespace {
// Anthropic stop_reason: the worker only distinguishes EOS ("stop") from the
// length cap ("length"); tool calls are detected from the decoded text.
std::string anthropic_stop_reason(const std::string& worker_reason) {
  return worker_reason == "length" ? "max_tokens" : "end_turn";
}

// Map an HTTP status onto the Anthropic error `type` (submit_request hands us
// OpenAI-style codes, which we ignore in favor of the status).
std::string anthropic_error_type(int status) {
  switch (status) {
    case 404:
      return "not_found_error";
    case 429:
      return "rate_limit_error";
    case 503:
      return "overloaded_error";
    default:
      return "invalid_request_error";
  }
}
}  // namespace

nlohmann::json HttpServer::run_blocking_messages(const std::shared_ptr<Request>& req,
                                                 const ChatRequest& cr) {
  const int input_tokens = static_cast<int>(req->prompt_ids.size());

  std::vector<int> out;
  int tok = 0;
  while (req->tokens.pop(tok)) out.push_back(tok);

  const std::string content = tok_->decode(out);
  const int output_tokens = static_cast<int>(out.size());
  std::string stop_reason = anthropic_stop_reason(req->finish_reason);

  json blocks = json::array();
  if (cr.tools_enabled()) {
    auto calls = parse_tool_calls(content);
    if (!calls.empty()) {
      blocks = make_tool_use_blocks(calls);
      stop_reason = "tool_use";
    }
  }
  if (blocks.empty()) blocks.push_back({{"type", "text"}, {"text", content}});

  return make_message_response(next_id("msg_"), model_name_, blocks, stop_reason, input_tokens,
                               output_tokens);
}

void HttpServer::stream_messages(const std::shared_ptr<Request>& req, httplib::Response& res,
                                 bool allow_tools) {
  const std::string id = next_id("msg_");
  const std::string model = model_name_;
  const Tokenizer* tok = tok_;
  const int input_tokens = static_cast<int>(req->prompt_ids.size());

  res.set_chunked_content_provider(
      "text/event-stream",
      [req, id, model, tok, allow_tools, input_tokens](size_t /*offset*/,
                                                       httplib::DataSink& sink) -> bool {
        auto send = [&](const std::string& frame) {
          if (sink.write(frame.data(), frame.size())) return true;
          req->cancelled.store(true);  // client disconnected -> worker evicts
          return false;
        };

        if (!send(sse_event("message_start", make_message_start(id, model, input_tokens))))
          return false;

        // The lone text block (index 0) is opened lazily on the first content so
        // a pure tool-call response carries no empty text block.
        bool text_open = false;
        auto open_text = [&]() -> bool {
          if (text_open) return true;
          text_open = true;
          return send(sse_event("content_block_start",
                                make_content_block_start(0, {{"type", "text"}, {"text", ""}})));
        };
        auto send_text = [&](const std::string& s) -> bool {
          if (s.empty()) return true;
          return open_text() && send(sse_event("content_block_delta", make_text_delta(0, s)));
        };

        // Mirror stream_chat: buffer until we know the output isn't a tool call.
        StreamingDetokenizer detok(*tok);
        std::string buffered;
        bool buffering = allow_tools;
        int output_tokens = 0;
        int t = 0;
        while (req->tokens.pop(t)) {
          ++output_tokens;
          std::string piece = detok.add(t);
          if (piece.empty()) continue;
          if (buffering) {
            buffered += piece;
            const size_t lead = buffered.find_first_not_of(" \t\r\n");
            if (lead == std::string::npos) continue;  // still only whitespace
            if (buffered[lead] == '{') continue;      // candidate tool call: keep buffering
            buffering = false;
            if (!send_text(buffered)) return false;
            buffered.clear();
          } else if (!send_text(piece)) {
            return false;
          }
        }
        const std::string tail = detok.finish();

        std::string stop_reason = anthropic_stop_reason(req->finish_reason);
        if (buffering) {
          buffered += tail;
          auto calls = parse_tool_calls(buffered);
          if (!calls.empty()) {
            // Each call is its own tool_use block; the input arrives as a single
            // input_json_delta (the engine produces the JSON atomically).
            for (int i = 0; i < static_cast<int>(calls.size()); ++i) {
              const json block = {{"type", "tool_use"},
                                  {"id", "toolu_" + std::to_string(i)},
                                  {"name", calls[i].name},
                                  {"input", json::object()}};
              if (!send(sse_event("content_block_start", make_content_block_start(i, block))))
                return false;
              if (!send(sse_event("content_block_delta",
                                  make_input_json_delta(i, calls[i].arguments))))
                return false;
              if (!send(sse_event("content_block_stop", make_content_block_stop(i)))) return false;
            }
            stop_reason = "tool_use";
          } else if (!send_text(buffered)) {
            return false;
          }
        } else if (!send_text(tail)) {
          return false;
        }

        // A non-tool response always carries a text block (possibly empty).
        if (stop_reason != "tool_use" && !open_text()) return false;
        if (text_open && !send(sse_event("content_block_stop", make_content_block_stop(0))))
          return false;

        if (!send(sse_event("message_delta", make_message_delta(stop_reason, output_tokens))))
          return false;
        send(sse_event("message_stop", kMessageStop));
        return false;  // stream complete
      });
}

std::shared_ptr<Request> HttpServer::submit_request(const ChatRequest& cr, const FailFn& fail) {
  // The server hosts a single model: a request may omit "model" (served as-is)
  // but if it names one it must be the loaded model, else 404.
  if (!cr.model.empty() && cr.model != model_name_) {
    fail(404,
         "model '" + cr.model + "' is not available; this server serves '" + model_name_ + "'",
         "invalid_request_error", "model_not_found");
    return nullptr;
  }
  auto req = make_request(cr);
  if (max_ctx_ > 0 && static_cast<int>(req->prompt_ids.size()) > max_ctx_) {
    fail(400, "prompt exceeds the maximum context length", "invalid_request_error",
         "context_length_exceeded");
    return nullptr;
  }
  if (!sched_->submit(req)) {
    fail(429, "server is overloaded; retry later", "rate_limit_error", "queue_full");
    return nullptr;
  }
  return req;
}

void HttpServer::setup_routes() {
  svr_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
    const bool ready = !ready_ || ready_();
    const WorkerMetrics m = metrics_ ? metrics_() : WorkerMetrics{};
    const long uptime = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - start_time_)
                            .count();
    const json body = {
        {"status", ready ? "ok" : "loading"},
        {"uptime_seconds", uptime},
        {"model",
         {{"name", model_name_},
          {"type", cfg_.model_type},
          {"context_size", cfg_.max_position_embeddings},
          {"max_context", max_ctx_},
          {"num_layers", cfg_.n_layers},
          {"num_heads", cfg_.n_heads},
          {"num_kv_heads", cfg_.n_kv_heads},
          {"vocab_size", cfg_.vocab},
          {"quantized", cfg_.quantized},
          {"quant_bits", cfg_.quant_bits}}},
        {"queue", {{"waiting", sched_->waiting_size()}, {"max_waiting", sched_->max_waiting()}}},
        {"batch", {{"active", m.active_batch}, {"peak", m.peak_batch}}},
        {"decode",
         {{"steps", m.decode_steps},
          {"requests_completed", m.requests_completed},
          {"prompt_tokens_total", m.prompt_tokens_total},
          {"completion_tokens_total", m.completion_tokens_total},
          {"avg_ttft_ms", m.avg_ttft_ms},
          {"avg_request_ms", m.avg_request_ms},
          {"avg_tokens_per_second", m.avg_tokens_per_second}}},
    };
    res.set_content(body.dump(), "application/json");
  });

  svr_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
    res.set_content(make_models_list(model_name_).dump(), "application/json");
  });

  auto fail = [](httplib::Response& res, int status, const std::string& msg, const std::string& type,
                 const std::string& code) {
    log::warn("request failed: {} {} ({})", status, code, msg);
    res.status = status;
    res.set_content(error_body(msg, type, code).dump(), "application/json");
  };

  auto handle = [this, fail](bool chat, const httplib::Request& http_req, httplib::Response& res) {
    log::debug("{} {} ({} bytes)", http_req.method, http_req.path, http_req.body.size());
    if (ready_ && !ready_()) {
      fail(res, 503, "model is still loading", "server_error", "model_loading");
      return;
    }
    const FailFn fail_req = [&](int s, const std::string& m, const std::string& t,
                               const std::string& c) { fail(res, s, m, t, c); };
    try {
      json body = json::parse(http_req.body);
      ChatRequest cr = chat ? parse_chat_request(body) : parse_completion_request(body);
      auto req = submit_request(cr, fail_req);
      if (!req) return;
      if (chat && cr.stream) {
        stream_chat(req, res, cr.tools_enabled());
      } else {
        res.set_content(run_blocking(req, cr).dump(), "application/json");
      }
    } catch (const json::parse_error& e) {
      fail(res, 400, e.what(), "invalid_request_error", "bad_json");
    } catch (const std::exception& e) {
      fail(res, 400, e.what(), "invalid_request_error", "invalid_params");
    }
  };

  svr_.Post("/v1/chat/completions", [handle](const httplib::Request& req, httplib::Response& res) {
    handle(true, req, res);
  });
  svr_.Post("/v1/completions", [handle](const httplib::Request& req, httplib::Response& res) {
    handle(false, req, res);
  });

  // OpenAI-compatible embeddings. Reuses the engine's embed seam (so the model's
  // detected conventions — last-token pooling + EOS for Qwen3-Embedding — apply).
  svr_.Post("/v1/embeddings", [this, fail](const httplib::Request& http_req,
                                           httplib::Response& res) {
    log::debug("{} {} ({} bytes)", http_req.method, http_req.path, http_req.body.size());
    if (!embed_) {
      fail(res, 501, "embeddings are not enabled on this server", "server_error",
           "not_implemented");
      return;
    }
    if (ready_ && !ready_()) {
      fail(res, 503, "model is still loading", "server_error", "model_loading");
      return;
    }
    try {
      EmbeddingsRequest er = parse_embeddings_request(json::parse(http_req.body));
      std::vector<std::vector<float>> vecs;
      vecs.reserve(er.input.size());
      int prompt_tokens = 0;
      for (const auto& text : er.input) {
        // Default options: the engine applies the model's detected conventions.
        std::vector<float> v = embed_(text, EmbedOptions{});
        if (v.empty()) {
          fail(res, 500, "embedding failed", "server_error", "embed_failed");
          return;
        }
        prompt_tokens += static_cast<int>(tok_->encode(text).size());
        vecs.push_back(std::move(v));
      }
      res.set_content(make_embeddings_response(model_name_, vecs, prompt_tokens).dump(),
                      "application/json");
    } catch (const json::parse_error& e) {
      fail(res, 400, e.what(), "invalid_request_error", "bad_json");
    } catch (const std::exception& e) {
      fail(res, 400, e.what(), "invalid_request_error", "invalid_params");
    }
  });

  // Anthropic Messages API. Same generation pipeline as the OpenAI handler; only
  // the request shape, the response/SSE framing, and the error body differ.
  svr_.Post("/v1/messages", [this](const httplib::Request& http_req, httplib::Response& res) {
    log::debug("{} {} ({} bytes)", http_req.method, http_req.path, http_req.body.size());
    // Frame failures as the Anthropic error body {type:"error", error:{type,
    // message}} (the shared submit_request passes OpenAI codes we don't use).
    auto a_fail = [&res](int status, const std::string& msg, const std::string& /*type*/,
                         const std::string& /*code*/) {
      const std::string type = anthropic_error_type(status);
      log::warn("request failed: {} {} ({})", status, type, msg);
      res.status = status;
      res.set_content(anthropic_error_body(type, msg).dump(), "application/json");
    };
    if (ready_ && !ready_()) {
      a_fail(503, "model is still loading", "", "");
      return;
    }
    try {
      ChatRequest cr = parse_messages_request(json::parse(http_req.body));
      auto req = submit_request(cr, a_fail);
      if (!req) return;
      if (cr.stream) {
        stream_messages(req, res, cr.tools_enabled());
      } else {
        res.set_content(run_blocking_messages(req, cr).dump(), "application/json");
      }
    } catch (const std::exception& e) {
      // Covers both json::parse_error and parse/validation runtime_errors.
      a_fail(400, e.what(), "invalid_request_error", "");
    }
  });
}

void HttpServer::listen(const std::string& host, int port) {
  log::info("http: listening on {}:{}", host, port);
  svr_.listen(host, port);
}

void HttpServer::stop() {
  log::info("http: stopping");
  svr_.stop();
}

}  // namespace mlxforge
