// The OpenAI-compatible HTTP server (cpp-httplib). Routes parse OpenAI requests,
// tokenize via the chat template, enqueue a Request to the scheduler, and
// assemble the response — both the non-streaming (blocking) path and the SSE
// streaming path with cancellation on client disconnect.
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "core/config.h"
#include "runtime/engine.h"
#include "runtime/metrics.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "server/openai.h"
#include "tokenizer/tokenizer.h"

namespace mlxforge {

class HttpServer {
 public:
  // Blocking embed seam (e.g. Engine::embed): text + options -> vector. The
  // server reuses it for /v1/embeddings so detection/EOS/instruction live in one
  // place. May be empty (then /v1/embeddings returns 501).
  using EmbedFn = std::function<std::vector<float>(const std::string&, const EmbedOptions&)>;

  // `ready` reports whether the model has finished loading (else 503); `max_ctx`
  // bounds the prompt length (else 400); `metrics` provides the live decode
  // snapshot for /health (reads only atomics, so it is safe off the worker thread).
  // `embed` is the optional embeddings seam (see EmbedFn).
  HttpServer(Scheduler* scheduler, const Tokenizer* tokenizer, ModelConfig config,
             std::string model_name, std::function<bool()> ready, int max_ctx,
             std::function<WorkerMetrics()> metrics, EmbedFn embed = nullptr);

  void listen(const std::string& host, int port);
  void stop();

  // Build a scheduler Request from a parsed OpenAI request (tokenizes the
  // prompt). Exposed for tests.
  std::shared_ptr<Request> make_request(const ChatRequest& cr) const;

  // Drain an already-submitted request and assemble the response JSON.
  nlohmann::json run_blocking(const std::shared_ptr<Request>& req, const ChatRequest& cr);

  // Stream an already-submitted request as SSE chat.completion.chunk frames.
  // Client disconnect (sink write fails) sets cancelled so the worker evicts.
  // When `allow_tools`, output that begins as a JSON object is buffered and, if
  // it parses as a tool call, emitted as a single tool_calls delta.
  void stream_chat(const std::shared_ptr<Request>& req, httplib::Response& res, bool allow_tools);

  // Anthropic Messages API analogues: drain into a {content:[blocks]} response,
  // or stream the message_start / content_block_* / message_delta / message_stop
  // event sequence. The tool-call buffering mirrors stream_chat.
  nlohmann::json run_blocking_messages(const std::shared_ptr<Request>& req, const ChatRequest& cr);
  void stream_messages(const std::shared_ptr<Request>& req, httplib::Response& res, bool allow_tools);

 private:
  // Validate the model name, tokenize, bound the context, and submit. On any
  // failure it calls `fail` (with the OpenAI-style status/type/code, which the
  // Anthropic handler remaps) and returns nullptr.
  using FailFn =
      std::function<void(int, const std::string&, const std::string&, const std::string&)>;
  std::shared_ptr<Request> submit_request(const ChatRequest& cr, const FailFn& fail);

  void setup_routes();
  std::string next_id(const char* prefix);

  Scheduler* sched_;
  const Tokenizer* tok_;
  ModelConfig cfg_;
  std::string model_name_;
  std::function<bool()> ready_;
  int max_ctx_;
  std::function<WorkerMetrics()> metrics_;
  EmbedFn embed_;
  std::chrono::steady_clock::time_point start_time_{std::chrono::steady_clock::now()};
  httplib::Server svr_;
  std::atomic<long> counter_{0};
};

}  // namespace mlxforge
