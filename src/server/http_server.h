// The OpenAI-compatible HTTP server (cpp-httplib). Routes parse OpenAI requests,
// tokenize via the chat template, enqueue a Request to the scheduler, and
// assemble the response — both the non-streaming (blocking) path and the SSE
// streaming path with cancellation on client disconnect.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "core/config.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "server/openai.h"
#include "tokenizer/tokenizer.h"

namespace mlxforge {

class HttpServer {
 public:
  // `ready` reports whether the model has finished loading (else 503); `max_ctx`
  // bounds the prompt length (else 400).
  HttpServer(Scheduler* scheduler, const Tokenizer* tokenizer, ModelConfig config,
             std::string model_name, std::function<bool()> ready, int max_ctx);

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

 private:
  void setup_routes();
  std::string next_id(const char* prefix);

  Scheduler* sched_;
  const Tokenizer* tok_;
  ModelConfig cfg_;
  std::string model_name_;
  std::function<bool()> ready_;
  int max_ctx_;
  httplib::Server svr_;
  std::atomic<long> counter_{0};
};

}  // namespace mlxforge
