// XLLM-022: the OpenAI-compatible HTTP server (cpp-httplib). Routes parse OpenAI
// requests, tokenize via the chat template, enqueue a Request to the scheduler,
// and assemble the response. XLLM-022 implements non-streaming; XLLM-023 adds
// SSE streaming + cancellation.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "core/config.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "server/openai.h"
#include "tokenizer/tokenizer.h"

namespace xllm {

class HttpServer {
 public:
  HttpServer(Scheduler* scheduler, const Tokenizer* tokenizer, ModelConfig config,
             std::string model_name);

  void listen(const std::string& host, int port);
  void stop();

  // Build a scheduler Request from a parsed OpenAI request (tokenizes the
  // prompt). Exposed for tests.
  std::shared_ptr<Request> make_request(const ChatRequest& cr) const;

  // Run a request to completion (non-streaming) and assemble the response JSON.
  nlohmann::json run_blocking(const ChatRequest& cr);

 private:
  void setup_routes();
  std::string next_id(const char* prefix);

  Scheduler* sched_;
  const Tokenizer* tok_;
  ModelConfig cfg_;
  std::string model_name_;
  httplib::Server svr_;
  std::atomic<long> counter_{0};
};

}  // namespace xllm
