#include "server/http_server.h"

#include <ctime>
#include <stdexcept>
#include <vector>

#include "core/logging.h"

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
                       std::string model_name, std::function<bool()> ready, int max_ctx)
    : sched_(scheduler),
      tok_(tokenizer),
      cfg_(std::move(config)),
      model_name_(std::move(model_name)),
      ready_(std::move(ready)),
      max_ctx_(max_ctx) {
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
  // tool_choice "none" suppresses the schemas (and later, the output parsing).
  const std::vector<std::string> tools = cr.tools_enabled() ? cr.tools : std::vector<std::string>{};
  req->prompt_ids = cr.is_chat ? tok_->apply_chat_template(cr.messages, true, "", tools)
                               : tok_->encode(cr.messages.front().content);
  req->params = cr.params;
  req->max_tokens = cr.max_tokens;
  req->eos_ids = cfg_.eos_token_ids;
  return req;
}

nlohmann::json HttpServer::run_blocking(const std::shared_ptr<Request>& req,
                                       const ChatRequest& cr) {
  const int prompt_tokens = static_cast<int>(req->prompt_ids.size());

  std::vector<int> out;
  int tok = 0;
  while (req->tokens.pop(tok)) out.push_back(tok);

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
    return make_chat_completion(next_id("chatcmpl-"), created, model_name_, content, finish,
                                prompt_tokens, completion_tokens);
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
        auto send_content = [&](const std::string& s) {
          return s.empty() || send(sse_frame(make_chat_chunk(id, created, model,
                                                             {{"content", s}}, nullptr)));
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

void HttpServer::setup_routes() {
  svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(json{{"status", "ok"}}.dump(), "application/json");
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
    try {
      json body = json::parse(http_req.body);
      ChatRequest cr = chat ? parse_chat_request(body) : parse_completion_request(body);
      auto req = make_request(cr);
      if (max_ctx_ > 0 && static_cast<int>(req->prompt_ids.size()) > max_ctx_) {
        fail(res, 400, "prompt exceeds the maximum context length", "invalid_request_error",
             "context_length_exceeded");
        return;
      }
      if (!sched_->submit(req)) {
        fail(res, 429, "server is overloaded; retry later", "rate_limit_error", "queue_full");
        return;
      }
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
