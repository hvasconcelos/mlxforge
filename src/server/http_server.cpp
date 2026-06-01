#include "server/http_server.h"

#include <ctime>
#include <stdexcept>
#include <vector>

namespace xllm {

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
                       std::string model_name)
    : sched_(scheduler),
      tok_(tokenizer),
      cfg_(std::move(config)),
      model_name_(std::move(model_name)) {
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
  req->prompt_ids = cr.is_chat ? tok_->apply_chat_template(cr.messages)
                               : tok_->encode(cr.messages.front().content);
  req->params = cr.params;
  req->max_tokens = cr.max_tokens;
  req->eos_ids = cfg_.eos_token_ids;
  return req;
}

nlohmann::json HttpServer::run_blocking(const ChatRequest& cr) {
  auto req = make_request(cr);
  const int prompt_tokens = static_cast<int>(req->prompt_ids.size());
  sched_->submit(req);

  std::vector<int> out;
  int tok = 0;
  while (req->tokens.pop(tok)) out.push_back(tok);

  const std::string content = tok_->decode(out);
  const std::string finish = finish_reason_of(req->finish_reason);
  const long created = static_cast<long>(std::time(nullptr));
  const int completion_tokens = static_cast<int>(out.size());

  if (cr.is_chat) {
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

void HttpServer::stream_chat(const ChatRequest& cr, httplib::Response& res) {
  auto req = make_request(cr);
  sched_->submit(req);
  const std::string id = next_id("chatcmpl-");
  const long created = static_cast<long>(std::time(nullptr));
  const std::string model = model_name_;
  const Tokenizer* tok = tok_;

  res.set_chunked_content_provider(
      "text/event-stream",
      [req, id, created, model, tok](size_t /*offset*/, httplib::DataSink& sink) -> bool {
        auto send = [&](const std::string& frame) {
          if (sink.write(frame.data(), frame.size())) return true;
          req->cancelled.store(true);  // client disconnected -> worker evicts
          return false;
        };

        // First chunk announces the assistant role.
        if (!send(sse_frame(make_chat_chunk(id, created, model, {{"role", "assistant"}}, nullptr))))
          return false;

        StreamingDetokenizer detok(*tok);
        int t = 0;
        while (req->tokens.pop(t)) {
          std::string piece = detok.add(t);
          if (piece.empty()) continue;
          if (!send(sse_frame(make_chat_chunk(id, created, model, {{"content", piece}}, nullptr))))
            return false;
        }
        std::string tail = detok.finish();
        if (!tail.empty() &&
            !send(sse_frame(make_chat_chunk(id, created, model, {{"content", tail}}, nullptr))))
          return false;

        // Final chunk carries the finish_reason, then the [DONE] sentinel.
        const std::string finish = finish_reason_of(req->finish_reason);
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

  auto handle = [this](bool chat, const httplib::Request& req, httplib::Response& res) {
    try {
      json body = json::parse(req.body);
      ChatRequest cr = chat ? parse_chat_request(body) : parse_completion_request(body);
      if (chat && cr.stream) {
        stream_chat(cr, res);
      } else {
        res.set_content(run_blocking(cr).dump(), "application/json");
      }
    } catch (const json::parse_error& e) {
      res.status = 400;
      res.set_content(error_body(e.what(), "invalid_request_error", "bad_json").dump(),
                      "application/json");
    } catch (const std::exception& e) {
      res.status = 400;
      res.set_content(error_body(e.what(), "invalid_request_error", "invalid_params").dump(),
                      "application/json");
    }
  };

  svr_.Post("/v1/chat/completions", [handle](const httplib::Request& req, httplib::Response& res) {
    handle(true, req, res);
  });
  svr_.Post("/v1/completions", [handle](const httplib::Request& req, httplib::Response& res) {
    handle(false, req, res);
  });
}

void HttpServer::listen(const std::string& host, int port) { svr_.listen(host, port); }

void HttpServer::stop() { svr_.stop(); }

}  // namespace xllm
