#include "server/openai.h"

#include <stdexcept>

namespace xllm {

namespace {
using nlohmann::json;

// stop may be a string or an array of strings.
std::vector<std::string> parse_stop(const json& body) {
  auto it = body.find("stop");
  if (it == body.end() || it->is_null()) return {};
  if (it->is_string()) return {it->get<std::string>()};
  if (it->is_array()) return it->get<std::vector<std::string>>();
  throw std::runtime_error("'stop' must be a string or array of strings");
}

void parse_common(const json& body, ChatRequest& r) {
  r.model = body.value("model", std::string("xllm"));
  r.max_tokens = body.value("max_tokens", 128);
  if (r.max_tokens <= 0) throw std::runtime_error("'max_tokens' must be positive");

  r.params.temperature = body.value("temperature", 1.0f);
  if (r.params.temperature < 0.0f) throw std::runtime_error("'temperature' must be >= 0");
  r.params.top_p = body.value("top_p", 1.0f);
  if (r.params.top_p <= 0.0f || r.params.top_p > 1.0f)
    throw std::runtime_error("'top_p' must be in (0, 1]");
  r.params.top_k = body.value("top_k", 0);
  if (body.contains("seed") && !body["seed"].is_null())
    r.params.seed = body["seed"].get<uint64_t>();

  r.stream = body.value("stream", false);
  r.n = body.value("n", 1);
  r.stop = parse_stop(body);
}
}  // namespace

ChatRequest parse_chat_request(const nlohmann::json& body) {
  ChatRequest r;
  r.is_chat = true;
  auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty())
    throw std::runtime_error("'messages' must be a non-empty array");
  for (const auto& m : *it) {
    if (!m.contains("role") || !m.contains("content"))
      throw std::runtime_error("each message needs 'role' and 'content'");
    r.messages.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
  }
  parse_common(body, r);
  return r;
}

ChatRequest parse_completion_request(const nlohmann::json& body) {
  ChatRequest r;
  r.is_chat = false;
  auto it = body.find("prompt");
  if (it == body.end() || !it->is_string())
    throw std::runtime_error("'prompt' must be a string");
  r.messages.push_back({"user", it->get<std::string>()});
  parse_common(body, r);
  return r;
}

nlohmann::json make_usage(int prompt_tokens, int completion_tokens) {
  return {{"prompt_tokens", prompt_tokens},
          {"completion_tokens", completion_tokens},
          {"total_tokens", prompt_tokens + completion_tokens}};
}

nlohmann::json make_chat_completion(const std::string& id, long created, const std::string& model,
                                    const std::string& content, const std::string& finish_reason,
                                    int prompt_tokens, int completion_tokens) {
  return {
      {"id", id},
      {"object", "chat.completion"},
      {"created", created},
      {"model", model},
      {"choices",
       json::array({{{"index", 0},
                     {"message", {{"role", "assistant"}, {"content", content}}},
                     {"finish_reason", finish_reason}}})},
      {"usage", make_usage(prompt_tokens, completion_tokens)},
  };
}

nlohmann::json make_chat_chunk(const std::string& id, long created, const std::string& model,
                               const nlohmann::json& delta, const nlohmann::json& finish_reason) {
  return {{"id", id},
          {"object", "chat.completion.chunk"},
          {"created", created},
          {"model", model},
          {"choices",
           json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish_reason}}})}};
}

std::string sse_frame(const nlohmann::json& payload) { return "data: " + payload.dump() + "\n\n"; }

const std::string kSseDone = "data: [DONE]\n\n";

nlohmann::json make_models_list(const std::string& model) {
  return {{"object", "list"},
          {"data", json::array({{{"id", model},
                                 {"object", "model"},
                                 {"created", 0},
                                 {"owned_by", "xllm"}}})}};
}

}  // namespace xllm
