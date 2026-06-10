#include "server/openai.h"

#include <sstream>
#include <stdexcept>

#include "server/base64.h"

namespace mlxforge {

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

// Each OpenAI tool is `{"type":"function","function":{name,description,parameters}}`;
// we render just the function schema (what the model was trained on), tolerating
// a bare schema (no "function" wrapper).
std::vector<std::string> parse_tools(const json& body) {
  auto it = body.find("tools");
  if (it == body.end() || it->is_null()) return {};
  if (!it->is_array()) throw std::runtime_error("'tools' must be an array");
  std::vector<std::string> out;
  for (const auto& t : *it) {
    const json& fn = t.contains("function") ? t["function"] : t;
    if (!fn.contains("name")) throw std::runtime_error("each tool needs a function 'name'");
    out.push_back(fn.dump(4));  // pretty-printed, matching the reference template
  }
  return out;
}

// tool_choice is a string ("auto"|"none"|"required") or an object naming a
// function; we collapse the object form to "required".
std::string parse_tool_choice(const json& body) {
  auto it = body.find("tool_choice");
  if (it == body.end() || it->is_null()) return "auto";
  if (it->is_string()) return it->get<std::string>();
  if (it->is_object()) return "required";
  throw std::runtime_error("'tool_choice' must be a string or object");
}

// Render an OpenAI assistant `tool_calls` array back into the single JSON object
// the model originally emitted: {"name": "fn", "parameters": {...}}. `arguments`
// arrives as a JSON string (per the OpenAI spec) but we tolerate an object too.
std::string render_assistant_tool_call(const json& tool_calls) {
  if (!tool_calls.is_array() || tool_calls.empty())
    throw std::runtime_error("'tool_calls' must be a non-empty array");
  const json& fn = tool_calls.front().at("function");
  json args = json::object();
  const json& raw = fn.at("arguments");
  if (raw.is_string()) {
    try {
      args = json::parse(raw.get<std::string>());
    } catch (const json::parse_error&) {
      args = json::object();
    }
  } else if (raw.is_object()) {
    args = raw;
  }
  return "{\"name\": " + json(fn.at("name").get<std::string>()).dump() +
         ", \"parameters\": " + args.dump() + "}";
}

// Decode an OpenAI image_url to raw bytes. Only base64 `data:` URIs are
// supported (an http(s) URL would require the server to fetch it).
std::string decode_image_url(const std::string& url) {
  const std::string marker = "base64,";
  const auto pos = url.find(marker);
  if (url.rfind("data:", 0) != 0 || pos == std::string::npos)
    throw std::runtime_error("only base64 'data:' image URLs are supported");
  return base64_decode(url.substr(pos + marker.size()));
}

// Parse an OpenAI `content` value (a string, or an array of {type:"text"} /
// {type:"image_url"} parts) into its text. Each image found is decoded and
// appended to `images_out`, in order.
std::string parse_content(const json& content, std::vector<std::string>& images_out) {
  if (content.is_string()) return content.get<std::string>();
  if (!content.is_array())
    throw std::runtime_error("'content' must be a string or an array of parts");
  std::string text;
  for (const auto& part : content) {
    if (!part.is_object()) continue;
    const std::string type = part.value("type", std::string());
    if (type == "text") {
      text += part.value("text", std::string());
    } else if (type == "image_url") {
      const auto iu = part.find("image_url");
      std::string url;
      if (iu != part.end() && iu->is_object()) url = iu->value("url", std::string());
      else if (iu != part.end() && iu->is_string()) url = iu->get<std::string>();
      if (!url.empty()) images_out.push_back(decode_image_url(url));
    }
  }
  return text;
}

// Parse one chat message into our Message, handling assistant tool_calls (content
// may be null/absent), array content with images, and tool-result messages. Any
// images are decoded and appended to `images_out`, in order.
Tokenizer::Message parse_message(const json& m, std::vector<std::string>& images_out) {
  if (!m.contains("role")) throw std::runtime_error("each message needs a 'role'");
  Tokenizer::Message msg;
  msg.role = m.at("role").get<std::string>();
  if (msg.role == "assistant" && m.contains("tool_calls") && !m["tool_calls"].is_null()) {
    msg.tool_call = render_assistant_tool_call(m["tool_calls"]);
    return msg;  // an assistant tool call carries no textual content
  }
  auto c = m.find("content");
  if (c == m.end() || c->is_null()) throw std::runtime_error("each message needs 'content'");
  msg.content = parse_content(*c, images_out);
  return msg;
}

void parse_common(const json& body, ChatRequest& r) {
  // Empty when omitted, so the server can tell "no model named" (serve the
  // loaded model) apart from an explicit name that must match it.
  r.model = body.value("model", std::string());
  r.max_tokens = body.value("max_tokens", 128);
  if (r.max_tokens <= 0) throw std::runtime_error("'max_tokens' must be positive");

  r.params.temperature = body.value("temperature", 1.0f);
  if (r.params.temperature < 0.0f) throw std::runtime_error("'temperature' must be >= 0");
  r.params.top_p = body.value("top_p", 1.0f);
  if (r.params.top_p <= 0.0f || r.params.top_p > 1.0f)
    throw std::runtime_error("'top_p' must be in (0, 1]");
  r.params.top_k = body.value("top_k", 0);
  r.params.min_p = body.value("min_p", 0.0f);
  if (r.params.min_p < 0.0f || r.params.min_p > 1.0f)
    throw std::runtime_error("'min_p' must be in [0, 1]");
  r.params.repetition_penalty = body.value("repetition_penalty", 1.0f);
  if (r.params.repetition_penalty <= 0.0f)
    throw std::runtime_error("'repetition_penalty' must be > 0");
  // OpenAI defines frequency/presence penalties over [-2, 2] (negatives encourage
  // repetition); we keep the same admissible range.
  r.params.frequency_penalty = body.value("frequency_penalty", 0.0f);
  r.params.presence_penalty = body.value("presence_penalty", 0.0f);
  if (r.params.frequency_penalty < -2.0f || r.params.frequency_penalty > 2.0f)
    throw std::runtime_error("'frequency_penalty' must be in [-2, 2]");
  if (r.params.presence_penalty < -2.0f || r.params.presence_penalty > 2.0f)
    throw std::runtime_error("'presence_penalty' must be in [-2, 2]");
  if (body.contains("seed") && !body["seed"].is_null())
    r.params.seed = body["seed"].get<uint64_t>();

  // OpenAI logprobs: `logprobs` (bool) turns on per-token reporting; with it,
  // `top_logprobs` (0–20) is the alternatives count. We carry both on the request
  // for serialization and let the HTTP layer fold them into params.top_logprobs.
  r.logprobs = body.value("logprobs", false);
  r.top_logprobs = body.value("top_logprobs", 0);
  if (r.top_logprobs < 0 || r.top_logprobs > 20)
    throw std::runtime_error("'top_logprobs' must be in [0, 20]");

  r.stream = body.value("stream", false);
  r.n = body.value("n", 1);
  r.stop = parse_stop(body);
  r.tools = parse_tools(body);
  r.tool_choice = parse_tool_choice(body);
  // Qwen3 reasoning toggle: accept a top-level `enable_thinking` or the
  // HF-conventional `chat_template_kwargs.enable_thinking`.
  r.enable_thinking = body.value("enable_thinking", true);
  if (auto kw = body.find("chat_template_kwargs"); kw != body.end() && kw->is_object())
    r.enable_thinking = kw->value("enable_thinking", r.enable_thinking);
}
}  // namespace

ChatRequest parse_chat_request(const nlohmann::json& body) {
  ChatRequest r;
  r.is_chat = true;
  auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty())
    throw std::runtime_error("'messages' must be a non-empty array");
  for (const auto& m : *it) {
    std::vector<std::string> imgs;  // this message's images, in order
    r.messages.push_back(parse_message(m, imgs));
    r.message_images.push_back(std::move(imgs));
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

EmbeddingsRequest parse_embeddings_request(const json& body) {
  EmbeddingsRequest r;
  r.model = body.value("model", "");
  if (body.contains("encoding_format") && body["encoding_format"].is_string())
    r.encoding_format = body["encoding_format"].get<std::string>();
  if (r.encoding_format != "float")
    throw std::runtime_error("only encoding_format 'float' is supported");

  auto it = body.find("input");
  if (it == body.end() || it->is_null()) throw std::runtime_error("'input' is required");
  if (it->is_string()) {
    r.input = {it->get<std::string>()};
  } else if (it->is_array()) {
    if (it->empty()) throw std::runtime_error("'input' array must not be empty");
    for (const auto& e : *it) {
      if (!e.is_string())
        throw std::runtime_error("'input' must be a string or array of strings");
      r.input.push_back(e.get<std::string>());
    }
  } else {
    throw std::runtime_error("'input' must be a string or array of strings");
  }
  return r;
}

nlohmann::json make_embeddings_response(const std::string& model,
                                        const std::vector<std::vector<float>>& embeddings,
                                        int prompt_tokens) {
  json data = json::array();
  for (size_t i = 0; i < embeddings.size(); ++i)
    data.push_back({{"object", "embedding"}, {"index", i}, {"embedding", embeddings[i]}});
  return {
      {"object", "list"},
      {"data", std::move(data)},
      {"model", model},
      // Embeddings produce no completion tokens; report prompt == total.
      {"usage", {{"prompt_tokens", prompt_tokens}, {"total_tokens", prompt_tokens}}},
  };
}

namespace {
// One {token, logprob, bytes} entry: decode `id` to its text and raw UTF-8 bytes.
json logprob_entry(int id, float logprob, const Tokenizer& tok) {
  const std::string text = tok.decode({id});
  json bytes = json::array();
  for (unsigned char c : text) bytes.push_back(static_cast<int>(c));
  return {{"token", text}, {"logprob", logprob}, {"bytes", std::move(bytes)}};
}
}  // namespace

nlohmann::json make_logprobs_content(const std::vector<TokenLogprob>& logprobs,
                                     const Tokenizer& tok) {
  json content = json::array();
  for (const TokenLogprob& lp : logprobs) {
    json entry = logprob_entry(lp.id, lp.logprob, tok);
    json top = json::array();
    for (const auto& alt : lp.top) top.push_back(logprob_entry(alt.first, alt.second, tok));
    entry["top_logprobs"] = std::move(top);
    content.push_back(std::move(entry));
  }
  return content;
}

nlohmann::json make_chat_completion(const std::string& id, long created, const std::string& model,
                                    const std::string& content, const std::string& finish_reason,
                                    int prompt_tokens, int completion_tokens,
                                    const nlohmann::json& logprobs_content) {
  json choice = {{"index", 0},
                 {"message", {{"role", "assistant"}, {"content", content}}},
                 {"finish_reason", finish_reason}};
  // null => logprobs disabled (omit the field); an array (even empty) => enabled.
  choice["logprobs"] =
      logprobs_content.is_null() ? json(nullptr) : json{{"content", logprobs_content}};
  return {
      {"id", id},
      {"object", "chat.completion"},
      {"created", created},
      {"model", model},
      {"choices", json::array({std::move(choice)})},
      {"usage", make_usage(prompt_tokens, completion_tokens)},
  };
}

namespace {
std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

// Fill `out` from a parsed JSON object and return true, or return false if it is
// not a call. The model emits "parameters"; we also accept "arguments".
bool object_to_call(const json& o, ToolCall& out) {
  if (!o.is_object() || !o.contains("name") || !o["name"].is_string()) return false;
  out.name = o["name"].get<std::string>();
  const json* args = o.contains("parameters") ? &o["parameters"]
                     : o.contains("arguments") ? &o["arguments"]
                                               : nullptr;
  out.arguments = (args && !args->is_null()) ? args->dump() : "{}";
  return true;
}
}  // namespace

std::vector<ToolCall> parse_tool_calls(const std::string& text) {
  std::string s = trim(text);
  // mlx-lm strips this special token, but guard against it surviving decode.
  constexpr const char* kPyTag = "<|python_tag|>";
  if (s.rfind(kPyTag, 0) == 0) s = trim(s.substr(std::char_traits<char>::length(kPyTag)));
  if (s.empty() || s.front() != '{') return {};

  std::vector<ToolCall> calls;
  // The common case: the whole output is a single JSON call object.
  try {
    ToolCall c;
    if (object_to_call(json::parse(s), c)) calls.push_back(c);
    return calls;
  } catch (const json::parse_error&) {
  }
  // Fallback: Llama emits parallel calls as ';'-separated objects.
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, ';')) {
    try {
      ToolCall c;
      if (object_to_call(json::parse(trim(part)), c)) calls.push_back(c);
    } catch (const json::parse_error&) {
    }
  }
  return calls;
}

nlohmann::json make_tool_calls(const std::vector<ToolCall>& calls) {
  json arr = json::array();
  for (size_t i = 0; i < calls.size(); ++i) {
    arr.push_back({{"id", "call_" + std::to_string(i)},
                   {"type", "function"},
                   {"function", {{"name", calls[i].name}, {"arguments", calls[i].arguments}}}});
  }
  return arr;
}

nlohmann::json make_chat_completion_tools(const std::string& id, long created,
                                          const std::string& model,
                                          const std::vector<ToolCall>& calls, int prompt_tokens,
                                          int completion_tokens) {
  return {
      {"id", id},
      {"object", "chat.completion"},
      {"created", created},
      {"model", model},
      {"choices", json::array({{{"index", 0},
                                {"message",
                                 {{"role", "assistant"},
                                  {"content", nullptr},
                                  {"tool_calls", make_tool_calls(calls)}}},
                                {"finish_reason", "tool_calls"}}})},
      {"usage", make_usage(prompt_tokens, completion_tokens)},
  };
}

nlohmann::json make_chat_chunk(const std::string& id, long created, const std::string& model,
                               const nlohmann::json& delta, const nlohmann::json& finish_reason,
                               const nlohmann::json& logprobs_content) {
  json choice = {{"index", 0}, {"delta", delta}, {"finish_reason", finish_reason}};
  // Attach logprobs for the tokens in this delta when present (null => omit).
  if (!logprobs_content.is_null()) choice["logprobs"] = json{{"content", logprobs_content}};
  return {{"id", id},
          {"object", "chat.completion.chunk"},
          {"created", created},
          {"model", model},
          {"choices", json::array({std::move(choice)})}};
}

std::string sse_frame(const nlohmann::json& payload) { return "data: " + payload.dump() + "\n\n"; }

const std::string kSseDone = "data: [DONE]\n\n";

nlohmann::json make_models_list(const std::string& model) {
  return {{"object", "list"},
          {"data", json::array({{{"id", model},
                                 {"object", "model"},
                                 {"created", 0},
                                 {"owned_by", "mlxforge"}}})}};
}

}  // namespace mlxforge
