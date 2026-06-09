#include "server/anthropic.h"

#include <stdexcept>

#include "server/base64.h"

namespace mlxforge {

namespace {
using nlohmann::json;

// Concatenate the text of a `content` value that is either a plain string or an
// array of {type:"text", text} blocks (other block types are ignored here).
std::string text_of(const json& content) {
  if (content.is_string()) return content.get<std::string>();
  std::string out;
  if (content.is_array()) {
    for (const auto& b : content) {
      if (b.is_object() && b.value("type", std::string()) == "text")
        out += b.value("text", std::string());
    }
  }
  return out;
}

// Render an Anthropic tool_use block {name, input} into the internal call form
// the chat template replays: {"name": "fn", "parameters": {...}}.
std::string render_tool_use(const json& block) {
  const json input = block.contains("input") ? block.at("input") : json::object();
  return "{\"name\": " + json(block.value("name", std::string())).dump() +
         ", \"parameters\": " + input.dump() + "}";
}

// Append the Tokenizer::Message(s) for one Anthropic chat message. A string
// content yields a single message; a block array is split so tool_use becomes an
// assistant tool-call turn and tool_result becomes a "tool" turn (matching the
// roles the chat template understands).
void append_message(const json& m, std::vector<Tokenizer::Message>& out,
                    std::vector<std::vector<std::string>>& message_images) {
  if (!m.contains("role")) throw std::runtime_error("each message needs a 'role'");
  const std::string role = m.at("role").get<std::string>();
  auto it = m.find("content");
  if (it == m.end() || it->is_null()) throw std::runtime_error("each message needs 'content'");

  // Every push to `out` gets a matching `message_images` entry (1:1).
  if (it->is_string()) {
    out.push_back({role, it->get<std::string>(), ""});
    message_images.push_back({});
    return;
  }
  if (!it->is_array()) throw std::runtime_error("'content' must be a string or array of blocks");

  // tool_result blocks (carried on a user turn) are fed back first, then any
  // free text; tool_use blocks (on an assistant turn) become tool-call turns;
  // base64 image blocks decode onto this turn (in order).
  std::string text;
  std::vector<json> tool_uses;
  std::vector<std::string> images;
  for (const auto& b : *it) {
    if (!b.is_object()) continue;
    const std::string type = b.value("type", std::string());
    if (type == "text") {
      text += b.value("text", std::string());
    } else if (type == "tool_use") {
      tool_uses.push_back(b);
    } else if (type == "tool_result") {
      out.push_back({"tool", text_of(b.value("content", json())), ""});
      message_images.push_back({});
    } else if (type == "image") {
      const auto src = b.find("source");
      if (src != b.end() && src->is_object() && src->value("type", std::string()) == "base64")
        images.push_back(base64_decode(src->value("data", std::string())));
    }
  }
  // The role turn carries this turn's text + images (emit it even for an
  // image-only turn so the image renders).
  if (!text.empty() || !images.empty()) {
    out.push_back({role, text, ""});
    message_images.push_back(std::move(images));
  }
  for (const auto& tu : tool_uses) {
    out.push_back({"assistant", "", render_tool_use(tu)});
    message_images.push_back({});
  }
}

// Convert an Anthropic tool definition {name, description, input_schema} into the
// internal function schema string the chat template injects (the OpenAI path
// uses `parameters`, so rename input_schema -> parameters).
std::vector<std::string> parse_tools(const json& body) {
  auto it = body.find("tools");
  if (it == body.end() || it->is_null()) return {};
  if (!it->is_array()) throw std::runtime_error("'tools' must be an array");
  std::vector<std::string> out;
  for (const auto& t : *it) {
    if (!t.contains("name")) throw std::runtime_error("each tool needs a 'name'");
    json fn = {{"name", t.at("name")}};
    if (t.contains("description")) fn["description"] = t.at("description");
    if (t.contains("input_schema")) fn["parameters"] = t.at("input_schema");
    out.push_back(fn.dump(4));  // pretty-printed, matching the reference template
  }
  return out;
}

// tool_choice is an object {type: "auto"|"any"|"tool"|"none", name?}; collapse it
// onto the internal vocabulary ("auto"|"none"|"required") used by the template.
std::string parse_tool_choice(const json& body) {
  auto it = body.find("tool_choice");
  if (it == body.end() || it->is_null()) return "auto";
  if (!it->is_object()) throw std::runtime_error("'tool_choice' must be an object");
  const std::string type = it->value("type", std::string("auto"));
  if (type == "none") return "none";
  if (type == "any" || type == "tool") return "required";
  return "auto";
}
}  // namespace

ChatRequest parse_messages_request(const nlohmann::json& body) {
  ChatRequest r;
  r.is_chat = true;
  r.model = body.value("model", std::string());

  // Anthropic requires max_tokens.
  if (!body.contains("max_tokens")) throw std::runtime_error("'max_tokens' is required");
  r.max_tokens = body.at("max_tokens").get<int>();
  if (r.max_tokens <= 0) throw std::runtime_error("'max_tokens' must be positive");

  // The top-level system prompt becomes the leading system message (the chat
  // template expects it first, especially for Qwen3).
  if (auto s = body.find("system"); s != body.end() && !s->is_null()) {
    const std::string sys = text_of(*s);
    if (!sys.empty()) {
      r.messages.push_back({"system", sys, ""});
      r.message_images.push_back({});  // keep aligned with messages
    }
  }

  auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty())
    throw std::runtime_error("'messages' must be a non-empty array");
  for (const auto& m : *it) append_message(m, r.messages, r.message_images);

  r.params.temperature = body.value("temperature", 1.0f);
  if (r.params.temperature < 0.0f) throw std::runtime_error("'temperature' must be >= 0");
  r.params.top_p = body.value("top_p", 1.0f);
  if (r.params.top_p <= 0.0f || r.params.top_p > 1.0f)
    throw std::runtime_error("'top_p' must be in (0, 1]");
  r.params.top_k = body.value("top_k", 0);
  if (body.contains("seed") && !body["seed"].is_null())
    r.params.seed = body["seed"].get<uint64_t>();

  r.stream = body.value("stream", false);
  if (auto stop = body.find("stop_sequences"); stop != body.end() && stop->is_array())
    r.stop = stop->get<std::vector<std::string>>();

  r.tools = parse_tools(body);
  r.tool_choice = parse_tool_choice(body);
  return r;
}

nlohmann::json make_tool_use_blocks(const std::vector<ToolCall>& calls) {
  json arr = json::array();
  for (size_t i = 0; i < calls.size(); ++i) {
    json input = json::object();
    try {
      input = json::parse(calls[i].arguments);
    } catch (const json::parse_error&) {
    }
    arr.push_back({{"type", "tool_use"},
                   {"id", "toolu_" + std::to_string(i)},
                   {"name", calls[i].name},
                   {"input", input}});
  }
  return arr;
}

nlohmann::json make_message_response(const std::string& id, const std::string& model,
                                     const nlohmann::json& content,
                                     const std::string& stop_reason, int input_tokens,
                                     int output_tokens) {
  return {
      {"id", id},
      {"type", "message"},
      {"role", "assistant"},
      {"model", model},
      {"content", content},
      {"stop_reason", stop_reason},
      {"stop_sequence", nullptr},
      {"usage", {{"input_tokens", input_tokens}, {"output_tokens", output_tokens}}},
  };
}

nlohmann::json anthropic_error_body(const std::string& type, const std::string& message) {
  return {{"type", "error"}, {"error", {{"type", type}, {"message", message}}}};
}

std::string sse_event(const std::string& name, const nlohmann::json& payload) {
  return "event: " + name + "\ndata: " + payload.dump() + "\n\n";
}

nlohmann::json make_message_start(const std::string& id, const std::string& model,
                                  int input_tokens) {
  return {{"type", "message_start"},
          {"message",
           {{"id", id},
            {"type", "message"},
            {"role", "assistant"},
            {"model", model},
            {"content", json::array()},
            {"stop_reason", nullptr},
            {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}}}}}};
}

nlohmann::json make_content_block_start(int index, const nlohmann::json& block) {
  return {{"type", "content_block_start"}, {"index", index}, {"content_block", block}};
}

nlohmann::json make_text_delta(int index, const std::string& text) {
  return {{"type", "content_block_delta"},
          {"index", index},
          {"delta", {{"type", "text_delta"}, {"text", text}}}};
}

nlohmann::json make_input_json_delta(int index, const std::string& partial_json) {
  return {{"type", "content_block_delta"},
          {"index", index},
          {"delta", {{"type", "input_json_delta"}, {"partial_json", partial_json}}}};
}

nlohmann::json make_content_block_stop(int index) {
  return {{"type", "content_block_stop"}, {"index", index}};
}

nlohmann::json make_message_delta(const std::string& stop_reason, int output_tokens) {
  return {{"type", "message_delta"},
          {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
          {"usage", {{"output_tokens", output_tokens}}}};
}

const nlohmann::json kMessageStop = {{"type", "message_stop"}};

}  // namespace mlxforge
