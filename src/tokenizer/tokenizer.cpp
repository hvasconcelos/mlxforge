#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/logging.h"
#include "tokenizer/bpe.h"

namespace mlxforge {

// Returns the chat format for a given model_type string: Qwen3/Qwen2 render the
// ChatML template, everything else falls back to Llama-3.2.
ChatFormat chat_format_from_model_type(const std::string& model_type) {
  if (model_type == "qwen3" || model_type == "qwen2") return ChatFormat::Qwen3;
  return ChatFormat::Llama3;
}

namespace {
std::string load_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Today's date as "01 Jun 2026" (the format the Llama-3.2 template uses).
std::string current_date() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%d %b %Y", &tm);
  return buf;
}

// Resolve the BOS id to actually prepend on encode. Some families (Qwen3) carry
// a `bos_token_id` in config.json for metadata but their tokenizer does NOT
// prepend it: tokenizer_config.json sets `add_bos_token: false` (and a null
// `bos_token`). Honor that sibling file so encode matches the HF tokenizer; when
// it's absent or doesn't disable BOS (Llama-3.2), keep the provided id.
int effective_bos_id(const std::string& tokenizer_json_path, int bos_id) {
  const size_t slash = tokenizer_json_path.find_last_of('/');
  const std::string dir = slash == std::string::npos ? "" : tokenizer_json_path.substr(0, slash + 1);
  std::ifstream f(dir + "tokenizer_config.json", std::ios::binary);
  if (!f) return bos_id;
  std::ostringstream ss;
  ss << f.rdbuf();
  nlohmann::json j = nlohmann::json::parse(ss.str(), /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return bos_id;
  if (!j.value("add_bos_token", true)) return -1;            // explicit opt-out (Qwen3)
  if (j.contains("bos_token") && j["bos_token"].is_null()) return -1;  // no BOS token at all
  return bos_id;
}

// Length of the longest prefix of `s` that is complete UTF-8 (no trailing
// partial multi-byte sequence).
size_t utf8_complete_len(const std::string& s) {
  if (s.empty()) return 0;
  size_t start = s.size() - 1;
  while (start > 0 && (static_cast<uint8_t>(s[start]) & 0xC0) == 0x80) --start;
  const uint8_t lead = static_cast<uint8_t>(s[start]);
  size_t need;
  if (lead < 0x80)
    need = 1;  // 0xxxxxxx (ASCII)
  else if (lead < 0xE0)
    need = 2;  // 110xxxxx
  else if (lead < 0xF0)
    need = 3;  // 1110xxxx
  else
    need = 4;  // 11110xxx
  return (s.size() - start >= need) ? s.size() : start;
}
}  // namespace

Tokenizer Tokenizer::from_file(const std::string& tokenizer_json_path, int bos_id, ChatFormat fmt) {
  const std::string blob = load_file(tokenizer_json_path);
  if (!BpeTokenizer::is_supported(blob))
    throw std::runtime_error("tokenizer: '" + tokenizer_json_path +
                             "' is not a byte-level BPE tokenizer (only Llama-3.2-style is "
                             "supported)");
  Tokenizer t;
  t.impl_ = std::make_shared<BpeTokenizer>(BpeTokenizer::from_blob(blob));
  t.bos_id_ = effective_bos_id(tokenizer_json_path, bos_id);
  t.chat_format_ = fmt;
  log::info("tokenizer: loaded vocab={} special_tokens={} bos_id={}", t.impl_->vocab_size(),
            t.impl_->special_ids().size(), t.bos_id_);
  return t;
}

Tokenizer Tokenizer::from_gguf(const std::vector<std::string>& tokens,
                               const std::vector<std::string>& merges,
                               const std::vector<int>& token_types, const std::string& pre,
                               int bos_id, ChatFormat fmt) {
  Tokenizer t;
  t.impl_ = std::make_shared<BpeTokenizer>(
      BpeTokenizer::from_gguf(tokens, merges, token_types, pre));
  t.bos_id_ = bos_id;
  t.chat_format_ = fmt;
  log::info("tokenizer: loaded from gguf vocab={} special_tokens={} bos_id={}",
            t.impl_->vocab_size(), t.impl_->special_ids().size(), bos_id);
  return t;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  // BpeTokenizer::encode does not run the BOS post-processor, so prepend the
  // configured begin-of-text id to match mlx-lm's tok.encode (none if < 0).
  std::vector<int> ids = impl_->encode(text);
  std::vector<int> out;
  out.reserve(ids.size() + 1);
  if (bos_id_ >= 0) out.push_back(bos_id_);
  out.insert(out.end(), ids.begin(), ids.end());
  log::debug("tokenizer: encoded {} chars -> {} tokens", text.size(), out.size());
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  // Drop the model's special tokens (parsed from tokenizer.json), matching
  // mlx-lm's skip_special_tokens default. BpeTokenizer::decode also skips them;
  // this pre-filter keeps behavior explicit and independent of that.
  const auto& special_ids = impl_->special_ids();
  std::vector<int> keep;
  keep.reserve(ids.size());
  for (int id : ids)
    if (!special_ids.count(id)) keep.push_back(id);
  return impl_->decode(keep);
}

namespace {

// The preamble Llama-3.2 expects ahead of the tool schemas in the first user
// turn (mirrors mlx-lm's `tools_in_user_message` rendering).
constexpr const char* kToolPreamble =
    "Given the following functions, please respond with a JSON for a function call "
    "with its proper arguments that best answers the given prompt.\n\n"
    "Respond in the format {\"name\": function name, \"parameters\": dictionary of "
    "argument name and its value}. Do not use variables.\n\n";

// Llama-3.2 chat template. The <|begin_of_text|> BOS is added by the encoder, so
// it is omitted here; the system block carries the default knowledge/date preamble.
// `tools` (each a function's JSON schema) is injected into the first user turn.
std::string render_llama3(const std::vector<Tokenizer::Message>& messages,
                          bool add_generation_prompt, const std::string& today_date,
                          const std::vector<std::string>& tools) {
  const std::string date = today_date.empty() ? current_date() : today_date;
  std::string sys_content;
  std::vector<Tokenizer::Message> rest;
  for (const auto& m : messages) {
    if (m.role == "system" && sys_content.empty())
      sys_content = m.content;
    else
      rest.push_back(m);
  }

  std::ostringstream os;
  os << "<|start_header_id|>system<|end_header_id|>\n\n"
     << "Cutting Knowledge Date: December 2023\nToday Date: " << date << "\n\n";
  if (!sys_content.empty()) os << sys_content;
  os << "<|eot_id|>";

  bool tools_injected = false;
  for (const auto& m : rest) {
    // A tool result is fed back under the model's "ipython" role.
    if (m.role == "tool") {
      os << "<|start_header_id|>ipython<|end_header_id|>\n\n" << m.content << "<|eot_id|>";
      continue;
    }
    // Replay a prior assistant tool call as the raw JSON the model emitted.
    if (m.role == "assistant" && !m.tool_call.empty()) {
      os << "<|start_header_id|>assistant<|end_header_id|>\n\n" << m.tool_call << "<|eot_id|>";
      continue;
    }
    os << "<|start_header_id|>" << m.role << "<|end_header_id|>\n\n";
    // Tool schemas precede the content of the first user turn.
    if (m.role == "user" && !tools.empty() && !tools_injected) {
      tools_injected = true;
      os << kToolPreamble;
      for (const auto& t : tools) os << t << "\n\n";
    }
    os << m.content << "<|eot_id|>";
  }
  if (add_generation_prompt) os << "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return os.str();
}

// Qwen3 ChatML template. No BOS token is emitted. Tools are injected Hermes-style
// into the leading system turn; a prior assistant tool call and tool results use
// <tool_call>/<tool_response> blocks. `enable_thinking == false` appends an empty
// <think></think> block after the assistant header to suppress reasoning. Mirrors
// Qwen3's tokenizer_config.json chat_template for the system/user/assistant/tool
// cases (historical assistant <think> stripping is not reproduced — prompts for
// fresh generation, the common case, match byte-for-byte; see the golden test).
std::string render_qwen3(const std::vector<Tokenizer::Message>& messages,
                         bool add_generation_prompt, bool enable_thinking,
                         const std::vector<std::string>& tools) {
  std::ostringstream os;

  // Leading system turn: the user's system message (if the first message is one)
  // plus, when tools are present, the Hermes tool-listing preamble.
  const bool have_system = !messages.empty() && messages.front().role == "system";
  if (!tools.empty()) {
    os << "<|im_start|>system\n";
    if (have_system) os << messages.front().content << "\n\n";
    os << "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
          "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    for (const auto& tool : tools) os << "\n" << tool;
    os << "\n</tools>\n\nFor each function call, return a json object with function name and "
          "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
          "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
  } else if (have_system) {
    os << "<|im_start|>system\n" << messages.front().content << "<|im_end|>\n";
  }

  // Remaining turns. Consecutive tool results are grouped under one user turn.
  for (size_t i = (have_system ? 1 : 0); i < messages.size(); ++i) {
    const auto& m = messages[i];
    if (m.role == "tool") {
      const bool first_tool = (i == 0) || messages[i - 1].role != "tool";
      const bool last_tool = (i + 1 == messages.size()) || messages[i + 1].role != "tool";
      if (first_tool) os << "<|im_start|>user";
      os << "\n<tool_response>\n" << m.content << "\n</tool_response>";
      if (last_tool) os << "<|im_end|>\n";
      continue;
    }
    if (m.role == "assistant") {
      os << "<|im_start|>assistant\n" << m.content;
      if (!m.tool_call.empty()) os << "\n<tool_call>\n" << m.tool_call << "\n</tool_call>";
      os << "<|im_end|>\n";
      continue;
    }
    os << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
  }

  if (add_generation_prompt) {
    os << "<|im_start|>assistant\n";
    if (!enable_thinking) os << "<think>\n\n</think>\n\n";
  }
  return os.str();
}

}  // namespace

std::string Tokenizer::render_chat_template(const std::vector<Message>& messages,
                                            bool add_generation_prompt,
                                            const std::string& today_date, ChatFormat fmt,
                                            const std::vector<std::string>& tools,
                                            bool enable_thinking) {
  if (fmt == ChatFormat::Qwen3)
    return render_qwen3(messages, add_generation_prompt, enable_thinking, tools);
  return render_llama3(messages, add_generation_prompt, today_date, tools);
}

std::vector<int> Tokenizer::apply_chat_template(const std::vector<Message>& messages,
                                                bool add_generation_prompt,
                                                const std::string& today_date,
                                                const std::vector<std::string>& tools,
                                                bool enable_thinking) const {
  return encode(render_chat_template(messages, add_generation_prompt, today_date, chat_format_,
                                     tools, enable_thinking));
}

std::string StreamingDetokenizer::add(int id) {
  ids_.push_back(id);
  std::string full = tok_.decode(ids_);
  const size_t complete = utf8_complete_len(full);
  if (complete <= emitted_) return "";
  std::string out = full.substr(emitted_, complete - emitted_);
  emitted_ = complete;
  return out;
}

std::string StreamingDetokenizer::finish() {
  std::string full = tok_.decode(ids_);
  std::string out = full.substr(emitted_);
  emitted_ = full.size();
  return out;
}

}  // namespace mlxforge
