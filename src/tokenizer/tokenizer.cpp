#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/logging.h"
#include "tokenizer/bpe.h"

namespace mlxforge {

// Returns the chat format for a given model_type string. Only Llama-3.2 is
// supported today; this seam is kept so new families can be mapped here later.
ChatFormat chat_format_from_model_type(const std::string& /*model_type*/) {
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
  t.bos_id_ = bos_id;
  t.chat_format_ = fmt;
  log::info("tokenizer: loaded vocab={} special_tokens={} bos_id={}", t.impl_->vocab_size(),
            t.impl_->special_ids().size(), bos_id);
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

// Llama-3.2 chat template. The <|begin_of_text|> BOS is added by the encoder, so
// it is omitted here; the system block carries the default knowledge/date preamble.
std::string render_llama3(const std::vector<Tokenizer::Message>& messages,
                          bool add_generation_prompt, const std::string& today_date) {
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

  for (const auto& m : rest) {
    os << "<|start_header_id|>" << m.role << "<|end_header_id|>\n\n"
       << m.content << "<|eot_id|>";
  }
  if (add_generation_prompt) os << "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return os.str();
}

}  // namespace

std::string Tokenizer::render_chat_template(const std::vector<Message>& messages,
                                            bool add_generation_prompt,
                                            const std::string& today_date, ChatFormat /*fmt*/) {
  return render_llama3(messages, add_generation_prompt, today_date);
}

std::vector<int> Tokenizer::apply_chat_template(const std::vector<Message>& messages,
                                                bool add_generation_prompt,
                                                const std::string& today_date) const {
  return encode(render_chat_template(messages, add_generation_prompt, today_date, chat_format_));
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
