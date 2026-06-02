#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
#include <tokenizers_cpp.h>

#include "core/logging.h"

namespace mlxforge {

ChatFormat chat_format_from_model_type(const std::string& model_type) {
  return model_type == "mistral" ? ChatFormat::Mistral : ChatFormat::Llama3;
}

namespace {
std::string load_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Collect the ids of every special token declared in a tokenizer.json blob
// (added_tokens[*] with "special": true). These are skipped on decode, which
// generalizes across model families (Llama reserves high ids, Mistral low ones).
std::unordered_set<int> parse_special_ids(const std::string& blob) {
  std::unordered_set<int> ids;
  nlohmann::json j = nlohmann::json::parse(blob, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (auto it = j.find("added_tokens"); it != j.end() && it->is_array()) {
    for (const auto& tok : *it) {
      if (tok.value("special", false) && tok.contains("id")) ids.insert(tok["id"].get<int>());
    }
  }
  return ids;
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
  Tokenizer t;
  t.impl_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  t.bos_id_ = bos_id;
  t.chat_format_ = fmt;
  *t.special_ids_ = parse_special_ids(blob);
  log::info("tokenizer: loaded vocab={} special_tokens={} bos_id={}", t.impl_->GetVocabSize(),
            t.special_ids_->size(), bos_id);
  return t;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  // tokenizers-cpp's Encode does not run the BOS post-processor, so prepend the
  // configured begin-of-text id to match mlx-lm's tok.encode (none if < 0).
  std::lock_guard<std::mutex> lk(*mu_);
  std::vector<int32_t> ids = impl_->Encode(text);
  std::vector<int> out;
  out.reserve(ids.size() + 1);
  if (bos_id_ >= 0) out.push_back(bos_id_);
  out.insert(out.end(), ids.begin(), ids.end());
  log::debug("tokenizer: encoded {} chars -> {} tokens", text.size(), out.size());
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  // Drop the model's special tokens (parsed from tokenizer.json), matching
  // mlx-lm's skip_special_tokens default, since tokenizers-cpp Decode renders them.
  std::vector<int32_t> keep;
  keep.reserve(ids.size());
  for (int id : ids)
    if (!special_ids_->count(id)) keep.push_back(id);
  std::lock_guard<std::mutex> lk(*mu_);
  return impl_->Decode(keep);
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

// Mistral [INST] template. The <s> BOS is added by the encoder; </s> (eos) ends
// each assistant turn; the model generates right after the final [/INST], so
// add_generation_prompt needs no extra marker. The canonical template has no
// system role, so a leading system message is folded into the first user turn.
//
// The leading space matches the SentencePiece prefix metaspace that mlx-lm emits
// (it tokenizes to a standalone "▁" before the first [INST]); tokenizers-cpp's
// Encode does not add it on its own, so it is part of the rendered string here.
std::string render_mistral(const std::vector<Tokenizer::Message>& messages) {
  std::string system;
  std::ostringstream os;
  os << " ";
  bool first_user = true;
  for (const auto& m : messages) {
    if (m.role == "system") {
      system = m.content;
    } else if (m.role == "assistant") {
      os << m.content << "</s>";
    } else {  // user (default)
      std::string content = m.content;
      if (first_user && !system.empty()) {
        content = system + "\n\n" + content;
        system.clear();
      }
      first_user = false;
      os << "[INST] " << content << " [/INST]";
    }
  }
  return os.str();
}

}  // namespace

std::string Tokenizer::render_chat_template(const std::vector<Message>& messages,
                                            bool add_generation_prompt,
                                            const std::string& today_date, ChatFormat fmt) {
  if (fmt == ChatFormat::Mistral) return render_mistral(messages);
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
