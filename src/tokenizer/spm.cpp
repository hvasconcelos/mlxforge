#include "tokenizer/spm.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/logging.h"

namespace mlxforge {
namespace {

// The SentencePiece metaspace marker U+2581 ("▁"), which stands in for a space.
constexpr const char* kMetaspace = "\xE2\x96\x81";

// Split a UTF-8 string into its characters (each a UTF-8 substring). Lengths come
// from the leading byte; a malformed lead byte is taken as a single byte so the
// raw bytes are always preserved.
std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    size_t len = 1;
    if (c >= 0xF0 && i + 3 < s.size()) {
      len = 4;
    } else if (c >= 0xE0 && i + 2 < s.size()) {
      len = 3;
    } else if (c >= 0xC0 && i + 1 < s.size()) {
      len = 2;
    }
    out.emplace_back(s, i, len);
    i += len;
  }
  return out;
}

// Uppercase two-hex-digit "<0xNN>" name for a byte, matching the SentencePiece
// byte-fallback vocab tokens.
std::string byte_token_name(int b) {
  static const char* hex = "0123456789ABCDEF";
  std::string s = "<0x";
  s.push_back(hex[(b >> 4) & 0xF]);
  s.push_back(hex[b & 0xF]);
  s.push_back('>');
  return s;
}

}  // namespace

void SpmBpeTokenizer::emit_symbol(const std::string& sym, std::vector<int>& out) const {
  auto it = token_to_id_.find(sym);
  if (it != token_to_id_.end()) {
    out.push_back(it->second);
    return;
  }
  // byte_fallback: a symbol absent from the vocab is emitted as its raw UTF-8
  // bytes via the "<0xNN>" byte tokens (fuse_unk: a byte with no token -> unk).
  for (unsigned char b : sym) {
    const int id = byte_fallback_[b];
    if (id >= 0)
      out.push_back(id);
    else if (unk_id_ >= 0)
      out.push_back(unk_id_);
  }
}

void SpmBpeTokenizer::encode_plain(const std::string& segment, std::vector<int>& out) const {
  if (segment.empty()) return;

  // Normalize: every space becomes the metaspace marker. (The vestigial space
  // pre-tokenizer then finds nothing to split, so the whole segment is one BPE
  // word — matching the HF pipeline for Gemma-style tokenizers.)
  std::string normalized;
  normalized.reserve(segment.size());
  for (char ch : segment) {
    if (ch == ' ')
      normalized += kMetaspace;
    else
      normalized.push_back(ch);
  }

  std::vector<std::string> symbols = utf8_chars(normalized);

  // Repeatedly merge the lowest-rank adjacent pair (merging all of its
  // occurrences left-to-right each pass), HF reference behavior.
  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    bool found = false;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        found = true;
      }
    }
    if (!found) break;

    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size()) {
        auto it = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
        if (it != merge_ranks_.end() && it->second == best_rank) {
          merged.push_back(symbols[i] + symbols[i + 1]);
          i += 2;
          continue;
        }
      }
      merged.push_back(symbols[i]);
      ++i;
    }
    symbols = std::move(merged);
  }

  for (const std::string& sym : symbols) emit_symbol(sym, out);
}

std::vector<int> SpmBpeTokenizer::encode(const std::string& text) const {
  std::vector<int> out;
  // Segment on special-token literals (longest-match first); the gaps between
  // them are plain text run through the SPM-BPE pipeline.
  size_t run_start = 0;
  size_t i = 0;
  while (i < text.size()) {
    bool matched = false;
    if (special_first_bytes_.count(static_cast<unsigned char>(text[i]))) {
      for (const auto& [lit, id] : special_tokens_) {  // sorted longest-first
        if (i + lit.size() <= text.size() && text.compare(i, lit.size(), lit) == 0) {
          encode_plain(text.substr(run_start, i - run_start), out);
          out.push_back(id);
          i += lit.size();
          run_start = i;
          matched = true;
          break;
        }
      }
    }
    if (!matched) ++i;
  }
  encode_plain(text.substr(run_start), out);
  return out;
}

std::string SpmBpeTokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  std::string pending_bytes;  // run of byte-fallback bytes, fused into UTF-8
  for (int id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
    if (special_ids_.count(id)) continue;

    auto bv = byte_token_value_.find(id);
    if (bv != byte_token_value_.end()) {
      pending_bytes.push_back(static_cast<char>(bv->second));
      continue;
    }
    out += pending_bytes;  // flush fused bytes before the next text token
    pending_bytes.clear();

    const std::string& tok = id_to_token_[id];
    // Metaspace -> space.
    for (size_t p = 0; p < tok.size();) {
      if (tok.compare(p, 3, kMetaspace) == 0) {
        out.push_back(' ');
        p += 3;
      } else {
        out.push_back(tok[p]);
        ++p;
      }
    }
  }
  out += pending_bytes;
  return out;
}

bool SpmBpeTokenizer::is_supported(const std::string& tokenizer_json) {
  nlohmann::json j = nlohmann::json::parse(tokenizer_json, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return false;
  auto model = j.find("model");
  if (model == j.end() || model->value("type", std::string()) != "BPE") return false;
  // byte_fallback (with a ByteFallback decoder) is what distinguishes a
  // SentencePiece-style BPE from the byte-level BPE handled by BpeTokenizer.
  if (!model->value("byte_fallback", false)) return false;
  auto dec = j.find("decoder");
  return dec != j.end() && dec->dump().find("ByteFallback") != std::string::npos;
}

SpmBpeTokenizer SpmBpeTokenizer::from_blob(const std::string& tokenizer_json) {
  nlohmann::json j = nlohmann::json::parse(tokenizer_json, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) throw std::runtime_error("spm: tokenizer.json is not valid JSON");

  auto model = j.find("model");
  if (model == j.end() || model->value("type", std::string()) != "BPE")
    throw std::runtime_error("spm: only SentencePiece-style BPE tokenizers are supported");

  SpmBpeTokenizer t;
  t.byte_fallback_.fill(-1);
  int max_id = -1;

  const auto& vocab = (*model)["vocab"];
  if (!vocab.is_object()) throw std::runtime_error("spm: model.vocab missing");
  t.token_to_id_.reserve(vocab.size() + 256);
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const int id = it.value().get<int>();
    t.token_to_id_.emplace(it.key(), id);
    max_id = std::max(max_id, id);
  }

  const auto& merges = (*model)["merges"];
  if (!merges.is_array()) throw std::runtime_error("spm: model.merges missing");
  t.merge_ranks_.reserve(merges.size());
  int rank = 0;
  for (const auto& m : merges) {
    std::string key;
    if (m.is_string()) {
      key = m.get<std::string>();  // "L R"
    } else if (m.is_array() && m.size() == 2) {
      key = m[0].get<std::string>() + " " + m[1].get<std::string>();  // ["L","R"]
    } else {
      throw std::runtime_error("spm: unexpected merges entry shape");
    }
    t.merge_ranks_.emplace(std::move(key), rank++);
  }

  // The "<0xNN>" byte-fallback tokens (in the vocab) and the unk id.
  for (int b = 0; b < 256; ++b) {
    auto it = t.token_to_id_.find(byte_token_name(b));
    if (it != t.token_to_id_.end()) {
      t.byte_fallback_[b] = it->second;
      t.byte_token_value_[it->second] = b;
    }
  }
  if (auto u = model->find("unk_token"); u != model->end() && u->is_string()) {
    auto it = t.token_to_id_.find(u->get<std::string>());
    if (it != t.token_to_id_.end()) t.unk_id_ = it->second;
  }

  // Added tokens: HF isolates EVERY added token atomically before BPE, whether or
  // not it is `special`. So segment on all of them; the `special` flag only
  // governs decode-skipping (special_ids_).
  if (auto at = j.find("added_tokens"); at != j.end() && at->is_array()) {
    for (const auto& tok : *at) {
      if (!tok.contains("id") || !tok.contains("content")) continue;
      const int id = tok["id"].get<int>();
      const std::string content = tok["content"].get<std::string>();
      max_id = std::max(max_id, id);
      t.token_to_id_[content] = id;
      t.special_tokens_.emplace_back(content, id);
      if (!content.empty()) t.special_first_bytes_.insert(static_cast<unsigned char>(content[0]));
      if (tok.value("special", false)) t.special_ids_.insert(id);
    }
  }

  // Longest literal first so e.g. "<start_of_turn>" wins over any shorter prefix.
  std::sort(t.special_tokens_.begin(), t.special_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  t.id_to_token_.assign(static_cast<size_t>(max_id + 1), std::string());
  for (const auto& [tok, id] : t.token_to_id_) t.id_to_token_[id] = tok;

  log::info("spm: loaded vocab={} merges={} byte_tokens={} special_tokens={}",
            t.token_to_id_.size(), t.merge_ranks_.size(), t.byte_token_value_.size(),
            t.special_ids_.size());
  return t;
}

}  // namespace mlxforge
