#include "tokenizer/bpe.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/logging.h"
#include "tokenizer/unicode_tables.h"

namespace mlxforge {
namespace {

// --- UTF-8 <-> codepoint helpers --------------------------------------------

// Append the UTF-8 encoding of `cp` to `out`.
void append_utf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Decode `s` into codepoints `cp` and byte offsets `off` (off has size
// cp.size()+1, with off.back() == s.size()). Invalid bytes are taken as a
// single codepoint equal to the byte value; pieces are later sliced from the
// original bytes by offset, so the raw bytes are preserved regardless.
void decode_utf8(const std::string& s, std::vector<uint32_t>& cp, std::vector<size_t>& off) {
  cp.clear();
  off.clear();
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    uint32_t value;
    size_t len;
    if (c < 0x80) {
      value = c;
      len = 1;
    } else if ((c >> 5) == 0x6 && i + 1 < n) {
      value = ((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F);
      len = 2;
    } else if ((c >> 4) == 0xE && i + 2 < n) {
      value = ((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
              (static_cast<uint8_t>(s[i + 2]) & 0x3F);
      len = 3;
    } else if ((c >> 3) == 0x1E && i + 3 < n) {
      value = ((c & 0x07) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
              ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) | (static_cast<uint8_t>(s[i + 3]) & 0x3F);
      len = 4;
    } else {
      value = c;  // invalid lead / truncated: fall back to one raw byte
      len = 1;
    }
    off.push_back(i);
    cp.push_back(value);
    i += len;
  }
  off.push_back(n);
}

// --- GPT-2 byte<->unicode alphabet ------------------------------------------
//
// The reversible byte<->unicode map used by HF ByteLevel: printable ASCII and a
// few Latin-1 ranges map to themselves; every other byte maps to a codepoint in
// U+0100+. The vocab is expressed in this alphabet (e.g. a space byte is 'Ġ').
struct ByteMap {
  std::array<std::string, 256> to_uni;          // byte -> UTF-8 of its codepoint
  std::array<int, 0x200> from_uni_dense;         // codepoint (<0x200) -> byte, or -1
  std::unordered_map<uint32_t, uint8_t> from_uni;  // any codepoint -> byte

  ByteMap() {
    from_uni_dense.fill(-1);
    std::array<bool, 256> printable{};
    auto mark = [&](int lo, int hi) {
      for (int b = lo; b <= hi; ++b) printable[b] = true;
    };
    mark('!', '~');      // 0x21..0x7E
    mark(0xA1, 0xAC);    // ¡..¬
    mark(0xAE, 0xFF);    // ®..ÿ
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      uint32_t cp = printable[b] ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
      std::string u;
      append_utf8(cp, u);
      to_uni[b] = u;
      from_uni[cp] = static_cast<uint8_t>(b);
      if (cp < 0x200) from_uni_dense[cp] = b;
    }
  }
};

const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

constexpr uint32_t kCR = 0x0D;  // '\r'
constexpr uint32_t kLF = 0x0A;  // '\n'
constexpr uint32_t kSpace = 0x20;
constexpr uint32_t kApostrophe = 0x27;

bool is_letter(uint32_t c) { return unicode::is_letter(c); }
bool is_number(uint32_t c) { return unicode::is_number(c); }
bool is_space(uint32_t c) { return unicode::is_whitespace(c); }
bool is_newline(uint32_t c) { return c == kCR || c == kLF; }

// ASCII-lowercase a codepoint (the contraction alternative is (?i:...), and all
// its letters are ASCII).
uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Match one pre-token at codepoint index `p`, returning its length in
// codepoints (always >= 1). Replicates, alternative-by-alternative and in
// order, the Llama-3.2 / Qwen Split regex (they differ only in the digit run
// cap `\p{N}{1,digit_run_max}` — 3 for Llama, 1 for Qwen):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}{1,digit_run_max}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
size_t match_piece(const std::vector<uint32_t>& cp, size_t p, int digit_run_max) {
  const size_t n = cp.size();

  // Alt 1: contractions, case-insensitive, tried in listed order.
  if (cp[p] == kApostrophe) {
    auto at = [&](size_t k, uint32_t want) { return p + k < n && ascii_lower(cp[p + k]) == want; };
    if (at(1, 's')) return 2;
    if (at(1, 't')) return 2;
    if (at(1, 'r') && at(2, 'e')) return 3;
    if (at(1, 'v') && at(2, 'e')) return 3;
    if (at(1, 'm')) return 2;
    if (at(1, 'l') && at(2, 'l')) return 3;
    if (at(1, 'd')) return 2;
  }

  // Alt 2: [^\r\n\p{L}\p{N}]? \p{L}+  (optional non-letter/number lead, then letters).
  {
    size_t q = p;
    if (!is_newline(cp[q]) && !is_letter(cp[q]) && !is_number(cp[q])) ++q;  // optional lead
    if (q < n && is_letter(cp[q])) {
      size_t e = q;
      while (e < n && is_letter(cp[e])) ++e;
      return e - p;
    }
    // No letter followed: this alternative fails (backtracking the optional
    // lead would still need a letter at p, but cp[p] is not one here).
  }

  // Alt 3: \p{N}{1,digit_run_max}
  if (is_number(cp[p])) {
    size_t e = p;
    while (e < n && e - p < static_cast<size_t>(digit_run_max) && is_number(cp[e])) ++e;
    return e - p;
  }

  // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*  (optional space, then symbols, then newlines).
  {
    size_t q = p;
    if (cp[q] == kSpace) ++q;  // optional leading space
    auto is_symbol = [](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };
    if (q < n && is_symbol(cp[q])) {
      size_t e = q;
      while (e < n && is_symbol(cp[e])) ++e;
      while (e < n && is_newline(cp[e])) ++e;  // trailing [\r\n]*
      return e - p;
    }
  }

  // Determine the maximal whitespace run [p, ws_end) for the remaining alts.
  size_t ws_end = p;
  while (ws_end < n && is_space(cp[ws_end])) ++ws_end;

  // Alt 5: \s*[\r\n]+ — whitespace ending at the last newline in the run.
  if (ws_end > p) {
    size_t last_nl = ws_end;  // exclusive sentinel
    for (size_t k = p; k < ws_end; ++k)
      if (is_newline(cp[k])) last_nl = k;
    if (last_nl != ws_end) return (last_nl + 1) - p;
  }

  // Alt 6: \s+(?!\S) — the whitespace run, giving back one char if a non-space
  // follows (the lookahead). Fails for a lone space followed by a non-space.
  if (ws_end > p) {
    if (ws_end == n) return ws_end - p;          // not followed by anything
    if (ws_end - p >= 2) return (ws_end - 1) - p;  // give back one trailing space
    // single whitespace followed by \S -> fall through to alt 7
  }

  // Alt 7: \s+
  if (ws_end > p) return ws_end - p;

  // No alternative matched (unreachable for valid input): emit one codepoint.
  return 1;
}

}  // namespace

void BpeTokenizer::bpe_piece(const std::string& piece, std::vector<int>& out) const {
  // ignore_merges: a whole pre-token that is itself a vocab entry is emitted
  // directly, without running the merge loop (Llama-3.2 sets this).
  if (auto it = token_to_id_.find(piece); it != token_to_id_.end()) {
    out.push_back(it->second);
    return;
  }

  // Split the piece into single-codepoint symbols (each a UTF-8 substring).
  std::vector<std::string> symbols;
  for (size_t i = 0; i < piece.size();) {
    const uint8_t c = static_cast<uint8_t>(piece[i]);
    size_t len = 1;
    if (c >= 0xF0)
      len = 4;
    else if (c >= 0xE0)
      len = 3;
    else if (c >= 0xC0)
      len = 2;
    symbols.emplace_back(piece, i, len);
    i += len;
  }

  // Repeatedly merge the lowest-rank adjacent pair (merging all of its
  // occurrences left-to-right each pass), GPT-2 / HF reference behavior.
  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        found = true;
      }
    }
    if (!found) break;

    const std::string& a = symbols[best_i];
    const std::string& b = symbols[best_i + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == a && symbols[i + 1] == b) {
        merged.push_back(a + b);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        ++i;
      }
    }
    symbols = std::move(merged);
  }

  for (const std::string& sym : symbols) {
    auto it = token_to_id_.find(sym);
    if (it != token_to_id_.end())
      out.push_back(it->second);
    else
      log::error("bpe: symbol not in vocab (len={})", sym.size());  // should never happen
  }
}

void BpeTokenizer::encode_plain(const std::string& segment, std::vector<int>& out) const {
  if (segment.empty()) return;
  std::vector<uint32_t> cp;
  std::vector<size_t> off;
  decode_utf8(segment, cp, off);

  const ByteMap& bm = byte_map();
  for (size_t p = 0; p < cp.size();) {
    const size_t len = match_piece(cp, p, digit_run_max_);
    // Map the piece's original bytes [off[p], off[p+len]) into the merge alphabet.
    std::string mapped;
    for (size_t b = off[p]; b < off[p + len]; ++b)
      mapped += bm.to_uni[static_cast<uint8_t>(segment[b])];
    bpe_piece(mapped, out);
    p += len;
  }
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
  std::vector<int> out;
  // Segment on special-token literals (longest-match first); the gaps between
  // them are plain text run through the BPE pipeline.
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

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
  const ByteMap& bm = byte_map();
  std::string out;
  std::vector<uint32_t> cp;
  std::vector<size_t> off;
  for (int id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
    if (special_ids_.count(id)) continue;
    const std::string& tok = id_to_token_[id];
    if (tok.empty()) continue;
    decode_utf8(tok, cp, off);
    for (uint32_t c : cp) {
      int b = (c < 0x200) ? bm.from_uni_dense[c] : -1;
      if (b < 0) {
        auto it = bm.from_uni.find(c);
        if (it != bm.from_uni.end()) b = it->second;
      }
      if (b >= 0) out.push_back(static_cast<char>(b));
    }
  }
  return out;
}

bool BpeTokenizer::is_supported(const std::string& tokenizer_json) {
  nlohmann::json j = nlohmann::json::parse(tokenizer_json, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return false;
  auto model = j.find("model");
  if (model == j.end() || model->value("type", std::string()) != "BPE") return false;
  // The byte-level pipeline (and our hand-rolled splitter) is correct only for a
  // ByteLevel decoder; Metaspace/SentencePiece models are not.
  auto dec = j.find("decoder");
  return dec != j.end() && dec->value("type", std::string()) == "ByteLevel";
}

BpeTokenizer BpeTokenizer::from_blob(const std::string& tokenizer_json) {
  nlohmann::json j = nlohmann::json::parse(tokenizer_json, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) throw std::runtime_error("bpe: tokenizer.json is not valid JSON");

  auto model = j.find("model");
  if (model == j.end() || model->value("type", std::string()) != "BPE")
    throw std::runtime_error("bpe: only byte-level BPE tokenizers are supported");

  BpeTokenizer t;
  int max_id = -1;

  // The number alternative of the pre-tokenizer split caps digit runs: Llama-3
  // uses `\p{N}{1,3}`, Qwen and most tiktoken BPEs use `\p{N}` (one digit). Detect
  // the cap from the pre_tokenizer subtree; default to 1 (single-digit).
  if (auto pt = j.find("pre_tokenizer");
      pt != j.end() && pt->dump().find("{1,3}") != std::string::npos)
    t.digit_run_max_ = 3;

  const auto& vocab = (*model)["vocab"];
  if (!vocab.is_object()) throw std::runtime_error("bpe: model.vocab missing");
  t.token_to_id_.reserve(vocab.size() + 256);
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const int id = it.value().get<int>();
    t.token_to_id_.emplace(it.key(), id);
    max_id = std::max(max_id, id);
  }

  const auto& merges = (*model)["merges"];
  if (!merges.is_array()) throw std::runtime_error("bpe: model.merges missing");
  t.merge_ranks_.reserve(merges.size());
  int rank = 0;
  for (const auto& m : merges) {
    std::string key;
    if (m.is_string()) {
      key = m.get<std::string>();  // "L R"
    } else if (m.is_array() && m.size() == 2) {
      key = m[0].get<std::string>() + " " + m[1].get<std::string>();  // ["L","R"]
    } else {
      throw std::runtime_error("bpe: unexpected merges entry shape");
    }
    t.merge_ranks_.emplace(std::move(key), rank++);
  }

  // Added tokens: HF isolates EVERY added token atomically before BPE, whether or
  // not it is `special` (e.g. Qwen3's `<think>`/`</think>` are non-special added
  // tokens but still single ids). So segment on all of them; the `special` flag
  // only governs decode-skipping (special_ids_). Mirrors tokenizer.cpp's decode.
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

  // Longest literal first so e.g. "<|eot_id|>" wins over any shorter prefix.
  std::sort(t.special_tokens_.begin(), t.special_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  t.id_to_token_.assign(static_cast<size_t>(max_id + 1), std::string());
  for (const auto& [tok, id] : t.token_to_id_) t.id_to_token_[id] = tok;

  log::info("bpe: loaded vocab={} merges={} special_tokens={}", t.token_to_id_.size(),
            t.merge_ranks_.size(), t.special_ids_.size());
  return t;
}

BpeTokenizer BpeTokenizer::from_gguf(const std::vector<std::string>& tokens,
                                     const std::vector<std::string>& merges,
                                     const std::vector<int>& token_types,
                                     const std::string& pre) {
  // The hand-rolled pre-tokenizer regex (match_piece) and byte-level pipeline
  // are correct only for the Llama-3 / Qwen byte-level BPE variants.
  if (pre != "llama-bpe" && pre != "llama3" && pre != "gpt2" && pre != "qwen2") {
    throw std::runtime_error("bpe: unsupported gguf pretokenizer '" + pre +
                             "' (expected llama-bpe/llama3/gpt2/qwen2)");
  }
  if (tokens.empty()) throw std::runtime_error("bpe: gguf has no tokenizer.ggml.tokens");

  BpeTokenizer t;
  // Llama-3 caps digit runs at 3 (`\p{N}{1,3}`); qwen2/gpt2 emit one digit each.
  t.digit_run_max_ = (pre == "llama-bpe" || pre == "llama3") ? 3 : 1;
  t.token_to_id_.reserve(tokens.size());
  t.id_to_token_.assign(tokens.begin(), tokens.end());  // dense id -> token
  for (int id = 0; id < static_cast<int>(tokens.size()); ++id) {
    t.token_to_id_.emplace(tokens[id], id);
    // token_type 3 (CONTROL) / 4 (USER_DEFINED) are the special/added tokens.
    const int tt = id < static_cast<int>(token_types.size()) ? token_types[id] : 1;
    if (tt == 3 || tt == 4) {
      t.special_ids_.insert(id);
      t.special_tokens_.emplace_back(tokens[id], id);
      if (!tokens[id].empty()) t.special_first_bytes_.insert(static_cast<unsigned char>(tokens[id][0]));
    }
  }

  t.merge_ranks_.reserve(merges.size());
  int rank = 0;
  for (const auto& m : merges) t.merge_ranks_.emplace(m, rank++);  // each entry is "L R"

  // Longest literal first so e.g. "<|eot_id|>" wins over any shorter prefix.
  std::sort(t.special_tokens_.begin(), t.special_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  log::info("bpe: loaded from gguf vocab={} merges={} special_tokens={} pre='{}'",
            t.token_to_id_.size(), t.merge_ranks_.size(), t.special_ids_.size(), pre);
  return t;
}

}  // namespace mlxforge
