// From-scratch SentencePiece-style BPE tokenizer (e.g. Gemma, Llama-2, Mistral),
// built only on nlohmann/json. It reproduces the HF "fast" tokenizer pipeline for
// a tokenizer.json with `model.type == "BPE"` and `byte_fallback == true`:
//
//   special-token segmentation -> Metaspace normalization (every ' ' -> U+2581
//   "▁") -> BPE merges over the whole segment's Unicode characters -> vocab
//   lookup, with byte_fallback (a character absent from the vocab is emitted as
//   its UTF-8 bytes via the "<0xNN>" byte tokens).
//
// This is a distinct family from the byte-level BPE (tokenizer/bpe.h): there is
// no GPT-2 byte->unicode remapping and no regex pre-tokenizer; spaces become the
// metaspace marker and OOV bytes fall back to byte tokens. Selected by
// Tokenizer's backend factory when is_supported() matches. Validated to produce
// byte-identical ids to the HF tokenizer (tests/tokenizer/spm_test.cpp). It is
// pure/const and therefore thread-safe.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tokenizer/encoder_backend.h"

namespace mlxforge {

// One concrete EncoderBackend: the SentencePiece-BPE engine. encode/decode/
// vocab_size/special_ids are the interface; is_supported/from_blob are
// SPM-specific factories used by Tokenizer's backend selection.
class SpmBpeTokenizer : public EncoderBackend {
 public:
  // Whether this tokenizer.json is a SentencePiece-style BPE we can handle
  // (model.type == "BPE" with byte_fallback and a Metaspace/ByteFallback
  // decoder). Returns false for a byte-level BPE (Llama-3/Qwen), which the caller
  // routes to BpeTokenizer instead.
  static bool is_supported(const std::string& tokenizer_json);

  // Parse a tokenizer.json blob. Throws std::runtime_error if malformed or not a
  // SentencePiece-BPE model (see is_supported).
  static SpmBpeTokenizer from_blob(const std::string& tokenizer_json);

  // Encode text to token ids. Does NOT prepend BOS — the Tokenizer wrapper owns
  // that. Special-token literals present in the input are emitted as their ids.
  std::vector<int> encode(const std::string& text) const override;

  // Decode ids back to text: metaspace -> space, byte tokens -> raw UTF-8 bytes
  // (fused). Special ids are skipped; unknown/out-of-range ids are ignored.
  std::string decode(const std::vector<int>& ids) const override;

  size_t vocab_size() const override { return token_to_id_.size(); }
  const std::unordered_set<int>& special_ids() const override { return special_ids_; }

 private:
  // Normalize a special-token-free segment (space -> metaspace), then run the BPE
  // merge loop over its Unicode characters and append the ids.
  void encode_plain(const std::string& segment, std::vector<int>& out) const;
  // Map one final BPE symbol to ids: its vocab id, or byte_fallback to "<0xNN>"
  // byte tokens, or the unk id.
  void emit_symbol(const std::string& sym, std::vector<int>& out) const;

  std::unordered_map<std::string, int> token_to_id_;  // token -> id
  std::vector<std::string>             id_to_token_;   // id -> token (dense, decode)
  std::unordered_map<std::string, int> merge_ranks_;   // "L R" -> rank (priority)
  std::unordered_set<int>              special_ids_;

  std::array<int, 256> byte_fallback_{};  // byte value -> "<0xNN>" id (-1 if absent)
  std::unordered_map<int, int> byte_token_value_;  // "<0xNN>" id -> byte value (decode)
  int unk_id_ = -1;

  // (literal, id) for every added-token, sorted by descending literal length so
  // longest-match wins during segmentation; first bytes are a cheap pre-filter.
  std::vector<std::pair<std::string, int>> special_tokens_;
  std::unordered_set<unsigned char>        special_first_bytes_;
};

}  // namespace mlxforge
