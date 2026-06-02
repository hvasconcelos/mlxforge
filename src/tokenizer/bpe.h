// From-scratch byte-level BPE tokenizer for Llama-3.2, built only on
// nlohmann/json (no Rust / tokenizers-cpp). It reproduces the HF "fast"
// tokenizer pipeline for a tokenizer.json with `model.type == "BPE"`:
//
//   special-token segmentation -> pre-tokenizer split (the llama3 tiktoken-style
//   regex, hand-rolled in bpe.cpp) -> GPT-2 byte->unicode -> BPE merges (with
//   `ignore_merges`: a whole pre-token that is itself a vocab entry is emitted
//   directly) -> vocab lookup.
//
// Validated to produce byte-identical ids to tokenizers-cpp (see
// tests/tokenizer/bpe_test.cpp). It is pure/const and therefore thread-safe.
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlxforge {

class BpeTokenizer {
 public:
  // Whether this tokenizer.json is a byte-level BPE we can handle (Llama-3.2
  // style: model.type == "BPE" with a ByteLevel decoder). Returns false for
  // a Metaspace/SentencePiece tokenizer, which the caller should route to a
  // different backend.
  static bool is_supported(const std::string& tokenizer_json);

  // Parse a tokenizer.json blob. Throws std::runtime_error if the file is
  // malformed or not a byte-level BPE model (see is_supported).
  static BpeTokenizer from_blob(const std::string& tokenizer_json);

  // Encode text to token ids. Does NOT prepend BOS — the Tokenizer wrapper owns
  // that, matching tokenizers-cpp's Encode (which skips the post-processor).
  // Special-token literals present in the input are emitted as their ids.
  std::vector<int> encode(const std::string& text) const;

  // Decode ids back to text. Special ids are skipped (matching the wrapper's
  // skip_special_tokens default); unknown/out-of-range ids are ignored.
  std::string decode(const std::vector<int>& ids) const;

  size_t vocab_size() const { return token_to_id_.size(); }
  const std::unordered_set<int>& special_ids() const { return special_ids_; }

 private:
  // Append the ids for one pre-token piece (given as a byte-level/merge-alphabet
  // string) to `out`, applying `ignore_merges` then the BPE merge loop.
  void bpe_piece(const std::string& piece, std::vector<int>& out) const;
  // Pre-tokenize a plain (special-token-free) segment and append its ids.
  void encode_plain(const std::string& segment, std::vector<int>& out) const;

  std::unordered_map<std::string, int> token_to_id_;  // byte-level token -> id
  std::vector<std::string> id_to_token_;              // id -> token (dense, for decode)
  std::unordered_map<std::string, int> merge_ranks_;  // "L R" -> rank (merge priority)
  std::unordered_set<int> special_ids_;
  // (literal, id) for every special added-token, sorted by descending literal
  // length so longest-match wins during segmentation.
  std::vector<std::pair<std::string, int>> special_tokens_;
  // First bytes of the special-token literals — a cheap pre-filter so the
  // segmentation scan only attempts matches where one could start.
  std::unordered_set<unsigned char> special_first_bytes_;
};

}  // namespace mlxforge
