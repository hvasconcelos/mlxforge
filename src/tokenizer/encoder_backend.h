// Backend-agnostic token encode/decode interface. The Tokenizer wrapper
// (tokenizer.h) holds one of these behind a pointer and layers the
// family-specific concerns on top — BOS prepending, chat templates, and
// special-token skipping — so swapping the underlying algorithm touches only
// the factory in tokenizer.cpp, not the wrapper or its callers.
//
// Today the only implementation is the from-scratch byte-level BPE engine
// (BpeTokenizer, tokenizer/bpe.h). A future SentencePiece/Metaspace family
// becomes a sibling implementation selected by Tokenizer::from_file; nothing
// else has to change because everything routes through this interface.
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace mlxforge {

class EncoderBackend {
 public:
  virtual ~EncoderBackend() = default;

  // Encode text to token ids. Does NOT prepend BOS — the Tokenizer wrapper owns
  // that. Special-token literals present in the input are emitted as their ids.
  virtual std::vector<int> encode(const std::string& text) const = 0;

  // Decode ids back to text. This backend's special ids are skipped, as are
  // unknown / out-of-range ids.
  virtual std::string decode(const std::vector<int>& ids) const = 0;

  virtual size_t vocab_size() const = 0;

  // The special-token ids (parsed from the tokenizer metadata) that decode
  // skips; the wrapper also consults these to pre-filter before decode.
  virtual const std::unordered_set<int>& special_ids() const = 0;
};

}  // namespace mlxforge
