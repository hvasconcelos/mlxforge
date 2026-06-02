// C++ tokenizer over the HF tokenizer.json. For Llama-3.2-style byte-level BPE,
// encode/decode use our own implementation (BpeTokenizer, src/tokenizer/bpe.h);
// other families (e.g. Mistral's Metaspace/SentencePiece) fall back to
// tokenizers-cpp until they too are reimplemented. Also provides the chat
// template (Llama-3.2 or Mistral) and a streaming detokenizer that never emits
// broken multi-byte UTF-8 / partial byte-BPE characters.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace mlxforge {
class BpeTokenizer;  // from tokenizer/bpe.h
}
namespace tokenizers {
class Tokenizer;  // from tokenizers_cpp.h (fallback backend)
}

namespace mlxforge {

// Which chat template to render. Selected from config.json's model_type so the
// forward pass (shared) and the prompt formatting (per-family) stay decoupled.
enum class ChatFormat { Llama3, Mistral };

// Map a config.json model_type to a chat format ("mistral" -> Mistral, else
// Llama3, which is the engine's original default).
ChatFormat chat_format_from_model_type(const std::string& model_type);

class Tokenizer {
 public:
  struct Message {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
  };

  // Load from a tokenizer.json file. `bos_id` is prepended on encode (the Rust
  // Encode does not run the BOS post-processor); pass -1 to add none. `fmt`
  // selects the chat template. Defaults match Llama-3.2 so existing call sites
  // are unchanged.
  static Tokenizer from_file(const std::string& tokenizer_json_path, int bos_id = 128000,
                             ChatFormat fmt = ChatFormat::Llama3);

  // Encode prepends the configured BOS id (when >= 0), matching mlx-lm's
  // tok.encode for the model.
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;

  // Render the model's chat template and encode it. For Llama-3.2 this injects
  // the default "Cutting Knowledge / Today Date" system preamble; `today_date`
  // (formatted like "01 Jun 2026", empty = current date) is Llama-only.
  std::vector<int> apply_chat_template(const std::vector<Message>& messages,
                                       bool add_generation_prompt = true,
                                       const std::string& today_date = "") const;

  // Build just the templated prompt string (no encoding) — used by the encoder
  // above and exposed for tests.
  static std::string render_chat_template(const std::vector<Message>& messages,
                                          bool add_generation_prompt, const std::string& today_date,
                                          ChatFormat fmt = ChatFormat::Llama3);

 private:
  // Exactly one backend is set per instance (see Tokenizer::from_file).
  // BpeTokenizer is pure/const and thread-safe; the tokenizers-cpp fallback
  // stashes per-handle state, so its calls are serialized by `mu_`.
  std::shared_ptr<BpeTokenizer> bpe_;
  std::shared_ptr<tokenizers::Tokenizer> hf_;
  std::shared_ptr<std::mutex> mu_ = std::make_shared<std::mutex>();  // guards hf_ only
  int bos_id_ = 128000;  // prepended on encode; -1 = none
  ChatFormat chat_format_ = ChatFormat::Llama3;
  // Special-token ids parsed from tokenizer.json (added_tokens[*].special);
  // skipped on decode, replacing the Llama-only "id >= 128000" heuristic.
  std::shared_ptr<std::unordered_set<int>> special_ids_ =
      std::make_shared<std::unordered_set<int>>();
};

// Incremental detokenizer: feed one new token id at a time; returns only the
// text that has become complete UTF-8 (may be empty until a character finishes).
class StreamingDetokenizer {
 public:
  explicit StreamingDetokenizer(const Tokenizer& tok) : tok_(tok) {}

  std::string add(int id);   // newly-complete text since the last call
  std::string finish();      // flush any trailing complete bytes

 private:
  const Tokenizer& tok_;
  std::vector<int> ids_;
  size_t emitted_ = 0;  // bytes already returned
};

}  // namespace mlxforge
