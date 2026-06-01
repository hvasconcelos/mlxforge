// XLLM-021: C++ tokenizer over the HF tokenizer.json (mlc-ai/tokenizers-cpp,
// which wraps the Rust `tokenizers` crate — no hand-rolled BPE). Provides
// encode, decode, the Llama-3.2 chat template, and a streaming detokenizer that
// never emits broken multi-byte UTF-8 / partial byte-BPE characters.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;  // from tokenizers_cpp.h
}

namespace xllm {

class Tokenizer {
 public:
  struct Message {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
  };

  // Load from a tokenizer.json file.
  static Tokenizer from_file(const std::string& tokenizer_json_path);

  // The tokenizer's post-processor adds the <|begin_of_text|> BOS, matching
  // mlx-lm's tok.encode.
  std::vector<int> encode(const std::string& text) const;
  std::string decode(const std::vector<int>& ids) const;

  // Render the Llama-3.2 chat template (incl. the default "Cutting Knowledge /
  // Today Date" system preamble) and encode it. `today_date` is formatted like
  // "01 Jun 2026"; empty means use the current date.
  std::vector<int> apply_chat_template(const std::vector<Message>& messages,
                                       bool add_generation_prompt = true,
                                       const std::string& today_date = "") const;

  // Build just the templated prompt string (no encoding) — used by the encoder
  // above and exposed for tests.
  static std::string render_chat_template(const std::vector<Message>& messages,
                                          bool add_generation_prompt,
                                          const std::string& today_date);

 private:
  std::shared_ptr<tokenizers::Tokenizer> impl_;
  // tokenizers-cpp stashes the last encode/decode result inside the handle, so
  // concurrent calls would race; serialize them. Shared across copies.
  std::shared_ptr<std::mutex> mu_ = std::make_shared<std::mutex>();
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

}  // namespace xllm
