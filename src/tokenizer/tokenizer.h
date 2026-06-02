// C++ tokenizer over the HF tokenizer.json. encode/decode use our own
// from-scratch byte-level BPE (BpeTokenizer, src/tokenizer/bpe.h) — no Rust /
// tokenizers-cpp. Currently supports Llama-3.2-style byte-level BPE only
// (from_file throws on anything else). Also provides the Llama-3.2 chat template
// and a streaming detokenizer that never emits broken multi-byte UTF-8 /
// partial byte-BPE characters.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mlxforge {
class BpeTokenizer;  // from tokenizer/bpe.h
}

namespace mlxforge {

// Which chat template to render. Selected from config.json's model_type so the
// forward pass (shared) and the prompt formatting (per-family) stay decoupled.
// Only Llama-3.2 is supported today; new families will add values here.
enum class ChatFormat { Llama3 };

// Map a config.json model_type to a chat format. Currently always Llama3; the
// seam is kept so new families can be mapped here when they are re-onboarded.
ChatFormat chat_format_from_model_type(const std::string& model_type);

class Tokenizer {
 public:
  struct Message {
    std::string role;     // "system" | "user" | "assistant" | "tool"
    std::string content;
    // For an assistant turn that called a tool: the rendered call object
    // `{"name": ..., "parameters": {...}}` replayed verbatim into the prompt.
    // Empty for ordinary messages. Llama-3.2 emits a single call per turn.
    std::string tool_call;
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
  // (formatted like "01 Jun 2026", empty = current date) is Llama-only. When
  // `tools` is non-empty each entry (a function's JSON schema, rendered
  // verbatim) is injected into the first user turn so the model can call it.
  std::vector<int> apply_chat_template(const std::vector<Message>& messages,
                                       bool add_generation_prompt = true,
                                       const std::string& today_date = "",
                                       const std::vector<std::string>& tools = {}) const;

  // Build just the templated prompt string (no encoding) — used by the encoder
  // above and exposed for tests.
  static std::string render_chat_template(const std::vector<Message>& messages,
                                          bool add_generation_prompt, const std::string& today_date,
                                          ChatFormat fmt = ChatFormat::Llama3,
                                          const std::vector<std::string>& tools = {});

 private:
  // BpeTokenizer is pure/const and thread-safe, so encode/decode need no mutex.
  // It also owns the special-token ids (parsed from tokenizer.json) that decode
  // skips, replacing the Llama-only "id >= 128000" heuristic.
  std::shared_ptr<BpeTokenizer> impl_;
  int bos_id_ = 128000;  // prepended on encode; -1 = none
  ChatFormat chat_format_ = ChatFormat::Llama3;
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
