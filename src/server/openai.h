// OpenAI request parsing + response serialization (pure functions, no
// server/GPU — unit tested). The HTTP layer (http_server) tokenizes the parsed
// messages and assembles responses from these.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "sample/sampler.h"
#include "tokenizer/tokenizer.h"

namespace mlxforge {

// A parsed /v1/chat/completions or /v1/completions request.
struct ChatRequest {
  std::string model;
  std::vector<Tokenizer::Message> messages;  // chat; for /v1/completions a single user msg
  bool is_chat = true;                        // chat vs raw completion
  SamplingParams params;
  int max_tokens = 128;
  bool stream = false;
  std::vector<std::string> stop;
  int n = 1;
};

// Parse a chat-completions body. Throws std::runtime_error on malformed JSON
// shape or out-of-range params (the server maps that to a 400).
ChatRequest parse_chat_request(const nlohmann::json& body);
// Parse a legacy completions body (`prompt` instead of `messages`).
ChatRequest parse_completion_request(const nlohmann::json& body);

// The OpenAI `usage` block (prompt/completion/total token counts), shared by the
// chat.completion and text_completion response shapes.
nlohmann::json make_usage(int prompt_tokens, int completion_tokens);

// Serialize a finished completion into the OpenAI chat.completion shape.
nlohmann::json make_chat_completion(const std::string& id, long created, const std::string& model,
                                    const std::string& content, const std::string& finish_reason,
                                    int prompt_tokens, int completion_tokens);

// GET /v1/models payload.
nlohmann::json make_models_list(const std::string& model);

// One streaming chunk object. `delta` is the partial message delta (e.g.
// {{"content","..."}} or {{"role","assistant"}}); `finish_reason` is the JSON
// finish reason (null until the final chunk).
nlohmann::json make_chat_chunk(const std::string& id, long created, const std::string& model,
                               const nlohmann::json& delta, const nlohmann::json& finish_reason);

// Wrap a JSON payload as an SSE frame: "data: <json>\n\n".
std::string sse_frame(const nlohmann::json& payload);

// The terminating SSE frame.
extern const std::string kSseDone;

}  // namespace mlxforge
