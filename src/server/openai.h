// OpenAI request parsing + response serialization (pure functions, no
// server/GPU — unit tested). The HTTP layer (http_server) tokenizes the parsed
// messages and assembles responses from these.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "sample/sampler.h"
#include "scheduler/request.h"  // TokenLogprob
#include "tokenizer/tokenizer.h"

namespace mlxforge {

// A parsed /v1/chat/completions or /v1/completions request.
struct ChatRequest {
  std::string model;
  std::vector<Tokenizer::Message> messages;  // chat; for /v1/completions a single user msg
  bool is_chat = true;                        // chat vs raw completion
  // Decoded image bytes per message, aligned 1:1 with `messages` (each entry is
  // that turn's images, in order; most are empty). When any are present the
  // request is served as a single-stream Qwen3-VL multimodal turn: each image is
  // decoded, ViT-encoded, and its placeholder run expanded into the prompt at the
  // position of the message it belongs to.
  std::vector<std::vector<std::string>> message_images;
  bool has_images() const {
    for (const auto& imgs : message_images)
      if (!imgs.empty()) return true;
    return false;
  }
  SamplingParams params;
  int max_tokens = 128;
  // OpenAI logprobs: `logprobs` enables per-token log-prob reporting; when set,
  // `top_logprobs` (0–20) is how many alternatives to include per token. Mapped
  // onto params.top_logprobs (-1 when off) by the HTTP layer.
  bool logprobs = false;
  int top_logprobs = 0;
  bool stream = false;
  std::vector<std::string> stop;
  int n = 1;
  // Each entry is one tool's function schema (JSON), rendered into the prompt by
  // the chat template. `tool_choice` is "auto" (default) | "none" | "required".
  std::vector<std::string> tools;
  std::string tool_choice = "auto";
  // Qwen3 reasoning toggle. Parsed from `enable_thinking` or
  // `chat_template_kwargs.enable_thinking`; false suppresses the <think> block.
  // Ignored by chat formats that don't support it (e.g. Llama-3.2).
  bool enable_thinking = true;

  // Whether the model should be offered tools and its output parsed for calls.
  bool tools_enabled() const { return !tools.empty() && tool_choice != "none"; }
};

// A tool call parsed from the model's output (or to be serialized in a response).
struct ToolCall {
  std::string name;       // function name
  std::string arguments;  // JSON object string (OpenAI `function.arguments`)
};

// A parsed /v1/embeddings request. `input` is normalized to a list of strings
// (OpenAI allows a single string or an array); each yields one embedding.
struct EmbeddingsRequest {
  std::string model;
  std::vector<std::string> input;
  std::string encoding_format = "float";  // only "float" is supported
};

// Parse a chat-completions body. Throws std::runtime_error on malformed JSON
// shape or out-of-range params (the server maps that to a 400).
ChatRequest parse_chat_request(const nlohmann::json& body);
// Parse a legacy completions body (`prompt` instead of `messages`).
ChatRequest parse_completion_request(const nlohmann::json& body);
// Parse an /v1/embeddings body (`input` is a string or array of strings).
EmbeddingsRequest parse_embeddings_request(const nlohmann::json& body);

// The OpenAI `usage` block (prompt/completion/total token counts), shared by the
// chat.completion and text_completion response shapes.
nlohmann::json make_usage(int prompt_tokens, int completion_tokens);

// Build the OpenAI logprobs `content` array from a request's per-token log-probs:
// one entry {token, logprob, bytes, top_logprobs:[{token, logprob, bytes}]} per
// token, where the token text and `bytes` come from decoding the id with `tok`.
nlohmann::json make_logprobs_content(const std::vector<TokenLogprob>& logprobs,
                                     const Tokenizer& tok);

// Serialize a finished completion into the OpenAI chat.completion shape. When
// `logprobs_content` is non-null it is attached as choices[0].logprobs.content
// (pass an empty array for an enabled-but-empty result; null/omitted = disabled).
nlohmann::json make_chat_completion(const std::string& id, long created, const std::string& model,
                                    const std::string& content, const std::string& finish_reason,
                                    int prompt_tokens, int completion_tokens,
                                    const nlohmann::json& logprobs_content = nlohmann::json());

// Detect a Llama-3.2 tool call in the model's decoded output. Returns the parsed
// calls, or an empty vector when the text is not a tool call (treat as content).
std::vector<ToolCall> parse_tool_calls(const std::string& text);

// Serialize tool calls into the OpenAI `message.tool_calls` array. Ids are
// deterministic ("call_0", "call_1", ...) so responses are reproducible.
nlohmann::json make_tool_calls(const std::vector<ToolCall>& calls);

// chat.completion whose assistant turn is a set of tool calls: content is null
// and finish_reason is "tool_calls".
nlohmann::json make_chat_completion_tools(const std::string& id, long created,
                                          const std::string& model,
                                          const std::vector<ToolCall>& calls, int prompt_tokens,
                                          int completion_tokens);

// Serialize embeddings into the OpenAI list shape: {object:"list", data:[{object:
// "embedding", index, embedding:[...]}], model, usage}. `prompt_tokens` is the
// summed input token count.
nlohmann::json make_embeddings_response(const std::string& model,
                                        const std::vector<std::vector<float>>& embeddings,
                                        int prompt_tokens);

// GET /v1/models payload.
nlohmann::json make_models_list(const std::string& model);

// One streaming chunk object. `delta` is the partial message delta (e.g.
// {{"content","..."}} or {{"role","assistant"}}); `finish_reason` is the JSON
// finish reason (null until the final chunk). When `logprobs_content` is non-null
// it is attached as choices[0].logprobs.content for the tokens in this delta.
nlohmann::json make_chat_chunk(const std::string& id, long created, const std::string& model,
                               const nlohmann::json& delta, const nlohmann::json& finish_reason,
                               const nlohmann::json& logprobs_content = nlohmann::json());

// Wrap a JSON payload as an SSE frame: "data: <json>\n\n".
std::string sse_frame(const nlohmann::json& payload);

// The terminating SSE frame.
extern const std::string kSseDone;

}  // namespace mlxforge
