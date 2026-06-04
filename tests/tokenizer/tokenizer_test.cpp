// C++ tokenizer — encode/decode round-trip, chat template parity, and
// streaming UTF-8-safe detokenization, all vs the mlx-lm reference.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/tokenizer.h"

using namespace mlxforge::test;

namespace {
bool valid_utf8(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    uint8_t c = static_cast<uint8_t>(s[i]);
    size_t n;
    if (c < 0x80)
      n = 1;  // 0xxxxxxx
    else if ((c >> 5) == 0x6)
      n = 2;  // 110xxxxx
    else if ((c >> 4) == 0xE)
      n = 3;  // 1110xxxx
    else if ((c >> 3) == 0x1E)
      n = 4;  // 11110xxx
    else
      n = 0;  // invalid lead byte
    if (n == 0 || i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k)
      if ((static_cast<uint8_t>(s[i + k]) & 0xC0) != 0x80) return false;
    i += n;
  }
  return true;
}
std::string tokenizer_path() { return std::string(MLXFORGE_MODEL_DIR) + "/tokenizer.json"; }
}  // namespace

TEST_CASE("encode/decode round-trip matches the reference tokenizer") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());

  // Encode matches mlx-lm's tokenization (with BOS) for the fixed prompt.
  CHECK(tok.encode("The capital of France is") == load_token_ids("prompt_0_ids.npy"));

  // Round-trip on a fixed string set, including multi-byte UTF-8.
  for (const std::string& s :
       {"Hello, world!", "The capital of France is Paris.", "caf\xC3\xA9 na\xC3\xAFve",
        "\xF0\x9F\x98\x80 emoji"}) {
    CHECK(tok.decode(tok.encode(s)) == s);
  }
}

TEST_CASE("chat template matches mlx-lm exactly") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());
  std::vector<mlxforge::Tokenizer::Message> messages = {
      {"user", "What is the capital of France?"}};
  // The reference dump was taken on 01 Jun 2026 (the template injects the date).
  std::vector<int> ids = tok.apply_chat_template(messages, /*add_generation_prompt=*/true,
                                                 /*today_date=*/"01 Jun 2026");
  CHECK(ids == load_token_ids("chat_ids.npy"));
}

TEST_CASE("chat template renders tools and tool turns (no model needed)") {
  using mlxforge::Tokenizer;
  const std::string schema = R"({"name": "get_weather"})";

  // Tools are injected into the first user turn, after the preamble.
  std::string with_tools = Tokenizer::render_chat_template(
      {{"user", "weather in SF?"}}, /*add_generation_prompt=*/true, /*today_date=*/"01 Jun 2026",
      mlxforge::ChatFormat::Llama3, /*tools=*/{schema});
  CHECK(with_tools.find("Respond in the format") != std::string::npos);
  CHECK(with_tools.find(schema) != std::string::npos);
  // The schema precedes the user's actual question within the user turn.
  CHECK(with_tools.find(schema) < with_tools.find("weather in SF?"));

  // No tools => no preamble, plain user turn.
  std::string no_tools = Tokenizer::render_chat_template(
      {{"user", "weather in SF?"}}, true, "01 Jun 2026", mlxforge::ChatFormat::Llama3, {});
  CHECK(no_tools.find("Respond in the format") == std::string::npos);

  // An assistant tool_call replays as raw JSON; a tool result uses the ipython role.
  Tokenizer::Message call{"assistant", ""};
  call.tool_call = R"({"name": "get_weather", "parameters": {"city": "SF"}})";
  std::string convo = Tokenizer::render_chat_template(
      {{"user", "weather in SF?"}, call, {"tool", "{\"temp\": 21}"}}, true, "01 Jun 2026");
  CHECK(convo.find(call.tool_call) != std::string::npos);
  CHECK(convo.find("ipython<|end_header_id|>\n\n{\"temp\": 21}") != std::string::npos);
}

TEST_CASE("Qwen3 ChatML template renders thinking, tools, and tool turns (no model needed)") {
  using mlxforge::ChatFormat;
  using mlxforge::Tokenizer;

  // Basic ChatML: no default system message, generation prompt opens an assistant
  // turn. Thinking is on by default (no <think> block emitted).
  std::string basic = Tokenizer::render_chat_template({{"user", "hi"}}, /*add_generation_prompt=*/
                                                      true, "", ChatFormat::Qwen3);
  CHECK(basic == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");

  // System message is rendered as a leading system turn.
  std::string with_sys = Tokenizer::render_chat_template(
      {{"system", "be terse"}, {"user", "hi"}}, true, "", ChatFormat::Qwen3);
  CHECK(with_sys.rfind("<|im_start|>system\nbe terse<|im_end|>\n", 0) == 0);

  // enable_thinking=false appends an empty <think></think> block after the header.
  std::string no_think = Tokenizer::render_chat_template({{"user", "hi"}}, true, "",
                                                         ChatFormat::Qwen3, {}, /*enable_thinking=*/
                                                         false);
  CHECK(no_think.find("<|im_start|>assistant\n<think>\n\n</think>\n\n") != std::string::npos);

  // Hermes-style tools: injected into the system turn, with <tool_call> guidance.
  const std::string schema = R"({"type": "function", "function": {"name": "get_weather"}})";
  std::string with_tools = Tokenizer::render_chat_template(
      {{"user", "weather?"}}, true, "", ChatFormat::Qwen3, /*tools=*/{schema});
  CHECK(with_tools.rfind("<|im_start|>system\n# Tools", 0) == 0);
  CHECK(with_tools.find("<tools>\n" + schema + "\n</tools>") != std::string::npos);
  CHECK(with_tools.find("<tool_call></tool_call>") != std::string::npos);

  // An assistant tool call replays as a <tool_call> block; a tool result is wrapped
  // in a <tool_response> under a user turn.
  Tokenizer::Message call{"assistant", ""};
  call.tool_call = R"({"name": "get_weather", "arguments": {"city": "SF"}})";
  std::string convo = Tokenizer::render_chat_template(
      {{"user", "weather?"}, call, {"tool", R"({"temp": 21})"}}, true, "", ChatFormat::Qwen3);
  CHECK(convo.find("<|im_start|>assistant\n\n<tool_call>\n" + call.tool_call + "\n</tool_call>") !=
        std::string::npos);
  CHECK(convo.find("<|im_start|>user\n<tool_response>\n{\"temp\": 21}\n</tool_response><|im_end|>") !=
        std::string::npos);
}

TEST_CASE("streaming detokenizer never emits broken UTF-8") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());
  std::vector<int> stream = load_token_ids("greedy_tokens.npy");

  mlxforge::StreamingDetokenizer detok(tok);
  std::string assembled;
  for (int id : stream) {
    std::string piece = detok.add(id);
    CHECK(valid_utf8(piece));  // every emitted piece is complete UTF-8
    assembled += piece;
  }
  assembled += detok.finish();

  // Incremental assembly equals a one-shot decode of the whole stream.
  CHECK(assembled == tok.decode(stream));
  CHECK(valid_utf8(assembled));
}
