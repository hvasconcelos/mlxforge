// MLXFORGE-022: OpenAI request parsing + response serialization (pure, no GPU).
#include <doctest/doctest.h>

#include "server/openai.h"

using namespace mlxforge;
using nlohmann::json;

TEST_CASE("MLXFORGE-022: parse_chat_request maps fields and applies defaults") {
  json body = json::parse(R"({
    "model": "mlxforge",
    "messages": [{"role": "system", "content": "be brief"},
                 {"role": "user", "content": "hi"}],
    "max_tokens": 32, "temperature": 0.7, "top_p": 0.9, "seed": 5, "stream": true,
    "stop": ["\n\n"]
  })");
  ChatRequest r = parse_chat_request(body);
  CHECK(r.model == "mlxforge");
  CHECK(r.is_chat);
  CHECK(r.messages.size() == 2);
  CHECK(r.messages[1].role == "user");
  CHECK(r.messages[1].content == "hi");
  CHECK(r.max_tokens == 32);
  CHECK(r.params.temperature == doctest::Approx(0.7f));
  CHECK(r.params.top_p == doctest::Approx(0.9f));
  CHECK(r.params.seed == 5);
  CHECK(r.stream);
  CHECK(r.stop == std::vector<std::string>{"\n\n"});

  // Defaults for omitted fields.
  ChatRequest d = parse_chat_request(json::parse(R"({"messages":[{"role":"user","content":"x"}]})"));
  CHECK(d.max_tokens == 128);
  CHECK(d.params.temperature == doctest::Approx(1.0f));
  CHECK_FALSE(d.stream);
}

TEST_CASE("MLXFORGE-022: parse_chat_request rejects malformed/out-of-range requests") {
  CHECK_THROWS_AS(parse_chat_request(json::parse(R"({})")), std::runtime_error);  // no messages
  CHECK_THROWS_AS(parse_chat_request(json::parse(R"({"messages":[]})")), std::runtime_error);
  CHECK_THROWS_AS(
      parse_chat_request(json::parse(R"({"messages":[{"role":"user","content":"x"}],"max_tokens":0})")),
      std::runtime_error);
  CHECK_THROWS_AS(parse_chat_request(json::parse(
                      R"({"messages":[{"role":"user","content":"x"}],"top_p":2})")),
                  std::runtime_error);
  CHECK_THROWS_AS(parse_chat_request(json::parse(
                      R"({"messages":[{"role":"user","content":"x"}],"temperature":-1})")),
                  std::runtime_error);
}

TEST_CASE("MLXFORGE-022: make_chat_completion emits the OpenAI chat.completion shape") {
  json c = make_chat_completion("chatcmpl-1", 1234, "mlxforge", "Paris.", "stop",
                                /*prompt_tokens=*/10, /*completion_tokens=*/3);
  CHECK(c["id"] == "chatcmpl-1");
  CHECK(c["object"] == "chat.completion");
  CHECK(c["created"] == 1234);
  CHECK(c["model"] == "mlxforge");
  CHECK(c["choices"][0]["index"] == 0);
  CHECK(c["choices"][0]["message"]["role"] == "assistant");
  CHECK(c["choices"][0]["message"]["content"] == "Paris.");
  CHECK(c["choices"][0]["finish_reason"] == "stop");
  CHECK(c["usage"]["prompt_tokens"] == 10);
  CHECK(c["usage"]["completion_tokens"] == 3);
  CHECK(c["usage"]["total_tokens"] == 13);
}

TEST_CASE("MLXFORGE-022: parse_completion_request reads a prompt string") {
  ChatRequest r = parse_completion_request(json::parse(R"({"prompt":"once upon","max_tokens":8})"));
  CHECK_FALSE(r.is_chat);
  CHECK(r.messages.front().content == "once upon");
  CHECK(r.max_tokens == 8);
  CHECK_THROWS_AS(parse_completion_request(json::parse(R"({})")), std::runtime_error);
}
