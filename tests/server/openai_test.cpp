// OpenAI request parsing + response serialization (pure, no GPU).
#include <doctest/doctest.h>

#include "server/openai.h"

using namespace mlxforge;
using nlohmann::json;

TEST_CASE("parse_chat_request maps fields and applies defaults") {
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

TEST_CASE("parse_chat_request rejects malformed/out-of-range requests") {
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

TEST_CASE("make_chat_completion emits the OpenAI chat.completion shape") {
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

TEST_CASE("parse_completion_request reads a prompt string") {
  ChatRequest r = parse_completion_request(json::parse(R"({"prompt":"once upon","max_tokens":8})"));
  CHECK_FALSE(r.is_chat);
  CHECK(r.messages.front().content == "once upon");
  CHECK(r.max_tokens == 8);
  CHECK_THROWS_AS(parse_completion_request(json::parse(R"({})")), std::runtime_error);
}

TEST_CASE("parse_chat_request extracts tools and tool_choice") {
  json body = json::parse(R"({
    "messages": [{"role": "user", "content": "weather in SF?"}],
    "tools": [{"type": "function",
               "function": {"name": "get_weather", "description": "current weather",
                            "parameters": {"type": "object"}}}]
  })");
  ChatRequest r = parse_chat_request(body);
  REQUIRE(r.tools.size() == 1);
  CHECK(r.tools[0].find("get_weather") != std::string::npos);
  CHECK(r.tool_choice == "auto");
  CHECK(r.tools_enabled());

  // tool_choice "none" disables tool offering even when tools are present.
  body["tool_choice"] = "none";
  CHECK_FALSE(parse_chat_request(body).tools_enabled());

  // No tools => not enabled.
  CHECK_FALSE(parse_chat_request(json::parse(
                  R"({"messages":[{"role":"user","content":"hi"}]})"))
                  .tools_enabled());
}

TEST_CASE("parse_chat_request replays assistant tool_calls and tool results") {
  json body = json::parse(R"({
    "messages": [
      {"role": "user", "content": "weather in SF?"},
      {"role": "assistant", "content": null,
       "tool_calls": [{"id": "call_0", "type": "function",
                       "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}}]},
      {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 21}"}
    ]
  })");
  ChatRequest r = parse_chat_request(body);
  REQUIRE(r.messages.size() == 3);
  // The assistant turn becomes a single rendered call object, no text content.
  CHECK(r.messages[1].role == "assistant");
  CHECK(r.messages[1].content.empty());
  CHECK(r.messages[1].tool_call == R"({"name": "get_weather", "parameters": {"city":"SF"}})");
  CHECK(r.messages[2].role == "tool");
  CHECK(r.messages[2].content == R"({"temp": 21})");
}

TEST_CASE("parse_tool_calls detects a JSON tool call, ignores plain text") {
  auto single = parse_tool_calls(R"({"name": "get_weather", "parameters": {"city": "SF"}})");
  REQUIRE(single.size() == 1);
  CHECK(single[0].name == "get_weather");
  CHECK(json::parse(single[0].arguments) == json::parse(R"({"city": "SF"})"));

  // Leading whitespace and a stray python tag are tolerated.
  auto tagged = parse_tool_calls("  <|python_tag|>{\"name\": \"f\", \"parameters\": {}}");
  REQUIRE(tagged.size() == 1);
  CHECK(tagged[0].name == "f");
  CHECK(tagged[0].arguments == "{}");

  // Parallel calls separated by ';'.
  auto many = parse_tool_calls(R"({"name": "a", "parameters": {}}; {"name": "b", "parameters": {}})");
  REQUIRE(many.size() == 2);
  CHECK(many[1].name == "b");

  // Plain prose is not a tool call.
  CHECK(parse_tool_calls("The weather is sunny.").empty());
  CHECK(parse_tool_calls("").empty());
  // A JSON object without a name is not a call.
  CHECK(parse_tool_calls(R"({"foo": 1})").empty());
}

TEST_CASE("make_chat_completion_tools emits tool_calls with null content") {
  std::vector<ToolCall> calls = {{"get_weather", R"({"city": "SF"})"}};
  json c = make_chat_completion_tools("chatcmpl-1", 1234, "mlxforge", calls,
                                      /*prompt_tokens=*/10, /*completion_tokens=*/5);
  auto& msg = c["choices"][0]["message"];
  CHECK(msg["role"] == "assistant");
  CHECK(msg["content"].is_null());
  CHECK(c["choices"][0]["finish_reason"] == "tool_calls");
  REQUIRE(msg["tool_calls"].size() == 1);
  CHECK(msg["tool_calls"][0]["id"] == "call_0");
  CHECK(msg["tool_calls"][0]["type"] == "function");
  CHECK(msg["tool_calls"][0]["function"]["name"] == "get_weather");
  CHECK(msg["tool_calls"][0]["function"]["arguments"] == R"({"city": "SF"})");
  CHECK(c["usage"]["total_tokens"] == 15);
}
