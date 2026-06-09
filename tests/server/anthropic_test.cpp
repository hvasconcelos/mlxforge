// Anthropic Messages API request parsing + response/SSE serialization (pure, no
// GPU). Mirrors openai_test.cpp / sse_test.cpp.
#include <doctest/doctest.h>

#include "server/anthropic.h"

using namespace mlxforge;
using nlohmann::json;

TEST_CASE("parse_messages_request maps fields and applies defaults") {
  json body = json::parse(R"({
    "model": "mlxforge",
    "max_tokens": 32,
    "system": "be brief",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 5, "stream": true,
    "stop_sequences": ["\n\n"]
  })");
  ChatRequest r = parse_messages_request(body);
  CHECK(r.model == "mlxforge");
  CHECK(r.is_chat);
  CHECK(r.max_tokens == 32);
  // The top-level system prompt becomes the leading system message.
  REQUIRE(r.messages.size() == 2);
  CHECK(r.messages[0].role == "system");
  CHECK(r.messages[0].content == "be brief");
  CHECK(r.messages[1].role == "user");
  CHECK(r.messages[1].content == "hi");
  CHECK(r.params.temperature == doctest::Approx(0.7f));
  CHECK(r.params.top_p == doctest::Approx(0.9f));
  CHECK(r.params.top_k == 40);
  CHECK(r.params.seed == 5);
  CHECK(r.stream);
  CHECK(r.stop == std::vector<std::string>{"\n\n"});

  // Defaults for omitted fields (max_tokens is required, so it is always set).
  ChatRequest d = parse_messages_request(
      json::parse(R"({"max_tokens":16,"messages":[{"role":"user","content":"x"}]})"));
  CHECK(d.params.temperature == doctest::Approx(1.0f));
  CHECK_FALSE(d.stream);
  CHECK(d.model.empty());
  CHECK(d.messages.size() == 1);  // no system prompt prepended
  CHECK(d.tool_choice == "auto");
}

TEST_CASE("parse_messages_request accepts content blocks and a system block array") {
  json body = json::parse(R"({
    "max_tokens": 16,
    "system": [{"type":"text","text":"sys "},{"type":"text","text":"prompt"}],
    "messages": [{"role":"user","content":[{"type":"text","text":"hello "},
                                           {"type":"text","text":"world"}]}]
  })");
  ChatRequest r = parse_messages_request(body);
  REQUIRE(r.messages.size() == 2);
  CHECK(r.messages[0].content == "sys prompt");
  CHECK(r.messages[1].role == "user");
  CHECK(r.messages[1].content == "hello world");
}

TEST_CASE("parse_messages_request round-trips tool_use and tool_result blocks") {
  json body = json::parse(R"({
    "max_tokens": 16,
    "messages": [
      {"role":"user","content":"weather in Paris?"},
      {"role":"assistant","content":[
        {"type":"text","text":"let me check"},
        {"type":"tool_use","id":"toolu_0","name":"get_weather","input":{"city":"Paris"}}]},
      {"role":"user","content":[
        {"type":"tool_result","tool_use_id":"toolu_0","content":"sunny"}]}
    ]
  })");
  ChatRequest r = parse_messages_request(body);
  REQUIRE(r.messages.size() == 4);
  CHECK(r.messages[0].role == "user");
  // Assistant block splits into a text turn then a tool-call turn.
  CHECK(r.messages[1].role == "assistant");
  CHECK(r.messages[1].content == "let me check");
  CHECK(r.messages[2].role == "assistant");
  CHECK(r.messages[2].tool_call == R"({"name": "get_weather", "parameters": {"city":"Paris"}})");
  // The tool_result becomes a "tool" turn the chat template understands.
  CHECK(r.messages[3].role == "tool");
  CHECK(r.messages[3].content == "sunny");
}

TEST_CASE("parse_messages_request converts tools and tool_choice") {
  json body = json::parse(R"({
    "max_tokens": 16,
    "messages": [{"role":"user","content":"x"}],
    "tools": [{"name":"get_weather","description":"Get weather",
               "input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],
    "tool_choice": {"type":"any"}
  })");
  ChatRequest r = parse_messages_request(body);
  REQUIRE(r.tools.size() == 1);
  // input_schema is rendered under "parameters" (matching the trained template).
  json fn = json::parse(r.tools[0]);
  CHECK(fn["name"] == "get_weather");
  CHECK(fn["description"] == "Get weather");
  CHECK(fn.contains("parameters"));
  CHECK_FALSE(fn.contains("input_schema"));
  CHECK(r.tool_choice == "required");  // "any" -> required
  CHECK(r.tools_enabled());

  // tool_choice "none" disables tools.
  json b2 = body;
  b2["tool_choice"] = {{"type", "none"}};
  CHECK(parse_messages_request(b2).tool_choice == "none");
  CHECK_FALSE(parse_messages_request(b2).tools_enabled());
}

TEST_CASE("parse_messages_request rejects malformed/out-of-range requests") {
  // max_tokens is required.
  CHECK_THROWS_AS(parse_messages_request(json::parse(R"({"messages":[{"role":"user","content":"x"}]})")),
                  std::runtime_error);
  CHECK_THROWS_AS(parse_messages_request(json::parse(R"({"max_tokens":0,"messages":[{"role":"user","content":"x"}]})")),
                  std::runtime_error);
  CHECK_THROWS_AS(parse_messages_request(json::parse(R"({"max_tokens":16})")),
                  std::runtime_error);  // no messages
  CHECK_THROWS_AS(parse_messages_request(json::parse(R"({"max_tokens":16,"messages":[]})")),
                  std::runtime_error);
  CHECK_THROWS_AS(parse_messages_request(
                      json::parse(R"({"max_tokens":16,"messages":[{"role":"user","content":"x"}],"top_p":2})")),
                  std::runtime_error);
}

TEST_CASE("make_message_response has the Anthropic message shape") {
  json content = json::array({{{"type", "text"}, {"text", "Paris"}}});
  json m = make_message_response("msg_1", "mlxforge", content, "end_turn", 10, 3);
  CHECK(m["id"] == "msg_1");
  CHECK(m["type"] == "message");
  CHECK(m["role"] == "assistant");
  CHECK(m["model"] == "mlxforge");
  CHECK(m["content"][0]["type"] == "text");
  CHECK(m["content"][0]["text"] == "Paris");
  CHECK(m["stop_reason"] == "end_turn");
  CHECK(m["stop_sequence"].is_null());
  CHECK(m["usage"]["input_tokens"] == 10);
  CHECK(m["usage"]["output_tokens"] == 3);
}

TEST_CASE("make_tool_use_blocks parses arguments into the input object") {
  std::vector<ToolCall> calls = {{"get_weather", R"({"city":"Paris"})"}};
  json blocks = make_tool_use_blocks(calls);
  REQUIRE(blocks.size() == 1);
  CHECK(blocks[0]["type"] == "tool_use");
  CHECK(blocks[0]["id"] == "toolu_0");
  CHECK(blocks[0]["name"] == "get_weather");
  CHECK(blocks[0]["input"]["city"] == "Paris");
}

TEST_CASE("sse_event frames a named SSE event") {
  std::string frame = sse_event("content_block_delta", make_text_delta(0, "hi"));
  CHECK(frame.rfind("event: content_block_delta\n", 0) == 0);
  CHECK(frame.find("\ndata: ") != std::string::npos);
  CHECK(frame.substr(frame.size() - 2) == "\n\n");

  const std::string prefix = "event: content_block_delta\ndata: ";
  json parsed = json::parse(frame.substr(prefix.size(), frame.size() - prefix.size() - 2));
  CHECK(parsed["type"] == "content_block_delta");
  CHECK(parsed["delta"]["type"] == "text_delta");
  CHECK(parsed["delta"]["text"] == "hi");
}

TEST_CASE("streaming event payloads carry the documented shapes") {
  CHECK(make_message_start("msg_1", "mlxforge", 5)["message"]["usage"]["input_tokens"] == 5);
  CHECK(make_content_block_start(0, {{"type", "text"}, {"text", ""}})["index"] == 0);
  CHECK(make_input_json_delta(1, R"({"a":1})")["delta"]["partial_json"] == R"({"a":1})");
  CHECK(make_content_block_stop(2)["index"] == 2);
  json md = make_message_delta("max_tokens", 7);
  CHECK(md["delta"]["stop_reason"] == "max_tokens");
  CHECK(md["usage"]["output_tokens"] == 7);
  CHECK(kMessageStop["type"] == "message_stop");
}

TEST_CASE("parse_messages_request extracts a base64 image block as bytes") {
  // base64 "aGVsbG8=" decodes to "hello".
  json body = json::parse(R"({
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "what animal?"},
      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "aGVsbG8="}}
    ]}],
    "max_tokens": 8
  })");
  ChatRequest r = parse_messages_request(body);
  REQUIRE(r.message_images.size() == 1);
  REQUIRE(r.message_images[0].size() == 1);
  CHECK(r.message_images[0][0] == "hello");
}
