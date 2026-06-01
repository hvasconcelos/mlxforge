// XLLM-023: SSE chunk framing + bounded token-queue backpressure (no server/GPU).
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "scheduler/request.h"
#include "server/openai.h"

using namespace xllm;
using nlohmann::json;

TEST_CASE("XLLM-023: SSE chunk frame is byte-exact") {
  json chunk = make_chat_chunk("chatcmpl-1", 1234, "xllm", {{"content", "Paris"}}, nullptr);
  std::string frame = sse_frame(chunk);

  // Framing contract: "data: " prefix, "\n\n" terminator.
  CHECK(frame.rfind("data: ", 0) == 0);
  CHECK(frame.substr(frame.size() - 2) == "\n\n");

  // Payload parses back to the expected chunk shape.
  json parsed = json::parse(frame.substr(6, frame.size() - 8));
  CHECK(parsed["object"] == "chat.completion.chunk");
  CHECK(parsed["choices"][0]["delta"]["content"] == "Paris");
  CHECK(parsed["choices"][0]["finish_reason"].is_null());

  // Final-chunk finish reason + the [DONE] sentinel.
  json final = make_chat_chunk("chatcmpl-1", 1234, "xllm", json::object(), "stop");
  CHECK(json::parse(sse_frame(final).substr(6))["choices"][0]["finish_reason"] == "stop");
  CHECK(kSseDone == "data: [DONE]\n\n");
}

TEST_CASE("XLLM-023: bounded token queue applies backpressure at capacity") {
  TokenQueue q(/*capacity=*/2);
  q.push(10);
  q.push(20);  // now full

  std::atomic<bool> pushed3{false};
  std::thread producer([&] {
    q.push(30);  // must block until a slot frees
    pushed3.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(pushed3.load());  // still blocked at capacity

  int v = 0;
  CHECK(q.pop(v));
  CHECK(v == 10);  // freeing a slot unblocks the producer
  producer.join();
  CHECK(pushed3.load());

  CHECK(q.pop(v));
  CHECK(v == 20);
  CHECK(q.pop(v));
  CHECK(v == 30);
}
