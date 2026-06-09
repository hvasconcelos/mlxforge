// Exercises the engine through ONLY the C ABI (capi/mlxforge.h) — no C++ engine
// types — proving the embeddable product surface works end to end, that errors
// are reported as messages (not exceptions), and that concurrent requests batch
// while preserving greedy determinism (batched greedy == single-stream greedy).
#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/mlxforge.h"
#include "support/model_fixture.h"

using namespace mlxforge::test;

namespace {

// Drain a request to completion through the C ABI, returning its decoded text.
std::string drain(mlxforge_request* req) {
  std::string out;
  char* text = nullptr;
  while (mlxforge_request_next(req, &text) == 0) {
    if (text) {
      out += text;
      mlxforge_string_free(text);
      text = nullptr;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("C ABI reports its version and ABI level") {
  CHECK(std::string(mlxforge_version()).size() > 0);
  CHECK(mlxforge_abi_version() == MLXFORGE_ABI_VERSION);
}

TEST_CASE("C ABI surfaces a bad spec as an error message, not a crash") {
  char* err = nullptr;
  mlxforge_engine* e = mlxforge_engine_create("", nullptr, &err);
  CHECK(e == nullptr);
  REQUIRE(err != nullptr);
  CHECK(std::string(err).size() > 0);
  mlxforge_string_free(err);
}

TEST_CASE("C ABI generates text and batches concurrent requests deterministically") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }

  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Greedy (zeroed sampling => argmax), deterministic baseline at batch size 1.
  mlxforge_sampling greedy = {};
  greedy.max_tokens = 16;

  mlxforge_msg msg = {"user", "What is the capital of France?"};
  mlxforge_request* r0 = mlxforge_submit_chat(eng, &msg, 1, &greedy, &err);
  REQUIRE_MESSAGE(r0 != nullptr, (err ? err : "submit_chat failed"));
  const std::string baseline = drain(r0);
  CHECK(std::string(mlxforge_request_finish_reason(r0)).size() > 0);  // "stop" | "length"
  mlxforge_request_free(r0);
  CHECK(baseline.size() > 0);

  // Submit several identical greedy requests at once: they share the batched
  // engine, and — being greedy — must reproduce the single-stream baseline
  // exactly. This validates both the ABI and that batching preserves output.
  const int N = 4;
  std::vector<mlxforge_request*> reqs;
  for (int i = 0; i < N; ++i) {
    mlxforge_request* r = mlxforge_submit_chat(eng, &msg, 1, &greedy, nullptr);
    REQUIRE(r != nullptr);
    reqs.push_back(r);
  }
  std::vector<std::string> outs(N);
  std::vector<std::thread> threads;
  for (int i = 0; i < N; ++i)
    threads.emplace_back([&, i] { outs[i] = drain(reqs[i]); });
  for (auto& t : threads) t.join();
  for (int i = 0; i < N; ++i) {
    CHECK(outs[i] == baseline);  // batched greedy == single-stream greedy
    mlxforge_request_free(reqs[i]);
  }

  mlxforge_engine_free(eng);
}

TEST_CASE("C ABI embeddings are unit-normalized, deterministic, and semantic") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto embed = [&](const char* text) {
    float* v = nullptr;
    size_t n = 0;
    int rc = mlxforge_embed(eng, text, /*pooling=*/0, &v, &n, &err);
    REQUIRE_MESSAGE(rc == 0, (err ? err : "embed failed"));
    std::vector<float> out(v, v + n);
    mlxforge_floats_free(v);
    return out;
  };
  auto dot = [](const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * b[i];
    return s;
  };

  const std::vector<float> a = embed("The cat sat on the warm mat.");
  const std::vector<float> b = embed("A kitten is resting on a soft rug.");
  const std::vector<float> c = embed("The stock market fell sharply amid economic fears.");

  REQUIRE(a.size() > 0);
  CHECK(a.size() == b.size());
  CHECK(a.size() == c.size());
  CHECK(dot(a, a) == doctest::Approx(1.0).epsilon(0.02));  // unit-normalized

  const std::vector<float> a2 = embed("The cat sat on the warm mat.");
  CHECK(dot(a, a2) == doctest::Approx(1.0).epsilon(1e-4));  // deterministic

  // The two animal sentences should be closer than either is to the finance one.
  CHECK(dot(a, b) > dot(a, c));

  // mlxforge_embed_ex with explicit options (last-token pooling + appended EOS +
  // an instruction wrap) returns a distinct but still unit-normalized vector.
  {
    mlxforge_embed_opts opts = {};
    opts.pooling = 1;     // last token
    opts.add_eos = 1;     // append the model's EOS
    opts.instruction = "Given a query, retrieve relevant passages";
    float* v = nullptr;
    size_t n = 0;
    int rc = mlxforge_embed_ex(eng, "The cat sat on the warm mat.", &opts, &v, &n, &err);
    REQUIRE_MESSAGE(rc == 0, (err ? err : "embed_ex failed"));
    std::vector<float> last(v, v + n);
    mlxforge_floats_free(v);
    CHECK(last.size() == a.size());
    CHECK(dot(last, last) == doctest::Approx(1.0).epsilon(0.02));  // unit-normalized

    // NULL opts == model defaults (here, the plain Llama mean path), still valid.
    rc = mlxforge_embed_ex(eng, "The cat sat on the warm mat.", nullptr, &v, &n, &err);
    REQUIRE_MESSAGE(rc == 0, (err ? err : "embed_ex(null) failed"));
    mlxforge_floats_free(v);
  }

  mlxforge_engine_free(eng);
}

TEST_CASE("C ABI constrained decoding forces well-formed JSON") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // JSON mode: the output must be valid JSON regardless of the (small) model.
  {
    mlxforge_sampling s = {};
    s.max_tokens = 96;
    s.json_schema = "json";
    mlxforge_msg msg = {"user", "Describe a person as a JSON object."};
    mlxforge_request* r = mlxforge_submit_chat(eng, &msg, 1, &s, &err);
    REQUIRE_MESSAGE(r != nullptr, (err ? err : "submit failed"));
    const std::string out = drain(r);
    const std::string reason = mlxforge_request_finish_reason(r);
    mlxforge_request_free(r);
    CAPTURE(out);
    // Whether it stops or hits the token budget, the output is a valid JSON
    // prefix; on a clean stop it is a complete, parseable value.
    if (reason == "stop") {
      auto parsed = nlohmann::json::parse(out, nullptr, /*allow_exceptions=*/false);
      CHECK_FALSE(parsed.is_discarded());
    }
  }

  // Schema mode: a top-level object with required typed keys, in order.
  {
    mlxforge_sampling s = {};
    s.max_tokens = 96;
    s.json_schema =
        R"({"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}}})";
    mlxforge_msg msg = {"user", "Give facts about Paris."};
    mlxforge_request* r = mlxforge_submit_chat(eng, &msg, 1, &s, &err);
    REQUIRE_MESSAGE(r != nullptr, (err ? err : "submit failed"));
    const std::string out = drain(r);
    const std::string reason = mlxforge_request_finish_reason(r);
    mlxforge_request_free(r);
    CAPTURE(out);
    if (reason == "stop") {
      auto parsed = nlohmann::json::parse(out, nullptr, /*allow_exceptions=*/false);
      REQUIRE_FALSE(parsed.is_discarded());
      CHECK(parsed.is_object());
      CHECK(parsed.contains("city"));
      CHECK(parsed.contains("population"));
      CHECK(parsed["city"].is_string());
      CHECK(parsed["population"].is_number_integer());
    }
  }

  mlxforge_engine_free(eng);
}
