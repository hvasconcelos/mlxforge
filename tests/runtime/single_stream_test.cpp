// single-stream greedy generation loop.
#include <doctest/doctest.h>

#include <vector>

#include "runtime/single_stream.h"
#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;

TEST_CASE("greedy CLI output matches mlx-lm token-for-token") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> prompt = load_token_ids("prompt_0_ids.npy");
  std::vector<int> ref = load_token_ids("greedy_tokens.npy");  // 20 tokens

  // Streamed tokens collected via the callback (proves incremental emission).
  std::vector<int> streamed;
  mlxforge::GenerateResult r = mlxforge::greedy_generate(
      model, prompt, /*max_tokens=*/static_cast<int>(ref.size()),
      model.config().eos_token_ids, [&](int id) { streamed.push_back(id); });

  assert_tokens_equal(r.tokens, ref);
  assert_tokens_equal(streamed, ref);
  CHECK_FALSE(r.hit_eos);
}

TEST_CASE("loop terminates on max_tokens") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> prompt = load_token_ids("prompt_0_ids.npy");
  mlxforge::GenerateResult r =
      greedy_generate(model, prompt, /*max_tokens=*/5, model.config().eos_token_ids);
  CHECK(r.tokens.size() == 5);
  CHECK_FALSE(r.hit_eos);
}

TEST_CASE("greedy_generate reports per-token logprobs aligned with the tokens") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> prompt = load_token_ids("prompt_0_ids.npy");

  mlxforge::GenerateResult r = mlxforge::greedy_generate(
      model, prompt, /*max_tokens=*/8, model.config().eos_token_ids, {}, /*top_logprobs=*/3);

  // One logprob record per emitted token, each aligned with its id.
  REQUIRE(r.token_logprobs.size() == r.tokens.size());
  for (size_t i = 0; i < r.tokens.size(); ++i) {
    const mlxforge::TokenLogprob& lp = r.token_logprobs[i];
    CHECK(lp.id == r.tokens[i]);
    CHECK(lp.logprob <= 0.0f);
    REQUIRE(lp.top.size() == 3);
    // Greedy: the chosen token is the most likely, so it heads the alternatives.
    CHECK(lp.top.front().first == lp.id);
    CHECK(lp.top.front().second == doctest::Approx(lp.logprob));
    for (size_t k = 1; k < lp.top.size(); ++k) CHECK(lp.top[k - 1].second >= lp.top[k].second);
  }

  // Off by default: no logprob work, no records.
  mlxforge::GenerateResult plain =
      mlxforge::greedy_generate(model, prompt, /*max_tokens=*/4, model.config().eos_token_ids);
  CHECK(plain.token_logprobs.empty());
}

TEST_CASE("loop terminates on EOS") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> prompt = load_token_ids("prompt_0_ids.npy");
  const int first = load_token_ids("greedy_tokens.npy")[0];

  // Make the very first generated token an EOS -> stop immediately, emit nothing.
  mlxforge::GenerateResult r = greedy_generate(model, prompt, /*max_tokens=*/20, {first});
  CHECK(r.tokens.empty());
  CHECK(r.hit_eos);
}
