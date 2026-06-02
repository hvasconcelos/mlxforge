// single-sequence KV cache — prove the prefill/decode split.
#include <doctest/doctest.h>

#include <vector>

#include "cache/kv_cache.h"
#include "support/model_fixture.h"
#include "support/reference.h"

#include "mlx/ops.h"

using namespace mlxforge::test;

namespace {
// Argmax token id over the last axis of a (1, vocab) row.
int argmax_row(const mx::array& row) {
  mx::array am = mx::astype(mx::argmax(row, -1), mx::int32);
  mx::eval(am);
  return am.data<int32_t>()[0];
}

// Argmax token id of the last position of a (1, L, vocab) logits tensor.
int last_argmax(const mx::array& logits) {
  const int L = logits.shape()[1];
  const int vocab = logits.shape()[2];
  return argmax_row(mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, vocab}), {1, vocab}));
}
}  // namespace

TEST_CASE("decode-with-cache equals a full recompute") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> ids = load_token_ids("prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());

  // Prefill the prompt, then decode the reference next token.
  mlxforge::KVCache cache(model.config().n_layers);
  mx::array prompt(ids.data(), {1, T}, mx::int32);
  mx::array prefill_logits = model.forward(prompt, &cache);
  CHECK(cache.offset() == T);
  int next = last_argmax(prefill_logits);

  mx::array step(&next, {1, 1}, mx::int32);
  mx::array cached_logits = model.forward(step, &cache);  // (1, 1, vocab)
  CHECK(cache.offset() == T + 1);

  // Full recompute over prompt + next, no cache.
  std::vector<int> full_ids = ids;
  full_ids.push_back(next);
  mx::array full(full_ids.data(), {1, T + 1}, mx::int32);
  mx::array recompute = model.forward(full);  // (1, T+1, vocab)
  mx::array recompute_last = mx::reshape(
      mx::slice(recompute, {0, T, 0}, {1, T + 1, recompute.shape()[2]}), {1, recompute.shape()[2]});
  mx::array cached_last = mx::reshape(cached_logits, {1, cached_logits.shape()[2]});
  // Both paths are valid fp16 computations of the same math; they differ only by
  // accumulation order, so use a logit-scale tolerance. The exact-token gate
  // below is the strict correctness check.
  assert_close(cached_last, recompute_last, /*rtol=*/2e-2f, /*atol=*/6e-2f);

  // The token actually chosen is identical either way.
  CHECK(argmax_row(cached_last) == argmax_row(recompute_last));
}

TEST_CASE("greedy stream with cache matches the reference exactly") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<int> ids = load_token_ids("prompt_0_ids.npy");

  mlxforge::KVCache cache(model.config().n_layers);
  mx::array prompt(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  int next = last_argmax(model.forward(prompt, &cache));

  std::vector<int> produced = {next};
  std::vector<int> ref = load_token_ids("greedy_tokens.npy");
  for (size_t i = 1; i < ref.size(); ++i) {
    mx::array step(&next, {1, 1}, mx::int32);
    next = last_argmax(model.forward(step, &cache));
    produced.push_back(next);
  }
  assert_tokens_equal(produced, ref);
}
