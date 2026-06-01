// MLXFORGE-013 (HIGH RISK): batched decode of ragged prompts must produce IDENTICAL
// tokens to single-sequence runs, and each batched step is a single eval over
// the whole batch (never per-row / per-layer).
#include <doctest/doctest.h>

#include <algorithm>
#include <vector>

#include "cache/batch_kv_cache.h"
#include "cache/kv_cache.h"
#include "support/model_fixture.h"
#include "support/reference.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge::test;

namespace {
constexpr int kSteps = 8;  // decode steps to compare

// Greedy next-token of the last position of each batch row: (B, L, vocab) -> (B,).
mx::array greedy_last(const mx::array& logits) {
  const int L = logits.shape()[1];
  const int vocab = logits.shape()[2];
  mx::array last =
      mx::reshape(mx::slice(logits, {0, L - 1, 0}, {logits.shape()[0], L, vocab}),
                  {logits.shape()[0], vocab});
  return mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);  // (B,)
}

std::vector<int> to_vec(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  const int32_t* p = c.data<int32_t>();
  return std::vector<int>(p, p + c.size());
}

// Solo greedy run for one prompt via the single-sequence cache.
std::vector<int> solo_run(mlxforge::LlamaModel& model, const std::vector<int>& ids, int steps) {
  mlxforge::KVCache cache(model.config().n_layers);
  mx::array prompt(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  int next = to_vec(greedy_last(model.forward(prompt, &cache)))[0];
  std::vector<int> out = {next};
  for (int s = 1; s < steps; ++s) {
    mx::array step(&next, {1, 1}, mx::int32);
    next = to_vec(greedy_last(model.forward(step, &cache)))[0];
    out.push_back(next);
  }
  return out;
}
}  // namespace

TEST_CASE("MLXFORGE-013: batched ragged decode matches single-sequence runs") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();

  // Three fixed prompts of different lengths (ragged -> left-padding).
  std::vector<std::vector<int>> prompts = {
      load_token_ids("prompt_0_ids.npy"),
      load_token_ids("prompt_1_ids.npy"),
      load_token_ids("prompt_2_ids.npy"),
  };
  const int B = static_cast<int>(prompts.size());

  // Solo reference per prompt.
  std::vector<std::vector<int>> solo(B);
  for (int b = 0; b < B; ++b) solo[b] = solo_run(model, prompts[b], kSteps);

  // Build the left-padded (B, P_max) prompt block and per-row left_padding.
  int p_max = 0;
  for (auto& p : prompts) p_max = std::max(p_max, static_cast<int>(p.size()));
  std::vector<int> left_padding(B);
  std::vector<int> padded(B * p_max, 0);  // pad id 0 (masked out)
  for (int b = 0; b < B; ++b) {
    const int pad = p_max - static_cast<int>(prompts[b].size());
    left_padding[b] = pad;
    for (size_t j = 0; j < prompts[b].size(); ++j) padded[b * p_max + pad + j] = prompts[b][j];
  }

  mlxforge::BatchKVCache cache(model.config().n_layers, left_padding);
  mx::array tokens(padded.data(), {B, p_max}, mx::int32);

  // Prefill: one eval over the whole batch.
  mx::array next = greedy_last(model.forward(tokens, cache));
  mx::eval(next);  // single eval per step, whole batch
  std::vector<std::vector<int>> batched(B);
  for (int b = 0; b < B; ++b) batched[b].push_back(to_vec(next)[b]);

  // Decode steps: each is exactly one eval over the (B, 1) batch.
  for (int s = 1; s < kSteps; ++s) {
    mx::array step = mx::reshape(next, {B, 1});
    next = greedy_last(model.forward(step, cache));
    mx::eval(next);  // <-- the one and only eval for this decode step
    std::vector<int> row = to_vec(next);
    for (int b = 0; b < B; ++b) batched[b].push_back(row[b]);
  }

  for (int b = 0; b < B; ++b) {
    INFO("row " << b);
    assert_tokens_equal(batched[b], solo[b]);
  }
}
