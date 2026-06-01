// MLXFORGE-017: prefill pass — left-pad, (chunked) dedicated forward, correct
// per-row offsets ready to merge into the decode cache.
#include <doctest/doctest.h>

#include <vector>

#include "cache/kv_cache.h"
#include "runtime/batching.h"
#include "support/model_fixture.h"
#include "support/reference.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge::test;

namespace {
std::vector<int> argmax_rows(const mx::array& logits2d) {  // (B, vocab) -> (B,)
  mx::array am = mx::contiguous(mx::astype(mx::argmax(logits2d, -1), mx::int32));
  mx::eval(am);
  return std::vector<int>(am.data<int32_t>(), am.data<int32_t>() + am.size());
}
std::vector<int> read_ints(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
}
// First next-token id from a single-sequence prefill.
int solo_first(mlxforge::LlamaModel& model, const std::vector<int>& ids) {
  mlxforge::KVCache cache(model.config().n_layers);
  mx::array prompt(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array logits = model.forward(prompt, &cache);
  const int T = static_cast<int>(ids.size());
  mx::array last =
      mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, logits.shape()[2]}), {1, logits.shape()[2]});
  return argmax_rows(last)[0];
}
}  // namespace

TEST_CASE("MLXFORGE-017: ragged prefill matches single-sequence first tokens + offsets") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<std::vector<int>> prompts = {load_token_ids("prompt_0_ids.npy"),
                                           load_token_ids("prompt_1_ids.npy"),
                                           load_token_ids("prompt_2_ids.npy")};

  mlxforge::PrefillResult pr = mlxforge::prefill(model, prompts);

  // Each row's next token equals its solo prefill next token.
  std::vector<int> batched = argmax_rows(pr.last_logits);
  for (size_t b = 0; b < prompts.size(); ++b) {
    INFO("row " << b);
    CHECK(batched[b] == solo_first(model, prompts[b]));
  }

  // Per-row offset == real token count (= prompt length); idx == P_max.
  CHECK(read_ints(pr.cache.offset()) ==
        std::vector<int>{(int)prompts[0].size(), (int)prompts[1].size(), (int)prompts[2].size()});
  int p_max = 0;
  for (auto& p : prompts) p_max = std::max<int>(p_max, p.size());
  CHECK(pr.cache.idx() == p_max);
}

TEST_CASE("MLXFORGE-017: chunked prefill equals a single-shot prefill") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  std::vector<std::vector<int>> prompts = {load_token_ids("prompt_2_ids.npy")};  // 12 tokens

  std::vector<int> one_shot = argmax_rows(mlxforge::prefill(model, prompts, /*step_size=*/4096).last_logits);
  std::vector<int> chunked = argmax_rows(mlxforge::prefill(model, prompts, /*step_size=*/4).last_logits);
  CHECK(one_shot == chunked);  // 3 chunks vs 1 -> identical next token
}
