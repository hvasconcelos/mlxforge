// Seeded (prefix-cache) prefill against the cold path on the real model: a
// cache built from harvested blocks + suffix-only prefill must reproduce the
// cold prefill's K/V and next-token choice. The harvested K/V is a bit-copy of
// the cold run's, so K/V compares close and the next token compares exact —
// the warm==cold equivalence is this feature's golden gate (the cold path is
// already gated against mlx-lm).
#include <doctest/doctest.h>

#include <vector>

#include "cache/prefix_cache.h"
#include "runtime/batching.h"
#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;
namespace mx = mlx::core;

namespace {

// Fixture prompts are tiny; concatenate them into a prefix-cache-sized prompt.
std::vector<int> long_prompt() {
  std::vector<int> p;
  for (const char* name : {"prompt_2_ids.npy", "prompt_0_ids.npy", "prompt_1_ids.npy",
                           "prompt_2_ids.npy", "prompt_0_ids.npy", "prompt_1_ids.npy"}) {
    std::vector<int> ids = load_token_ids(name);
    p.insert(p.end(), ids.begin(), ids.end());
  }
  return p;  // 46 tokens
}

int argmax_id(const mx::array& logits_row) {
  mx::array am = mx::argmax(logits_row, /*axis=*/-1);
  mx::eval(am);
  return static_cast<int>(am.item<uint32_t>());
}

}  // namespace

TEST_CASE("prefill_with_prefix matches the cold prefill (dense fp16)") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  const int n_layers = model.config().n_layers;
  const std::vector<int> prompt = long_prompt();
  const int len = static_cast<int>(prompt.size());

  // Cold prefill, then harvest its rows into a fresh pool (block 16 so the
  // short fixture prompt spans multiple blocks).
  mlxforge::PrefillResult cold = mlxforge::prefill(model, {prompt});
  mlxforge::PrefixCache pc({true, /*block_size=*/16, 1ull << 30, /*salt=*/42});
  pc.harvest(prompt, len, n_layers,
             [&](int l) { return cold.cache.fetch_row_components(l, 0, 0, len); });
  CHECK(pc.pool_blocks() == len / 16);

  mlxforge::PrefixCache::Match m = pc.match(prompt);
  REQUIRE(m.tokens == (len / 16) * 16);

  mlxforge::PrefillResult warm = mlxforge::prefill_with_prefix(model, prompt, m.blocks, m.tokens);

  // Same next token (exact), same cache content (fp16-close; the suffix is
  // recomputed in a different graph context, so raw-logit equality is not the
  // gate — see the decode-vs-recompute gotcha).
  CHECK(argmax_id(warm.last_logits) == argmax_id(cold.last_logits));
  CHECK(warm.cache.idx() == cold.cache.idx());
  for (int l = 0; l < n_layers; ++l) {
    auto [ck, cv] = cold.cache.fetch(l);
    auto [wk, wv] = warm.cache.fetch(l);
    mlxforge::test::assert_close(wk, ck);
    mlxforge::test::assert_close(wv, cv);
  }
}

TEST_CASE("prefill_with_prefix matches the cold prefill (quantized KV)") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();
  const mlxforge::KVQuantConfig qc{8, 64};
  const int n_layers = model.config().n_layers;
  const std::vector<int> prompt = long_prompt();
  const int len = static_cast<int>(prompt.size());

  mlxforge::PrefillResult cold =
      mlxforge::prefill(model, {prompt}, mlxforge::kPrefillStepSize, 0, qc);
  mlxforge::PrefixCache pc({true, 16, 1ull << 30, 42});
  pc.harvest(prompt, len, n_layers,
             [&](int l) { return cold.cache.fetch_row_components(l, 0, 0, len); });

  mlxforge::PrefixCache::Match m = pc.match(prompt);
  REQUIRE(m.tokens > 0);
  mlxforge::PrefillResult warm =
      mlxforge::prefill_with_prefix(model, prompt, m.blocks, m.tokens,
                                    mlxforge::kPrefillStepSize, qc);

  // Quantized matmuls are fusion-context-sensitive (see the kv-quant gates):
  // the recomputed suffix legitimately shifts within quantization noise, so the
  // gate is the choice (argmax) plus exact reuse of the cached region — the
  // pooled triplets must dequantize identically to the cold run's.
  CHECK(argmax_id(warm.last_logits) == argmax_id(cold.last_logits));
  auto prefix_of = [&](const mx::array& a) {
    const auto& s = a.shape();
    return mx::slice(a, {0, 0, 0, 0}, {s[0], s[1], m.tokens, s[3]});
  };
  for (int l = 0; l < n_layers; ++l) {
    auto [ck, cv] = cold.cache.fetch_dequantized(l);
    auto [wk, wv] = warm.cache.fetch_dequantized(l);
    mlxforge::test::assert_close(prefix_of(wk), prefix_of(ck));
    mlxforge::test::assert_close(prefix_of(wv), prefix_of(cv));
  }
}
