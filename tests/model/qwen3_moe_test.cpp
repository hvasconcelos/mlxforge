// Qwen3 MoE (sparse expert-routing) golden-reference checks: the stacked expert
// weights load, the sparse MoE block (layer 0) matches in isolation, decoder
// block 0 matches end-to-end, and the full-forward first-token argmax + greedy
// stream match the fixtures dumped by `reference/dump_ref.py --model qwen3_moe`.
// Self-skips unless both the model (MLXFORGE_MODEL_DIR_QWEN3_MOE — the large
// 30B-A3B-4bit checkpoint) and its committed fixtures are present.
#include <doctest/doctest.h>

#include <vector>

#include "core/config.h"
#include "mlx/ops.h"
#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;

namespace {
bool qwen3_moe_fixtures_present() {
  return std::ifstream(qwen3_moe_ref_path("manifest.json")).good();
}
bool qwen3_moe_ready() { return qwen3_moe_model_available() && qwen3_moe_fixtures_present(); }
}  // namespace

TEST_CASE("Qwen3 MoE: stacked expert + router weights load") {
  if (!qwen3_moe_ready()) {
    MESSAGE("Qwen3 MoE model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3MoeModel& model = shared_qwen3_moe_model();
  // The router gate and the per-layer stacked SwitchLinear experts must be present
  // (the load-time sanitize stacks raw per-expert tensors; mlx repos ship stacked).
  CHECK(model.config().num_experts > 0);
  CHECK(model.weights().has("model.layers.0.mlp.gate.weight"));
  CHECK(model.weights().has("model.layers.0.mlp.switch_mlp.gate_proj.weight"));
  CHECK(model.weights().has("model.layers.0.mlp.switch_mlp.up_proj.weight"));
  CHECK(model.weights().has("model.layers.0.mlp.switch_mlp.down_proj.weight"));
  // The stacked expert weight has the expert axis as dim 0.
  CHECK(model.weights().at("model.layers.0.mlp.switch_mlp.gate_proj.weight").shape()[0] ==
        model.config().num_experts);
}

TEST_CASE("Qwen3 MoE: sparse MoE block (layer 0) matches the reference") {
  if (!qwen3_moe_ready()) {
    MESSAGE("Qwen3 MoE model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3MoeModel& model = shared_qwen3_moe_model();
  REQUIRE(model.config().is_moe_layer(0));

  std::vector<int> ids = load_qwen3_moe_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  // Reconstruct the MoE input the same way the decoder block does: the
  // post-attention residual, then post_attention_layernorm. This is exactly the
  // tensor reference/dump_ref.py feeds to the sparse block.
  mx::array h = mx::add(emb, model.attention(emb, /*layer=*/0));
  mx::array moe_in =
      model.rms_norm(h, model.weights().at("model.layers.0.post_attention_layernorm.weight"));

  // The sparse expert block in isolation (router top-k + per-expert SwiGLU +
  // score-weighted sum) — the defining Qwen3 MoE delta.
  assert_close(model.moe_mlp(moe_in, /*layer=*/0), load_qwen3_moe_npy("moe_out0.npy"));
}

TEST_CASE("Qwen3 MoE: decoder block 0 output matches the reference") {
  if (!qwen3_moe_ready()) {
    MESSAGE("Qwen3 MoE model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3MoeModel& model = shared_qwen3_moe_model();
  std::vector<int> ids = load_qwen3_moe_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  assert_close(model.decoder_block(emb, /*layer=*/0), load_qwen3_moe_npy("block0.npy"));
}

TEST_CASE("Qwen3 MoE: full forward first-token argmax + greedy stream match") {
  if (!qwen3_moe_ready()) {
    MESSAGE("Qwen3 MoE model/fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3MoeModel& model = shared_qwen3_moe_model();
  std::vector<int> ids = load_qwen3_moe_token_ids("prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, T}, mx::int32);

  mx::array logits = model.forward(tokens);  // (1, T, vocab)
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
  // 4-bit, 48-layer model: raw logits drift more in fp16, so the exact argmax /
  // greedy stream below are the real correctness gate; this is a sanity bound.
  assert_close(last, load_qwen3_moe_npy("logits_last.npy"), /*rtol=*/5e-2f, /*atol=*/5e-2f);

  std::vector<int> argmax = load_qwen3_moe_token_ids("argmax.npy");
  mx::array got = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
  mx::eval(got);
  CHECK(got.data<int32_t>()[0] == argmax[0]);

  // Greedy continuation (full-recompute, no cache) must reproduce the mlx-lm
  // reference token-for-token — the end-to-end gate on the sparse forward pass.
  std::vector<int> expected = load_qwen3_moe_token_ids("greedy_tokens.npy");
  std::vector<int> seq = ids;
  std::vector<int> greedy;
  for (size_t i = 0; i < expected.size(); ++i) {
    mx::array cur(seq.data(), {1, static_cast<int>(seq.size())}, mx::int32);
    mx::array lg = model.forward(cur);
    const int L = static_cast<int>(seq.size());
    mx::array lastlg = mx::reshape(mx::slice(lg, {0, L - 1, 0}, {1, L, vocab}), {1, vocab});
    mx::array nxt = mx::astype(mx::argmax(lastlg, /*axis=*/-1), mx::int32);
    mx::eval(nxt);
    const int tok = nxt.data<int32_t>()[0];
    greedy.push_back(tok);
    seq.push_back(tok);
  }
  assert_tokens_equal(greedy, expected);
}
