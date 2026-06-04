// embedding + RMSNorm + Q/K/V projections + RoPE, validated against
// the golden reference (fp16 rel ~1e-2). RoPE is the classic silent bug, so the
// post-RoPE Q/K tensors are asserted, not assumed.
#include <doctest/doctest.h>

#include <vector>

#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;

TEST_CASE("fast::rope(const array& offset, ...) overload exists on the pinned MLX") {
  CHECK(mlxforge::rope_array_offset_overload_available());
}

TEST_CASE("embedding + RMSNorm + RoPE'd Q/K/V match the reference") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();

  // RoPE frequencies (llama3 rescaling) must match before anything downstream.
  assert_close(model.rope_freqs(), load_npy("rope_freqs.npy"));

  std::vector<int> ids = load_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);  // (1, T)

  // Embedding lookup.
  mx::array emb = model.embed(tokens);
  assert_close(emb, load_npy("embeddings.npy"));

  // Post-RMSNorm (input_layernorm of layer 0).
  mx::array normed =
      model.rms_norm(emb, model.weights().at("model.layers.0.input_layernorm.weight"));
  assert_close(normed, load_npy("attn_norm0.npy"));

  // RoPE in isolation: apply to the reference pre-RoPE Q, expect reference Q.
  assert_close(model.apply_rope(load_npy("q_pre0.npy")), load_npy("q_rope0.npy"));

  // Post-RoPE Q/K and un-roped V, each (1, heads, T, head_dim). V has heavy
  // cancellation, so it is the strictest check that the projection is correct.
  mlxforge::DecoderModel::QKV qkv = model.attn_qkv(emb, /*layer=*/0);
  assert_close(qkv.q, load_npy("q_rope0.npy"));
  assert_close(qkv.k, load_npy("k_rope0.npy"));
  assert_close(qkv.v, load_npy("v0.npy"));
}
