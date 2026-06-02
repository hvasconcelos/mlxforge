// a single decoder block (prefill) — attention (causal SDPA, GQA) +
// SwiGLU MLP with residuals — must match the reference block-0 output.
#include <doctest/doctest.h>

#include <vector>

#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;

TEST_CASE("decoder block 0 output matches the reference") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();

  std::vector<int> ids = load_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);

  mx::array block0 = model.decoder_block(emb, /*layer=*/0);  // (1, T, hidden)
  assert_close(block0, load_npy("block0.npy"));
}
