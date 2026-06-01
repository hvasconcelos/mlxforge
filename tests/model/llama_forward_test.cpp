// MLXFORGE-008 (MILESTONE): full forward pass to logits. The hard numerical part is
// done once argmax matches mlx-lm exactly.
#include <doctest/doctest.h>

#include <vector>

#include "support/model_fixture.h"
#include "support/reference.h"

#include "mlx/ops.h"

using namespace mlxforge::test;

TEST_CASE("MLXFORGE-008: full forward logits + first-token argmax match the reference") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_model();

  std::vector<int> ids = load_token_ids("prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, T}, mx::int32);

  mx::array logits = model.forward(tokens);  // (1, T, vocab)
  const int vocab = logits.shape()[2];

  // Last-position logits (1, vocab).
  mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
  assert_close(last, load_npy("logits_last.npy"));

  // Exact greedy-token match (Phase 4 done criterion).
  std::vector<int> argmax = load_token_ids("argmax.npy");
  mx::array got = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
  mx::eval(got);
  CHECK(got.data<int32_t>()[0] == argmax[0]);
}
