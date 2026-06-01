// MLXFORGE-019: batch-size bucketing + proof that masked dummy rows do not affect
// real rows' outputs.
#include <doctest/doctest.h>

#include <vector>

#include "runtime/batching.h"
#include "support/model_fixture.h"
#include "support/reference.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
using namespace mlxforge::test;

TEST_CASE("MLXFORGE-019: next_bucket rounds up to {1,2,4,8,16,32} then multiples of 32") {
  CHECK(next_bucket(1) == 1);
  CHECK(next_bucket(2) == 2);
  CHECK(next_bucket(3) == 4);
  CHECK(next_bucket(5) == 8);
  CHECK(next_bucket(9) == 16);
  CHECK(next_bucket(17) == 32);
  CHECK(next_bucket(32) == 32);
  CHECK(next_bucket(33) == 64);
  CHECK(next_bucket(40) == 64);
}

TEST_CASE("MLXFORGE-019: masked dummy rows do not change real rows' next-token logits") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  LlamaModel& model = shared_model();
  std::vector<std::vector<int>> reals = {load_token_ids("prompt_0_ids.npy"),
                                         load_token_ids("prompt_1_ids.npy"),
                                         load_token_ids("prompt_2_ids.npy")};  // B=3

  // Real-only prefill: each row's next-token logits.
  PrefillResult real_only = prefill(model, reals);

  // Pad B=3 up to bucket 4 with a masked dummy row, then trim it back, and run a
  // dummy decode step in between -> real rows must be unchanged.
  PrefillResult padded = prefill(model, reals);
  padded.cache.pad_dummies(next_bucket(3) - 3);  // one dummy row
  CHECK(padded.cache.batch_size() == 4);
  // A decode step over the bucketed batch (dummy row fed a pad token).
  std::vector<int> feed = {1, 1, 1, 0};  // dummy is row 3
  mx::array inputs(feed.data(), {4, 1}, mx::int32);
  mx::array logits = model.forward(inputs, padded.cache);  // (4, 1, vocab)
  padded.cache.filter({0, 1, 2});  // drop the dummy
  CHECK(padded.cache.batch_size() == 3);

  // The real rows' decode logits must match a real-only run of the same step.
  std::vector<int> feed3 = {1, 1, 1};
  mx::array inputs3(feed3.data(), {3, 1}, mx::int32);
  mx::array logits3 = model.forward(inputs3, real_only.cache);

  mx::array real_rows = mx::slice(logits, {0, 0, 0}, {3, 1, logits.shape()[2]});
  assert_close(real_rows, logits3, /*rtol=*/2e-2f, /*atol=*/5e-2f);
}
