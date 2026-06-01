// MLXFORGE-005: self-tests for the golden-reference compare harness.
#include "support/reference.h"

#include <doctest/doctest.h>

using namespace mlxforge::test;

TEST_CASE("load_npy reads a fixture into an MLX array") {
  mx::array emb = load_npy("embeddings.npy");
  CHECK(emb.shape() == mx::Shape{1, 6, 2048});
  CHECK(emb.dtype() == mx::float16);
}

TEST_CASE("compare_close passes when an array is compared to itself") {
  mx::array emb = load_npy("embeddings.npy");
  CHECK(compare_close(emb, emb).ok);
}

TEST_CASE("compare_close detects a divergence and reports the first index") {
  mx::array a = mx::array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  mx::array b = mx::array({1.0f, 2.0f, 9.0f, 4.0f}, {2, 2});  // differs at (1,0)
  CompareResult r = compare_close(a, b);
  CHECK_FALSE(r.ok);
  CHECK(r.message.find("(1, 0)") != std::string::npos);
}

TEST_CASE("compare_close tolerates small fp16-scale noise within rtol") {
  mx::array a = mx::array({100.0f, -50.0f, 0.0f});
  mx::array b = mx::array({100.5f, -50.2f, 0.005f});  // within 1e-2 rel + 1e-2 abs
  CHECK(compare_close(a, b).ok);
}

TEST_CASE("compare_close flags a shape mismatch") {
  mx::array a = mx::array({1.0f, 2.0f, 3.0f});
  mx::array b = mx::array({1.0f, 2.0f, 3.0f, 4.0f});
  CHECK_FALSE(compare_close(a, b).ok);
}

TEST_CASE("token-stream comparison: equal vs first mismatch") {
  std::vector<int> ref = load_token_ids("greedy_tokens.npy");
  CHECK(ref.size() == 20);
  CHECK(first_token_mismatch(ref, ref) == -1);

  std::vector<int> perturbed = ref;
  perturbed[5] = -1;
  CHECK(first_token_mismatch(perturbed, ref) == 5);

  std::vector<int> shorter(ref.begin(), ref.end() - 1);
  CHECK(first_token_mismatch(shorter, ref) == 19);  // length divergence
}
