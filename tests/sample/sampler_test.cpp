// MLXFORGE-014: sampling ops on synthetic logits (no model).
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "sample/sampler.h"

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;

namespace {
mx::array logits2d(const std::vector<std::vector<float>>& rows) {
  const int B = static_cast<int>(rows.size());
  const int V = static_cast<int>(rows[0].size());
  std::vector<float> flat;
  for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());
  return mx::array(flat.data(), {B, V}, mx::float32);
}
std::vector<int> ints(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
}
std::vector<float> floats(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(c);
  return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}
int count_finite(const std::vector<float>& v) {
  int n = 0;
  for (float x : v) n += std::isfinite(x) ? 1 : 0;
  return n;
}
}  // namespace

TEST_CASE("MLXFORGE-014: greedy returns the argmax index") {
  mx::array logits = logits2d({{1.0f, 3.0f, 2.0f}, {5.0f, 0.0f, -1.0f}});
  CHECK(ints(Sampler::greedy(logits)) == std::vector<int>{1, 0});
}

TEST_CASE("MLXFORGE-014: temperature 0 collapses to greedy") {
  mx::array logits = logits2d({{0.2f, 0.9f, 0.1f, 0.4f}});
  SamplingParams p;
  p.temperature = 0.0f;
  SampleResult r = Sampler::sample(logits, p, mx::random::key(0));
  CHECK(ints(r.tokens) == ints(Sampler::greedy(logits)));
}

TEST_CASE("MLXFORGE-014: top-k keeps exactly k candidates") {
  mx::array logits = logits2d({{5.0f, 1.0f, 4.0f, 2.0f, 3.0f}});  // distinct
  std::vector<float> f = floats(Sampler::apply_top_k(logits, 2));
  CHECK(count_finite(f) == 2);
  CHECK(std::isfinite(f[0]));        // 5
  CHECK(std::isfinite(f[2]));        // 4
  CHECK_FALSE(std::isfinite(f[1]));  // dropped
}

TEST_CASE("MLXFORGE-014: top-p keeps the smallest prefix whose mass >= p") {
  // probs = [0.5, 0.3, 0.15, 0.05]; p=0.7 keeps {0.5, 0.3} (cum 0.8 >= 0.7).
  std::vector<float> probs = {0.5f, 0.3f, 0.15f, 0.05f};
  std::vector<float> lg(probs.size());
  for (size_t i = 0; i < probs.size(); ++i) lg[i] = std::log(probs[i]);
  std::vector<float> f = floats(Sampler::apply_top_p(logits2d({lg}), 0.7f));
  CHECK(count_finite(f) == 2);
  CHECK(std::isfinite(f[0]));
  CHECK(std::isfinite(f[1]));
  CHECK_FALSE(std::isfinite(f[2]));
  CHECK_FALSE(std::isfinite(f[3]));
}

TEST_CASE("MLXFORGE-014: same seed reproduces, different seed diverges") {
  mx::array logits = mx::zeros({64, 16}, mx::float32);  // uniform
  SamplingParams p;  // temperature 1, no top-k/p
  std::vector<int> a1 = ints(Sampler::sample(logits, p, mx::random::key(42)).tokens);
  std::vector<int> a2 = ints(Sampler::sample(logits, p, mx::random::key(42)).tokens);
  std::vector<int> b = ints(Sampler::sample(logits, p, mx::random::key(7)).tokens);
  CHECK(a1 == a2);   // reproducible
  CHECK(a1 != b);    // different seed -> different draws (overwhelmingly likely)
}

TEST_CASE("MLXFORGE-014: sample returns batched tokens and logprobs") {
  mx::array logits = logits2d({{1.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 1.0f}});
  SampleResult r = Sampler::sample(logits, SamplingParams{}, mx::random::key(1));
  CHECK(r.tokens.shape() == mx::Shape{2});
  CHECK(r.logprobs.shape() == mx::Shape{2});
  for (float lp : floats(r.logprobs)) CHECK(lp <= 0.0f);  // log-probabilities
}
