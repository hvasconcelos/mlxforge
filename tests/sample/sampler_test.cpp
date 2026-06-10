// sampling ops on synthetic logits (no model).
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

TEST_CASE("greedy returns the argmax index") {
  mx::array logits = logits2d({{1.0f, 3.0f, 2.0f}, {5.0f, 0.0f, -1.0f}});
  CHECK(ints(Sampler::greedy(logits)) == std::vector<int>{1, 0});
}

TEST_CASE("temperature 0 collapses to greedy") {
  mx::array logits = logits2d({{0.2f, 0.9f, 0.1f, 0.4f}});
  SamplingParams p;
  p.temperature = 0.0f;
  SampleResult r = Sampler::sample(logits, p, mx::random::key(0));
  CHECK(ints(r.tokens) == ints(Sampler::greedy(logits)));
}

TEST_CASE("top-k keeps exactly k candidates") {
  mx::array logits = logits2d({{5.0f, 1.0f, 4.0f, 2.0f, 3.0f}});  // distinct
  std::vector<float> f = floats(Sampler::apply_top_k(logits, 2));
  CHECK(count_finite(f) == 2);
  CHECK(std::isfinite(f[0]));        // 5
  CHECK(std::isfinite(f[2]));        // 4
  CHECK_FALSE(std::isfinite(f[1]));  // dropped
}

TEST_CASE("top-p keeps the smallest prefix whose mass >= p") {
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

TEST_CASE("min-p keeps tokens with prob >= min_p * max_prob") {
  // probs = [0.5, 0.3, 0.15, 0.05]; min_p=0.4 keeps prob >= 0.2 => {0.5, 0.3}.
  std::vector<float> probs = {0.5f, 0.3f, 0.15f, 0.05f};
  std::vector<float> lg(probs.size());
  for (size_t i = 0; i < probs.size(); ++i) lg[i] = std::log(probs[i]);
  std::vector<float> f = floats(Sampler::apply_min_p(logits2d({lg}), 0.4f));
  CHECK(count_finite(f) == 2);
  CHECK(std::isfinite(f[0]));
  CHECK(std::isfinite(f[1]));
  CHECK_FALSE(std::isfinite(f[2]));
  CHECK_FALSE(std::isfinite(f[3]));
}

TEST_CASE("repetition penalty pushes a seen token below an unseen one") {
  // Token 0 leads but is in the history; a strong penalty should drop it under 1.
  mx::array logits = logits2d({{2.0f, 1.0f, 0.5f}});
  mx::array history = mx::array(std::vector<int>{0}.data(), {1, 1}, mx::int32);
  std::vector<float> f = floats(Sampler::apply_repetition_penalty(logits, history, 4.0f));
  CHECK(f[0] == doctest::Approx(0.5f));  // 2.0 / 4.0
  CHECK(f[1] == doctest::Approx(1.0f));  // untouched
  CHECK(f[0] < f[1]);                    // argmax moved off the repeated token
}

TEST_CASE("frequency penalty scales with count, presence is flat") {
  // History: token 1 twice, token 2 once.
  mx::array logits = logits2d({{0.0f, 0.0f, 0.0f, 0.0f}});
  mx::array history = mx::array(std::vector<int>{1, 1, 2}.data(), {1, 3}, mx::int32);

  std::vector<float> freq = floats(Sampler::apply_frequency_presence(logits, history, 0.5f, 0.0f));
  CHECK(freq[0] == doctest::Approx(0.0f));   // unseen
  CHECK(freq[1] == doctest::Approx(-1.0f));  // 2 * 0.5
  CHECK(freq[2] == doctest::Approx(-0.5f));  // 1 * 0.5
  CHECK(freq[3] == doctest::Approx(0.0f));

  std::vector<float> pres = floats(Sampler::apply_frequency_presence(logits, history, 0.0f, 0.7f));
  CHECK(pres[0] == doctest::Approx(0.0f));   // unseen
  CHECK(pres[1] == doctest::Approx(-0.7f));  // flat, regardless of count
  CHECK(pres[2] == doctest::Approx(-0.7f));
}

TEST_CASE("repetition penalty changes greedy choice via sample()") {
  mx::array logits = logits2d({{2.0f, 1.0f, 0.5f}});
  SamplingParams p;
  p.temperature = 0.0f;            // greedy
  p.repetition_penalty = 4.0f;     // demote the repeated token
  mx::array history = mx::array(std::vector<int>{0}.data(), {1, 1}, mx::int32);
  SampleResult r = Sampler::sample(logits, p, mx::random::key(0), history);
  CHECK(ints(r.tokens) == std::vector<int>{1});  // not 0, which greedy would pick
}

TEST_CASE("same seed reproduces, different seed diverges") {
  mx::array logits = mx::zeros({64, 16}, mx::float32);  // uniform
  SamplingParams p;  // temperature 1, no top-k/p
  std::vector<int> a1 = ints(Sampler::sample(logits, p, mx::random::key(42)).tokens);
  std::vector<int> a2 = ints(Sampler::sample(logits, p, mx::random::key(42)).tokens);
  std::vector<int> b = ints(Sampler::sample(logits, p, mx::random::key(7)).tokens);
  CHECK(a1 == a2);   // reproducible
  CHECK(a1 != b);    // different seed -> different draws (overwhelmingly likely)
}

TEST_CASE("sample returns batched tokens and logprobs") {
  mx::array logits = logits2d({{1.0f, 2.0f, 3.0f}, {3.0f, 2.0f, 1.0f}});
  SampleResult r = Sampler::sample(logits, SamplingParams{}, mx::random::key(1));
  CHECK(r.tokens.shape() == mx::Shape{2});
  CHECK(r.logprobs.shape() == mx::Shape{2});
  for (float lp : floats(r.logprobs)) CHECK(lp <= 0.0f);  // log-probabilities
}

TEST_CASE("top_logprobs off leaves the top arrays empty") {
  mx::array logits = logits2d({{1.0f, 2.0f, 3.0f}});
  SampleResult r = Sampler::sample(logits, SamplingParams{}, mx::random::key(0));  // default -1
  CHECK(r.top_tokens.shape() == mx::Shape{1, 0});
  CHECK(r.top_logprobs.shape() == mx::Shape{1, 0});
}

TEST_CASE("top_logprobs == 0 reports the chosen logprob but no alternatives") {
  mx::array logits = logits2d({{1.0f, 2.0f, 3.0f}});
  SamplingParams p;
  p.temperature = 0.0f;  // greedy
  p.top_logprobs = 0;
  SampleResult r = Sampler::sample(logits, p, mx::random::key(0));
  CHECK(ints(r.tokens) == std::vector<int>{2});  // argmax
  CHECK(floats(r.logprobs)[0] <= 0.0f);
  CHECK(r.top_tokens.shape() == mx::Shape{1, 0});
  CHECK(r.top_logprobs.shape() == mx::Shape{1, 0});
}

TEST_CASE("top_logprobs > 0 returns the top-k ids in descending logprob order") {
  // logits 1,4,2,3,0 -> ranked ids 1(4), 3(3), 2(2), 0(1), 4(0).
  mx::array logits = logits2d({{1.0f, 4.0f, 2.0f, 3.0f, 0.0f}});
  SamplingParams p;
  p.temperature = 0.0f;
  p.top_logprobs = 3;
  SampleResult r = Sampler::sample(logits, p, mx::random::key(0));
  CHECK(r.top_tokens.shape() == mx::Shape{1, 3});
  CHECK(r.top_logprobs.shape() == mx::Shape{1, 3});
  CHECK(ints(r.top_tokens) == std::vector<int>{1, 3, 2});

  std::vector<float> lp = floats(r.top_logprobs);
  CHECK(lp[0] >= lp[1]);
  CHECK(lp[1] >= lp[2]);
  for (float x : lp) CHECK(x <= 0.0f);
  // The chosen token is the argmax and its logprob equals the top alternative.
  CHECK(ints(r.tokens) == std::vector<int>{1});
  CHECK(floats(r.logprobs)[0] == doctest::Approx(lp[0]));
}

TEST_CASE("a full top-k recovers a normalized distribution") {
  mx::array logits = logits2d({{0.5f, 1.5f, -0.5f, 2.0f}});
  SamplingParams p;
  p.temperature = 0.0f;
  p.top_logprobs = 4;  // == vocab
  SampleResult r = Sampler::sample(logits, p, mx::random::key(0));
  float sum = 0.0f;
  for (float x : floats(r.top_logprobs)) sum += std::exp(x);
  CHECK(sum == doctest::Approx(1.0f).epsilon(0.01));
}

TEST_CASE("top_logprobs is reported under temperature sampling too") {
  mx::array logits = logits2d({{1.0f, 4.0f, 2.0f, 3.0f}});
  SamplingParams p;
  p.temperature = 0.8f;
  p.top_logprobs = 2;
  SampleResult r = Sampler::sample(logits, p, mx::random::key(3));
  CHECK(r.top_tokens.shape() == mx::Shape{1, 2});
  // Alternatives come from the pre-temperature distribution: highest-logit first.
  CHECK(ints(r.top_tokens) == std::vector<int>{1, 3});
}
