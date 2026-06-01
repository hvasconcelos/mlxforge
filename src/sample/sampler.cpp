#include "sample/sampler.h"

#include <limits>

#include "mlx/ops.h"
#include "mlx/random.h"

namespace mlxforge {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// log-prob of each chosen token under softmax(logits): (B, vocab), (B,) -> (B,).
mx::array gather_logprob(const mx::array& logits, const mx::array& tokens) {
  const int batch = tokens.shape()[0];
  mx::array logp = mx::log(mx::softmax(logits, /*axis=*/-1, /*precise=*/true));
  mx::array idx = mx::reshape(tokens, {batch, 1});
  return mx::reshape(mx::take_along_axis(logp, idx, /*axis=*/-1), {batch});
}
}  // namespace

mx::array Sampler::greedy(const mx::array& logits) {
  return mx::astype(mx::argmax(logits, /*axis=*/-1), mx::int32);
}

mx::array Sampler::apply_top_k(const mx::array& logits, int k) {
  // k-th largest per row = min of the top-k values; drop anything below it.
  mx::array kth = mx::topk(logits, k, /*axis=*/-1);                  // (B, k)
  mx::array threshold = mx::min(kth, /*axis=*/-1, /*keepdims=*/true);  // (B, 1)
  return mx::where(mx::less(logits, threshold), mx::array(kNegInf, logits.dtype()), logits);
}

mx::array Sampler::apply_top_p(const mx::array& logits, float p) {
  // Sort descending (argsort of the negated logits is ascending of -logits).
  mx::array order = mx::argsort(mx::negative(logits), /*axis=*/-1);
  mx::array sorted_logits = mx::take_along_axis(logits, order, /*axis=*/-1);
  mx::array probs = mx::softmax(sorted_logits, /*axis=*/-1, /*precise=*/true);
  mx::array cum = mx::cumsum(probs, /*axis=*/-1, /*reverse=*/false, /*inclusive=*/true);
  // Keep token i while the mass strictly before it is < p (smallest prefix whose
  // cumulative mass reaches p).
  mx::array prior = mx::subtract(cum, probs);
  mx::array keep = mx::less(prior, mx::array(p, probs.dtype()));
  mx::array sorted_filtered =
      mx::where(keep, sorted_logits, mx::array(kNegInf, logits.dtype()));
  // Scatter back to the original vocab order.
  mx::array base = mx::full(logits.shape(), kNegInf, logits.dtype());
  return mx::put_along_axis(base, order, sorted_filtered, /*axis=*/-1);
}

SampleResult Sampler::sample(const mx::array& logits, const SamplingParams& params,
                             const mx::array& key) {
  if (params.temperature <= 0.0f) {
    mx::array tokens = greedy(logits);
    return {tokens, gather_logprob(logits, tokens)};
  }

  mx::array scaled = mx::divide(logits, mx::array(params.temperature, logits.dtype()));
  if (params.top_k > 0) scaled = apply_top_k(scaled, params.top_k);
  if (params.top_p < 1.0f) scaled = apply_top_p(scaled, params.top_p);

  mx::array tokens = mx::astype(mx::random::categorical(scaled, /*axis=*/-1, key), mx::int32);
  return {tokens, gather_logprob(scaled, tokens)};
}

}  // namespace mlxforge
