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

mx::array Sampler::apply_min_p(const mx::array& logits, float min_p) {
  // Keep tokens whose prob is at least min_p times the most-likely token's prob.
  mx::array probs = mx::softmax(logits, /*axis=*/-1, /*precise=*/true);
  mx::array peak = mx::max(probs, /*axis=*/-1, /*keepdims=*/true);  // (B, 1)
  mx::array threshold = mx::multiply(peak, mx::array(min_p, probs.dtype()));
  return mx::where(mx::less(probs, threshold), mx::array(kNegInf, logits.dtype()), logits);
}

mx::array Sampler::apply_repetition_penalty(const mx::array& logits,
                                            const mx::array& history, float penalty) {
  // Gather the logit of each seen token, scale it (divide if >0 else multiply,
  // so a penalty > 1 always pushes it toward -inf), and scatter back. Repeated
  // ids in `history` scatter the same scaled value, so overwrite is idempotent.
  mx::array seen = mx::take_along_axis(logits, history, /*axis=*/-1);  // (B, N)
  mx::array pen = mx::array(penalty, logits.dtype());
  mx::array scaled = mx::where(mx::greater(seen, mx::array(0.0f, logits.dtype())),
                               mx::divide(seen, pen), mx::multiply(seen, pen));
  return mx::put_along_axis(logits, history, scaled, /*axis=*/-1);
}

mx::array Sampler::apply_frequency_presence(const mx::array& logits,
                                            const mx::array& history, float frequency,
                                            float presence) {
  // Per-vocab occurrence counts: scatter-add a 1 at each history id (along-axis
  // scatter, the additive analog of put_along_axis).
  mx::array base = mx::zeros(logits.shape(), logits.dtype());
  mx::array ones = mx::ones(history.shape(), logits.dtype());
  mx::array counts = mx::scatter_add_axis(base, history, ones, /*axis=*/-1);  // (B, vocab)
  mx::array seen = mx::astype(mx::greater(counts, mx::array(0.0f, logits.dtype())),
                              logits.dtype());
  mx::array adjusted = mx::subtract(
      logits, mx::multiply(counts, mx::array(frequency, logits.dtype())));
  return mx::subtract(adjusted, mx::multiply(seen, mx::array(presence, logits.dtype())));
}

namespace {
// Apply the configured penalties to logits given the seen-token history.
mx::array penalize(const mx::array& logits, const SamplingParams& params,
                   const mx::array& history) {
  mx::array out = logits;
  if (params.repetition_penalty != 1.0f)
    out = Sampler::apply_repetition_penalty(out, history, params.repetition_penalty);
  if (params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f)
    out = Sampler::apply_frequency_presence(out, history, params.frequency_penalty,
                                            params.presence_penalty);
  return out;
}
}  // namespace

SampleResult Sampler::sample(const mx::array& logits, const SamplingParams& params,
                             const mx::array& key) {
  return sample(logits, params, key, mx::zeros({logits.shape()[0], 0}, mx::int32));
}

SampleResult Sampler::sample(const mx::array& logits, const SamplingParams& params,
                             const mx::array& key, const mx::array& history) {
  // Penalties reshape the logit landscape and so apply to greedy too.
  mx::array adj = history.shape().back() > 0 ? penalize(logits, params, history) : logits;

  if (params.temperature <= 0.0f) {
    mx::array tokens = greedy(adj);
    return {tokens, gather_logprob(adj, tokens)};
  }

  mx::array scaled = mx::divide(adj, mx::array(params.temperature, adj.dtype()));
  if (params.top_k > 0) scaled = apply_top_k(scaled, params.top_k);
  if (params.top_p < 1.0f) scaled = apply_top_p(scaled, params.top_p);
  if (params.min_p > 0.0f) scaled = apply_min_p(scaled, params.min_p);

  mx::array tokens = mx::astype(mx::random::categorical(scaled, /*axis=*/-1, key), mx::int32);
  return {tokens, gather_logprob(scaled, tokens)};
}

}  // namespace mlxforge
