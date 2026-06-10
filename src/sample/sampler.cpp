#include "sample/sampler.h"

#include <algorithm>
#include <limits>

#include "mlx/ops.h"
#include "mlx/random.h"

namespace mlxforge {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Log-prob distribution over the vocab: log(softmax(logits)), (B, vocab).
mx::array log_probs(const mx::array& logits) {
  return mx::log(mx::softmax(logits, /*axis=*/-1, /*precise=*/true));
}

// log-prob of each chosen token, given a precomputed (B, vocab) log-prob array
// and the (B,) chosen ids -> (B,).
mx::array gather_logprob(const mx::array& logp, const mx::array& tokens) {
  const int batch = tokens.shape()[0];
  mx::array idx = mx::reshape(tokens, {batch, 1});
  return mx::reshape(mx::take_along_axis(logp, idx, /*axis=*/-1), {batch});
}

// Top-k (id, log-prob) per row in descending log-prob order, from a (B, vocab)
// log-prob array. Returns ((B, k) int32 ids, (B, k) fp32 log-probs). Pure graph
// ops — no eval, so the caller keeps the single-async_eval-per-step invariant.
std::pair<mx::array, mx::array> top_k_logprobs(const mx::array& logp, int k) {
  const int batch = logp.shape()[0];
  // argsort of the negated log-probs is ascending of -logp = descending of logp;
  // the first k columns are the k most-likely ids, already in order.
  mx::array order = mx::argsort(mx::negative(logp), /*axis=*/-1);  // (B, vocab)
  mx::array top_ids = mx::slice(order, {0, 0}, {batch, k});        // (B, k)
  mx::array top_lp = mx::take_along_axis(logp, top_ids, /*axis=*/-1);
  return {mx::astype(top_ids, mx::int32), top_lp};
}

// Empty (B, 0) placeholders for the top-k fields when no alternatives were asked.
SampleResult with_no_top(const mx::array& tokens, const mx::array& logprobs) {
  const int batch = tokens.shape()[0];
  return {tokens, logprobs, mx::zeros({batch, 0}, mx::int32),
          mx::zeros({batch, 0}, mx::float32)};
}

// Attach the chosen-token logprob and (when params.top_logprobs > 0) the top-k
// alternatives, both derived from `dist` (a coherent (B, vocab) log-prob array).
SampleResult attach_logprobs(const mx::array& tokens, const mx::array& dist,
                             const SamplingParams& params, int vocab) {
  mx::array chosen = gather_logprob(dist, tokens);
  const int k = std::min(params.top_logprobs, vocab);
  if (k <= 0) return with_no_top(tokens, chosen);
  std::pair<mx::array, mx::array> top = top_k_logprobs(dist, k);
  return {tokens, chosen, top.first, top.second};
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
  const int vocab = logits.shape().back();

  // Reported log-probs come from `adj` — the penalized, pre-temperature,
  // pre-filter distribution. That is a coherent softmax over the whole vocab (the
  // temperature-scaled/top-k-masked logits are not), so the chosen-token logprob
  // and its top-k alternatives are computed from the same array.
  auto finish = [&](const mx::array& tokens) {
    if (params.top_logprobs < 0) return with_no_top(tokens, gather_logprob(log_probs(adj), tokens));
    return attach_logprobs(tokens, log_probs(adj), params, vocab);
  };

  if (params.temperature <= 0.0f) return finish(greedy(adj));

  mx::array scaled = mx::divide(adj, mx::array(params.temperature, adj.dtype()));
  // Clamp top_k to the vocab size: mx::topk throws for k > axis length, and a
  // caller-supplied k larger than the vocab is just "keep everything" anyway.
  if (params.top_k > 0) scaled = apply_top_k(scaled, std::min(params.top_k, vocab));
  if (params.top_p < 1.0f) scaled = apply_top_p(scaled, params.top_p);
  if (params.min_p > 0.0f) scaled = apply_min_p(scaled, params.min_p);

  return finish(mx::astype(mx::random::categorical(scaled, /*axis=*/-1, key), mx::int32));
}

}  // namespace mlxforge
