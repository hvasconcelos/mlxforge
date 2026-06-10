#include "runtime/single_stream.h"

#include <algorithm>
#include <chrono>

#include "cache/kv_cache.h"
#include "sample/sampler.h"

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
std::vector<int> to_ints(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
}
std::vector<float> to_floats(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(c);
  return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}

// Greedy token id from the last position of (1, L, vocab) logits. When `lp` is
// non-null, also fill it with the chosen token's log-prob and (per `top_logprobs`)
// its top-K alternatives, reusing the sampler's greedy + logprob path.
int greedy_last(const mx::array& logits, int top_logprobs, TokenLogprob* lp) {
  const int L = logits.shape()[1];
  const int V = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, V}), {1, V});
  if (!lp) {
    mx::array tok = Sampler::greedy(last);
    mx::eval(tok);
    return tok.item<int>();
  }
  SamplingParams p;
  p.temperature = 0.0f;  // greedy
  p.top_logprobs = top_logprobs;
  SampleResult res = Sampler::sample(last, p, mx::random::key(0));
  // Read through to_ints/to_floats (which astype first): the chosen logprob is in
  // the model's fp16, so a raw item<float>() would misread the 2-byte value.
  const int id = to_ints(res.tokens)[0];
  lp->id = id;
  lp->logprob = to_floats(res.logprobs)[0];
  lp->top.clear();
  const std::vector<int> tids = to_ints(res.top_tokens);
  const std::vector<float> tlp = to_floats(res.top_logprobs);
  for (size_t k = 0; k < tids.size(); ++k) lp->top.emplace_back(tids[k], tlp[k]);
  return id;
}
}  // namespace

GenerateResult greedy_generate(const DecoderModel& model, const std::vector<int>& prompt_ids,
                               int max_tokens, const std::vector<int>& eos_ids,
                               const std::function<void(int)>& on_token, int top_logprobs) {
  auto is_eos = [&](int id) {
    return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
  };

  using Clock = std::chrono::steady_clock;
  auto ms_since = [](Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };

  GenerateResult result;
  KVCache cache(model.config().n_layers);
  mx::array prompt(prompt_ids.data(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);

  // `next_lp` carries the log-prob record for `next`, pushed when (and only when)
  // `next` is actually emitted (EOS is excluded, like its token id).
  const bool want_lp = top_logprobs >= 0;
  TokenLogprob next_lp;

  // Prefill + first sample: time to first token. greedy_last() eval()s, so the
  // elapsed wall time covers the actual GPU work, not just graph construction.
  const auto t_start = Clock::now();
  int next = greedy_last(model.forward(prompt, &cache), top_logprobs, want_lp ? &next_lp : nullptr);
  result.ttft_ms = ms_since(t_start);

  const auto t_decode_start = Clock::now();
  for (int i = 0; i < max_tokens; ++i) {
    if (is_eos(next)) {
      result.hit_eos = true;
      break;
    }
    result.tokens.push_back(next);
    if (want_lp) result.token_logprobs.push_back(next_lp);
    if (on_token) on_token(next);

    mx::array step(&next, {1, 1}, mx::int32);
    next = greedy_last(model.forward(step, &cache), top_logprobs, want_lp ? &next_lp : nullptr);
  }
  // The first token came from prefill; throughput here is the steady-state
  // decode rate over the tokens generated after it.
  result.decode_tokens = std::max<int>(0, static_cast<int>(result.tokens.size()) - 1);
  if (result.decode_tokens > 0) result.decode_ms = ms_since(t_decode_start);
  return result;
}

}  // namespace mlxforge
