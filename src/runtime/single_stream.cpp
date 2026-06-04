#include "runtime/single_stream.h"

#include <algorithm>
#include <chrono>

#include "cache/kv_cache.h"
#include "sample/sampler.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
// Greedy token id from the last position of (1, L, vocab) logits.
int greedy_last(const mx::array& logits) {
  const int L = logits.shape()[1];
  const int V = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, V}), {1, V});
  mx::array tok = Sampler::greedy(last);
  mx::eval(tok);
  return tok.item<int>();
}
}  // namespace

GenerateResult greedy_generate(const DecoderModel& model, const std::vector<int>& prompt_ids,
                               int max_tokens, const std::vector<int>& eos_ids,
                               const std::function<void(int)>& on_token) {
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

  // Prefill + first sample: time to first token. greedy_last() eval()s, so the
  // elapsed wall time covers the actual GPU work, not just graph construction.
  const auto t_start = Clock::now();
  int next = greedy_last(model.forward(prompt, &cache));
  result.ttft_ms = ms_since(t_start);

  const auto t_decode_start = Clock::now();
  for (int i = 0; i < max_tokens; ++i) {
    if (is_eos(next)) {
      result.hit_eos = true;
      break;
    }
    result.tokens.push_back(next);
    if (on_token) on_token(next);

    mx::array step(&next, {1, 1}, mx::int32);
    next = greedy_last(model.forward(step, &cache));
  }
  // The first token came from prefill; throughput here is the steady-state
  // decode rate over the tokens generated after it.
  result.decode_tokens = std::max<int>(0, static_cast<int>(result.tokens.size()) - 1);
  if (result.decode_tokens > 0) result.decode_ms = ms_since(t_decode_start);
  return result;
}

}  // namespace mlxforge
