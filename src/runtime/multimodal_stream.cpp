#include "runtime/multimodal_stream.h"

#include <algorithm>
#include <chrono>

#include "cache/kv_cache.h"
#include "sample/sampler.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
// Greedy token id from a (1, vocab) logits row.
int greedy_row(const mx::array& row) {
  mx::array tok = Sampler::greedy(row);
  mx::eval(tok);
  return tok.item<int>();
}
// Greedy token id from the last position of (1, L, vocab) logits.
int greedy_last(const mx::array& logits) {
  const int L = logits.shape()[1], V = logits.shape()[2];
  return greedy_row(mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, V}), {1, V}));
}
}  // namespace

GenerateResult greedy_generate_multimodal(const Qwen3VLModel& model,
                                          const std::vector<int>& prompt_ids,
                                          const mx::array& image_features,
                                          const std::vector<mx::array>& deepstack,
                                          const mx::array& position_ids, int max_tokens,
                                          const std::vector<int>& eos_ids,
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

  // Prefill + first sample (time to first token). A generated token continues one
  // past the prompt's max M-RoPE position (the prompt's positions jump over the
  // image's spatial extent).
  const auto t_start = Clock::now();
  int next = greedy_last(model.prefill(prompt_ids, image_features, deepstack, position_ids, cache));
  int pos = static_cast<int>(mx::max(position_ids).item<int>()) + 1;
  result.ttft_ms = ms_since(t_start);

  const auto t_decode_start = Clock::now();
  for (int i = 0; i < max_tokens; ++i) {
    if (is_eos(next)) {
      result.hit_eos = true;
      break;
    }
    result.tokens.push_back(next);
    if (on_token) on_token(next);
    next = greedy_row(model.decode_step(next, pos++, cache));
  }
  result.decode_tokens = std::max<int>(0, static_cast<int>(result.tokens.size()) - 1);
  if (result.decode_tokens > 0) result.decode_ms = ms_since(t_decode_start);
  return result;
}

}  // namespace mlxforge
