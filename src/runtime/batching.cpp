#include "runtime/batching.h"

#include <algorithm>

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

int next_bucket(int n) {
  for (int b : {1, 2, 4, 8, 16, 32}) {
    if (n <= b) return b;
  }
  return ((n + 31) / 32) * 32;  // beyond 32, round up to a multiple of 32
}

PrefillResult prefill(const DecoderModel& model, const std::vector<std::vector<int>>& prompts,
                      int step_size, int pad_id, KVQuantConfig kv_quant) {
  const int B = static_cast<int>(prompts.size());
  int p_max = 0;
  for (const auto& p : prompts) p_max = std::max(p_max, static_cast<int>(p.size()));

  // Left-pad each prompt to P_max; per-row left_padding = pad count.
  std::vector<int> left_padding(B);
  std::vector<int> padded(static_cast<size_t>(B) * p_max, pad_id);
  for (int b = 0; b < B; ++b) {
    const int pad = p_max - static_cast<int>(prompts[b].size());
    left_padding[b] = pad;
    for (size_t j = 0; j < prompts[b].size(); ++j) padded[b * p_max + pad + j] = prompts[b][j];
  }
  mx::array tokens(padded.data(), {B, p_max}, mx::int32);

  BatchKVCache cache(model.config().n_layers, left_padding, kv_quant);

  // Dedicated prefill forward, chunked to bound graph growth on long prompts.
  mx::array logits = mx::zeros({B, 1, model.config().vocab}, mx::float16);
  for (int c = 0; c < p_max; c += step_size) {
    const int n = std::min(step_size, p_max - c);
    mx::array chunk = mx::slice(tokens, {0, c}, {B, c + n});
    logits = model.forward(chunk, cache);
    cache.eval_state();  // eval(cache.state) at each chunk boundary
  }

  // Every row's last real token is at P_max-1 == the last column of the final
  // chunk's logits.
  const int n_last = logits.shape()[1];
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, n_last - 1, 0}, {B, n_last, vocab}),
                              {B, vocab});
  mx::eval(last);
  return {std::move(cache), last, std::move(left_padding)};
}

PrefillResult prefill_with_prefix(const DecoderModel& model, const std::vector<int>& prompt,
                                  const std::vector<std::shared_ptr<const KVBlock>>& blocks,
                                  int cached_len, int step_size, KVQuantConfig kv_quant) {
  const int n = static_cast<int>(prompt.size());
  BatchKVCache cache =
      BatchKVCache::from_prefix(model.config().n_layers, blocks, cached_len, kv_quant);
  cache.eval_state();  // materialize the seeded storage before the forward

  // Suffix prefill, chunked like the cold path. The cache's offset/idx already
  // sit at cached_len, so RoPE positions and the mask line up unchanged.
  const int suffix = n - cached_len;
  mx::array logits = mx::zeros({1, 1, model.config().vocab}, mx::float16);
  for (int c = 0; c < suffix; c += step_size) {
    const int m = std::min(step_size, suffix - c);
    mx::array chunk(prompt.data() + cached_len + c, {1, m}, mx::int32);
    logits = model.forward(chunk, cache);
    cache.eval_state();
  }

  const int n_last = logits.shape()[1];
  const int vocab = logits.shape()[2];
  mx::array last =
      mx::reshape(mx::slice(logits, {0, n_last - 1, 0}, {1, n_last, vocab}), {1, vocab});
  mx::eval(last);
  return {std::move(cache), last, std::vector<int>{0}};
}

}  // namespace mlxforge
