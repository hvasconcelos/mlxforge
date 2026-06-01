#include "runtime/single_stream.h"

#include <algorithm>

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

GenerateResult greedy_generate(const LlamaModel& model, const std::vector<int>& prompt_ids,
                               int max_tokens, const std::vector<int>& eos_ids,
                               const std::function<void(int)>& on_token) {
  auto is_eos = [&](int id) {
    return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
  };

  KVCache cache(model.config().n_layers);
  mx::array prompt(prompt_ids.data(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  int next = greedy_last(model.forward(prompt, &cache));

  GenerateResult result;
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
  return result;
}

}  // namespace mlxforge
