#include "runtime/multimodal_stream.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <string>
#include <vector>

#include "cache/kv_cache.h"
#include "sample/sampler.h"
#include "vision/preprocess.h"

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

MultimodalPrefillInputs prepare_multimodal_prefill(const Qwen3VLModel& model, const VitEncoder& vit,
                                                   const std::vector<int>& prompt_ids,
                                                   const std::vector<mx::array>& images_rgb,
                                                   const PreprocessConfig* pcfg) {
  const ModelConfig& cfg = model.config();
  if (images_rgb.empty()) throw std::runtime_error("prepare_multimodal_prefill: no images");
  const PreprocessConfig pc = pcfg ? *pcfg : PreprocessConfig::from(*cfg.vision);

  // Smart-resize + preprocess + ViT-encode each image; collect per-image merged
  // features, per-layer DeepStack features, and the patch grids (in order).
  std::vector<std::array<int, 3>> grids;
  std::vector<mx::array> hidden_parts;
  std::vector<std::vector<mx::array>> deepstack_parts;  // [image][layer]
  int merged_total = 0;
  for (const auto& rgb : images_rgb) {
    Preprocessed pre = preprocess_image(rgb, pc);
    mx::array grid(pre.grid_thw.data(), {1, 3}, mx::int32);
    VitEncoder::Output v = vit.forward(pre.pixel_values, grid);
    grids.push_back(pre.grid_thw);
    hidden_parts.push_back(v.hidden);
    deepstack_parts.push_back(v.deepstack);
    merged_total += pre.grid_thw[0] * pre.grid_thw[1] * pre.grid_thw[2] / cfg.vision->merge_unit();
  }

  // The prompt's placeholder runs must total the merged-patch count, or the
  // feature scatter (merge_image_features) misaligns.
  const int pads =
      static_cast<int>(std::count(prompt_ids.begin(), prompt_ids.end(), cfg.image_token_id));
  if (pads != merged_total) {
    throw std::runtime_error("multimodal prompt has " + std::to_string(pads) +
                             " image placeholder(s) but the image(s) yield " +
                             std::to_string(merged_total));
  }

  // Concatenate features (and each DeepStack layer) across images, in order.
  auto cat = [](const std::vector<mx::array>& parts) {
    return parts.size() == 1 ? parts[0] : mx::concatenate(parts, /*axis=*/0);
  };
  mx::array features = cat(hidden_parts);
  std::vector<mx::array> deepstack;
  for (size_t layer = 0; layer < deepstack_parts[0].size(); ++layer) {
    std::vector<mx::array> layer_parts;
    for (const auto& per_image : deepstack_parts) layer_parts.push_back(per_image[layer]);
    deepstack.push_back(cat(layer_parts));
  }
  return {features, std::move(deepstack), mrope_position_ids(prompt_ids, grids, cfg)};
}

GenerateResult generate_multimodal(const Qwen3VLModel& model, const VitEncoder& vit,
                                   const std::vector<int>& prompt_ids,
                                   const std::vector<mx::array>& images_rgb, int max_tokens,
                                   const std::vector<int>& eos_ids,
                                   const std::function<void(int)>& on_token,
                                   const PreprocessConfig* pcfg) {
  MultimodalPrefillInputs in = prepare_multimodal_prefill(model, vit, prompt_ids, images_rgb, pcfg);
  return greedy_generate_multimodal(model, prompt_ids, in.features, in.deepstack, in.position_ids,
                                    max_tokens, eos_ids, on_token);
}

std::vector<int> render_multimodal_prompt(const Tokenizer& tokenizer, const Qwen3VLModel& model,
                                          const std::string& user_text,
                                          const std::vector<mx::array>& images_rgb,
                                          const PreprocessConfig* pcfg) {
  // Size each placeholder run from its image's dimensions (CPU math) and render a
  // one-user-message prompt with that many image blocks. Must use the SAME
  // preprocessing config the ViT prefill will use, or the placeholder count and
  // the merged-patch count disagree.
  const PreprocessConfig pc = pcfg ? *pcfg : PreprocessConfig::from(*model.config().vision);
  std::vector<int> counts;
  counts.reserve(images_rgb.size());
  for (const auto& rgb : images_rgb)
    counts.push_back(image_token_count(rgb.shape()[0], rgb.shape()[1], pc));

  Tokenizer::Message msg;
  msg.role = "user";
  msg.content = user_text;
  msg.image_token_counts = counts;
  return tokenizer.apply_chat_template({msg}, /*add_generation_prompt=*/true);
}

GenerateResult generate_from_images(const Qwen3VLModel& model, const VitEncoder& vit,
                                    const Tokenizer& tokenizer, const std::string& user_text,
                                    const std::vector<mx::array>& images_rgb, int max_tokens,
                                    const std::vector<int>& eos_ids,
                                    const std::function<void(int)>& on_token,
                                    const PreprocessConfig* pcfg) {
  // Single-turn convenience: template the prompt, then generate. The full chat
  // history is handled by the caller building prompt_ids directly for
  // generate_multimodal (the server path).
  const PreprocessConfig pc = pcfg ? *pcfg : PreprocessConfig::from(*model.config().vision);
  std::vector<int> ids = render_multimodal_prompt(tokenizer, model, user_text, images_rgb, &pc);
  return generate_multimodal(model, vit, ids, images_rgb, max_tokens, eos_ids, on_token, &pc);
}

MultimodalPrefill prefill_multimodal_batched(const Qwen3VLModel& model,
                                             const std::vector<int>& prompt_ids,
                                             const mx::array& features,
                                             const std::vector<mx::array>& deepstack,
                                             const mx::array& position_ids) {
  // Reuse the golden-gated single-stream prefill verbatim (it writes the prompt's
  // K/V into a single-sequence cache), then adopt that K/V into a batch-1
  // BatchKVCache. The decode RoPE position continues one past the prompt's max
  // M-RoPE position (the prompt's positions jump over the image's spatial extent),
  // which is well below the token count — so it is seeded explicitly, decoupled
  // from the cache's physical length.
  KVCache kv(model.config().n_layers);
  mx::array logits = model.prefill(prompt_ids, features, deepstack, position_ids, kv);
  const int seq = static_cast<int>(prompt_ids.size());
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, seq - 1, 0}, {1, seq, vocab}), {1, vocab});
  const int decode_offset = static_cast<int>(mx::max(position_ids).item<int>()) + 1;

  std::vector<std::pair<mx::array, mx::array>> kv_per_layer;
  kv_per_layer.reserve(kv.n_layers());
  for (int l = 0; l < kv.n_layers(); ++l) kv_per_layer.push_back(kv.fetch(l));
  BatchKVCache cache = BatchKVCache::from_single_sequence(std::move(kv_per_layer), seq,
                                                          decode_offset);
  cache.eval_state();  // materialize K/V + bookkeeping (detach from the prefill graph)
  mx::eval(last);
  return {std::move(cache), last};
}

GenerateResult greedy_generate_multimodal_batched(const Qwen3VLModel& model,
                                                  const std::vector<int>& prompt_ids,
                                                  const mx::array& features,
                                                  const std::vector<mx::array>& deepstack,
                                                  const mx::array& position_ids, int max_tokens,
                                                  const std::vector<int>& eos_ids,
                                                  const std::function<void(int)>& on_token) {
  auto is_eos = [&](int id) {
    return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
  };
  MultimodalPrefill pf =
      prefill_multimodal_batched(model, prompt_ids, features, deepstack, position_ids);

  GenerateResult result;
  int next = greedy_row(pf.last_logits);  // (1, vocab)
  for (int i = 0; i < max_tokens; ++i) {
    if (is_eos(next)) {
      result.hit_eos = true;
      break;
    }
    result.tokens.push_back(next);
    if (on_token) on_token(next);
    mx::array step(&next, {1, 1}, mx::int32);
    mx::array logits = model.forward(step, pf.cache);  // (1, 1, vocab), batched path
    next = greedy_row(mx::reshape(logits, {1, logits.shape()[2]}));
  }
  return result;
}

GenerateResult generate_from_image(const Qwen3VLModel& model, const VitEncoder& vit,
                                   const Tokenizer& tokenizer, const std::string& user_text,
                                   const mx::array& image_rgb, int max_tokens,
                                   const std::vector<int>& eos_ids,
                                   const std::function<void(int)>& on_token,
                                   const PreprocessConfig* pcfg) {
  return generate_from_images(model, vit, tokenizer, user_text, {image_rgb}, max_tokens, eos_ids,
                              on_token, pcfg);
}

}  // namespace mlxforge
