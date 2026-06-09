// Greedy generation for Qwen3-VL (image -> text), single-stream and batched.
//
// The multimodal sibling of greedy_generate(): prefill the prompt — with the ViT
// image features scattered into the image_pad rows, DeepStack injection, and 3D
// interleaved M-RoPE positions — into a KV cache, then decode text tokens
// incrementally (each a scalar M-RoPE position one past the prompt's max). The
// single-stream path is what the CLI uses; the lower half of this header packages
// the same prefill for the continuous-batching worker (prefill-single,
// decode-batched), where a VL prompt's K/V joins the shared decode batch and its
// pure-text generated tokens decode alongside text rows.
#pragma once

#include <functional>
#include <vector>

#include "mlx/array.h"

#include "cache/batch_kv_cache.h"
#include "model/qwen3_vl.h"
#include "model/vision/vit.h"
#include "runtime/single_stream.h"  // GenerateResult
#include "tokenizer/tokenizer.h"

namespace mlxforge {

namespace mx = mlx::core;

struct PreprocessConfig;  // vision/preprocess.h (only a pointer crosses this header)

// Greedy (argmax) multimodal generation. `image_features` / `deepstack` are the
// ViT encoder outputs; `position_ids` is (3, prompt_len) from mrope_position_ids.
// Calls `on_token(id)` for each emitted token; stops on EOS or max_tokens.
GenerateResult greedy_generate_multimodal(const Qwen3VLModel& model,
                                          const std::vector<int>& prompt_ids,
                                          const mx::array& image_features,
                                          const std::vector<mx::array>& deepstack,
                                          const mx::array& position_ids, int max_tokens,
                                          const std::vector<int>& eos_ids,
                                          const std::function<void(int)>& on_token = {});

// Generate from an already-templated multimodal prompt: `prompt_ids` must already
// contain the image placeholder runs (image_token_id × Nᵢ) in order — the caller
// renders the full chat history and sizes each Nᵢ from that image's dimensions.
// Preprocesses + ViT-encodes each of `images_rgb`, concatenates their features /
// DeepStack outputs in order, checks the total placeholder count matches, builds
// M-RoPE positions over all images, and greedily generates. `pcfg` overrides the
// preprocessing config; NULL uses the model's defaults.
GenerateResult generate_multimodal(const Qwen3VLModel& model, const VitEncoder& vit,
                                   const std::vector<int>& prompt_ids,
                                   const std::vector<mx::array>& images_rgb, int max_tokens,
                                   const std::vector<int>& eos_ids,
                                   const std::function<void(int)>& on_token = {},
                                   const PreprocessConfig* pcfg = nullptr);

// High-level single-turn orchestration: render a one-user-message ChatML prompt
// for `user_text` with a placeholder run per image (sized from each image's
// dimensions), then generate over all `images_rgb` (decoded H×W×3 uint8). Ties
// the whole vision pipeline together behind one call (the CLI / C-ABI image-to-
// text core). `pcfg` overrides the preprocessing config; NULL uses the defaults.
GenerateResult generate_from_images(const Qwen3VLModel& model, const VitEncoder& vit,
                                    const Tokenizer& tokenizer, const std::string& user_text,
                                    const std::vector<mx::array>& images_rgb, int max_tokens,
                                    const std::vector<int>& eos_ids,
                                    const std::function<void(int)>& on_token = {},
                                    const PreprocessConfig* pcfg = nullptr);

// Single-image convenience: generate_from_images with one image.
GenerateResult generate_from_image(const Qwen3VLModel& model, const VitEncoder& vit,
                                   const Tokenizer& tokenizer, const std::string& user_text,
                                   const mx::array& image_rgb, int max_tokens,
                                   const std::vector<int>& eos_ids,
                                   const std::function<void(int)>& on_token = {},
                                   const PreprocessConfig* pcfg = nullptr);

// --- Prefill-single, decode-batched (continuous-batching) multimodal serving ---
//
// The vision prompt is prefilled single-stream (the ViT can't batch ragged image
// grids, and M-RoPE positions are 3D and per-prompt), then handed to the text
// continuous-batching decode pool: generated tokens are pure text (t==h==w), so
// decode runs through the ordinary batched forward, mixing freely with text rows.

// ViT features + DeepStack + (3, seq) M-RoPE positions for a templated prompt.
// Factored out of generate_multimodal so the batched worker and the single-stream
// path build their prefill inputs the same way.
struct MultimodalPrefillInputs {
  mx::array features;                // (num_image_tokens, hidden) merged ViT features, in order
  std::vector<mx::array> deepstack;  // per-layer image features
  mx::array position_ids;            // (3, seq) int32 interleaved M-RoPE positions
};
MultimodalPrefillInputs prepare_multimodal_prefill(const Qwen3VLModel& model, const VitEncoder& vit,
                                                   const std::vector<int>& prompt_ids,
                                                   const std::vector<mx::array>& images_rgb,
                                                   const PreprocessConfig* pcfg = nullptr);

// A vision prompt prefilled and packaged for admission into the decode pool: a
// batch-1 BatchKVCache (decode RoPE offset seeded one past the prompt's max
// M-RoPE position) plus the prompt's last-token logits (1, vocab) to sample the
// first generated token from.
struct MultimodalPrefill {
  BatchKVCache cache;
  mx::array last_logits;  // (1, vocab)
};
MultimodalPrefill prefill_multimodal_batched(const Qwen3VLModel& model,
                                             const std::vector<int>& prompt_ids,
                                             const mx::array& features,
                                             const std::vector<mx::array>& deepstack,
                                             const mx::array& position_ids);

// Render a single-user-turn ChatML prompt for `user_text` with one image-pad run
// per image (sized from each image's dimensions). The templating half of
// generate_from_images, shared with the batched worker path.
std::vector<int> render_multimodal_prompt(const Tokenizer& tokenizer, const Qwen3VLModel& model,
                                          const std::string& user_text,
                                          const std::vector<mx::array>& images_rgb,
                                          const PreprocessConfig* pcfg = nullptr);

// Greedy generation through the BATCHED decode path for a single row: prefill the
// prompt, then decode via forward(BatchKVCache&). Proves the batched serving path
// is numerically identical to greedy_generate_multimodal (golden/QA seam).
GenerateResult greedy_generate_multimodal_batched(const Qwen3VLModel& model,
                                                  const std::vector<int>& prompt_ids,
                                                  const mx::array& features,
                                                  const std::vector<mx::array>& deepstack,
                                                  const mx::array& position_ids, int max_tokens,
                                                  const std::vector<int>& eos_ids,
                                                  const std::function<void(int)>& on_token = {});

}  // namespace mlxforge
