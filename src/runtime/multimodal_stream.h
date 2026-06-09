// Single-stream greedy generation for Qwen3-VL (image -> text).
//
// The multimodal sibling of greedy_generate(): prefill the prompt — with the ViT
// image features scattered into the image_pad rows, DeepStack injection, and 3D
// interleaved M-RoPE positions — into a KV cache, then decode text tokens
// incrementally (each a scalar M-RoPE position one past the prompt's max). The
// continuous-batching worker is still text-only; this is the single-stream path
// the CLI uses to run a vision-language prompt end to end.
#pragma once

#include <functional>
#include <vector>

#include "mlx/array.h"

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

// High-level single-image orchestration: smart-resize + preprocess `image_rgb`
// (decoded H×W×3 uint8), encode the ViT, render the ChatML prompt with the right
// number of image placeholders, build M-RoPE positions, and greedily generate.
// Ties the whole vision pipeline together behind one call (the CLI's image-to-
// text core). `pcfg` overrides the preprocessing config (resize bounds etc.);
// NULL uses the model's defaults.
GenerateResult generate_from_image(const Qwen3VLModel& model, const VitEncoder& vit,
                                   const Tokenizer& tokenizer, const std::string& user_text,
                                   const mx::array& image_rgb, int max_tokens,
                                   const std::vector<int>& eos_ids,
                                   const std::function<void(int)>& on_token = {},
                                   const PreprocessConfig* pcfg = nullptr);

}  // namespace mlxforge
