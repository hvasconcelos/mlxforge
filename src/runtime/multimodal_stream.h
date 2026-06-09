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
#include "runtime/single_stream.h"  // GenerateResult

namespace mlxforge {

namespace mx = mlx::core;

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

}  // namespace mlxforge
