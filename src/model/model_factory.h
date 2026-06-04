// Concrete-model selection. Given a parsed config + loaded weights, pick the
// DecoderModel subclass that matches the checkpoint and return it owned. The
// apps and the worker use this rather than naming a subclass directly.
#pragma once

#include <memory>

#include "core/config.h"
#include "core/weights.h"
#include "model/decoder_model.h"

namespace mlxforge {

// Dispatch on the checkpoint's architecture markers:
//   - num_experts > 0          -> Qwen3MoeModel (sparse MoE; also has QK-Norm)
//   - has q_norm weights        -> Qwen3Model (dense + QK-Norm)
//   - otherwise                 -> LlamaModel (plain dense)
std::unique_ptr<DecoderModel> create_model(ModelConfig config, Weights weights);

}  // namespace mlxforge
