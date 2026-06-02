// Load a self-contained GGUF checkpoint (llama.cpp / Ollama single-file format).
//
// A GGUF file bundles the weights, the model hyperparameters, and the tokenizer
// in one file's metadata, so there is no config.json / tokenizer.json on disk.
// load_gguf_model parses all three via MLX's mx::load_gguf:
//   - weight tensors are remapped from the ggml naming convention
//     ("blk.0.attn_q.weight") to the canonical HF form
//     ("model.layers.0.self_attn.q_proj.weight") and tagged with their quant
//     params (GGUF Q4_0/Q4_1/Q8_0 stay quantized at group_size 32; K-quants and
//     F16/F32 arrive dense fp16);
//   - the ModelConfig is built from the "llama.*" metadata keys (llama-family
//     only — other architectures are rejected);
//   - the tokenizer's raw material (tokens / merges / token types / special ids)
//     is extracted so a BpeTokenizer can be rebuilt without a tokenizer.json.
//
// The llama3 RoPE rescaling is baked into a "rope_freqs.weight" tensor (the
// scaling params are absent from the metadata); its values are lifted into
// ModelConfig::rope_freq_factors and the tensor itself is dropped.
#pragma once

#include <string>
#include <vector>

#include "core/config.h"
#include "core/weights.h"

namespace mlxforge {

// Everything parsed from a GGUF file in a single pass. The tokenizer fields are
// the raw arrays needed to reconstruct a BpeTokenizer (see Tokenizer::from_gguf).
struct GgufModel {
  ModelConfig config;
  Weights weights;
  std::vector<std::string> tokens;       // tokenizer.ggml.tokens (id -> token)
  std::vector<std::string> merges;       // tokenizer.ggml.merges ("L R" per rank)
  std::vector<int> token_types;          // tokenizer.ggml.token_type (per id)
  int bos_id = -1;                        // tokenizer.ggml.bos_token_id
  int eos_id = -1;                        // tokenizer.ggml.eos_token_id
  std::string pre;                        // tokenizer.ggml.pre (e.g. "llama-bpe")
};

// Whether `spec` names a GGUF file (case-insensitive ".gguf" suffix).
bool is_gguf_path(const std::string& spec);

// Parse a GGUF file into config + weights + tokenizer material. Throws on a
// non-llama architecture or malformed/missing required metadata. Loads weight
// tensors (MLX arrays) so must run on the thread that will own the model.
GgufModel load_gguf_model(const std::string& gguf_path);

// Parse only the config + tokenizer material (no weights) from the GGUF
// metadata. Creates no MLX arrays, so it is safe to call on any thread — the
// server uses it on the main thread while the worker loads the weights.
GgufModel load_gguf_config_and_tokenizer(const std::string& gguf_path);

}  // namespace mlxforge
