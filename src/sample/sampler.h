// Sampling implemented entirely as MLX graph ops (logits stay on the GPU —
// pulling them to the CPU would break the decode pipeline). Greedy is the
// deterministic baseline (matches the golden reference); temperature / top-k /
// top-p layer on top of it.
#pragma once

#include <cstdint>

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

struct SamplingParams {
  float temperature = 1.0f;  // <= 0 => greedy (argmax)
  int top_k = 0;             // <= 0 => disabled
  float top_p = 1.0f;        // >= 1 => disabled
  uint64_t seed = 0;
};

struct SampleResult {
  mx::array tokens;    // (B,) int32
  mx::array logprobs;  // (B,) fp32 — log-prob of each chosen token
};

class Sampler {
 public:
  // Sample one token per row from batched logits (B, vocab). `key` is an MLX
  // random key (random::key(seed)); reused for reproducibility.
  static SampleResult sample(const mx::array& logits, const SamplingParams& params,
                             const mx::array& key);

  // Greedy argmax over the vocab axis: (B, vocab) -> (B,) int32.
  static mx::array greedy(const mx::array& logits);

  // Filtering helpers (graph ops): return logits with removed entries set to
  // -inf. top_k keeps the k largest per row; top_p keeps the smallest prefix
  // (by descending prob) whose cumulative mass reaches p.
  static mx::array apply_top_k(const mx::array& logits, int k);
  static mx::array apply_top_p(const mx::array& logits, float p);
};

}  // namespace mlxforge
