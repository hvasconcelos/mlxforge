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
  float min_p = 0.0f;        // <= 0 => disabled (keep prob >= min_p * max_prob)
  float repetition_penalty = 1.0f;  // 1 => disabled (divides/multiplies seen logits)
  float frequency_penalty = 0.0f;   // 0 => disabled (subtracts penalty * count)
  float presence_penalty = 0.0f;    // 0 => disabled (subtracts penalty if seen)
  uint64_t seed = 0;

  // Per-token log-probability reporting (OpenAI logprobs/top_logprobs). -1 = off
  // (no logprob work); 0 = report the chosen token's logprob only; N > 0 = also
  // report the N most-likely alternatives. Off by default so the hot path is
  // untouched unless a consumer asks for it.
  int top_logprobs = -1;

  // Whether any penalty is active (skips the per-row history machinery when not).
  bool has_penalties() const {
    return repetition_penalty != 1.0f || frequency_penalty != 0.0f ||
           presence_penalty != 0.0f;
  }
};

struct SampleResult {
  mx::array tokens;    // (B,) int32
  mx::array logprobs;  // (B,) fp32 — log-prob of each chosen token
  // Top-N alternatives, when params.top_logprobs > 0 (else (B, 0) placeholders).
  // Both are (B, K), aligned column-wise and ordered by descending log-prob.
  mx::array top_tokens;    // (B, K) int32 — the K most-likely token ids
  mx::array top_logprobs;  // (B, K) fp32 — their log-probs
};

class Sampler {
 public:
  // Sample one token per row from batched logits (B, vocab). `key` is an MLX
  // random key (random::key(seed)); reused for reproducibility.
  static SampleResult sample(const mx::array& logits, const SamplingParams& params,
                             const mx::array& key);

  // As above, but penalize tokens in `history` (a (B, N) int32 array of prior
  // token ids — typically one row, prompt+generated) before sampling. Used by
  // the worker so repetition/frequency/presence penalties see what was emitted.
  static SampleResult sample(const mx::array& logits, const SamplingParams& params,
                             const mx::array& key, const mx::array& history);

  // Greedy argmax over the vocab axis: (B, vocab) -> (B,) int32.
  static mx::array greedy(const mx::array& logits);

  // Filtering helpers (graph ops): return logits with removed entries set to
  // -inf. top_k keeps the k largest per row; top_p keeps the smallest prefix
  // (by descending prob) whose cumulative mass reaches p; min_p keeps tokens
  // whose prob is at least min_p times the per-row max prob.
  static mx::array apply_top_k(const mx::array& logits, int k);
  static mx::array apply_top_p(const mx::array& logits, float p);
  static mx::array apply_min_p(const mx::array& logits, float min_p);

  // Penalty helpers (graph ops). `history` is a (B, N) int32 array of prior
  // token ids; logits is (B, vocab). repetition divides/multiplies the logit of
  // each seen token; frequency/presence subtract penalty*count and penalty*(seen).
  static mx::array apply_repetition_penalty(const mx::array& logits,
                                            const mx::array& history, float penalty);
  static mx::array apply_frequency_presence(const mx::array& logits,
                                            const mx::array& history, float frequency,
                                            float presence);
};

}  // namespace mlxforge
