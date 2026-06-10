// Single-stream greedy generation loop (the CLI's core).
// prefill -> sample -> append -> stream -> until EOS or max_tokens. Token ids
// are produced incrementally via the on_token callback (the CLI detokenizes and
// prints each piece).
#pragma once

#include <functional>
#include <vector>

#include "model/decoder_model.h"
#include "scheduler/request.h"  // TokenLogprob

namespace mlxforge {

struct GenerateResult {
  std::vector<int> tokens;  // generated token ids (EOS excluded)
  // Per-token log-probs (OpenAI logprobs), aligned with `tokens`; populated only
  // when greedy_generate is asked for them (top_logprobs >= 0), else empty.
  std::vector<TokenLogprob> token_logprobs;
  bool hit_eos = false;     // stopped because an EOS token was produced
  double ttft_ms = 0.0;     // prefill + first sample (time to first token)
  double decode_ms = 0.0;   // wall time spent generating tokens after the first
  int decode_tokens = 0;    // tokens produced during decode (excludes the first)

  // Mean decode throughput, tokens/s. 0 if fewer than two tokens were produced.
  double decode_tokens_per_second() const {
    return decode_ms > 0.0 ? decode_tokens * 1000.0 / decode_ms : 0.0;
  }
};

// Greedy (argmax) single-stream generation. Calls `on_token(id)` for each
// emitted token. Stops when an EOS token would be produced or after max_tokens.
// `top_logprobs` mirrors SamplingParams: -1 = off (no logprob work); 0 = record
// each emitted token's own log-prob into result.token_logprobs; N > 0 = also
// record its N most-likely alternatives.
GenerateResult greedy_generate(const DecoderModel& model, const std::vector<int>& prompt_ids,
                               int max_tokens, const std::vector<int>& eos_ids,
                               const std::function<void(int)>& on_token = {},
                               int top_logprobs = -1);

}  // namespace mlxforge
