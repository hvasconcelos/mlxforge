// Single-stream greedy generation loop (the CLI's core).
// prefill -> sample -> append -> stream -> until EOS or max_tokens. Token ids
// are produced incrementally via the on_token callback (the CLI detokenizes and
// prints each piece).
#pragma once

#include <functional>
#include <vector>

#include "model/decoder_model.h"

namespace mlxforge {

struct GenerateResult {
  std::vector<int> tokens;  // generated token ids (EOS excluded)
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
GenerateResult greedy_generate(const DecoderModel& model, const std::vector<int>& prompt_ids,
                               int max_tokens, const std::vector<int>& eos_ids,
                               const std::function<void(int)>& on_token = {});

}  // namespace mlxforge
