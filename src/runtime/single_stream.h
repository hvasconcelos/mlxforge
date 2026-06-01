// MLXFORGE-015: single-stream greedy generation loop (the CLI's core).
// prefill -> sample -> append -> stream -> until EOS or max_tokens. Token ids
// are produced incrementally via the on_token callback (the CLI prints them; a
// real tokenizer arrives in MLXFORGE-021).
#pragma once

#include <functional>
#include <vector>

#include "model/llama.h"

namespace mlxforge {

struct GenerateResult {
  std::vector<int> tokens;  // generated token ids (EOS excluded)
  bool hit_eos = false;     // stopped because an EOS token was produced
};

// Greedy (argmax) single-stream generation. Calls `on_token(id)` for each
// emitted token. Stops when an EOS token would be produced or after max_tokens.
GenerateResult greedy_generate(const LlamaModel& model, const std::vector<int>& prompt_ids,
                               int max_tokens, const std::vector<int>& eos_ids,
                               const std::function<void(int)>& on_token = {});

}  // namespace mlxforge
