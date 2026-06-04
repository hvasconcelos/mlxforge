// The prefill pass. Waiting requests are left-padded to a common P_max and
// prefilled in a DEDICATED forward (a separate pass, then joined — not
// chunk-interleaved into the decode batch). Long prompts are chunked at
// prefill_step_size with eval(cache.state) at each boundary to bound graph
// growth. The result merges into the worker's decode cache.
#pragma once

#include <memory>
#include <vector>

#include "cache/batch_kv_cache.h"
#include "model/decoder_model.h"
#include "scheduler/request.h"

namespace mlxforge {

// Scheduler sizing knobs.
constexpr int kPrefillBatchSize = 8;
constexpr int kPrefillStepSize = 2048;

// Round an active batch size up to a fixed decode bucket so the forward graph
// shape recurs (avoiding a per-step regraph). Buckets are
// {1,2,4,8,16,32}; beyond 32, round up to a multiple of 32.
int next_bucket(int n);

struct PrefillResult {
  BatchKVCache cache;          // populated, left-padded, ready for decode
  mx::array last_logits;       // (B, vocab): next-token logits for each row
  std::vector<int> left_padding;
};

// Left-pad `prompts` to a common P_max and prefill them into a fresh
// BatchKVCache, chunking at `step_size`. Returns the cache plus the last-real-
// position logits (rows are left-padded, so every row's last token sits at
// P_max-1). `pad_id` fills the masked-out left padding.
PrefillResult prefill(const DecoderModel& model, const std::vector<std::vector<int>>& prompts,
                      int step_size = kPrefillStepSize, int pad_id = 0);

}  // namespace mlxforge
