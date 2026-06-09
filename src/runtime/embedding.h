// Sentence embeddings from a decoder model: run the forward pass to the final
// hidden states (no LM head), pool over the sequence, and L2-normalize. This
// reuses the existing decoder + byte-level BPE tokenizer, so any LLaMA/Qwen
// checkpoint produces an embedding; an embedding-tuned checkpoint (e.g.
// Qwen3-Embedding) produces a good one.
#pragma once

#include <vector>

#include "model/decoder_model.h"

namespace mlxforge {

enum class Pooling : int {
  Mean = 0,  // average the hidden states over the sequence (robust default)
  Last = 1,  // the last token's hidden state (causal last-token pooling)
};

// Embed one already-tokenized prompt: forward_hidden -> pool -> (optional)
// L2-normalize. Returns a `hidden`-dimensional vector (unit-norm when
// `normalize`). Must run on the model's thread. For last-token pooling the
// caller is expected to have appended the model's EOS id (Qwen3-Embedding
// convention) so the pooled position is the sentence-final hidden state.
std::vector<float> embed_pooled(const DecoderModel& model, const std::vector<int>& ids,
                                Pooling pooling = Pooling::Mean, bool normalize = true);

}  // namespace mlxforge
