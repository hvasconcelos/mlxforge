// The single GPU worker thread. It owns ALL MLX state — it LOADS the model on
// its own thread (MLX arrays are thread-bound) and is the only thread that calls
// mx::eval/async_eval. Other threads interact solely through the Scheduler and
// their own Request.
//
// Continuous batching: a persistent decode batch is kept in a single
// BatchKVCache. Each loop iteration admits waiting requests (prefill -> merge),
// runs exactly ONE async_eval decode step over the whole batch, pushes each
// row's token, and evicts finished/cancelled rows via filter.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cache/batch_kv_cache.h"
#include "cache/prefix_cache.h"
#include "model/decoder_model.h"
#include "runtime/metrics.h"
#include "scheduler/request.h"  // Request, TokenLogprob
#include "scheduler/scheduler.h"

#include "mlx/array.h"

namespace mlxforge {

class Tokenizer;    // for per-token byte strings used by grammar masking
class VitEncoder;   // lazily built for multimodal requests (borrows model weights)
class BlockStore;   // SSD spill tier for the prefix cache (cache/block_store.h)

class Worker {
 public:
  using ModelFactory = std::function<std::unique_ptr<DecoderModel>()>;

  // `tok` (optional) supplies the per-token byte strings used for constrained
  // decoding; when null, grammar-constrained requests fall back to unconstrained.
  // `kv_quant` selects the decode cache's storage (dense fp16 by default) and
  // `prefix` the prefix-cache setting; the Engine validates both against the
  // model before construction. Defined out-of-line (with the destructor)
  // because the unique_ptr<VitEncoder> member needs the complete type for
  // cleanup.
  Worker(ModelFactory factory, Scheduler* scheduler, const Tokenizer* tok = nullptr,
         KVQuantConfig kv_quant = {}, PrefixCacheConfig prefix = {});
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  void start();  // launch the GPU thread (loads the model, then loops)
  void stop();   // signal the scheduler to drain, then join

  // Number of batched decode steps executed (== async_eval calls). Far below the
  // total tokens generated under load proves one eval covers the whole batch.
  long decode_steps() const { return decode_steps_.load(); }

  // True once the model has finished loading on the worker thread (else 503).
  bool ready() const { return ready_.load(); }

  // Snapshot of live decode metrics for /health. Reads only atomics, so it is
  // safe to call from the HTTP threads (never touches the worker-owned batch).
  WorkerMetrics metrics() const;

 private:
  void run();  // the loop; the sole caller of MLX eval/async_eval

  // Prefill `incoming` and merge it into the decode batch (emitting each row's
  // first token). With the prefix cache on, requests whose prompt matches
  // pooled blocks are admitted one-by-one via prefill_with_prefix (seeded
  // cache + suffix-only prefill); the rest take the batched cold path.
  void admit(const std::vector<std::shared_ptr<Request>>& incoming);
  // Seal finished rows' PROMPT K/V into the prefix pool (called before the
  // cache rows are filtered away). Prompt-only: decode-produced K/V differs
  // from a recompute by fp16 accumulation order, so pooling it would break the
  // warm==cold token gate. Skips multimodal rows — their K/V embeds image
  // content and 3D positions that a token-id hash cannot identify.
  void harvest_finished();
  // Register freshly-admitted rows into the decode-batch state (already merged
  // into the cache) and sample each row's first token from `last_logits` (rows
  // aligned to the new tail). Shared by the text and multimodal admit paths.
  void register_rows(const std::vector<std::shared_ptr<Request>>& incoming,
                     const mx::array& last_logits);
  // One decode step over the whole batch: forward -> sample -> async_eval ->
  // push each row's token, marking finished rows.
  void decode_step();

  // Result of sampling the active batch in one graph: the chosen tokens, plus —
  // for the rows that requested log-probs (params.top_logprobs >= 0) — their
  // per-row log-prob arrays. All are built into the same graph so one async_eval
  // covers them; the parallel vectors are aligned with `lp_rows`.
  struct SampledRows {
    mx::array tokens;                   // (count,) int32
    std::vector<int> lp_rows;           // row-local indices (0..count-1) wanting logprobs
    std::vector<mx::array> lp_chosen;   // aligned: (1,) chosen-token log-prob
    std::vector<mx::array> lp_top_ids;  // aligned: (1,K) int32 alternatives
    std::vector<mx::array> lp_top_lp;   // aligned: (1,K) fp32 alternatives
  };
  // Sample one token for `count` rows of `logits`, where logits row i belongs to
  // request reqs_[row_offset + i]. Applies each request's SamplingParams + penalty
  // history and advances its RNG key. Builds one graph (no eval) so the caller
  // keeps the single-async_eval-per-step invariant.
  SampledRows sample_rows(const mx::array& logits, int row_offset, int count);
  // Async-eval a sampled batch (the ONE eval per step) and read it back: fills
  // `ids` with the chosen token id per local row, and — for logprob-enabled rows —
  // `row_lp`/`has_lp` (both sized `count`) with the reconstructed TokenLogprob.
  void finalize_sample(const SampledRows& s, int count, std::vector<int>& ids,
                       std::vector<TokenLogprob>& row_lp, std::vector<char>& has_lp);
  // Drop rows marked finished (filter the cache, compact the row vectors,
  // close their token queues).
  void evict_finished();

  // Handle a one-shot embedding request: forward_hidden -> pool -> normalize into
  // req.embedding_result, then close its token queue. Runs on the worker thread.
  void handle_embedding(Request& req);

  // Admit a multimodal (Qwen3-VL) request into the decode batch: decode the
  // image(s), run the ViT, render/validate the prompt, prefill single-stream, then
  // merge the prompt's K/V into the shared BatchKVCache so its (pure-text)
  // generated tokens decode batched alongside text rows (prefill-single,
  // decode-batched). Runs on the worker thread (it owns all MLX state). On error
  // (e.g. the loaded model is not a VL model) the request is failed, not batched.
  void admit_multimodal(const std::shared_ptr<Request>& req);

  // Constrained decoding helpers. ensure_token_bytes builds the id->output-bytes
  // table once (from the tokenizer); grammar_mask returns an additive (1, vocab)
  // fp32 mask (-inf on tokens the grammar forbids at its current state); advance
  // commits the chosen token's bytes to a row's grammar.
  void ensure_token_bytes(int vocab);
  mx::array grammar_mask(const Request& req, const JsonGrammar& g, int vocab);
  void advance_grammar(Request& req, int chosen_id);

  ModelFactory      factory_;
  Scheduler*        sched_;
  const Tokenizer*  tok_;  // for per-token bytes (grammar masking); may be null
  KVQuantConfig     kv_quant_;  // decode-cache storage (dense when bits == 0)
  PrefixCacheConfig prefix_cfg_;
  // SSD tier; declared before prefix_ so the pool's hooks (which reference it)
  // are destroyed first. Byte-only across threads; null when spill is off.
  std::unique_ptr<BlockStore> block_store_;
  // Worker-thread-only (holds MLX arrays); null when the feature is off.
  std::unique_ptr<PrefixCache> prefix_;
  std::vector<std::string> token_bytes_;  // id -> output bytes ("" for specials)
  bool token_bytes_built_ = false;
  std::unique_ptr<DecoderModel> model_;  // constructed and owned on the worker thread
  std::unique_ptr<VitEncoder> vit_;      // built on first multimodal request (VL models)

  // Decode-batch state (worker thread only). All vectors are row-aligned with
  // the cache's batch axis.
  std::unique_ptr<BatchKVCache> cache_;
  std::vector<std::shared_ptr<Request>> reqs_;
  std::vector<int> produced_;  // tokens emitted per row
  std::vector<int> feed_;      // next input token per row (host side)
  std::vector<char> finished_;  // row marked for eviction
  std::vector<std::vector<int>> history_;  // prompt+generated ids per row (penalties)
  std::vector<mx::array> rng_keys_;        // per-row RNG key, advanced each step

  std::atomic<long> decode_steps_{0};
  std::atomic<bool> ready_{false};

  // Live metrics (worker writes, HTTP threads read via metrics()). active_/peak_
  // track batch occupancy; the *_sum_ accumulators feed the computed averages.
  std::atomic<int> active_batch_{0};
  std::atomic<int> peak_batch_{0};
  std::atomic<long> requests_completed_{0};
  std::atomic<long long> prompt_tokens_total_{0};
  std::atomic<long long> completion_tokens_total_{0};
  std::atomic<long long> ttft_us_sum_{0};
  std::atomic<long long> gen_us_sum_{0};
  std::atomic<long long> request_us_sum_{0};
  // Prefix-cache counters (worker writes after each admit/harvest).
  std::atomic<long> prefix_hits_{0};
  std::atomic<long long> prefix_tokens_reused_{0};
  std::atomic<long long> prefix_pool_bytes_{0};
  std::atomic<long> prefix_pool_blocks_{0};
  std::atomic<long> spill_writes_{0};  // blocks queued to the SSD tier
  std::atomic<long> spill_reads_{0};   // blocks revived from the SSD tier

  std::thread thread_;
};

}  // namespace mlxforge
