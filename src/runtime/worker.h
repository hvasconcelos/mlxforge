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
#include "model/decoder_model.h"
#include "runtime/metrics.h"
#include "scheduler/scheduler.h"

#include "mlx/array.h"

namespace mlxforge {

class Tokenizer;    // for per-token byte strings used by grammar masking
class VitEncoder;   // lazily built for multimodal requests (borrows model weights)

class Worker {
 public:
  using ModelFactory = std::function<std::unique_ptr<DecoderModel>()>;

  // `tok` (optional) supplies the per-token byte strings used for constrained
  // decoding; when null, grammar-constrained requests fall back to unconstrained.
  // Defined out-of-line (with the destructor) because the unique_ptr<VitEncoder>
  // member needs the complete type for cleanup.
  Worker(ModelFactory factory, Scheduler* scheduler, const Tokenizer* tok = nullptr);
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
  // first token).
  void admit(const std::vector<std::shared_ptr<Request>>& incoming);
  // One decode step over the whole batch: forward -> sample -> async_eval ->
  // push each row's token, marking finished rows.
  void decode_step();
  // Sample one token for `count` rows of `logits`, where logits row i belongs to
  // request reqs_[row_offset + i]. Applies each request's SamplingParams + penalty
  // history and advances its RNG key. Builds one graph (no eval) so the caller
  // keeps the single-async_eval-per-step invariant. Returns a (count,) int32 array.
  mx::array sample_rows(const mx::array& logits, int row_offset, int count);
  // Drop rows marked finished (filter the cache, compact the row vectors,
  // close their token queues).
  void evict_finished();

  // Handle a one-shot embedding request: forward_hidden -> pool -> normalize into
  // req.embedding_result, then close its token queue. Runs on the worker thread.
  void handle_embedding(Request& req);

  // Handle a one-shot multimodal (Qwen3-VL) request: decode the image, run the
  // ViT, render the chat prompt, and stream generated tokens into req.tokens
  // single-stream. Runs on the worker thread (it owns all MLX state). Errors if
  // the loaded model is not a vision-language model.
  void handle_multimodal(Request& req);

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

  std::thread thread_;
};

}  // namespace mlxforge
