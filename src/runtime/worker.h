// XLLM-016/018: the single GPU worker thread. It owns ALL MLX state — it LOADS
// the model on its own thread (MLX arrays are thread-bound) and is the only
// thread that calls mx::eval/async_eval. Other threads interact solely through
// the Scheduler and their own Request.
//
// Continuous batching (XLLM-018): a persistent decode batch is kept in a single
// BatchKVCache. Each loop iteration admits waiting requests (prefill -> merge),
// runs exactly ONE async_eval decode step over the whole batch, pushes each
// row's token, and evicts finished/cancelled rows via filter.
#pragma once

#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "cache/batch_kv_cache.h"
#include "model/llama.h"
#include "scheduler/scheduler.h"

namespace xllm {

class Worker {
 public:
  using ModelFactory = std::function<std::unique_ptr<LlamaModel>()>;

  Worker(ModelFactory factory, Scheduler* scheduler)
      : factory_(std::move(factory)), sched_(scheduler) {}
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  void start();  // launch the GPU thread (loads the model, then loops)
  void stop();   // signal the scheduler to drain, then join

 private:
  void run();  // the loop; the sole caller of MLX eval/async_eval

  // Prefill `incoming` and merge it into the decode batch (emitting each row's
  // first token).
  void admit(const std::vector<std::shared_ptr<Request>>& incoming);
  // One decode step over the whole batch: forward -> sample -> async_eval ->
  // push each row's token, marking finished rows.
  void decode_step();
  // Drop rows marked finished (filter the cache, compact the row vectors,
  // close their token queues).
  void evict_finished();

  ModelFactory factory_;
  Scheduler* sched_;
  std::unique_ptr<LlamaModel> model_;  // constructed and owned on the worker thread

  // Decode-batch state (worker thread only). All vectors are row-aligned with
  // the cache's batch axis.
  std::unique_ptr<BatchKVCache> cache_;
  std::vector<std::shared_ptr<Request>> reqs_;
  std::vector<int> produced_;  // tokens emitted per row
  std::vector<int> feed_;      // next input token per row (host side)
  std::vector<char> finished_;  // row marked for eviction

  std::thread thread_;
};

}  // namespace xllm
