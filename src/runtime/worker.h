// XLLM-016: the single GPU worker thread. It owns ALL MLX state — it LOADS the
// model on its own thread and is the only thread that calls mx::eval/async_eval.
// (MLX GPU arrays are tied to the thread that created them, so loading and
// every forward pass must run here.) Other threads interact solely through the
// Scheduler and their own Request.
//
// XLLM-016 drives requests one at a time (batch of 1) through prefill+decode;
// XLLM-017/018 evolve this into true continuous batching.
#pragma once

#include <functional>
#include <memory>
#include <thread>

#include "model/llama.h"
#include "scheduler/scheduler.h"

namespace xllm {

class Worker {
 public:
  // The factory is invoked ON the worker thread to build the model, so all of
  // its arrays belong to this thread.
  using ModelFactory = std::function<std::unique_ptr<LlamaModel>()>;

  Worker(ModelFactory factory, Scheduler* scheduler)
      : factory_(std::move(factory)), sched_(scheduler) {}
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  void start();  // launch the GPU thread (loads the model, then loops)
  void stop();   // signal the scheduler to drain, then join

 private:
  void run();  // the loop; the sole caller of MLX eval
  void process_solo(const std::shared_ptr<Request>& req);

  ModelFactory factory_;
  Scheduler* sched_;
  std::unique_ptr<LlamaModel> model_;  // constructed and owned on the worker thread
  std::thread thread_;
};

}  // namespace xllm
