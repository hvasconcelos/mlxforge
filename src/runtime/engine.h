// Engine: the inference engine as a single embeddable object, with no HTTP
// dependency. It performs the load-and-start sequence — resolve the model spec,
// load the config + tokenizer (GGUF or safetensors), then construct and start
// the GPU Worker over a Scheduler — and exposes the handful of accessors a
// consumer needs (the HTTP server is one such consumer; an embedding app is
// another).
//
// Submission goes through scheduler(): build a Request and call
// scheduler().submit(req), then drain req->tokens. The Engine owns the Worker
// thread (the only thread that touches MLX) for its whole lifetime.
#pragma once

#include <string>

#include "core/config.h"
#include "runtime/metrics.h"
#include "runtime/worker.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"

namespace mlxforge {

struct EngineConfig {
  std::string model_spec;  // local dir, HF repo id, or .gguf file (resolved internally)
  int max_waiting = 256;   // scheduler waiting-queue bound (0 = unbounded)
};

class Engine {
 public:
  // Resolve the spec, load config + tokenizer on the calling thread, then start
  // the worker (which loads the weights on its own thread). Throws on a bad spec
  // or load failure.
  explicit Engine(EngineConfig cfg);
  ~Engine();  // drains in-flight requests (via Worker::stop) on destruction

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // The submission seam: consumers build a Request and scheduler().submit() it,
  // then drain its token queue. Non-const because submit() mutates the queue.
  Scheduler& scheduler() { return scheduler_; }

  const Tokenizer& tokenizer() const { return tok_; }
  const ModelConfig& config() const { return cfg_; }
  // The spec the user passed; echoed as the served model name (/v1/models).
  const std::string& model_name() const { return model_name_; }

  bool ready() const { return worker_.ready(); }       // model finished loading?
  WorkerMetrics metrics() const { return worker_.metrics(); }

  // Drain the waiting queue and join the worker thread. Idempotent; also run by
  // the destructor, so an explicit call is only needed to drain early.
  void stop() { worker_.stop(); }

 private:
  // The config + tokenizer parsed up front (no MLX arrays), plus where the
  // weights live and how to load them — enough to build the worker's factory.
  struct Loaded {
    std::string dir;
    bool is_gguf = false;
    ModelConfig config;
    Tokenizer tokenizer;
  };
  static Loaded load_head(const std::string& spec);
  static Worker::ModelFactory make_factory(std::string dir, bool is_gguf);

  // Delegated-to constructor: runs once load_head() has produced the head, so
  // the members can be initialized (and ordered) directly.
  Engine(EngineConfig cfg, Loaded loaded);

  std::string model_name_;
  ModelConfig cfg_;
  Tokenizer tok_;
  // scheduler_ is declared before worker_: the worker stores &scheduler_, so the
  // scheduler must outlive it (members destruct in reverse order).
  Scheduler scheduler_;
  Worker worker_;
};

}  // namespace mlxforge
