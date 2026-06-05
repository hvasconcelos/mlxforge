// Engine: core inference orchestrator — an embeddable object, not dependent on HTTP.
// Handles the end-to-end model startup: resolves the user model spec (local dir, repo id, or .gguf),
// loads config and tokenizer (GGUF or safetensors), and boots a Worker (on its own thread)
// that interacts with the GPU. Sits above the Scheduler, which handles request submission.
// The Engine exposes only minimal access points for a consumer: providing config, tokenizer,
// the request submission seam, and status/metrics. Used both in HTTP servers and embedded apps.
//
// Usage pattern: create an Engine, build a Request, submit via scheduler().submit(req),
// then drain output tokens from req->tokens. The Engine owns its Worker for its lifetime;
// only that thread directly calls into MLX, preserving thread-safety.
#pragma once

#include <string>

#include "core/config.h"
#include "runtime/metrics.h"
#include "runtime/worker.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"

namespace mlxforge {

// Lightweight configuration for engine construction.
struct EngineConfig {
  std::string model_spec;  // Model description: local directory, HuggingFace repo id, or .gguf file path (to be resolved internally)
  int max_waiting = 256;   // Maximum length of the Scheduler's waiting queue; 0 disables the cap
};

class Engine {
 public:
  // Constructor: resolves the model spec, loads config/tokenizer on the caller's thread,
  // then spawns a Worker which loads weights and must be run on its own thread due to MLX constraints.
  // Throws if the spec is invalid or there is a loading failure.
  explicit Engine(EngineConfig cfg);
  ~Engine();  // Ensures all in-flight requests drain and the Worker thread stops cleanly

  // Disable copying/moving: Engine must remain unique, as it manages thread-bound objects.
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Main submission entrypoint: get a reference to the Scheduler for Request lifecycle/queueing.
  // Non-const: users will submit() Requests, mutating scheduler state.
  Scheduler& scheduler() { return scheduler_; }

  // Access the loaded Tokenizer (thread-safe constant interface)
  const Tokenizer& tokenizer() const { return tok_; }
  // Access the loaded ModelConfig (i.e. model hyperparameters, architecture, etc)
  const ModelConfig& config() const { return cfg_; }
  // Access the resolved model name (as supplied by the user, useful for HTTP responses etc.)
  const std::string& model_name() const { return model_name_; }

  // Query whether the model is loaded and ready for inference (Worker finished initialization)
  bool ready() const { return worker_.ready(); }
  // Fetch running metrics (queues, timing, active requests, etc.) from the Worker
  WorkerMetrics metrics() const { return worker_.metrics(); }

  // Explicitly drain all remaining requests/queues and join/stop the Worker thread.
  // Idempotent: also called by the destructor, so only use if early shutdown/drain is desired.
  void stop() { worker_.stop(); }

 private:
  // Internal struct: stores everything loaded on the caller thread prior to Worker start
  // (no MLX arrays instantiated yet — just config, tokenizer, and resolved model dir).
  struct Loaded {
    std::string dir;           // Fully resolved model/weights directory
    bool is_gguf = false;      // Was the model GGUF format?
    ModelConfig config;        // Parsed model config
    Tokenizer tokenizer;       // Parsed tokenizer
  };

  // Resolve model spec path, parse config/tokenizer, etc.
  static Loaded load_head(const std::string& spec);
  // Factory builder: creates a Worker::ModelFactory, handling weight loading with proper backend
  static Worker::ModelFactory make_factory(std::string dir, bool is_gguf);

  // Private delegating ctor, used internally after head-loading step is complete.
  Engine(EngineConfig cfg, Loaded loaded);

  // --- Engine state (order is important for destruction) ---

  std::string model_name_;    // Echoes user-supplied spec (for e.g. server listing endpoints)
  ModelConfig cfg_;           // Model config, immutable after load
  Tokenizer tok_;             // Tokenizer, immutable after load

  // Order of these two is critical:
  // Scheduler must outlive Worker, because Worker holds a pointer to scheduler_.
  Scheduler scheduler_;
  Worker worker_;
};

}  // namespace mlxforge
