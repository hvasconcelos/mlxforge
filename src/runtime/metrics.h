// A plain snapshot of the worker's live decode metrics, produced by
// Worker::metrics() and consumed by the HTTP /health route. Kept free of MLX
// (and any heavy) includes so the server header can hold the type without
// pulling the model/cache headers into the HTTP layer.
#pragma once

namespace mlxforge {

struct WorkerMetrics {
  long decode_steps = 0;               // total async_eval decode steps
  int active_batch = 0;                // rows currently in the decode batch
  int peak_batch = 0;                  // high-water mark of active_batch
  long requests_completed = 0;         // requests evicted (finished/cancelled)
  long long prompt_tokens_total = 0;
  long long completion_tokens_total = 0;
  double avg_ttft_ms = 0.0;            // enqueue -> first token
  double avg_request_ms = 0.0;         // enqueue -> finished
  double avg_tokens_per_second = 0.0;  // aggregate: completion_tokens / gen_seconds
};

}  // namespace mlxforge
