// The waiting-queue half of the three-queue state machine. Requests flow
//   waiting -> prefill_batch -> decode_batch
// The HTTP/test threads only push to `waiting`; the worker thread owns the
// prefill/decode sets and is the only thread that touches MLX. Handoff is via a
// mutex + condition_variable.
#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "scheduler/request.h"

namespace mlxforge {

class Scheduler {
 public:
  // Bound the waiting queue (0 = unbounded); submit() rejects beyond this.
  void set_max_waiting(int max) { max_waiting_ = max; }

  // The configured waiting-queue bound (0 = unbounded). For /health reporting.
  int max_waiting() const { return max_waiting_; }

  // Submit a request from any thread; wakes the worker. Returns false if the
  // waiting queue is full (the caller should reply 429).
  bool submit(const std::shared_ptr<Request>& req);

  // Worker-side: block until a waiting request is available or stop() is called.
  // Returns nullptr when the scheduler is stopping and drained.
  std::shared_ptr<Request> next_waiting();

  // Drain up to `max` waiting requests without blocking (used to fill a prefill
  // batch). Returns however many are ready (possibly empty).
  std::vector<std::shared_ptr<Request>> take_waiting(int max);

  size_t waiting_size() const;

  // Signal the worker loop to exit once the waiting queue is drained.
  void stop();
  bool stopping() const;

 private:
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> waiting_;
  int max_waiting_ = 0;  // 0 = unbounded
  bool stop_ = false;
};

}  // namespace mlxforge
