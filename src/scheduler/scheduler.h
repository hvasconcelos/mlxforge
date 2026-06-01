// XLLM-016: the three-queue state machine. Requests flow
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

namespace xllm {

class Scheduler {
 public:
  // Submit a request from any thread; wakes the worker.
  void submit(const std::shared_ptr<Request>& req);

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
  bool stop_ = false;
};

}  // namespace xllm
