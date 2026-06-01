// XLLM-016: a Request is the unit of work shared between an HTTP/test thread and
// the GPU worker. The submitting thread touches only its own Request (MLX is not
// thread-safe for concurrent eval); the worker is the sole producer of tokens.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "sample/sampler.h"

namespace xllm {

// Bounded, blocking, single-producer (worker) / single-consumer (request thread)
// token queue. push() applies backpressure when full (XLLM-023); the consumer
// pop()s until the producer close()s and the queue drains.
class TokenQueue {
 public:
  explicit TokenQueue(std::size_t capacity = 1024) : capacity_(capacity) {}

  // Producer: append a token, blocking while full unless closed.
  void push(int token) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < capacity_ || closed_; });
    if (closed_) return;
    q_.push(token);
    not_empty_.notify_one();
  }

  // Producer: signal that no more tokens will be pushed.
  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  // Consumer: pop the next token; returns false once closed and drained.
  bool pop(int& out) {
    std::unique_lock<std::mutex> lk(m_);
    not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = q_.front();
    q_.pop();
    not_full_.notify_one();
    return true;
  }

 private:
  std::size_t capacity_;
  std::queue<int> q_;
  bool closed_ = false;
  std::mutex m_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

struct Request {
  std::vector<int> prompt_ids;
  SamplingParams params;
  int max_tokens = 64;
  std::vector<int> eos_ids;

  // Set by the submitting thread (e.g. client disconnect); read by the worker.
  std::atomic<bool> cancelled{false};

  TokenQueue tokens;                 // worker pushes generated ids, then close()s
  std::string finish_reason;         // "stop" | "length" | "cancel" (worker sets)
};

}  // namespace xllm
