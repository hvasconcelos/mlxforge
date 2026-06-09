// A Request is the unit of work shared between an HTTP/test thread and the GPU
// worker. The submitting thread touches only its own Request (MLX is not
// thread-safe for concurrent eval); the worker is the sole producer of tokens.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "sample/json_grammar.h"
#include "sample/sampler.h"

namespace mlxforge {

// Bounded, blocking, single-producer (worker) / single-consumer (request thread)
// token queue. push() applies backpressure when full (slow SSE consumers); the
// consumer pop()s until the producer close()s and the queue drains.
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

  // Constrained decoding: empty = unconstrained; "json" = any valid JSON value;
  // otherwise a JSON-Schema string (the supported subset, see json_grammar.h).
  // The worker compiles this into `grammar` on admit and masks logits so the
  // output can only be valid JSON.
  std::string json_schema;
  std::unique_ptr<JsonGrammar> grammar;  // runtime grammar state (worker-owned)

  // Embedding request: when true the worker runs a single forward pass, pools +
  // (optionally) normalizes the hidden states into `embedding_result`, and
  // closes `tokens` (no generation). `pooling` is a mlxforge::Pooling value;
  // `embedding_normalize` L2-normalizes the pooled vector (the default).
  bool embedding = false;
  int pooling = 0;
  bool embedding_normalize = true;
  std::vector<float> embedding_result;

  // Multimodal (Qwen3-VL) one-shot generation: when `mm_image` is non-empty the
  // worker decodes the image, runs the ViT, renders the chat prompt from
  // `mm_text` with the image placeholders, and streams generated tokens
  // single-stream (not merged into the continuous-decode batch). `prompt_ids` is
  // unused on this path — the worker builds the prompt itself.
  std::string mm_text;                 // the user's text prompt
  std::vector<std::uint8_t> mm_image;  // raw encoded image bytes (JPEG/PNG/…)
  bool is_multimodal() const { return !mm_image.empty(); }

  // Set by the submitting thread (e.g. client disconnect); read by the worker.
  std::atomic<bool> cancelled{false};

  TokenQueue tokens;                 // worker pushes generated ids, then close()s
  std::string finish_reason;         // "stop" | "length" | "cancel" | "embed"

  // Metrics: enqueue stamped on submit, first_token/finish stamped by the worker.
  using Clock = std::chrono::steady_clock;
  Clock::time_point enqueue_time{};
  Clock::time_point first_token_time{};
};

}  // namespace mlxforge
