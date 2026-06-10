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

#include <utility>

#include "sample/json_grammar.h"
#include "sample/sampler.h"

namespace mlxforge {

// Bounded, blocking, single-producer (worker) / single-consumer (request thread)
// queue. push() applies backpressure when full (slow SSE consumers); the consumer
// pop()s until the producer close()s and the queue drains. Used for both the
// generated token ids (TokenQueue) and their optional per-token log-probs
// (LogprobQueue), which the worker pushes in lockstep.
template <class T>
class SpscQueue {
 public:
  explicit SpscQueue(std::size_t capacity = 1024) : capacity_(capacity) {}

  // Producer: append an item, blocking while full unless closed.
  void push(T item) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < capacity_ || closed_; });
    if (closed_) return;
    q_.push(std::move(item));
    not_empty_.notify_one();
  }

  // Producer: signal that no more items will be pushed.
  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  // Consumer: pop the next item; returns false once closed and drained.
  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop();
    not_full_.notify_one();
    return true;
  }

 private:
  std::size_t capacity_;
  std::queue<T> q_;
  bool closed_ = false;
  std::mutex m_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
};

using TokenQueue = SpscQueue<int>;

// One emitted token's log-probability data (OpenAI logprobs). `id`/`logprob` are
// the chosen token and its log-prob; `top` holds the requested alternatives as
// (id, log-prob) pairs in descending order (empty when only the chosen logprob
// was requested). Carried on Request::logprobs in lockstep with Request::tokens.
struct TokenLogprob {
  int id = 0;
  float logprob = 0.0f;
  std::vector<std::pair<int, float>> top;
};

using LogprobQueue = SpscQueue<TokenLogprob>;

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

  // Multimodal (Qwen3-VL) one-shot generation: when `mm_images` is non-empty the
  // worker decodes each image, runs the ViT, and streams generated tokens
  // single-stream (not merged into the continuous-decode batch). Two prompt forms:
  //   - `prompt_ids` set (the server): already chat-templated with the image
  //     placeholder runs expanded in order — the worker uses it as-is.
  //   - `prompt_ids` empty (C ABI / CLI): `mm_text` is a single user turn the
  //     worker renders itself (one image).
  // Images are consumed in `mm_images` order, matching the <|image_pad|> runs.
  std::string mm_text;                              // single-turn user text (mm_text path)
  std::vector<std::vector<std::uint8_t>> mm_images;  // raw encoded image bytes, in order
  bool is_multimodal() const { return !mm_images.empty(); }

  // Set by the submitting thread (e.g. client disconnect); read by the worker.
  std::atomic<bool> cancelled{false};

  TokenQueue tokens;                 // worker pushes generated ids, then close()s
  // Per-token log-probs, pushed in lockstep with `tokens` when
  // params.top_logprobs >= 0 (otherwise never touched). Closed alongside `tokens`.
  LogprobQueue logprobs;
  std::string finish_reason;         // "stop" | "length" | "cancel" | "embed"

  // Metrics: enqueue stamped on submit, first_token/finish stamped by the worker.
  using Clock = std::chrono::steady_clock;
  Clock::time_point enqueue_time{};
  Clock::time_point first_token_time{};
};

}  // namespace mlxforge
