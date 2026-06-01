#include "runtime/worker.h"

#include <algorithm>

#include "cache/batch_kv_cache.h"
#include "sample/sampler.h"

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

namespace xllm {

namespace {
// Sample one token from the last position of (1, L, vocab) logits.
int sample_last(const mx::array& logits, const SamplingParams& params, const mx::array& key) {
  const int L = logits.shape()[1];
  const int V = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, L - 1, 0}, {1, L, V}), {1, V});
  mx::array tok = Sampler::sample(last, params, key).tokens;
  mx::eval(tok);
  return tok.item<int>();
}
}  // namespace

Worker::~Worker() { stop(); }

void Worker::start() {
  thread_ = std::thread([this] { run(); });
}

void Worker::stop() {
  sched_->stop();
  if (thread_.joinable()) thread_.join();
}

void Worker::run() {
  model_ = factory_();  // load the model on this thread so its arrays live here
  while (true) {
    std::shared_ptr<Request> req = sched_->next_waiting();
    if (!req) break;  // stopping and drained
    process_solo(req);
  }
}

void Worker::process_solo(const std::shared_ptr<Request>& req) {
  auto is_eos = [&](int id) {
    return std::find(req->eos_ids.begin(), req->eos_ids.end(), id) != req->eos_ids.end();
  };

  BatchKVCache cache(model_->config().n_layers, /*left_padding=*/{0});  // batch of 1
  mx::array prompt(req->prompt_ids.data(), {1, static_cast<int>(req->prompt_ids.size())},
                   mx::int32);
  mx::array logits = model_->forward(prompt, cache);

  std::string reason = "length";
  for (int step = 0; step < req->max_tokens; ++step) {
    if (req->cancelled.load()) {
      reason = "cancel";
      break;
    }
    mx::array key = mx::random::key(req->params.seed + static_cast<uint64_t>(step));
    int next = sample_last(logits, req->params, key);
    if (is_eos(next)) {
      reason = "stop";
      break;
    }
    req->tokens.push(next);

    mx::array step_arr(&next, {1, 1}, mx::int32);
    logits = model_->forward(step_arr, cache);
  }

  req->finish_reason = reason;
  req->tokens.close();
}

}  // namespace xllm
