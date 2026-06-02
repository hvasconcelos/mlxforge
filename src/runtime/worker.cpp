#include "runtime/worker.h"

#include <algorithm>
#include <chrono>

#include "core/logging.h"
#include "runtime/batching.h"
#include "sample/sampler.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
std::vector<int> read_ids(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
}

bool is_eos(const Request& req, int id) {
  return std::find(req.eos_ids.begin(), req.eos_ids.end(), id) != req.eos_ids.end();
}

// Emit token `id` for a request, returning true if the request is now finished.
bool consume(Request& req, int& produced, int id) {
  if (is_eos(req, id)) {
    req.finish_reason = "stop";
    return true;
  }
  if (produced == 0) req.first_token_time = Request::Clock::now();  // TTFT marker
  req.tokens.push(id);
  if (++produced >= req.max_tokens) {
    req.finish_reason = "length";
    return true;
  }
  return false;
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
  log::info("worker: loading model...");
  model_ = factory_();  // load the model on this thread so its arrays live here
  ready_.store(true);
  log::info("worker: model loaded, ready");

  while (true) {
    std::vector<std::shared_ptr<Request>> incoming;
    if (reqs_.empty()) {
      auto r = sched_->next_waiting();  // block until work or stop+drained
      if (!r) break;
      incoming.push_back(r);
      auto more = sched_->take_waiting(kPrefillBatchSize - 1);
      incoming.insert(incoming.end(), more.begin(), more.end());
    } else {
      incoming = sched_->take_waiting(kPrefillBatchSize);  // non-blocking top-up
    }

    try {
      if (!incoming.empty()) admit(incoming);
      evict_finished();  // a row may finish on its very first token
      if (reqs_.empty()) continue;

      decode_step();
      evict_finished();
    } catch (const std::exception& e) {
      log::error("worker: decode loop error: {}", e.what());
      throw;
    }
  }
  log::info("worker: stopped after {} decode steps", decode_steps_.load());
}

void Worker::admit(const std::vector<std::shared_ptr<Request>>& incoming) {
  std::vector<std::vector<int>> prompts;
  prompts.reserve(incoming.size());
  for (const auto& r : incoming) prompts.push_back(r->prompt_ids);

  log::debug("worker: admitting {} request(s) (batch {} -> {})", incoming.size(), reqs_.size(),
             reqs_.size() + incoming.size());
  PrefillResult pr = prefill(*model_, prompts);
  std::vector<int> first = read_ids(Sampler::greedy(pr.last_logits));  // each row's first token

  if (!cache_) {
    cache_ = std::make_unique<BatchKVCache>(std::move(pr.cache));
  } else {
    cache_->merge(pr.cache);
  }

  const int base = static_cast<int>(reqs_.size());
  for (size_t i = 0; i < incoming.size(); ++i) {
    reqs_.push_back(incoming[i]);
    produced_.push_back(0);
    feed_.push_back(first[i]);  // feed the first token next step
    finished_.push_back(false);
  }
  for (size_t i = 0; i < incoming.size(); ++i) {
    const int b = base + static_cast<int>(i);
    if (reqs_[b]->cancelled.load()) {
      reqs_[b]->finish_reason = "cancel";
      finished_[b] = true;
    } else if (consume(*reqs_[b], produced_[b], first[i])) {
      finished_[b] = true;
    }
  }
}

void Worker::decode_step() {
  ++decode_steps_;
  const int B = static_cast<int>(reqs_.size());
  log::debug("worker: decode step {} (batch={})", decode_steps_.load(), B);

  // Pad the active batch up to a fixed bucket with masked dummy rows so the
  // decode forward graph shape recurs. Dummy rows are batch-independent and
  // masked, so they cannot affect the real rows; they are trimmed after.
  const int bucket = next_bucket(B);
  const int extra = bucket - B;
  if (extra > 0) cache_->pad_dummies(extra);
  std::vector<int> fed = feed_;
  fed.resize(bucket, 0);  // dummy rows feed a pad token

  mx::array inputs(fed.data(), {bucket, 1}, mx::int32);
  mx::array logits = model_->forward(inputs, *cache_);  // (bucket, 1, vocab)
  mx::array next = Sampler::greedy(mx::reshape(logits, {bucket, logits.shape()[2]}));

  mx::async_eval(next);  // the ONE eval per decode step, over the whole batch
  std::vector<int> ids = read_ids(next);

  for (int b = 0; b < B; ++b) {
    if (finished_[b]) continue;
    if (reqs_[b]->cancelled.load()) {
      reqs_[b]->finish_reason = "cancel";
      finished_[b] = true;
      continue;
    }
    feed_[b] = ids[b];
    if (consume(*reqs_[b], produced_[b], ids[b])) finished_[b] = true;
  }

  // Drop the dummy rows so the cache holds only real rows again.
  if (extra > 0) {
    std::vector<int> keep(B);
    for (int b = 0; b < B; ++b) keep[b] = b;
    cache_->filter(keep);
  }
}

void Worker::evict_finished() {
  using ms = std::chrono::duration<double, std::milli>;
  using sec = std::chrono::duration<double>;
  const auto now = Request::Clock::now();

  std::vector<int> keep;
  for (int b = 0; b < static_cast<int>(finished_.size()); ++b) {
    if (finished_[b]) {
      const Request& r = *reqs_[b];
      const double ttft = ms(r.first_token_time - r.enqueue_time).count();
      const double gen_s = sec(now - r.first_token_time).count();
      const double tps = gen_s > 0 ? produced_[b] / gen_s : 0.0;
      // Per-request metrics: TTFT, tokens/s, batch occupancy, queue depth.
      log::info("done reason={} prompt={} gen={} ttft={:.1f}ms tok/s={:.1f} batch={} queue={}",
                r.finish_reason, r.prompt_ids.size(), produced_[b], ttft, tps,
                static_cast<int>(reqs_.size()), sched_->waiting_size());
      reqs_[b]->tokens.close();  // signal the consumer
    } else {
      keep.push_back(b);
    }
  }
  if (keep.size() == reqs_.size()) return;  // nothing evicted

  if (keep.empty()) {
    cache_.reset();
  } else {
    cache_->filter(keep);
  }

  // Compact each row-aligned vector down to the kept rows (keep is ascending,
  // so dst <= src and an in-place gather is safe).
  for (int dst = 0; dst < static_cast<int>(keep.size()); ++dst) {
    const int src = keep[dst];
    reqs_[dst] = std::move(reqs_[src]);
    produced_[dst] = produced_[src];
    feed_[dst] = feed_[src];
    finished_[dst] = finished_[src];
  }
  reqs_.resize(keep.size());
  produced_.resize(keep.size());
  feed_.resize(keep.size());
  finished_.resize(keep.size());
}

}  // namespace mlxforge
