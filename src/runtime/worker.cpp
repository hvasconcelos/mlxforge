#include "runtime/worker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>

#include "core/logging.h"
#include "model/qwen3_vl.h"
#include "model/vision/vit.h"
#include "runtime/batching.h"
#include "runtime/embedding.h"
#include "runtime/multimodal_stream.h"
#include "sample/sampler.h"
#include "tokenizer/tokenizer.h"
#include "vision/image_decode.h"

#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"

namespace mlxforge {

namespace {
std::vector<int> read_ids(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::int32));
  mx::eval(c);
  return std::vector<int>(c.data<int32_t>(), c.data<int32_t>() + c.size());
}

std::vector<float> read_floats(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(c);
  return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}

bool is_eos(const Request& req, int id) {
  return std::find(req.eos_ids.begin(), req.eos_ids.end(), id) != req.eos_ids.end();
}

// Emit token `id` for a request, returning true if the request is now finished.
// `lp` (when non-null) is the token's log-prob record, pushed in lockstep with
// the token; EOS is never emitted, so it carries no logprob.
bool consume(Request& req, int& produced, int id, const TokenLogprob* lp) {
  if (is_eos(req, id)) {
    req.finish_reason = "stop";
    return true;
  }
  if (produced == 0) req.first_token_time = Request::Clock::now();  // TTFT marker
  req.tokens.push(id);
  if (lp) req.logprobs.push(*lp);
  if (++produced >= req.max_tokens) {
    req.finish_reason = "length";
    return true;
  }
  return false;
}
}  // namespace

Worker::Worker(ModelFactory factory, Scheduler* scheduler, const Tokenizer* tok)
    : factory_(std::move(factory)), sched_(scheduler), tok_(tok) {}

Worker::~Worker() { stop(); }

void Worker::handle_embedding(Request& req) {
  try {
    req.embedding_result = embed_pooled(*model_, req.prompt_ids,
                                        static_cast<Pooling>(req.pooling),
                                        req.embedding_normalize);
    req.finish_reason = "embed";
  } catch (const std::exception& e) {
    log::error("worker: embedding error: {}", e.what());
    req.finish_reason = "error";
  }
  req.tokens.close();  // unblock the waiting submitter
  req.logprobs.close();
}

void Worker::admit_multimodal(const std::shared_ptr<Request>& req) {
  try {
    auto* vl = dynamic_cast<Qwen3VLModel*>(model_.get());
    if (vl == nullptr || !model_->config().has_vision_tower()) {
      throw std::runtime_error("loaded model is not a vision-language model");
    }
    // The ViT borrows the model's weights; build it once and reuse it.
    if (!vit_) {
      vit_ = std::make_unique<VitEncoder>(*model_->config().vision, model_->weights());
    }
    std::vector<mx::array> images;
    images.reserve(req->mm_images.size());
    for (const auto& bytes : req->mm_images)
      images.push_back(decode_image(bytes.data(), bytes.size()));

    // A caller that pre-rendered the full chat history (the server) supplies
    // prompt_ids with the image placeholders already expanded (any number of
    // images); the simple path (C ABI / CLI) supplies just mm_text + image(s).
    std::vector<int> prompt_ids = req->prompt_ids;
    if (prompt_ids.empty()) {
      if (tok_ == nullptr) throw std::runtime_error("multimodal text prompt needs a tokenizer");
      prompt_ids = render_multimodal_prompt(*tok_, *vl, req->mm_text, images);
    }
    // The row decodes as ordinary text from here, so its history/metrics treat the
    // expanded prompt (image placeholders included) as the prompt.
    req->prompt_ids = prompt_ids;

    // Prefill single-stream (all fallible work — ViT, scatter, M-RoPE — happens
    // before we touch the shared batch), then admit into the decode pool.
    MultimodalPrefillInputs in = prepare_multimodal_prefill(*vl, *vit_, prompt_ids, images);
    MultimodalPrefill pf =
        prefill_multimodal_batched(*vl, prompt_ids, in.features, in.deepstack, in.position_ids);

    if (!cache_) {
      cache_ = std::make_unique<BatchKVCache>(std::move(pf.cache));
    } else {
      cache_->merge(pf.cache);
    }
    register_rows({req}, pf.last_logits);
    log::debug("worker: admitted multimodal request (prompt={}, batch now {})", prompt_ids.size(),
               reqs_.size());
  } catch (const std::exception& e) {
    log::error("worker: multimodal admit error: {}", e.what());
    req->finish_reason = "error";
    req->tokens.close();
    req->logprobs.close();
  }
}

void Worker::ensure_token_bytes(int vocab) {
  if (token_bytes_built_ || !tok_) return;
  token_bytes_.assign(vocab, std::string());
  for (int id = 0; id < vocab; ++id) token_bytes_[id] = tok_->decode({id});
  token_bytes_built_ = true;
}

// Build an additive (1, vocab) fp32 mask: 0 for tokens the grammar allows at its
// current state, -inf for the rest. Forces EOS once the JSON value is complete,
// and forbids non-EOS special tokens mid-JSON.
mx::array Worker::grammar_mask(const Request& req, const JsonGrammar& g, int vocab) {
  const float NEG = -std::numeric_limits<float>::infinity();
  std::vector<float> add(static_cast<size_t>(vocab), 0.0f);
  const bool complete = g.complete();

  // Cheap pre-filter: which first bytes can the grammar accept now? Most tokens
  // are rejected on their first byte, avoiding a full accepts() copy per token.
  std::array<bool, 256> first_ok{};
  if (!complete)
    for (int b = 0; b < 256; ++b)
      first_ok[b] = g.accepts(std::string(1, static_cast<char>(b)));

  auto is_eos = [&](int id) {
    return std::find(req.eos_ids.begin(), req.eos_ids.end(), id) != req.eos_ids.end();
  };

  int allowed = 0;
  for (int id = 0; id < vocab; ++id) {
    bool allow;
    if (is_eos(id)) {
      allow = complete;  // may only stop once the JSON value is complete
    } else if (complete) {
      allow = false;     // force an EOS token once complete
    } else {
      const std::string& bytes = token_bytes_[id];
      if (bytes.empty()) allow = false;  // non-EOS special token: not valid JSON
      else if (!first_ok[static_cast<uint8_t>(bytes[0])]) allow = false;
      else allow = g.accepts(bytes);
    }
    if (allow) ++allowed;
    else add[id] = NEG;
  }

  // Never emit an all -inf mask (it would NaN the softmax). If nothing is
  // allowed (e.g. complete but the model defines no EOS id), leave logits as-is.
  if (allowed == 0) return mx::zeros({1, vocab}, mx::float32);
  return mx::array(add.data(), {1, vocab}, mx::float32);
}

void Worker::advance_grammar(Request& req, int chosen_id) {
  if (!req.grammar) return;
  if (chosen_id >= 0 && chosen_id < static_cast<int>(token_bytes_.size()))
    req.grammar->advance(token_bytes_[chosen_id]);  // EOS/specials decode to "" (no-op)
}

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
      if (!incoming.empty()) {
        // Embedding requests are one-shot (handled inline). Multimodal requests are
        // prefilled single-stream then admitted into the decode batch (one prefill
        // each — the ViT can't batch ragged grids). Text-generation requests are
        // prefilled together and admitted as a group.
        std::vector<std::shared_ptr<Request>> gen;
        gen.reserve(incoming.size());
        for (auto& r : incoming) {
          if (r->embedding) handle_embedding(*r);
          else if (r->is_multimodal()) admit_multimodal(r);
          else gen.push_back(std::move(r));
        }
        if (!gen.empty()) admit(gen);
      }
      evict_finished();  // a row may finish on its very first token
      if (reqs_.empty()) continue;

      decode_step();
      evict_finished();
    } catch (const std::exception& e) {
      log::error("worker: decode loop error: {}", e.what());
      throw;
    }

    // Publish batch occupancy for /health (single writer -> plain load-max-store).
    const int active = static_cast<int>(reqs_.size());
    active_batch_.store(active);
    if (active > peak_batch_.load()) peak_batch_.store(active);
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

  if (!cache_) {
    cache_ = std::make_unique<BatchKVCache>(std::move(pr.cache));
  } else {
    cache_->merge(pr.cache);
  }
  register_rows(incoming, pr.last_logits);
}

void Worker::register_rows(const std::vector<std::shared_ptr<Request>>& incoming,
                           const mx::array& last_logits) {
  // Register the new rows before sampling so sample_rows() can read their params,
  // penalty history (seeded with the prompt) and RNG key. `last_logits` rows are
  // aligned to the new tail of the batch (row i -> reqs_[base + i]).
  const int base = static_cast<int>(reqs_.size());
  for (size_t i = 0; i < incoming.size(); ++i) {
    reqs_.push_back(incoming[i]);
    produced_.push_back(0);
    feed_.push_back(0);  // set to the first token below
    finished_.push_back(false);
    history_.push_back(incoming[i]->prompt_ids);
    rng_keys_.push_back(mx::random::key(incoming[i]->params.seed));
    // Compile the constrained-decoding grammar before the first token is sampled.
    // Compact mode forbids inter-token whitespace so greedy decoding cannot stall
    // emitting endless separators.
    if (tok_ && !incoming[i]->json_schema.empty()) {
      auto g = std::make_unique<JsonGrammar>(
          JsonGrammar::from_schema_string(incoming[i]->json_schema));
      g->set_compact(true);
      incoming[i]->grammar = std::move(g);
    }
  }

  const int count = static_cast<int>(incoming.size());
  std::vector<int> first;
  std::vector<TokenLogprob> row_lp;
  std::vector<char> has_lp;
  finalize_sample(sample_rows(last_logits, base, count), count, first, row_lp, has_lp);

  for (int i = 0; i < count; ++i) {
    const int b = base + i;
    feed_[b] = first[i];               // feed the first token next step
    history_[b].push_back(first[i]);   // and let later penalties see it
    advance_grammar(*reqs_[b], first[i]);
    if (reqs_[b]->cancelled.load()) {
      reqs_[b]->finish_reason = "cancel";
      finished_[b] = true;
    } else if (consume(*reqs_[b], produced_[b], first[i], has_lp[i] ? &row_lp[i] : nullptr)) {
      finished_[b] = true;
    }
  }
}

Worker::SampledRows Worker::sample_rows(const mx::array& logits, int row_offset, int count) {
  const int vocab = logits.shape()[1];
  std::vector<mx::array> tokens;
  tokens.reserve(count);
  SampledRows out{mx::zeros({0}, mx::int32), {}, {}, {}, {}};
  for (int i = 0; i < count; ++i) {
    const int r = row_offset + i;
    mx::array row = mx::slice(logits, {i, 0}, {i + 1, vocab});  // (1, vocab)
    // Constrained decoding: mask the logits so only grammar-valid tokens remain.
    if (tok_ && reqs_[r]->grammar) {
      ensure_token_bytes(vocab);
      row = mx::add(row, grammar_mask(*reqs_[r], *reqs_[r]->grammar, vocab));
    }
    const SamplingParams& p = reqs_[r]->params;

    // Advance the per-row key so successive steps draw independently but
    // reproducibly for a fixed seed; greedy rows still split (cheap, keeps state).
    std::pair<mx::array, mx::array> ks = mx::random::split(rng_keys_[r]);
    rng_keys_[r] = ks.first;

    SampleResult res = [&] {
      if (!p.has_penalties() || history_[r].empty()) return Sampler::sample(row, p, ks.second);
      const std::vector<int>& h = history_[r];
      mx::array history(h.data(), {1, static_cast<int>(h.size())}, mx::int32);
      return Sampler::sample(row, p, ks.second, history);
    }();
    tokens.push_back(res.tokens);  // (1,)
    // Collect the log-prob arrays only for rows that asked, so the logprob
    // subgraph stays dead (MLX prunes it) for everyone else.
    if (p.top_logprobs >= 0) {
      out.lp_rows.push_back(i);
      out.lp_chosen.push_back(res.logprobs);      // (1,)
      out.lp_top_ids.push_back(res.top_tokens);   // (1, K)
      out.lp_top_lp.push_back(res.top_logprobs);  // (1, K)
    }
  }
  out.tokens = mx::concatenate(tokens, /*axis=*/0);  // (count,)
  return out;
}

void Worker::finalize_sample(const SampledRows& s, int count, std::vector<int>& ids,
                             std::vector<TokenLogprob>& row_lp, std::vector<char>& has_lp) {
  // The ONE async_eval per step, over the whole batch: the chosen tokens plus
  // every logprob-enabled row's log-prob arrays, so they ride the same eval.
  std::vector<mx::array> to_eval;
  to_eval.reserve(1 + s.lp_chosen.size() + s.lp_top_ids.size() + s.lp_top_lp.size());
  to_eval.push_back(s.tokens);
  for (const auto& a : s.lp_chosen) to_eval.push_back(a);
  for (const auto& a : s.lp_top_ids) to_eval.push_back(a);
  for (const auto& a : s.lp_top_lp) to_eval.push_back(a);
  mx::async_eval(to_eval);

  ids = read_ids(s.tokens);
  row_lp.assign(count, TokenLogprob{});
  has_lp.assign(count, 0);
  for (size_t j = 0; j < s.lp_rows.size(); ++j) {
    const int i = s.lp_rows[j];
    has_lp[i] = 1;
    TokenLogprob& lp = row_lp[i];
    lp.id = ids[i];
    lp.logprob = read_floats(s.lp_chosen[j])[0];
    const std::vector<int> tids = read_ids(s.lp_top_ids[j]);
    const std::vector<float> tlp = read_floats(s.lp_top_lp[j]);
    lp.top.reserve(tids.size());
    for (size_t k = 0; k < tids.size(); ++k) lp.top.emplace_back(tids[k], tlp[k]);
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
  // Sample only the B real rows (dummy rows are excluded from the graph).
  SampledRows sampled = sample_rows(mx::reshape(logits, {bucket, logits.shape()[2]}), 0, B);

  // One async_eval over the whole batch (tokens + any logprob arrays).
  std::vector<int> ids;
  std::vector<TokenLogprob> row_lp;
  std::vector<char> has_lp;
  finalize_sample(sampled, B, ids, row_lp, has_lp);

  for (int b = 0; b < B; ++b) {
    if (finished_[b]) continue;
    if (reqs_[b]->cancelled.load()) {
      reqs_[b]->finish_reason = "cancel";
      finished_[b] = true;
      continue;
    }
    feed_[b] = ids[b];
    history_[b].push_back(ids[b]);  // penalties see the full sequence so far
    advance_grammar(*reqs_[b], ids[b]);
    if (consume(*reqs_[b], produced_[b], ids[b], has_lp[b] ? &row_lp[b] : nullptr))
      finished_[b] = true;
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
  using us = std::chrono::duration<double, std::micro>;
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

      // Accumulate the same numbers for /health (sums + counts -> averages).
      ++requests_completed_;
      prompt_tokens_total_ += static_cast<long long>(r.prompt_ids.size());
      completion_tokens_total_ += produced_[b];
      ttft_us_sum_ += static_cast<long long>(ttft * 1000.0);
      gen_us_sum_ += static_cast<long long>(gen_s * 1e6);
      request_us_sum_ += static_cast<long long>(us(now - r.enqueue_time).count());

      reqs_[b]->tokens.close();  // signal the consumer
      reqs_[b]->logprobs.close();
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
    history_[dst] = std::move(history_[src]);
    rng_keys_[dst] = rng_keys_[src];
  }
  reqs_.resize(keep.size());
  produced_.resize(keep.size());
  feed_.resize(keep.size());
  finished_.resize(keep.size());
  history_.resize(keep.size());
  // mx::array is not default-constructible, so resize() won't compile; erase the
  // tail instead (only needs move-assignment).
  rng_keys_.erase(rng_keys_.begin() + keep.size(), rng_keys_.end());
}

WorkerMetrics Worker::metrics() const {
  WorkerMetrics m;
  m.decode_steps = decode_steps_.load();
  m.active_batch = active_batch_.load();
  m.peak_batch = peak_batch_.load();
  m.requests_completed = requests_completed_.load();
  m.prompt_tokens_total = prompt_tokens_total_.load();
  m.completion_tokens_total = completion_tokens_total_.load();

  const long completed = m.requests_completed;
  if (completed > 0) {
    m.avg_ttft_ms = ttft_us_sum_.load() / 1000.0 / completed;
    m.avg_request_ms = request_us_sum_.load() / 1000.0 / completed;
  }
  const long long gen_us = gen_us_sum_.load();
  if (gen_us > 0) m.avg_tokens_per_second = m.completion_tokens_total * 1e6 / gen_us;
  return m;
}

}  // namespace mlxforge
