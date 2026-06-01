// XLLM-020 (MILESTONE): scheduler correctness & throughput. N concurrent
// requests (identical + distinct, mixed lengths admitted/evicted at different
// times) each produce the same tokens as a solo run; throughput rises with
// concurrency; the one-eval-per-decode-step invariant holds under load.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include "core/config.h"
#include "core/weights.h"
#include "runtime/single_stream.h"
#include "runtime/worker.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "support/model_fixture.h"
#include "support/reference.h"

using namespace xllm::test;

namespace {
xllm::Worker::ModelFactory factory_for(const std::string& dir) {
  return [dir] {
    return std::make_unique<xllm::LlamaModel>(
        xllm::ModelConfig::from_file(dir + "/config.json"), xllm::load_weights(dir));
  };
}

std::shared_ptr<xllm::Request> make_req(const std::vector<int>& prompt, int max_tokens,
                                        const std::vector<int>& eos) {
  auto r = std::make_shared<xllm::Request>();
  r->prompt_ids = prompt;
  r->params.temperature = 0.0f;
  r->max_tokens = max_tokens;
  r->eos_ids = eos;
  return r;
}
}  // namespace

TEST_CASE("XLLM-020: N concurrent mixed requests each match their solo run") {
  if (!model_available()) {
    MESSAGE("XLLM_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  xllm::ModelConfig cfg = xllm::ModelConfig::from_file(dir + "/config.json");
  xllm::LlamaModel& solo = shared_model();

  std::vector<std::vector<int>> base = {load_token_ids("prompt_0_ids.npy"),
                                        load_token_ids("prompt_1_ids.npy"),
                                        load_token_ids("prompt_2_ids.npy")};

  // 10 requests (> prefill_batch_size 8 -> admitted in two waves), identical and
  // distinct prompts, varying max_tokens -> eviction at different steps.
  std::vector<std::shared_ptr<xllm::Request>> reqs;
  std::vector<std::vector<int>> expected;
  for (int i = 0; i < 10; ++i) {
    const std::vector<int>& p = base[i % base.size()];
    const int max_tokens = 6 + (i % 5);
    reqs.push_back(make_req(p, max_tokens, cfg.eos_token_ids));
    expected.push_back(xllm::greedy_generate(solo, p, max_tokens, cfg.eos_token_ids).tokens);
  }

  xllm::Scheduler sched;
  xllm::Worker worker(factory_for(dir), &sched);
  worker.start();
  for (auto& r : reqs) sched.submit(r);

  // Consume each stream concurrently so admit/evict interleave under real load.
  std::vector<std::vector<int>> got(reqs.size());
  std::vector<std::thread> consumers;
  for (size_t i = 0; i < reqs.size(); ++i) {
    consumers.emplace_back([&, i] {
      int tok = 0;
      while (reqs[i]->tokens.pop(tok)) got[i].push_back(tok);
    });
  }
  for (auto& t : consumers) t.join();
  worker.stop();

  for (size_t i = 0; i < reqs.size(); ++i) {
    INFO("request " << i);
    assert_tokens_equal(got[i], expected[i]);
  }
}

TEST_CASE("XLLM-020: throughput rises with concurrency; one eval per decode step") {
  if (!model_available()) {
    MESSAGE("XLLM_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  xllm::ModelConfig cfg = xllm::ModelConfig::from_file(dir + "/config.json");
  std::vector<int> prompt = load_token_ids("prompt_0_ids.npy");
  const int kTokens = 24;

  auto run_concurrency = [&](int C, long& decode_steps) {
    xllm::Scheduler sched;
    xllm::Worker worker(factory_for(dir), &sched);
    worker.start();
    // Warm up (model load + graph build) so timing measures steady state.
    {
      auto w = make_req(prompt, 4, cfg.eos_token_ids);
      sched.submit(w);
      int t = 0;
      while (w->tokens.pop(t)) {
      }
    }
    const long steps_before = worker.decode_steps();

    std::vector<std::shared_ptr<xllm::Request>> reqs;
    for (int i = 0; i < C; ++i) reqs.push_back(make_req(prompt, kTokens, cfg.eos_token_ids));
    auto t0 = std::chrono::steady_clock::now();
    for (auto& r : reqs) sched.submit(r);
    std::vector<long> counts(C, 0);
    std::vector<std::thread> consumers;
    for (int i = 0; i < C; ++i) {
      consumers.emplace_back([&, i] {
        int tok = 0;
        while (reqs[i]->tokens.pop(tok)) ++counts[i];
      });
    }
    for (auto& t : consumers) t.join();
    auto t1 = std::chrono::steady_clock::now();
    long total = 0;
    for (long c : counts) total += c;
    decode_steps = worker.decode_steps() - steps_before;
    worker.stop();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    return total / secs;  // tokens/s aggregate
  };

  long steps1 = 0, steps4 = 0, steps8 = 0;
  double tput1 = run_concurrency(1, steps1);
  double tput4 = run_concurrency(4, steps4);
  double tput8 = run_concurrency(8, steps8);
  std::printf("throughput tok/s: C1=%.1f C4=%.1f C8=%.1f (decode steps: %ld/%ld/%ld)\n",
              tput1, tput4, tput8, steps1, steps4, steps8);

  // Aggregate throughput rises with concurrency (documented sanity target ~4x@8).
  CHECK(tput8 > tput1);

  // One eval per decode step: 8 concurrent requests of 24 tokens generate ~192
  // tokens but take ~24 decode steps (batched) — far fewer steps than tokens.
  CHECK(steps8 < 8 * kTokens);
  CHECK(steps8 <= static_cast<long>(kTokens) + 4);  // ~one batched step per token
}
