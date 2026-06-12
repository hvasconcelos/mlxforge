// the GPU worker picks up a cross-thread request and generates the
// correct tokens (it is the only thread touching MLX).
#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include "core/config.h"
#include "core/weights.h"
#include "runtime/single_stream.h"
#include "runtime/worker.h"
#include "scheduler/request.h"
#include "scheduler/scheduler.h"
#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;

TEST_CASE("worker processes a request submitted from another thread") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");  // no MLX

  mlxforge::Scheduler sched;
  mlxforge::Worker worker(
      [dir] {
        mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
        auto w = mlxforge::load_weights(dir, c);
        return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
      },
      &sched);
  worker.start();  // loads the model on the worker thread

  auto req = std::make_shared<mlxforge::Request>();
  req->prompt_ids = load_token_ids("prompt_0_ids.npy");
  req->params.temperature = 0.0f;  // greedy -> deterministic, matches reference
  req->max_tokens = 20;
  req->eos_ids = cfg.eos_token_ids;

  sched.submit(req);  // from this (non-worker) thread

  // Consume streamed tokens until the worker closes the queue.
  std::vector<int> got;
  int tok = 0;
  while (req->tokens.pop(tok)) got.push_back(tok);

  worker.stop();

  assert_tokens_equal(got, load_token_ids("greedy_tokens.npy"));
  CHECK(req->finish_reason == "length");
}

TEST_CASE("chunked prefill reproduces the reference greedy stream across chunk sizes") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");

  // A prompt long enough that prefill_chunk = 8 spans several chunks; the
  // expectation comes from the validated single-stream loop.
  std::vector<int> prompt;
  for (const char* name : {"prompt_0_ids.npy", "prompt_1_ids.npy", "prompt_2_ids.npy"}) {
    std::vector<int> ids = load_token_ids(name);
    prompt.insert(prompt.end(), ids.begin(), ids.end());
  }
  const int kMax = 16;
  const std::vector<int> expect =
      mlxforge::greedy_generate(shared_model(), prompt, kMax, cfg.eos_token_ids).tokens;

  for (int chunk : {0, 8}) {  // monolithic and aggressively chunked must agree
    CAPTURE(chunk);
    mlxforge::Scheduler sched;
    mlxforge::Worker worker(
        [dir] {
          mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
          auto w = mlxforge::load_weights(dir, c);
          return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
        },
        &sched, /*tok=*/nullptr, /*kv_quant=*/{}, /*prefix=*/{}, chunk);
    worker.start();

    // Two simultaneous submissions land in one batched (left-padded) cold unit.
    auto make = [&] {
      auto r = std::make_shared<mlxforge::Request>();
      r->prompt_ids = prompt;
      r->params.temperature = 0.0f;
      r->max_tokens = kMax;
      r->eos_ids = cfg.eos_token_ids;
      return r;
    };
    auto a = make(), b = make();
    REQUIRE(sched.submit(a));
    REQUIRE(sched.submit(b));
    for (const auto& r : {a, b}) {
      std::vector<int> got;
      int tok = 0;
      while (r->tokens.pop(tok)) got.push_back(tok);
      assert_tokens_equal(got, expect);
    }
    worker.stop();
  }
}

TEST_CASE("skinny_mm decode kernels reproduce the stock-matmul greedy stream") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");

  std::vector<int> prompt;
  for (const char* name : {"prompt_0_ids.npy", "prompt_1_ids.npy", "prompt_2_ids.npy"}) {
    std::vector<int> ids = load_token_ids(name);
    prompt.insert(prompt.end(), ids.begin(), ids.end());
  }
  const int kMax = 16;
  const int kBatch = 6;  // B > 4 routes decode linears through the MMA tile kernel

  // Reuse may only change speed, never tokens: the kernel-on batch must match
  // the kernel-off batch row for row (both greedy on identical prompts).
  std::vector<std::vector<int>> streams[2];
  for (bool skinny : {false, true}) {
    mlxforge::Scheduler sched;
    mlxforge::Worker worker(
        [dir] {
          mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
          auto w = mlxforge::load_weights(dir, c);
          return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
        },
        &sched, /*tok=*/nullptr, /*kv_quant=*/{}, /*prefix=*/{}, /*prefill_chunk=*/256, skinny);
    worker.start();

    std::vector<std::shared_ptr<mlxforge::Request>> reqs;
    for (int i = 0; i < kBatch; ++i) {
      auto r = std::make_shared<mlxforge::Request>();
      r->prompt_ids = prompt;
      r->params.temperature = 0.0f;
      r->max_tokens = kMax;
      r->eos_ids = cfg.eos_token_ids;
      REQUIRE(sched.submit(r));
      reqs.push_back(std::move(r));
    }
    for (const auto& r : reqs) {
      std::vector<int> got;
      int tok = 0;
      while (r->tokens.pop(tok)) got.push_back(tok);
      streams[skinny].push_back(std::move(got));
    }
    worker.stop();
  }
  for (int i = 0; i < kBatch; ++i) {
    CAPTURE(i);
    assert_tokens_equal(streams[1][i], streams[0][i]);
  }
}
