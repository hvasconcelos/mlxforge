// continuous-batching decode — concurrent requests each produce the
// same greedy stream as a solo run; finished/cancelled rows are evicted.
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

TEST_CASE("concurrent requests each match their solo greedy run") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");

  // Solo expectations (computed with the validated single-stream loop).
  mlxforge::LlamaModel& solo_model = shared_model();
  std::vector<std::vector<int>> prompts = {load_token_ids("prompt_0_ids.npy"),
                                           load_token_ids("prompt_1_ids.npy"),
                                           load_token_ids("prompt_2_ids.npy")};
  const int kMax = 16;
  std::vector<std::vector<int>> expected;
  for (auto& p : prompts)
    expected.push_back(mlxforge::greedy_generate(solo_model, p, kMax, cfg.eos_token_ids).tokens);

  // Run them concurrently through the continuous-batching worker.
  mlxforge::Scheduler sched;
  mlxforge::Worker worker(
      [dir] {
        mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
        auto w = mlxforge::load_weights(dir, c);
        return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
      },
      &sched);
  worker.start();

  std::vector<std::shared_ptr<mlxforge::Request>> reqs;
  for (auto& p : prompts) {
    auto req = std::make_shared<mlxforge::Request>();
    req->prompt_ids = p;
    req->params.temperature = 0.0f;
    req->max_tokens = kMax;
    req->eos_ids = cfg.eos_token_ids;
    reqs.push_back(req);
    sched.submit(req);
  }

  // Drain each request's stream (queues are bounded but large enough to buffer).
  std::vector<std::vector<int>> got(reqs.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    int tok = 0;
    while (reqs[i]->tokens.pop(tok)) got[i].push_back(tok);
  }
  worker.stop();

  for (size_t i = 0; i < reqs.size(); ++i) {
    INFO("request " << i);
    assert_tokens_equal(got[i], expected[i]);
  }
}

TEST_CASE("a cancelled request is evicted") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");

  mlxforge::Scheduler sched;
  mlxforge::Worker worker(
      [dir] {
        mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
        auto w = mlxforge::load_weights(dir, c);
        return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
      },
      &sched);
  worker.start();

  auto req = std::make_shared<mlxforge::Request>();
  req->prompt_ids = load_token_ids("prompt_0_ids.npy");
  req->params.temperature = 0.0f;
  req->max_tokens = 1000;
  req->eos_ids = cfg.eos_token_ids;
  req->cancelled.store(true);  // cancel before it is admitted
  sched.submit(req);

  int tok = 0;
  int count = 0;
  while (req->tokens.pop(tok)) ++count;  // drains immediately
  worker.stop();

  CHECK(count == 0);
  CHECK(req->finish_reason == "cancel");
}
