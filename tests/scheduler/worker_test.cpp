// the GPU worker picks up a cross-thread request and generates the
// correct tokens (it is the only thread touching MLX).
#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include "core/config.h"
#include "core/weights.h"
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
