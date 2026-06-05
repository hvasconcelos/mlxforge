// The Engine facade loads a model and drives generation end to end: construct
// it from a local model dir, wait for readiness, submit a Request through its
// scheduler, and drain the streamed tokens — the same path the HTTP server uses,
// with no HTTP layer involved.
#include <doctest/doctest.h>

#include <memory>
#include <thread>
#include <vector>

#include "runtime/engine.h"
#include "scheduler/request.h"
#include "support/model_fixture.h"

using namespace mlxforge::test;

TEST_CASE("engine loads a model and generates tokens via its scheduler") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }

  // The spec is the local snapshot dir; resolve_model_dir uses it as-is.
  mlxforge::Engine engine(mlxforge::EngineConfig{model_dir(), /*max_waiting=*/256});

  // The worker loads the weights on its own thread; wait until it is ready.
  while (!engine.ready()) std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto req = std::make_shared<mlxforge::Request>();
  req->prompt_ids = engine.tokenizer().encode("The capital of France is");
  req->params.temperature = 0.0f;  // greedy -> deterministic
  req->max_tokens = 8;
  req->eos_ids = engine.config().eos_token_ids;

  CHECK(engine.scheduler().submit(req));

  std::vector<int> got;
  int tok = 0;
  while (req->tokens.pop(tok)) got.push_back(tok);

  CHECK(got.size() == 8);                 // hit max_tokens (none of these are EOS)
  CHECK(req->finish_reason == "length");
}
