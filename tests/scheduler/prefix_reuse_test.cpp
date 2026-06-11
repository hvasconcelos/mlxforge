// End-to-end prefix-cache gate through the continuous-batching worker: with
// the prefix cache on, a warm request (same or extended prompt) must produce
// the exact greedy stream of a cold solo run — reuse may only change speed,
// never tokens — and the worker's metrics must show the reuse happened.
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

namespace {

std::vector<int> long_prompt() {
  std::vector<int> p;
  for (const char* name : {"prompt_2_ids.npy", "prompt_0_ids.npy", "prompt_1_ids.npy",
                           "prompt_2_ids.npy", "prompt_0_ids.npy", "prompt_1_ids.npy"}) {
    std::vector<int> ids = load_token_ids(name);
    p.insert(p.end(), ids.begin(), ids.end());
  }
  return p;  // 46 tokens: several 16-token blocks
}

// Submit one greedy request and drain its stream (the worker harvests the row
// into the prefix pool when it finishes).
std::vector<int> run_one(mlxforge::Scheduler& sched, const std::vector<int>& prompt,
                         const std::vector<int>& eos_ids, int max_tokens) {
  auto req = std::make_shared<mlxforge::Request>();
  req->prompt_ids = prompt;
  req->params.temperature = 0.0f;
  req->max_tokens = max_tokens;
  req->eos_ids = eos_ids;
  REQUIRE(sched.submit(req));
  std::vector<int> got;
  int tok = 0;
  while (req->tokens.pop(tok)) got.push_back(tok);
  return got;
}

}  // namespace

TEST_CASE("prefix-cache reuse reproduces the cold greedy stream exactly") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  const int kMax = 16;

  const std::vector<int> turn1 = long_prompt();
  // Solo expectations from the validated single-stream loop. Each turn's
  // prompt extends the full prior conversation (prompt + answer + new text),
  // the multi-turn shape prefix caching exists for.
  mlxforge::LlamaModel& solo = shared_model();
  const std::vector<int> expect1 =
      mlxforge::greedy_generate(solo, turn1, kMax, cfg.eos_token_ids).tokens;
  auto extend = [](std::vector<int> conv, const std::vector<int>& answer, const char* next) {
    conv.insert(conv.end(), answer.begin(), answer.end());
    const std::vector<int> extra = load_token_ids(next);
    conv.insert(conv.end(), extra.begin(), extra.end());
    return conv;
  };
  const std::vector<int> turn2 = extend(turn1, expect1, "prompt_0_ids.npy");
  const std::vector<int> expect2 =
      mlxforge::greedy_generate(solo, turn2, kMax, cfg.eos_token_ids).tokens;
  const std::vector<int> turn3 = extend(turn2, expect2, "prompt_1_ids.npy");
  const std::vector<int> expect3 =
      mlxforge::greedy_generate(solo, turn3, kMax, cfg.eos_token_ids).tokens;

  // Worker with the prefix cache on (block 16: the fixture prompts are short;
  // the engine's >= 16 floor still holds).
  mlxforge::PrefixCacheConfig pcfg;
  pcfg.enabled = true;
  pcfg.block_size = 16;
  pcfg.pool_bytes = 1ull << 30;
  pcfg.salt = 7;
  mlxforge::Scheduler sched;
  mlxforge::Worker worker(
      [dir] {
        mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
        auto w = mlxforge::load_weights(dir, c);
        return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
      },
      &sched, /*tok=*/nullptr, /*kv_quant=*/{}, pcfg);
  worker.start();

  // Cold: no pool content yet.
  CHECK(run_one(sched, turn1, cfg.eos_token_ids, kMax) == expect1);
  CHECK(worker.metrics().prefix_hits == 0);
  CHECK(worker.metrics().prefix_pool_blocks > 0);  // the finished row was harvested

  // Warm 1: identical prompt — served from the pool, same tokens.
  CHECK(run_one(sched, turn1, cfg.eos_token_ids, kMax) == expect1);
  // Warm 2: the multi-turn continuation. Its prefix hit covers turn 1's
  // prompt blocks; harvesting turn 2's own (seeded) prefill then pools the
  // conversation through turn 1's answer.
  CHECK(run_one(sched, turn2, cfg.eos_token_ids, kMax) == expect2);
  // Warm 3 reuses blocks spanning turn 1's *generated* text — pooled by turn
  // 2's prompt prefill (decode-produced K/V itself is never pooled; see
  // Worker::harvest_finished).
  CHECK(run_one(sched, turn3, cfg.eos_token_ids, kMax) == expect3);

  const mlxforge::WorkerMetrics m = worker.metrics();
  CHECK(m.prefix_hits == 3);
  const long long bs = 16;
  // turn1 warm reuses turn1's full blocks; turn2 reuses the same; turn3
  // reuses turn2's full blocks (its prompt was pooled when turn2 finished).
  const long long expected_reuse = (static_cast<long long>(turn1.size()) / bs) * bs * 2 +
                                   (static_cast<long long>(turn2.size()) / bs) * bs;
  CHECK(m.prefix_tokens_reused == expected_reuse);
  CHECK(m.prefix_pool_bytes > 0);
  CHECK(m.prefix_pool_blocks >= static_cast<long>(turn3.size() / bs));

  worker.stop();
}
