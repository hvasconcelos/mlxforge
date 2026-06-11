// End-to-end SSD spill gate: with a RAM pool deliberately too small to hold a
// prompt's blocks, reuse must round-trip through the spill tier (evict ->
// serialize -> SSD -> revive) and still reproduce the cold greedy stream
// exactly — including from a fresh Worker on the same spill dir (restart
// persistence).
#include <doctest/doctest.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
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
namespace fs = std::filesystem;

namespace {

std::vector<int> long_prompt() {
  std::vector<int> p;
  for (const char* name : {"prompt_2_ids.npy", "prompt_0_ids.npy", "prompt_1_ids.npy",
                           "prompt_2_ids.npy", "prompt_0_ids.npy", "prompt_1_ids.npy"}) {
    std::vector<int> ids = load_token_ids(name);
    p.insert(p.end(), ids.begin(), ids.end());
  }
  return p;  // 46 tokens: two full 16-token blocks
}

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

TEST_CASE("prefix blocks spill to SSD, revive exactly, and survive a restart") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  const std::string dir = model_dir();
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");
  const int kMax = 16;

  const std::vector<int> prompt = long_prompt();
  const std::vector<int> expected =
      mlxforge::greedy_generate(shared_model(), prompt, kMax, cfg.eos_token_ids).tokens;

  char tmpl[] = "/tmp/mlxforge_prefix_spill_XXXXXX";
  const std::string spill_dir = mkdtemp(tmpl);

  // One 16-token block of this model is n_layers * (K+V) * 8 heads * 64 dims
  // * fp16 = 512 KiB; a 600 KB pool holds exactly one, forcing the other
  // through the spill tier.
  mlxforge::PrefixCacheConfig pcfg;
  pcfg.enabled = true;
  pcfg.block_size = 16;
  pcfg.pool_bytes = 600'000;
  pcfg.salt = 7;
  pcfg.spill_dir = spill_dir;
  auto factory = [dir] {
    mlxforge::ModelConfig c = mlxforge::ModelConfig::from_file(dir + "/config.json");
    auto w = mlxforge::load_weights(dir, c);
    return std::make_unique<mlxforge::LlamaModel>(std::move(c), std::move(w));
  };

  {
    mlxforge::Scheduler sched;
    mlxforge::Worker worker(factory, &sched, nullptr, {}, pcfg);
    worker.start();

    CHECK(run_one(sched, prompt, cfg.eos_token_ids, kMax) == expected);  // cold
    // Harvest inserted 2 blocks into a 1-block pool: one was spilled.
    CHECK(worker.metrics().spill_writes >= 1);

    // Warm: the evicted block must come back from the SSD tier.
    CHECK(run_one(sched, prompt, cfg.eos_token_ids, kMax) == expected);
    const mlxforge::WorkerMetrics m = worker.metrics();
    CHECK(m.prefix_hits == 1);
    CHECK(m.spill_reads >= 1);
    worker.stop();
  }  // ~Worker drains the spill writer: everything queued is now on disk

  // Restart: a fresh worker on the same spill dir starts warm.
  {
    mlxforge::Scheduler sched;
    mlxforge::Worker worker(factory, &sched, nullptr, {}, pcfg);
    worker.start();
    CHECK(run_one(sched, prompt, cfg.eos_token_ids, kMax) == expected);
    const mlxforge::WorkerMetrics m = worker.metrics();
    CHECK(m.prefix_hits == 1);
    CHECK(m.prefix_tokens_reused > 0);
    CHECK(m.spill_reads >= 1);
    worker.stop();
  }

  std::error_code ec;
  fs::remove_all(spill_dir, ec);
}
