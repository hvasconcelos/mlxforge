// MLXFORGE-012: KV memory projection + admission gate (pure arithmetic, no GPU).
#include <doctest/doctest.h>

#include "cache/kv_budget.h"
#include "core/config.h"

using namespace mlxforge;

namespace {
// Llama-3.2-1B shape relevant to KV size.
ModelConfig llama32_1b() {
  ModelConfig c;
  c.n_layers = 16;
  c.n_kv_heads = 8;
  c.head_dim = 64;
  return c;
}
constexpr std::size_t kMiB = 1024 * 1024;
constexpr std::size_t kGiB = 1024 * 1024 * 1024;
}  // namespace

TEST_CASE("MLXFORGE-012: KV size matches the spec's memory math") {
  KVBudget budget(llama32_1b(), /*budget_bytes=*/0);
  CHECK(budget.bytes_per_token() == 32 * 1024);  // 32 KiB/token

  // 1 sequence @ 2048 tokens = 64 MiB.
  CHECK(budget.project_bytes(/*max_len=*/2048, /*max_new=*/0, /*batch=*/1) == 64 * kMiB);
  // Batch 32 @ 2048 ~= 2 GiB.
  CHECK(budget.project_bytes(2048, 0, 32) == 2 * kGiB);
}

TEST_CASE("MLXFORGE-012: admission is refused when projection exceeds the budget") {
  KVBudget budget(llama32_1b(), /*budget_bytes=*/2 * kGiB);

  CHECK(budget.can_admit(2048, 0, 32));         // exactly 2 GiB, fits
  CHECK_FALSE(budget.can_admit(2048, 0, 33));   // over budget -> refuse/queue
  CHECK(budget.can_admit(2048, 0, 16));         // 1 GiB, fits
  CHECK_FALSE(budget.can_admit(2048, 256, 32)); // max_new pushes it over
}

TEST_CASE("MLXFORGE-012: a zero budget is treated as unbounded") {
  KVBudget budget(llama32_1b(), /*budget_bytes=*/0);
  CHECK(budget.can_admit(131072, 4096, 256));
}
