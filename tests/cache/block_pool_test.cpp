// BlockPool pure-logic tests: chained prefix hashing, LRU eviction under a
// byte budget, and the Phase-2 evict hook — tiny synthetic tensors, no model.
#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include "cache/block_pool.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;

namespace {

// A minimal pooled block: one layer, dense K/V of `tokens` positions.
std::shared_ptr<KVBlock> make_block(int tokens, float fill) {
  auto b = std::make_shared<KVBlock>();
  b->k = {{mx::full({1, 2, tokens, 8}, fill, mx::float16)}};
  b->v = {{mx::full({1, 2, tokens, 8}, -fill, mx::float16)}};
  mx::eval(b->k[0][0], b->v[0][0]);
  b->bytes = block_bytes(*b);
  return b;
}

}  // namespace

TEST_CASE("chain_hash identifies the whole prefix, not just the block") {
  const std::vector<int> a = {1, 2, 3, 4};
  const std::vector<int> b = {5, 6, 7, 8};

  // Deterministic for the same chain.
  const uint64_t h1 = chain_hash(chain_hash(0, a.data(), 4), b.data(), 4);
  const uint64_t h2 = chain_hash(chain_hash(0, a.data(), 4), b.data(), 4);
  CHECK(h1 == h2);

  // The same block ids behind a different first block hash differently.
  CHECK(chain_hash(chain_hash(0, b.data(), 4), b.data(), 4) != h1);

  // A different salt (seed) changes every key in the chain.
  CHECK(chain_hash(chain_hash(99, a.data(), 4), b.data(), 4) != h1);

  // Single-token difference inside a block changes its key.
  std::vector<int> a2 = a;
  a2[2] = 30;
  CHECK(chain_hash(0, a2.data(), 4) != chain_hash(0, a.data(), 4));
}

TEST_CASE("block_bytes sums every component buffer") {
  auto b = make_block(4, 1.0f);
  // 2 tensors (K+V) * 1 layer * 1 component * (1*2*4*8) fp16 elements.
  CHECK(b->bytes == 2 * 2 * 4 * 8 * sizeof(uint16_t));
}

TEST_CASE("BlockPool insert/get with LRU eviction under the byte budget") {
  auto b = make_block(4, 1.0f);  // 256 bytes each
  BlockPool pool(/*budget_bytes=*/b->bytes * 2);

  pool.insert(1, make_block(4, 1.0f));
  pool.insert(2, make_block(4, 2.0f));
  CHECK(pool.size() == 2);
  CHECK(pool.bytes() == b->bytes * 2);

  // Touch 1 so 2 becomes the LRU victim of the next insert.
  CHECK(pool.get(1) != nullptr);
  pool.insert(3, make_block(4, 3.0f));
  CHECK(pool.size() == 2);
  CHECK(pool.get(2) == nullptr);  // evicted
  CHECK(pool.get(1) != nullptr);
  CHECK(pool.get(3) != nullptr);
}

TEST_CASE("BlockPool re-insert of an existing key is a no-op") {
  BlockPool pool(/*budget_bytes=*/0);  // unbounded
  pool.insert(7, make_block(4, 1.0f));
  const std::size_t bytes = pool.bytes();
  pool.insert(7, make_block(4, 2.0f));
  CHECK(pool.size() == 1);
  CHECK(pool.bytes() == bytes);
  // First write wins (blocks are immutable; same key == same content).
  mx::array k = pool.get(7)->k[0][0];
  mx::eval(k);
  CHECK(static_cast<float>(k.data<mx::float16_t>()[0]) == doctest::Approx(1.0f));
}

TEST_CASE("BlockPool rejects a block larger than the whole budget") {
  auto small = make_block(4, 1.0f);
  BlockPool pool(small->bytes);
  pool.insert(1, std::move(small));
  CHECK(pool.size() == 1);
  pool.insert(2, make_block(8, 1.0f));  // 2x the budget: not admitted
  CHECK(pool.size() == 1);
  CHECK(pool.get(1) != nullptr);  // and the existing entry survived
}

TEST_CASE("BlockPool evict hook sees each victim before it is dropped") {
  auto b = make_block(4, 1.0f);
  BlockPool pool(b->bytes * 2);
  std::vector<uint64_t> evicted;
  pool.set_evict_hook([&](uint64_t h, const std::shared_ptr<const KVBlock>&) {
    evicted.push_back(h);
  });
  pool.insert(1, make_block(4, 1.0f));
  pool.insert(2, make_block(4, 2.0f));
  pool.insert(3, make_block(4, 3.0f));
  pool.insert(4, make_block(4, 4.0f));
  CHECK(evicted == std::vector<uint64_t>{1, 2});
}

TEST_CASE("in-flight references survive pool eviction") {
  auto b = make_block(4, 5.0f);
  BlockPool pool(b->bytes);
  pool.insert(1, std::move(b));
  std::shared_ptr<const KVBlock> ref = pool.get(1);  // an admission holding the block
  pool.insert(2, make_block(4, 6.0f));               // evicts key 1
  CHECK(pool.get(1) == nullptr);
  REQUIRE(ref != nullptr);  // but the gathered copy is still usable
  mx::array k = ref->k[0][0];
  mx::eval(k);
  CHECK(static_cast<float>(k.data<mx::float16_t>()[0]) == doctest::Approx(5.0f));
}
