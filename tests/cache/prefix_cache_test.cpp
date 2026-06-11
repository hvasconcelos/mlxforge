// PrefixCache pure-logic tests: longest-prefix matching, the last-token
// recompute clamp, harvest sealing/dedup, and seeding a BatchKVCache from
// matched blocks (dense and quantized) — tiny synthetic tensors, no model.
#include <doctest/doctest.h>

#include <vector>

#include "cache/batch_kv_cache.h"
#include "cache/kv_quant.h"
#include "cache/prefix_cache.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;

namespace {

constexpr int kH = 2, kD = 64;  // head_dim a multiple of the quant group size

// Deterministic varied values (constants would quantize trivially).
mx::array varied(int len, float phase) {
  mx::array a = mx::arange(static_cast<float>(kH * len * kD));
  a = mx::sin(mx::add(mx::multiply(a, mx::array(0.37f)), mx::array(phase)));
  return mx::astype(mx::reshape(a, {1, kH, len, kD}), mx::float16);
}

bool same(const mx::array& a, const mx::array& b) {
  mx::array eq = mx::allclose(a, b, /*rtol=*/0.0, /*atol=*/1e-6);
  mx::eval(eq);
  return eq.item<bool>();
}

std::vector<int> iota_ids(int n, int start = 0) {
  std::vector<int> ids(n);
  for (int i = 0; i < n; ++i) ids[i] = start + i;
  return ids;
}

// A row's dense K/V (one component) per layer, len positions.
struct FakeRow {
  std::vector<mx::array> k, v;  // [layer], (1, kH, len, kD)
  FakeRow(int n_layers, int len) {
    for (int l = 0; l < n_layers; ++l) {
      k.push_back(varied(len, 0.1f + l));
      v.push_back(varied(len, 0.2f + l));
    }
  }
  PrefixCache::LayerFetch fetch() const {
    return [this](int l) {
      return std::make_pair(std::vector<mx::array>{k[l]}, std::vector<mx::array>{v[l]});
    };
  }
};

}  // namespace

TEST_CASE("match on an empty pool reuses nothing") {
  PrefixCache pc({true, 16, 1ull << 20, 0});
  PrefixCache::Match m = pc.match(iota_ids(64));
  CHECK(m.tokens == 0);
  CHECK(m.blocks.empty());
}

TEST_CASE("harvest seals only full blocks; match clamps to len-1") {
  const int bs = 16;
  PrefixCache pc({true, bs, 1ull << 20, 0});
  FakeRow row(/*n_layers=*/2, /*len=*/40);  // 2 full blocks + 8-token tail
  pc.harvest(iota_ids(40), 40, 2, row.fetch());
  CHECK(pc.pool_blocks() == 2);  // the partial tail is never sealed

  // A longer prompt sharing both blocks reuses all 32 cached tokens.
  PrefixCache::Match m = pc.match(iota_ids(64));
  CHECK(m.tokens == 32);
  CHECK(m.blocks.size() == 2);
  // The pooled content is exactly the harvested span.
  CHECK(same(m.blocks[0]->k[0][0], mx::slice(row.k[0], {0, 0, 0, 0}, {1, kH, bs, kD})));
  CHECK(same(m.blocks[1]->v[1][0], mx::slice(row.v[1], {0, 0, bs, 0}, {1, kH, 2 * bs, kD})));

  // A prompt that IS the cached span still recomputes its last token.
  m = pc.match(iota_ids(32));
  CHECK(m.tokens == 31);
  CHECK(m.blocks.size() == 2);  // both needed to cover [0, 31)
}

TEST_CASE("a diverging block ends the match (keys chain over the whole prefix)") {
  const int bs = 16;
  PrefixCache pc({true, bs, 1ull << 20, 0});
  FakeRow row(1, 32);
  pc.harvest(iota_ids(32), 32, 1, row.fetch());

  std::vector<int> ids = iota_ids(64);
  ids[20] = 9999;  // mutate inside block 1
  PrefixCache::Match m = pc.match(ids);
  CHECK(m.tokens == bs);  // block 0 still matches, block 1 no longer can
  ids[3] = 9999;  // mutate inside block 0: nothing matches
  CHECK(pc.match(ids).tokens == 0);
}

TEST_CASE("re-harvesting the same row is a no-op (dedup against the pool)") {
  PrefixCache pc({true, 16, 1ull << 20, 0});
  FakeRow row(1, 32);
  pc.harvest(iota_ids(32), 32, 1, row.fetch());
  const std::size_t bytes = pc.pool_bytes();
  pc.harvest(iota_ids(32), 32, 1, row.fetch());
  CHECK(pc.pool_bytes() == bytes);
  CHECK(pc.pool_blocks() == 2);

  // Extending the row later seals only the new block.
  FakeRow longer(1, 48);
  pc.harvest(iota_ids(48), 48, 1, longer.fetch());
  CHECK(pc.pool_blocks() == 3);
}

TEST_CASE("pool eviction under a small budget loses the oldest prefix") {
  const int bs = 16;
  FakeRow a(1, bs), b(1, bs);
  PrefixCache pc({true, bs, 0, 0});  // measure one block first
  pc.harvest(iota_ids(bs), bs, 1, a.fetch());
  const std::size_t one_block = pc.pool_bytes();

  PrefixCache small({true, bs, one_block, 0});
  small.harvest(iota_ids(bs, 0), bs, 1, a.fetch());
  small.harvest(iota_ids(bs, 1000), bs, 1, b.fetch());  // different prefix -> evicts the first
  CHECK(small.pool_blocks() == 1);
  CHECK(small.match(iota_ids(bs + 1, 0)).tokens == 0);
  CHECK(small.match(iota_ids(bs + 1, 1000)).tokens == bs);
}

TEST_CASE("salt separates pools (a persisted block can never cross models)") {
  FakeRow row(1, 16);
  PrefixCache pc1({true, 16, 1ull << 20, 1});
  pc1.harvest(iota_ids(16), 16, 1, row.fetch());
  PrefixCache pc2({true, 16, 1ull << 20, 2});
  // Same ids, different salt: pc2's keys don't collide with pc1's content.
  CHECK(pc2.match(iota_ids(17)).tokens == 0);
  CHECK(pc1.match(iota_ids(17)).tokens == 16);
}

TEST_CASE("BatchKVCache::from_prefix seeds a batch-1 cache from matched blocks") {
  const int bs = 16, n_layers = 2, len = 48;
  PrefixCache pc({true, bs, 1ull << 24, 0});
  FakeRow row(n_layers, len);
  pc.harvest(iota_ids(len), len, n_layers, row.fetch());

  PrefixCache::Match m = pc.match(iota_ids(len));  // clamps to 47
  CHECK(m.tokens == len - 1);
  BatchKVCache cache = BatchKVCache::from_prefix(n_layers, m.blocks, m.tokens);
  CHECK(cache.batch_size() == 1);
  CHECK(cache.idx() == m.tokens);
  CHECK(cache.s_cap() % BatchKVCache::kStep == 0);  // standard block-grow rounding

  // RoPE offset == cached length; no left padding.
  mx::array off = cache.offset();
  mx::eval(off);
  CHECK(off.item<int>() == m.tokens);
  CHECK(cache.left_padding_host() == std::vector<int>{0});

  for (int l = 0; l < n_layers; ++l) {
    auto [k, v] = cache.fetch(l);
    CHECK(same(k, mx::slice(row.k[l], {0, 0, 0, 0}, {1, kH, m.tokens, kD})));
    CHECK(same(v, mx::slice(row.v[l], {0, 0, 0, 0}, {1, kH, m.tokens, kD})));
  }

  // Appending after the seed continues at the right position.
  mx::array k1 = varied(1, 7.0f), v1 = varied(1, 8.0f);
  auto [ks, vs] = cache.update_and_fetch(0, k1, v1);
  CHECK(ks.shape()[2] == m.tokens + 1);
}

TEST_CASE("from_prefix round-trips quantized (triplet) blocks") {
  const int bs = 16, len = 32;
  const KVQuantConfig qc{8, 64};
  PrefixCache pc({true, bs, 1ull << 24, 0});

  // The harvested row stores triplets, exactly like a quantized BatchKVCache.
  mx::array kd = varied(len, 0.3f), vd = varied(len, 0.6f);
  std::vector<mx::array> kt = mx::quantize(kd, qc.group_size, qc.bits);
  std::vector<mx::array> vt = mx::quantize(vd, qc.group_size, qc.bits);
  pc.harvest(iota_ids(len), len, 1, [&](int) { return std::make_pair(kt, vt); });
  CHECK(pc.pool_blocks() == 2);

  PrefixCache::Match m = pc.match(iota_ids(len + 8));
  CHECK(m.tokens == len);
  BatchKVCache cache = BatchKVCache::from_prefix(1, m.blocks, m.tokens, qc);
  auto [k, v] = cache.fetch_dequantized(0);
  CHECK(same(k, mx::dequantize(kt[0], kt[1], kt[2], qc.group_size, qc.bits)));
  CHECK(same(v, mx::dequantize(vt[0], vt[1], vt[2], qc.group_size, qc.bits)));
}
