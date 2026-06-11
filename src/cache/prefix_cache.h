// Prefix cache: longest-prefix matching + harvest policy over a BlockPool.
//
// match() turns a prompt's token ids into the longest chain of consecutive
// cached full blocks; the worker seeds the row's cache with them and prefills
// only the suffix. harvest() seals a finished row's K/V back into the pool so
// the next turn of the same conversation (or the next request sharing the
// system prompt) hits. Both run on the worker thread only (MLX arrays are
// thread-bound; see BlockPool).
//
// At least the prompt's last token is always recomputed (cached_len is clamped
// to prompt_len - 1) so the admission still produces next-token logits — the
// same rule vLLM/SGLang use.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "cache/block_pool.h"

namespace mlxforge {

// Engine-wide prefix-cache setting (the pool stores the engine's single storage
// layout, so this cannot be per-request — same reasoning as KVQuantConfig).
struct PrefixCacheConfig {
  bool enabled = false;
  // Hashing/pooling granularity in tokens. Smaller blocks match shorter shared
  // prefixes but cost more hash/slice overhead per token. Power of two in
  // [16, 4096] (validated at engine creation).
  int block_size = 256;
  // RAM budget for pooled blocks (LRU beyond it); 0 = unbounded.
  std::size_t pool_bytes = 1ull << 30;
  // Salt folded into every block key: model fingerprint + storage config, so
  // persisted blocks can never cross models or quantization settings.
  uint64_t salt = 0;
  // SSD spill tier: RAM-evicted blocks persist under this directory and are
  // reloaded on a pool miss (also across engine restarts). Empty = no spill.
  std::string spill_dir;
  // Disk budget for spilled blocks (LRU-deleted beyond it); 0 = unbounded.
  std::size_t spill_bytes = 0;
};

class PrefixCache {
 public:
  explicit PrefixCache(PrefixCacheConfig cfg) : cfg_(cfg), pool_(cfg.pool_bytes) {}

  struct Match {
    std::vector<std::shared_ptr<const KVBlock>> blocks;  // consecutive from position 0
    int tokens = 0;  // cached token count to reuse (<= blocks * block_size, >= 0)
  };

  // Longest run of consecutive cached full blocks covering ids[0..], clamped to
  // ids.size() - 1 tokens. Hits bump the blocks' LRU recency.
  Match match(const std::vector<int>& ids);

  // Per-layer accessor a harvest caller provides: the row's populated K/V
  // component vectors, each component (1, n_kv_heads, len, comp_dim) covering
  // ids[0..len). Views are fine — harvest materializes its own copies.
  using LayerFetch =
      std::function<std::pair<std::vector<mx::array>, std::vector<mx::array>>(int layer)>;

  // Seal every full block of ids[0..len) not already pooled and insert it.
  // Slices are materialized (mx::contiguous + one eval over all new blocks) so
  // the pool never pins the batch cache's buffers.
  void harvest(const std::vector<int>& ids, int len, int n_layers, const LayerFetch& fetch);

  // Second-level lookup consulted when the RAM pool misses (the SSD tier):
  // returns the revived block or nullptr. The revived block is re-promoted
  // into the pool. Runs inside match() on the worker thread.
  using MissFn = std::function<std::shared_ptr<const KVBlock>(uint64_t)>;
  void set_miss_fn(MissFn fn) { on_miss_ = std::move(fn); }

  const PrefixCacheConfig& config() const { return cfg_; }
  std::size_t pool_bytes() const { return pool_.bytes(); }
  std::size_t pool_blocks() const { return pool_.size(); }
  BlockPool& pool() { return pool_; }  // spill (evict) hook installation

 private:
  PrefixCacheConfig cfg_;
  BlockPool pool_;
  MissFn on_miss_;
};

}  // namespace mlxforge
