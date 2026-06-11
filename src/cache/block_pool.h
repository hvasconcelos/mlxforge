// Block-pool ("paged") KV storage for prefix caching.
//
// The decode batch itself stays contiguous (BatchKVCache) — MLX has no paged
// SDPA kernel and the golden reference (mlx-lm) has none to gate one against.
// Pages live here instead: immutable KVBlocks of `block_size` tokens (all
// layers, all storage components), keyed by a chained hash of the token-id
// prefix. On admission matched blocks are gathered (copied) into the row's
// cache; on eviction a finished row's K/V is harvested back into the pool.
// Unified memory makes those copies cheap next to the prefill they replace.
//
// A block's key hashes its OWN ids chained onto the previous block's key, so a
// key identifies the entire token prefix up to the block's end — two prompts
// share a block only if they share every token before it. Keys are salted
// (model fingerprint + storage config) so persisted blocks can never cross
// models or quantization settings.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"

namespace mlxforge {

namespace mx = mlx::core;

// Chain `n` token ids onto `prev` (FNV-1a-64). Deterministic across runs so
// hashes can key on-disk blocks.
uint64_t chain_hash(uint64_t prev, const int* ids, int n);

// FNV-1a-64 of a byte string; used to derive the pool's salt (the chain seed)
// from the model fingerprint + storage config.
uint64_t fnv1a(const void* data, std::size_t n, uint64_t seed = 14695981039346656037ull);

// One block_size-token span of cached K/V: per layer the same component vector
// the caches store (1 array dense fp16, 3 quantized), each
// (1, n_kv_heads, block_size, comp_dim). Immutable once pooled.
struct KVBlock {
  std::vector<std::vector<mx::array>> k;  // [layer][component]
  std::vector<std::vector<mx::array>> v;
  std::size_t bytes = 0;  // summed component bytes (LRU budget accounting)
};

// Sum of all component buffer sizes — the block's RAM cost.
std::size_t block_bytes(const KVBlock& b);

// hash -> KVBlock map with LRU eviction under a byte budget. Worker-thread-only
// (it holds MLX arrays, which are thread-bound); cross-thread visibility goes
// through the Worker's atomics, never through this object.
class BlockPool {
 public:
  // budget_bytes == 0 means "unbounded" (matching the kv_budget convention).
  explicit BlockPool(std::size_t budget_bytes) : budget_(budget_bytes) {}

  // Look up a block, bumping it to most-recently-used. nullptr on miss.
  std::shared_ptr<const KVBlock> get(uint64_t hash);

  // Membership test without touching recency (harvest uses it to skip slicing
  // blocks that are already pooled).
  bool contains(uint64_t hash) const { return map_.count(hash) != 0; }

  // Insert (no-op if present), then evict LRU entries until under budget. A
  // single block larger than the whole budget is not admitted at all. In-flight
  // references (shared_ptr held by an admission) survive eviction.
  void insert(uint64_t hash, std::shared_ptr<const KVBlock> block);

  // Phase-2 spill hook: called with each evicted (hash, block) before it is
  // dropped from RAM, so an SSD tier can persist it.
  using EvictFn = std::function<void(uint64_t, const std::shared_ptr<const KVBlock>&)>;
  void set_evict_hook(EvictFn fn) { on_evict_ = std::move(fn); }

  std::size_t bytes() const { return bytes_; }
  std::size_t size() const { return map_.size(); }
  std::size_t budget_bytes() const { return budget_; }

 private:
  void evict_to_budget();

  struct Entry {
    std::shared_ptr<const KVBlock> block;
    std::list<uint64_t>::iterator lru_it;
  };
  std::list<uint64_t> lru_;  // front = most recent, back = eviction victim
  std::unordered_map<uint64_t, Entry> map_;
  std::size_t budget_;
  std::size_t bytes_ = 0;
  EvictFn on_evict_;
};

}  // namespace mlxforge
