#include "cache/block_pool.h"

namespace mlxforge {

uint64_t fnv1a(const void* data, std::size_t n, uint64_t seed) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t h = seed;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

uint64_t chain_hash(uint64_t prev, const int* ids, int n) {
  // Fold the previous block's key in first so the result identifies the whole
  // prefix, then the block's own ids.
  uint64_t h = fnv1a(&prev, sizeof(prev));
  return fnv1a(ids, static_cast<std::size_t>(n) * sizeof(int), h);
}

std::size_t block_bytes(const KVBlock& b) {
  std::size_t n = 0;
  for (const auto& layer : b.k)
    for (const auto& c : layer) n += c.nbytes();
  for (const auto& layer : b.v)
    for (const auto& c : layer) n += c.nbytes();
  return n;
}

std::shared_ptr<const KVBlock> BlockPool::get(uint64_t hash) {
  auto it = map_.find(hash);
  if (it == map_.end()) return nullptr;
  lru_.splice(lru_.begin(), lru_, it->second.lru_it);  // bump to most recent
  return it->second.block;
}

void BlockPool::insert(uint64_t hash, std::shared_ptr<const KVBlock> block) {
  if (map_.count(hash) != 0) return;  // immutable content; first write wins
  // A block that alone exceeds the budget would just evict the whole pool and
  // then be evicted itself on the next insert — don't admit it.
  if (budget_ != 0 && block->bytes > budget_) return;
  lru_.push_front(hash);
  bytes_ += block->bytes;
  map_.emplace(hash, Entry{std::move(block), lru_.begin()});
  evict_to_budget();
}

void BlockPool::evict_to_budget() {
  if (budget_ == 0) return;
  while (bytes_ > budget_ && !lru_.empty()) {
    const uint64_t victim = lru_.back();
    auto it = map_.find(victim);
    if (on_evict_) on_evict_(victim, it->second.block);
    bytes_ -= it->second.block->bytes;
    map_.erase(it);
    lru_.pop_back();
  }
}

}  // namespace mlxforge
