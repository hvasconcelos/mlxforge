#include "cache/prefix_cache.h"

#include <algorithm>

#include "cache/kv_quant.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlxforge {

PrefixCache::Match PrefixCache::match(const std::vector<int>& ids) {
  Match m;
  const int bs = cfg_.block_size;
  const int n_full = static_cast<int>(ids.size()) / bs;
  uint64_t h = cfg_.salt;
  for (int b = 0; b < n_full; ++b) {
    h = chain_hash(h, ids.data() + static_cast<std::size_t>(b) * bs, bs);
    std::shared_ptr<const KVBlock> blk = pool_.get(h);
    if (!blk && on_miss_) {
      blk = on_miss_(h);  // SSD tier
      // Re-promote: the revived block is hot again. (A block bigger than the
      // whole pool budget stays un-pooled; the shared_ptr still serves this
      // match.)
      if (blk) pool_.insert(h, blk);
    }
    if (!blk) break;  // keys chain, so the first miss ends every longer match
    m.blocks.push_back(std::move(blk));
  }
  m.tokens = std::max(0, std::min(static_cast<int>(m.blocks.size()) * bs,
                                  static_cast<int>(ids.size()) - 1));
  return m;
}

void PrefixCache::harvest(const std::vector<int>& ids, int len, int n_layers,
                          const LayerFetch& fetch) {
  const int bs = cfg_.block_size;
  const int n_full = std::min(len, static_cast<int>(ids.size())) / bs;
  if (n_full == 0) return;

  // Chain the keys, keeping only blocks the pool doesn't already hold.
  std::vector<uint64_t> hashes;
  std::vector<int> fresh;
  uint64_t h = cfg_.salt;
  for (int b = 0; b < n_full; ++b) {
    h = chain_hash(h, ids.data() + static_cast<std::size_t>(b) * bs, bs);
    hashes.push_back(h);
    if (!pool_.contains(h)) fresh.push_back(b);
  }
  if (fresh.empty()) return;

  // Slice each fresh block out of the row and materialize it (a lazy slice
  // would pin the whole batch buffer past the row's eviction), batching one
  // eval over every new component.
  std::vector<std::shared_ptr<KVBlock>> blocks(fresh.size());
  for (auto& b : blocks) {
    b = std::make_shared<KVBlock>();
    b->k.resize(n_layers);
    b->v.resize(n_layers);
  }
  std::vector<mx::array> to_eval;
  for (int l = 0; l < n_layers; ++l) {
    auto [kc, vc] = fetch(l);
    for (std::size_t j = 0; j < fresh.size(); ++j) {
      const int start = fresh[j] * bs;
      for (const auto& c : kc) {
        blocks[j]->k[l].push_back(mx::contiguous(slice_seq(c, start, start + bs)));
        to_eval.push_back(blocks[j]->k[l].back());
      }
      for (const auto& c : vc) {
        blocks[j]->v[l].push_back(mx::contiguous(slice_seq(c, start, start + bs)));
        to_eval.push_back(blocks[j]->v[l].back());
      }
    }
  }
  mx::eval(to_eval);

  for (std::size_t j = 0; j < fresh.size(); ++j) {
    blocks[j]->bytes = block_bytes(*blocks[j]);
    pool_.insert(hashes[fresh[j]], std::move(blocks[j]));
  }
}

}  // namespace mlxforge
