// SSD spill tier for the prefix-cache block pool.
//
// Blocks evicted from the RAM pool (BlockPool's evict hook) are serialized and
// written to one file per block (<hash>.kvb) under a spill directory; a pool
// miss loads the file back (PrefixCache's miss hook). The directory is
// rescanned at construction, so the prefix cache survives engine restarts.
//
// Threading (the MLX thread-bound rule): BlockStore itself never touches MLX
// arrays — its writer thread and file index handle only raw byte buffers. The
// array <-> bytes conversions (serialize_block / deserialize_block) must run on
// the worker thread, which owns every pooled array. Writes are asynchronous
// (queued to the writer thread); reads are synchronous on the caller's thread —
// an SSD read of a few MB replaces a far more expensive prefill.
//
// On-disk format (version 1), little-endian, one block per file:
//   magic u64 ("MLXFKVB1"), version u32, salt u64, n_layers u32,
//   k_comps u32, v_comps u32, then per layer, K components then V components:
//   dtype u32 (0 = float16, 1 = uint32), shape i32[4], nbytes u64, raw bytes.
// The salt (model fingerprint + storage config + block size) is verified on
// load, and block keys are salted, so a file can never be revived for a
// different model or quantization setting. Files are created 0600 — the cache
// holds conversation content.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache/block_pool.h"

namespace mlxforge {

// KVBlock -> bytes. Worker thread only (reads array buffers; defensively
// contiguous+eval'd first, per the data<T>()-on-views gotcha).
std::vector<char> serialize_block(const KVBlock& block, uint64_t salt);

// bytes -> KVBlock. Worker thread only (creates MLX arrays). Returns nullptr on
// any mismatch (magic, version, salt) or truncation — the caller treats it as
// a plain miss, never an error.
std::shared_ptr<KVBlock> deserialize_block(const std::vector<char>& bytes, uint64_t salt);

class BlockStore {
 public:
  // Creates `dir` if needed and rescans existing *.kvb files into the index
  // (LRU-seeded by file modification time). budget_bytes == 0 means unbounded.
  BlockStore(std::string dir, std::size_t budget_bytes, uint64_t salt);
  ~BlockStore();  // drains queued writes, then joins the writer thread

  BlockStore(const BlockStore&) = delete;
  BlockStore& operator=(const BlockStore&) = delete;

  // Queue a serialized block for writing (any thread; the writer thread does
  // the file IO, then enforces the disk budget by deleting LRU files).
  void put(uint64_t hash, std::vector<char> bytes);

  // Read a block's bytes back (any thread, synchronous). Bumps LRU recency.
  // nullopt on miss or unreadable file.
  std::optional<std::vector<char>> get(uint64_t hash);

  bool contains(uint64_t hash) const;
  std::size_t bytes() const;
  std::size_t size() const;

 private:
  void writer_loop();
  std::string path_of(uint64_t hash) const;
  // Index mutation under m_; eviction scans for the smallest recency stamp
  // (the index stays small enough that a linear scan beats bookkeeping).
  void evict_to_budget_locked();

  struct Entry {
    std::size_t bytes = 0;
    uint64_t stamp = 0;  // recency: higher = more recently used
  };

  const std::string dir_;
  const std::size_t budget_;
  const uint64_t salt_;

  mutable std::mutex m_;
  std::condition_variable cv_;
  std::unordered_map<uint64_t, Entry> index_;
  std::size_t bytes_ = 0;
  uint64_t stamp_ = 0;
  std::deque<std::pair<uint64_t, std::vector<char>>> queue_;
  bool stop_ = false;
  std::thread writer_;
};

}  // namespace mlxforge
