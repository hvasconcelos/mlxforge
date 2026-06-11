#include "cache/block_store.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#include "core/logging.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace fs = std::filesystem;

namespace mlxforge {

namespace {

constexpr uint64_t kMagic = 0x3142564b46584c4dull;  // "MLXFKVB1" little-endian
constexpr uint32_t kVersion = 1;

// dtype codes in the file format. Only the two storage dtypes exist: dense and
// quantized scales/biases are fp16, quantized packed words are uint32.
uint32_t dtype_code(const mx::array& a) {
  if (a.dtype() == mx::float16) return 0;
  if (a.dtype() == mx::uint32) return 1;
  throw std::logic_error("serialize_block: unexpected KV storage dtype");
}

template <typename T>
void append(std::vector<char>& out, const T& v) {
  const char* p = reinterpret_cast<const char*>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

// Bounds-checked sequential reader over the serialized buffer.
struct Reader {
  const char* p;
  std::size_t left;
  template <typename T>
  bool read(T& v) {
    if (left < sizeof(T)) return false;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    left -= sizeof(T);
    return true;
  }
};

}  // namespace

std::vector<char> serialize_block(const KVBlock& block, uint64_t salt) {
  // Defensive contiguous+eval before data<char>(): pooled blocks are already
  // materialized contiguous copies, but a lazy view here would silently
  // serialize the wrong elements. Order matches deserialize_block: per layer,
  // K components then V components.
  std::vector<mx::array> comps;
  for (std::size_t l = 0; l < block.k.size(); ++l) {
    for (const auto& c : block.k[l]) comps.push_back(mx::contiguous(c));
    for (const auto& c : block.v[l]) comps.push_back(mx::contiguous(c));
  }
  mx::eval(comps);

  std::vector<char> out;
  out.reserve(64 + block_bytes(block));
  append(out, kMagic);
  append(out, kVersion);
  append(out, salt);
  append(out, static_cast<uint32_t>(block.k.size()));
  append(out, static_cast<uint32_t>(block.k.empty() ? 0 : block.k[0].size()));
  append(out, static_cast<uint32_t>(block.v.empty() ? 0 : block.v[0].size()));
  for (const auto& c : comps) {
    append(out, dtype_code(c));
    for (int d = 0; d < 4; ++d) append(out, static_cast<int32_t>(c.shape()[d]));
    append(out, static_cast<uint64_t>(c.nbytes()));
    const char* p = c.data<char>();
    out.insert(out.end(), p, p + c.nbytes());
  }
  return out;
}

std::shared_ptr<KVBlock> deserialize_block(const std::vector<char>& bytes, uint64_t salt) {
  Reader r{bytes.data(), bytes.size()};
  uint64_t magic = 0, file_salt = 0;
  uint32_t version = 0, n_layers = 0, k_comps = 0, v_comps = 0;
  if (!r.read(magic) || magic != kMagic) return nullptr;
  if (!r.read(version) || version != kVersion) return nullptr;
  if (!r.read(file_salt) || file_salt != salt) return nullptr;
  if (!r.read(n_layers) || !r.read(k_comps) || !r.read(v_comps)) return nullptr;
  if (n_layers == 0 || n_layers > 4096 || k_comps > 3 || v_comps > 3) return nullptr;

  auto read_comp = [&](mx::array& out) {
    uint32_t code = 0;
    int32_t shape[4];
    uint64_t nbytes = 0;
    if (!r.read(code) || code > 1) return false;
    for (int d = 0; d < 4; ++d) {
      if (!r.read(shape[d]) || shape[d] <= 0) return false;
    }
    if (!r.read(nbytes) || r.left < nbytes) return false;
    const mx::Shape s{shape[0], shape[1], shape[2], shape[3]};
    const std::size_t elems =
        static_cast<std::size_t>(shape[0]) * shape[1] * shape[2] * shape[3];
    if (code == 0) {
      if (nbytes != elems * sizeof(uint16_t)) return false;
      out = mx::array(reinterpret_cast<const mx::float16_t*>(r.p), s);
    } else {
      if (nbytes != elems * sizeof(uint32_t)) return false;
      out = mx::array(reinterpret_cast<const uint32_t*>(r.p), s);
    }
    r.p += nbytes;
    r.left -= nbytes;
    return true;
  };

  auto block = std::make_shared<KVBlock>();
  block->k.resize(n_layers);
  block->v.resize(n_layers);
  for (uint32_t l = 0; l < n_layers; ++l) {
    for (uint32_t i = 0; i < k_comps; ++i) {
      mx::array c = mx::zeros({0});
      if (!read_comp(c)) return nullptr;
      block->k[l].push_back(std::move(c));
    }
    for (uint32_t i = 0; i < v_comps; ++i) {
      mx::array c = mx::zeros({0});
      if (!read_comp(c)) return nullptr;
      block->v[l].push_back(std::move(c));
    }
  }
  block->bytes = block_bytes(*block);
  return block;
}

BlockStore::BlockStore(std::string dir, std::size_t budget_bytes, uint64_t salt)
    : dir_(std::move(dir)), budget_(budget_bytes), salt_(salt) {
  fs::create_directories(dir_);
  // Rescan surviving blocks, LRU-seeded by modification time so the budget
  // evicts the stalest first. Unparseable names are ignored (foreign files).
  struct Found {
    fs::file_time_type mtime;
    uint64_t hash;
    std::size_t bytes;
  };
  std::vector<Found> found;
  for (const auto& e : fs::directory_iterator(dir_)) {
    if (!e.is_regular_file() || e.path().extension() != ".kvb") continue;
    uint64_t hash = 0;
    try {
      hash = std::stoull(e.path().stem().string(), nullptr, 16);
    } catch (...) {
      continue;
    }
    found.push_back({e.last_write_time(), hash, static_cast<std::size_t>(e.file_size())});
  }
  std::sort(found.begin(), found.end(),
            [](const Found& a, const Found& b) { return a.mtime < b.mtime; });
  for (const Found& f : found) {
    index_[f.hash] = Entry{f.bytes, ++stamp_};
    bytes_ += f.bytes;
  }
  log::info("block store: {} ({} blocks, {} bytes on disk)", dir_, index_.size(), bytes_);
  writer_ = std::thread([this] { writer_loop(); });
}

BlockStore::~BlockStore() {
  {
    std::lock_guard<std::mutex> lk(m_);
    stop_ = true;
  }
  cv_.notify_all();
  if (writer_.joinable()) writer_.join();
}

std::string BlockStore::path_of(uint64_t hash) const {
  char name[32];
  std::snprintf(name, sizeof(name), "%016llx.kvb", static_cast<unsigned long long>(hash));
  return dir_ + "/" + name;
}

void BlockStore::put(uint64_t hash, std::vector<char> bytes) {
  {
    std::lock_guard<std::mutex> lk(m_);
    if (index_.count(hash) != 0) return;  // already on disk
    queue_.emplace_back(hash, std::move(bytes));
  }
  cv_.notify_one();
}

std::optional<std::vector<char>> BlockStore::get(uint64_t hash) {
  {
    std::lock_guard<std::mutex> lk(m_);
    auto it = index_.find(hash);
    if (it == index_.end()) {
      // Not on disk yet — it may still be in flight in the write queue (an
      // immediate re-request right after a spill). Serve it from there so the
      // async writer can never lose a hit.
      for (const auto& [h, bytes] : queue_) {
        if (h == hash) return bytes;
      }
      return std::nullopt;
    }
    it->second.stamp = ++stamp_;
  }
  std::ifstream f(path_of(hash), std::ios::binary | std::ios::ate);
  if (!f) return std::nullopt;
  std::vector<char> bytes(static_cast<std::size_t>(f.tellg()));
  f.seekg(0);
  f.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!f) return std::nullopt;
  return bytes;
}

bool BlockStore::contains(uint64_t hash) const {
  std::lock_guard<std::mutex> lk(m_);
  if (index_.count(hash) != 0) return true;
  for (const auto& [h, _] : queue_) {
    if (h == hash) return true;
  }
  return false;
}

std::size_t BlockStore::bytes() const {
  std::lock_guard<std::mutex> lk(m_);
  return bytes_;
}

std::size_t BlockStore::size() const {
  std::lock_guard<std::mutex> lk(m_);
  return index_.size();
}

void BlockStore::writer_loop() {
  for (;;) {
    std::pair<uint64_t, std::vector<char>> job;
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
      if (queue_.empty()) return;  // stop_ and drained
      // Copy (don't pop): the entry must stay visible to get()/contains()
      // while the file is being written, or an immediate re-request would
      // land in the gap and silently miss.
      job = queue_.front();
    }
    // Write to a temp name then rename so a crash never leaves a torn block,
    // and 0600 the file: the cache is conversation content. No fsync — this is
    // a cache, not a durability contract.
    const std::string path = path_of(job.first);
    const std::string tmp = path + ".tmp";
    bool ok = false;
    {
      std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      if (f) {
        f.write(job.second.data(), static_cast<std::streamsize>(job.second.size()));
        ok = static_cast<bool>(f);
      }
    }
    if (ok) {
      ::chmod(tmp.c_str(), 0600);
      std::error_code ec;
      fs::rename(tmp, path, ec);
      ok = !ec;
    }
    if (!ok) {
      log::warn("block store: failed to persist {}", path);
      std::error_code ec;
      fs::remove(tmp, ec);
    }

    std::lock_guard<std::mutex> lk(m_);
    if (ok) {
      index_[job.first] = Entry{job.second.size(), ++stamp_};
      bytes_ += job.second.size();
      evict_to_budget_locked();
    }
    queue_.pop_front();  // the index entry (or the drop) is now authoritative
  }
}

void BlockStore::evict_to_budget_locked() {
  if (budget_ == 0) return;
  while (bytes_ > budget_ && index_.size() > 1) {
    auto victim = index_.begin();
    for (auto it = index_.begin(); it != index_.end(); ++it) {
      if (it->second.stamp < victim->second.stamp) victim = it;
    }
    std::error_code ec;
    fs::remove(path_of(victim->first), ec);
    bytes_ -= victim->second.bytes;
    index_.erase(victim);
  }
}

}  // namespace mlxforge
