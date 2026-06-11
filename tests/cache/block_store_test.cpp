// BlockStore pure-logic tests: serialize/deserialize byte round-trips (dense
// and quantized layouts), salt/corruption rejection, async write + reload,
// restart rescan, and the disk budget — temp dirs, no model.
#include <doctest/doctest.h>

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include "cache/block_store.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"

using namespace mlxforge;
namespace mx = mlx::core;
namespace fs = std::filesystem;

namespace {

constexpr uint64_t kSalt = 0xabcdef;

// A scoped temp dir so failed runs don't leak files.
struct TempDir {
  std::string path;
  TempDir() {
    char tmpl[] = "/tmp/mlxforge_block_store_XXXXXX";
    path = mkdtemp(tmpl);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

std::shared_ptr<KVBlock> dense_block(int n_layers, int tokens, float phase) {
  auto b = std::make_shared<KVBlock>();
  for (int l = 0; l < n_layers; ++l) {
    mx::array base = mx::arange(static_cast<float>(2 * tokens * 8));
    base = mx::sin(mx::add(mx::multiply(base, mx::array(0.37f)), mx::array(phase + l)));
    mx::array k = mx::astype(mx::reshape(base, {1, 2, tokens, 8}), mx::float16);
    mx::array v = mx::astype(mx::negative(mx::reshape(base, {1, 2, tokens, 8})), mx::float16);
    mx::eval(k, v);
    b->k.push_back({k});
    b->v.push_back({v});
  }
  b->bytes = block_bytes(*b);
  return b;
}

std::shared_ptr<KVBlock> quantized_block(int tokens) {
  mx::array base = mx::arange(static_cast<float>(2 * tokens * 64));
  base = mx::astype(mx::reshape(mx::sin(base), {1, 2, tokens, 64}), mx::float16);
  auto b = std::make_shared<KVBlock>();
  b->k.push_back(mx::quantize(base, 64, 8));
  b->v.push_back(mx::quantize(mx::negative(base), 64, 8));
  b->bytes = block_bytes(*b);
  return b;
}

bool blocks_equal(const KVBlock& a, const KVBlock& b) {
  if (a.k.size() != b.k.size()) return false;
  auto comps_equal = [](const std::vector<mx::array>& x, const std::vector<mx::array>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
      mx::array eq = mx::array_equal(x[i], y[i]);
      mx::eval(eq);
      if (!eq.item<bool>()) return false;
    }
    return true;
  };
  for (size_t l = 0; l < a.k.size(); ++l) {
    if (!comps_equal(a.k[l], b.k[l]) || !comps_equal(a.v[l], b.v[l])) return false;
  }
  return true;
}

// Wait until the async writer has persisted `n` blocks (bounded poll).
void wait_for_size(const BlockStore& store, std::size_t n) {
  for (int i = 0; i < 500 && store.size() < n; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

}  // namespace

TEST_CASE("serialize/deserialize round-trips a dense block exactly") {
  auto b = dense_block(/*n_layers=*/3, /*tokens=*/16, 0.5f);
  std::vector<char> bytes = serialize_block(*b, kSalt);
  std::shared_ptr<KVBlock> back = deserialize_block(bytes, kSalt);
  REQUIRE(back != nullptr);
  CHECK(back->bytes == b->bytes);
  CHECK(blocks_equal(*b, *back));
}

TEST_CASE("serialize/deserialize round-trips a quantized (triplet) block exactly") {
  auto b = quantized_block(/*tokens=*/16);
  std::vector<char> bytes = serialize_block(*b, kSalt);
  std::shared_ptr<KVBlock> back = deserialize_block(bytes, kSalt);
  REQUIRE(back != nullptr);
  CHECK(back->k[0].size() == 3);
  CHECK(blocks_equal(*b, *back));
}

TEST_CASE("deserialize rejects the wrong salt, corruption, and truncation") {
  auto b = dense_block(1, 8, 0.1f);
  std::vector<char> bytes = serialize_block(*b, kSalt);

  CHECK(deserialize_block(bytes, kSalt + 1) == nullptr);  // another model/config

  std::vector<char> corrupt = bytes;
  corrupt[0] ^= 0xff;  // magic
  CHECK(deserialize_block(corrupt, kSalt) == nullptr);

  std::vector<char> truncated(bytes.begin(), bytes.begin() + bytes.size() / 2);
  CHECK(deserialize_block(truncated, kSalt) == nullptr);

  CHECK(deserialize_block(std::vector<char>{}, kSalt) == nullptr);
}

TEST_CASE("BlockStore writes asynchronously and reads back the same bytes") {
  TempDir dir;
  auto b = dense_block(2, 16, 0.2f);
  std::vector<char> bytes = serialize_block(*b, kSalt);
  BlockStore store(dir.path, /*budget=*/0, kSalt);
  CHECK(!store.contains(11));
  store.put(11, bytes);
  wait_for_size(store, 1);
  REQUIRE(store.contains(11));
  auto back = store.get(11);
  REQUIRE(back.has_value());
  CHECK(*back == bytes);
  CHECK(store.bytes() == bytes.size());
}

TEST_CASE("BlockStore rescan revives the index across restarts") {
  TempDir dir;
  std::vector<char> bytes = serialize_block(*dense_block(1, 8, 0.3f), kSalt);
  {
    BlockStore store(dir.path, 0, kSalt);
    store.put(21, bytes);
    store.put(22, bytes);
  }  // dtor drains the write queue

  // Foreign files in the dir are ignored by the rescan.
  std::ofstream(dir.path + "/notes.txt") << "not a block";
  std::ofstream(dir.path + "/zzzz.kvb") << "bad name";  // unparseable hex is fine to skip...

  BlockStore revived(dir.path, 0, kSalt);
  CHECK(revived.size() >= 2);
  CHECK(revived.contains(21));
  CHECK(revived.contains(22));
  auto back = revived.get(21);
  REQUIRE(back.has_value());
  CHECK(deserialize_block(*back, kSalt) != nullptr);
}

TEST_CASE("BlockStore disk budget deletes the least recently used file") {
  TempDir dir;
  std::vector<char> bytes = serialize_block(*dense_block(1, 8, 0.4f), kSalt);
  BlockStore store(dir.path, /*budget=*/bytes.size() * 2, kSalt);
  store.put(1, bytes);
  store.put(2, bytes);
  wait_for_size(store, 2);
  CHECK(store.get(1).has_value());  // bump 1; 2 becomes the victim
  store.put(3, bytes);
  for (int i = 0; i < 500 && store.contains(2); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  CHECK(store.contains(1));
  CHECK(!store.contains(2));
  CHECK(store.contains(3));
  CHECK(store.bytes() == bytes.size() * 2);
}
