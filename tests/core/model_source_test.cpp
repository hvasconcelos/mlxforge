// Model-source resolution + HF download helpers (pure logic, no network/GPU).
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/hf_download.h"
#include "core/model_source.h"

using namespace mlxforge;

namespace {
std::string fixtures() { return std::string(MLXFORGE_TEST_FIXTURES_DIR); }
std::string hf_cache() { return fixtures() + "/hf_cache"; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Set an env var for the duration of a scope, restoring the prior value after.
struct ScopedEnv {
  std::string key;
  bool had_prev = false;
  std::string prev;
  ScopedEnv(const char* k, const char* v) : key(k) {
    if (const char* p = std::getenv(k)) {
      had_prev = true;
      prev = p;
    }
    setenv(k, v, 1);
  }
  ~ScopedEnv() {
    if (had_prev)
      setenv(key.c_str(), prev.c_str(), 1);
    else
      unsetenv(key.c_str());
  }
};
struct ScopedUnset {
  std::string key;
  bool had_prev = false;
  std::string prev;
  explicit ScopedUnset(const char* k) : key(k) {
    if (const char* p = std::getenv(k)) {
      had_prev = true;
      prev = p;
    }
    unsetenv(k);
  }
  ~ScopedUnset() {
    if (had_prev) setenv(key.c_str(), prev.c_str(), 1);
  }
};
}  // namespace

TEST_CASE("looks_like_repo_id accepts well-formed org/name and rejects everything else") {
  CHECK(looks_like_repo_id("mlx-community/Llama-3.2-1B-Instruct-4bit"));
  CHECK(looks_like_repo_id("org/name"));
  CHECK(looks_like_repo_id("a.b_c-d/e.f"));

  CHECK_FALSE(looks_like_repo_id(""));
  CHECK_FALSE(looks_like_repo_id("name"));        // no slash
  CHECK_FALSE(looks_like_repo_id("/name"));       // empty org
  CHECK_FALSE(looks_like_repo_id("name/"));       // empty name
  CHECK_FALSE(looks_like_repo_id("a/b/c"));       // two slashes
  CHECK_FALSE(looks_like_repo_id("a b/c"));       // illegal char (space)
  CHECK_FALSE(looks_like_repo_id("~/x"));         // illegal char (~)
  CHECK_FALSE(looks_like_repo_id(fixtures()));    // existing path, not a repo id
}

TEST_CASE("is_model_dir is true only for a dir containing config.json") {
  CHECK(is_model_dir(hf_cache() + "/models--org--name/snapshots/deadbeef"));
  CHECK_FALSE(is_model_dir(fixtures()));               // no config.json (has config_*.json)
  CHECK_FALSE(is_model_dir(fixtures() + "/nope-1234"));  // does not exist
}

TEST_CASE("hf_cache_snapshot_dir resolves refs/main to the snapshot, else empty") {
  const std::string snap = hf_cache_snapshot_dir(hf_cache(), "org/name");
  CHECK(snap == hf_cache() + "/models--org--name/snapshots/deadbeef");
  CHECK(is_model_dir(snap));

  // refs/main points at a snapshot dir that doesn't exist.
  CHECK(hf_cache_snapshot_dir(hf_cache(), "org/dangling").empty());
  // no on-disk cache entry at all.
  CHECK(hf_cache_snapshot_dir(hf_cache(), "org/missing").empty());
  // empty root.
  CHECK(hf_cache_snapshot_dir("", "org/name").empty());
}

TEST_CASE("hf_hub_cache_root honors HF_HUB_CACHE, then HF_HOME/hub, then default") {
  {
    ScopedEnv e("HF_HUB_CACHE", "/tmp/custom-hub");
    CHECK(hf_hub_cache_root() == "/tmp/custom-hub");
  }
  {
    ScopedUnset u("HF_HUB_CACHE");
    ScopedEnv e("HF_HOME", "/tmp/hf");
    CHECK(hf_hub_cache_root() == "/tmp/hf/hub");
  }
  {
    ScopedUnset u1("HF_HUB_CACHE");
    ScopedUnset u2("HF_HOME");
    ScopedEnv home("HOME", "/tmp/fakehome");
    CHECK(hf_hub_cache_root() == "/tmp/fakehome/.cache/huggingface/hub");
  }
}

TEST_CASE("mlxforge_cache_root honors MLXFORGE_CACHE, then ~/.cache/mlxforge") {
  {
    ScopedEnv e("MLXFORGE_CACHE", "/tmp/my-cache");
    CHECK(mlxforge_cache_root() == "/tmp/my-cache");
  }
  {
    ScopedUnset u("MLXFORGE_CACHE");
    ScopedEnv home("HOME", "/tmp/fakehome");
    CHECK(mlxforge_cache_root() == "/tmp/fakehome/.cache/mlxforge");
  }
}

TEST_CASE("resolve_model_dir resolves an HF cache parent dir to its snapshot") {
  // The reported bug: passing .../models--org--name (the parent) should resolve
  // to its snapshots/<rev>/ subdir instead of failing.
  const std::string parent = hf_cache() + "/models--org--name";
  CHECK(resolve_model_dir(parent) == parent + "/snapshots/deadbeef");

  // A plain local model dir is returned as-is.
  const std::string snap = parent + "/snapshots/deadbeef";
  CHECK(resolve_model_dir(snap) == snap);
}

TEST_CASE("resolve_model_dir rejects a non-model dir and a bogus spec") {
  CHECK_THROWS_AS(resolve_model_dir(fixtures()), std::runtime_error);  // dir, no config.json
  CHECK_THROWS_AS(resolve_model_dir("not a path or repo"), std::runtime_error);
}

TEST_CASE("parse_repo_siblings reads siblings[].rfilename") {
  const auto files = parse_repo_siblings(read_file(fixtures() + "/hf_api_model_info.json"));
  CHECK(files.size() == 8);
  CHECK(std::find(files.begin(), files.end(), "config.json") != files.end());
  CHECK(std::find(files.begin(), files.end(), "model.safetensors") != files.end());
  CHECK(std::find(files.begin(), files.end(), "README.md") != files.end());
}

TEST_CASE("select_files_to_download keeps the inference set and drops the rest") {
  const std::vector<std::string> siblings = {
      ".gitattributes", "README.md",  "config.json",
      "model.safetensors", "model.safetensors.index.json",
      "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
      "pytorch_model.bin", "model.gguf"};
  const auto sel = select_files_to_download(siblings);
  auto has = [&](const std::string& f) {
    return std::find(sel.begin(), sel.end(), f) != sel.end();
  };
  CHECK(has("config.json"));
  CHECK(has("model.safetensors"));
  CHECK(has("model.safetensors.index.json"));
  CHECK(has("tokenizer.json"));
  CHECK(has("tokenizer_config.json"));
  CHECK(has("special_tokens_map.json"));
  CHECK_FALSE(has("README.md"));
  CHECK_FALSE(has(".gitattributes"));
  CHECK_FALSE(has("pytorch_model.bin"));
  CHECK_FALSE(has("model.gguf"));
}

TEST_CASE("select_files_to_download throws when weights or config are missing") {
  CHECK_THROWS_AS(select_files_to_download({"config.json", "tokenizer.json"}),
                  std::runtime_error);  // no safetensors
  CHECK_THROWS_AS(select_files_to_download({"model.safetensors", "tokenizer.json"}),
                  std::runtime_error);  // no config.json
}
