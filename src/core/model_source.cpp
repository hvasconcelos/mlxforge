#include "core/model_source.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/env.h"
#include "core/hf_download.h"
#include "core/logging.h"

namespace fs = std::filesystem;

namespace mlxforge {

namespace {

std::string home_dir() { return env_or("HOME", ""); }

// "org/name" -> "models--org--name" (the HF hub on-disk directory name).
std::string hf_repo_dirname(const std::string& repo_id) {
  std::string d = "models--";
  for (char c : repo_id) d += (c == '/') ? std::string("--") : std::string(1, c);
  return d;
}

// "org/name" -> "org__name" for the flat mlxforge download cache.
std::string flat_cache_name(const std::string& repo_id) {
  std::string d;
  for (char c : repo_id) d += (c == '/') ? std::string("__") : std::string(1, c);
  return d;
}

// Read a one-line ref file (refs/<rev>) and return the trimmed commit hash.
std::string read_ref(const fs::path& ref_path) {
  std::ifstream f(ref_path);
  if (!f) return "";
  std::string hash;
  std::getline(f, hash);
  // trim surrounding whitespace/newline
  const auto a = hash.find_first_not_of(" \t\r\n");
  const auto b = hash.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? "" : hash.substr(a, b - a + 1);
}

// Resolve the snapshot dir given the on-disk "models--..." dir, or "" if absent.
std::string snapshot_from_repo_cache(const fs::path& repo_cache_dir) {
  const std::string hash = read_ref(repo_cache_dir / "refs" / "main");
  if (hash.empty()) return "";
  const fs::path snap = repo_cache_dir / "snapshots" / hash;
  std::error_code ec;
  return fs::is_directory(snap, ec) ? snap.string() : "";
}

}  // namespace

bool is_model_dir(const std::string& dir) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return false;
  return fs::is_regular_file(fs::path(dir) / "config.json", ec);
}

bool looks_like_repo_id(const std::string& spec) {
  if (spec.empty()) return false;
  std::error_code ec;
  if (fs::exists(spec, ec)) return false;  // it's a path, not a repo id
  const auto slash = spec.find('/');
  if (slash == std::string::npos) return false;
  if (spec.find('/', slash + 1) != std::string::npos) return false;  // only one '/'
  if (slash == 0 || slash == spec.size() - 1) return false;          // non-empty halves
  for (char c : spec) {
    if (c == '/') continue;
    const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

std::string hf_cache_snapshot_dir(const std::string& hub_cache_root, const std::string& repo_id) {
  if (hub_cache_root.empty()) return "";
  const fs::path repo_dir = fs::path(hub_cache_root) / hf_repo_dirname(repo_id);
  return snapshot_from_repo_cache(repo_dir);
}

std::string hf_hub_cache_root() {
  std::string root = env_or("HF_HUB_CACHE", "");
  if (!root.empty()) return root;
  const std::string hf_home = env_or("HF_HOME", "");
  if (!hf_home.empty()) return (fs::path(hf_home) / "hub").string();
  const std::string home = home_dir();
  if (home.empty()) return "";
  return (fs::path(home) / ".cache" / "huggingface" / "hub").string();
}

std::string mlxforge_cache_root() {
  const std::string env = env_or("MLXFORGE_CACHE", "");
  if (!env.empty()) return env;
  const std::string home = home_dir();
  if (home.empty()) return ".mlxforge-cache";  // last resort if $HOME is unset
  return (fs::path(home) / ".cache" / "mlxforge").string();
}

std::string resolve_model_dir(const std::string& spec) {
  // Tier 1: an explicit local model directory.
  if (is_model_dir(spec)) {
    log::info("model: using local dir '{}'", spec);
    return spec;
  }

  std::error_code ec;
  if (fs::exists(spec, ec) && fs::is_directory(spec, ec)) {
    // An existing dir without config.json. Common mistake: passing the HF cache
    // parent (.../models--org--name) instead of its snapshots/<rev>/ subdir.
    const std::string snap = snapshot_from_repo_cache(spec);
    if (!snap.empty() && is_model_dir(snap)) {
      log::info("model: resolved HF cache parent '{}' to snapshot '{}'", spec, snap);
      return snap;
    }
    throw std::runtime_error("model: directory '" + spec +
                             "' has no config.json (did you mean its snapshots/<rev>/ subdir?)");
  }

  // Tiers 2-4: an HF repo id.
  if (looks_like_repo_id(spec)) {
    // Reuse a model already in the standard HF hub cache.
    const std::string hub_snap = hf_cache_snapshot_dir(hf_hub_cache_root(), spec);
    if (!hub_snap.empty() && is_model_dir(hub_snap)) {
      log::info("model: found '{}' in HF cache at '{}'", spec, hub_snap);
      return hub_snap;
    }

    // Reuse a prior mlxforge download.
    const std::string dest = (fs::path(mlxforge_cache_root()) / flat_cache_name(spec)).string();
    if (is_model_dir(dest)) {
      log::info("model: using cached download '{}'", dest);
      return dest;
    }

    // Download.
    log::info("model: '{}' not cached; downloading to '{}'", spec, dest);
    return hf_download_repo(spec, dest);
  }

  throw std::runtime_error("model: '" + spec +
                           "' is neither a directory with config.json nor a HuggingFace repo id "
                           "(expected 'org/name')");
}

}  // namespace mlxforge
