#include "core/model_source.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/env.h"
#include "core/logging.h"
#ifdef MLXFORGE_ENABLE_HF_DOWNLOAD
#include "core/hf_download.h"
#endif

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

namespace {
std::string to_lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool has_gguf_suffix(const std::string& s) {
  if (s.size() < 5) return false;
  return to_lower(s.substr(s.size() - 5)) == ".gguf";
}

// In a previously-downloaded GGUF cache dir, find a `.gguf` whose name contains
// `tag` (case-insensitive), so a re-run reuses it without a network round-trip.
// Returns the path, or "" if the dir is absent or holds no matching file.
std::string find_cached_gguf(const fs::path& dir, const std::string& tag) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return "";
  const std::string want = to_lower(tag);
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (!e.is_regular_file(ec)) continue;
    const std::string name = e.path().filename().string();
    if (has_gguf_suffix(name) && to_lower(name).find(want) != std::string::npos)
      return e.path().string();
  }
  return "";
}
}  // namespace

std::string resolve_model_dir(const std::string& spec) {
  // Tier 0: a local GGUF file (self-contained: config + tokenizer + weights).
  // Returned verbatim; callers detect it with is_gguf_path and route to the
  // GGUF loader instead of expecting a config.json directory.
  std::error_code gec;
  if (has_gguf_suffix(spec) && fs::is_regular_file(spec, gec)) {
    log::info("model: using local GGUF file '{}'", spec);
    return spec;
  }

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

  // GGUF download: a "org/name:VARIANT" spec (the ':' signals a single-file GGUF
  // fetch; the variant, default Q4_0, selects which quant). The colon keeps this
  // distinct from a bare repo id, which still resolves to the safetensors path.
  if (!fs::exists(spec, ec)) {
    const auto colon = spec.find(':');
    if (colon != std::string::npos) {
      const std::string repo = spec.substr(0, colon);
      std::string tag = spec.substr(colon + 1);
      if (tag.empty()) tag = "Q4_0";
      if (looks_like_repo_id(repo)) {
        const fs::path dir =
            fs::path(mlxforge_cache_root()) / (flat_cache_name(repo) + "-gguf");
        const std::string cached = find_cached_gguf(dir, tag);
        if (!cached.empty()) {
          log::info("model: using cached GGUF '{}'", cached);
          return cached;
        }
#ifdef MLXFORGE_ENABLE_HF_DOWNLOAD
        log::info("model: '{}' GGUF ({}) not cached; downloading to '{}'", repo, tag,
                  dir.string());
        return hf_download_gguf(repo, dir.string(), tag);
#else
        throw std::runtime_error("model: '" + spec +
                                 "' is not cached and HuggingFace download is disabled "
                                 "(built without MLXFORGE_ENABLE_HF_DOWNLOAD)");
#endif
      }
    }
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
#ifdef MLXFORGE_ENABLE_HF_DOWNLOAD
    log::info("model: '{}' not cached; downloading to '{}'", spec, dest);
    return hf_download_repo(spec, dest);
#else
    throw std::runtime_error("model: '" + spec +
                             "' is not cached locally and HuggingFace download is disabled "
                             "(built without MLXFORGE_ENABLE_HF_DOWNLOAD)");
#endif
  }

  throw std::runtime_error("model: '" + spec +
                           "' is neither a directory with config.json nor a HuggingFace repo id "
                           "(expected 'org/name')");
}

}  // namespace mlxforge
