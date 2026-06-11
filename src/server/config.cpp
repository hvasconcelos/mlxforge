#include "server/config.h"

#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mlxforge {

namespace {

// Returns the value of the given environment variable, or `fallback` if unset.
// Used to optionally override config values via the environment.
std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : fallback;
}

// Returns the value of the given environment variable parsed as long, or `fallback` if unset.
// Used for env overrides of integer config fields.
long env_long(const char* key, long fallback) {
  const char* v = std::getenv(key);
  return v ? std::stol(v) : fallback;
}

// Fetches a typed config-file field, wrapping a type mismatch in a descriptive
// error. Mirrors the require<T> helper in core/config.cpp.
template <typename T>
T require_type(const nlohmann::json& j, const char* key) {
  try {
    return j.at(key).get<T>();
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("config file: field '") + key +
                             "' has wrong type: " + e.what());
  }
}

}  // namespace

// Loads + validates a JSON config file into a ServerConfig. See header.
ServerConfig ServerConfig::from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    throw std::runtime_error("config file: cannot open '" + path + "'");
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("config file: parse error in '" + path + "': " + e.what());
  }
  if (!j.is_object()) {
    throw std::runtime_error("config file: '" + path + "' must contain a JSON object");
  }

  // Reject unknown keys up front so typos (e.g. "prot") fail loudly.
  static const std::set<std::string> kKnownKeys = {
      "model",     "host",    "port",         "max_ctx",  "max_waiting",  "kv_budget",
      "kv_bits",   "prefix_cache", "kv_block", "kv_pool", "kv_spill_dir", "kv_spill_bytes"};
  for (const auto& [key, _] : j.items()) {
    if (kKnownKeys.find(key) == kKnownKeys.end()) {
      throw std::runtime_error("config file: unknown key '" + key + "' in '" + path + "'");
    }
  }

  // Apply onto a default-constructed config; omitted keys keep struct defaults.
  ServerConfig c;
  if (j.contains("model")) c.model_dir = require_type<std::string>(j, "model");
  if (j.contains("host")) c.host = require_type<std::string>(j, "host");
  if (j.contains("port")) {
    c.port = require_type<int>(j, "port");
    if (c.port < 1 || c.port > 65535)
      throw std::runtime_error("config file: 'port' must be in 1..65535");
  }
  if (j.contains("max_ctx")) {
    c.max_ctx = require_type<int>(j, "max_ctx");
    if (c.max_ctx <= 0) throw std::runtime_error("config file: 'max_ctx' must be > 0");
  }
  if (j.contains("max_waiting")) {
    c.max_waiting = require_type<int>(j, "max_waiting");
    if (c.max_waiting < 0) throw std::runtime_error("config file: 'max_waiting' must be >= 0");
  }
  if (j.contains("kv_budget")) {
    long long budget = require_type<long long>(j, "kv_budget");
    if (budget < 0) throw std::runtime_error("config file: 'kv_budget' must be >= 0");
    c.kv_budget_bytes = static_cast<std::size_t>(budget);
  }
  if (j.contains("kv_bits")) {
    c.kv_bits = require_type<int>(j, "kv_bits");
    if (c.kv_bits != 0 && c.kv_bits != 4 && c.kv_bits != 8)
      throw std::runtime_error("config file: 'kv_bits' must be 0, 4, or 8");
  }
  if (j.contains("prefix_cache")) c.prefix_cache = require_type<bool>(j, "prefix_cache");
  if (j.contains("kv_block")) {
    c.kv_block = require_type<int>(j, "kv_block");
    if (c.kv_block <= 0) throw std::runtime_error("config file: 'kv_block' must be > 0");
  }
  if (j.contains("kv_pool")) {
    long long pool = require_type<long long>(j, "kv_pool");
    if (pool < 0) throw std::runtime_error("config file: 'kv_pool' must be >= 0");
    c.kv_pool_bytes = static_cast<std::size_t>(pool);
  }
  if (j.contains("kv_spill_dir")) c.kv_spill_dir = require_type<std::string>(j, "kv_spill_dir");
  if (j.contains("kv_spill_bytes")) {
    long long spill = require_type<long long>(j, "kv_spill_bytes");
    if (spill < 0) throw std::runtime_error("config file: 'kv_spill_bytes' must be >= 0");
    c.kv_spill_bytes = static_cast<std::size_t>(spill);
  }
  return c;
}

// Construct a ServerConfig from a CLI-style argument vector, layering sources by
// precedence: defaults < config file (-c/--config) < environment < CLI flags.
// The model is given via -m/--model (or the config file's "model" key).
ServerConfig ServerConfig::parse(const std::vector<std::string>& args) {
  // Helper: extract flag name, stripping any "=" and trailing value
  auto name_of = [](const std::string& a) -> std::string {
    auto eq = a.find('=');
    return eq == std::string::npos ? a : a.substr(0, eq);
  };
  auto is_config_flag = [&](const std::string& a) {
    const std::string flag = name_of(a);
    return flag == "-c" || flag == "--config";
  };

  // Pass 1: locate the config file flag (-c / --config / --config=path) so the
  // file forms the base layer beneath env vars and the remaining CLI flags.
  ServerConfig c;
  for (size_t i = 0; i < args.size(); ++i) {
    if (!is_config_flag(args[i])) continue;
    std::string path;
    if (auto eq = args[i].find('='); eq != std::string::npos) {
      path = args[i].substr(eq + 1);
    } else if (i + 1 < args.size()) {
      path = args[i + 1];
    } else {
      throw std::runtime_error("missing value for flag " + name_of(args[i]));
    }
    c = ServerConfig::from_file(path);
    break;  // first one wins; CLI flags below still override its values
  }

  // Env layer: override the file/defaults, using the current values as fallbacks
  // so env only takes effect when actually set.
  c.host = env_or("MLXFORGE_HOST", c.host);
  c.port = static_cast<int>(env_long("MLXFORGE_PORT", c.port));
  c.max_ctx = static_cast<int>(env_long("MLXFORGE_MAX_CTX", c.max_ctx));
  c.max_waiting = static_cast<int>(env_long("MLXFORGE_MAX_WAITING", c.max_waiting));
  c.kv_budget_bytes =
      static_cast<std::size_t>(env_long("MLXFORGE_KV_BUDGET", static_cast<long>(c.kv_budget_bytes)));
  c.kv_bits = static_cast<int>(env_long("MLXFORGE_KV_BITS", c.kv_bits));
  c.prefix_cache = env_long("MLXFORGE_PREFIX_CACHE", c.prefix_cache ? 1 : 0) != 0;
  c.kv_block = static_cast<int>(env_long("MLXFORGE_KV_BLOCK", c.kv_block));
  c.kv_pool_bytes =
      static_cast<std::size_t>(env_long("MLXFORGE_KV_POOL", static_cast<long>(c.kv_pool_bytes)));
  c.kv_spill_dir = env_or("MLXFORGE_KV_SPILL_DIR", c.kv_spill_dir);
  c.kv_spill_bytes = static_cast<std::size_t>(
      env_long("MLXFORGE_KV_SPILL_BYTES", static_cast<long>(c.kv_spill_bytes)));

  // Helper: extract value for a flag (accepts "--flag value" or "--flag=value")
  auto value_of = [&](const std::string& a, size_t& i) -> std::string {
    auto eq = a.find('=');
    if (eq != std::string::npos) return a.substr(eq + 1);  // "--flag=foo"
    if (i + 1 >= args.size()) throw std::runtime_error("missing value for flag " + a);
    return args[++i];  // "--flag foo"
  };

  // Pass 2: CLI parsing — flags override env/file/defaults.
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const std::string flag = name_of(a);
    // Config flag was consumed in Pass 1; skip it (and its detached value).
    if (is_config_flag(a)) {
      if (a.find('=') == std::string::npos && i + 1 < args.size()) ++i;
      continue;
    }
    if (flag == "-m" || flag == "--model")
      c.model_dir = value_of(a, i);  // overrides any "model" from the config file
    else if (flag == "--host")
      c.host = value_of(a, i);
    else if (flag == "--port")
      c.port = std::stoi(value_of(a, i));
    else if (flag == "--max-ctx")
      c.max_ctx = std::stoi(value_of(a, i));
    else if (flag == "--max-waiting")
      c.max_waiting = std::stoi(value_of(a, i));
    else if (flag == "--kv-budget")
      c.kv_budget_bytes = static_cast<std::size_t>(std::stoll(value_of(a, i)));
    else if (flag == "--kv-bits")
      c.kv_bits = std::stoi(value_of(a, i));
    else if (flag == "--prefix-cache")
      c.prefix_cache = std::stoi(value_of(a, i)) != 0;
    else if (flag == "--kv-block")
      c.kv_block = std::stoi(value_of(a, i));
    else if (flag == "--kv-pool")
      c.kv_pool_bytes = static_cast<std::size_t>(std::stoll(value_of(a, i)));
    else if (flag == "--kv-spill-dir")
      c.kv_spill_dir = value_of(a, i);
    else if (flag == "--kv-spill-bytes")
      c.kv_spill_bytes = static_cast<std::size_t>(std::stoll(value_of(a, i)));
    else
      throw std::runtime_error("unknown flag: " + flag);
  }
  if (c.kv_bits != 0 && c.kv_bits != 4 && c.kv_bits != 8)
    throw std::runtime_error("--kv-bits must be 0, 4, or 8");
  return c;
}

}  // namespace mlxforge
