#include "server/config.h"

#include <cstdlib>
#include <stdexcept>

namespace mlxforge {

namespace {
// Environment override: returns the env value or `fallback` if unset.
std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : fallback;
}
long env_long(const char* key, long fallback) {
  const char* v = std::getenv(key);
  return v ? std::stol(v) : fallback;
}
}  // namespace

ServerConfig ServerConfig::parse(const std::vector<std::string>& args) {
  ServerConfig c;
  // Environment defaults first; CLI flags override below.
  c.host = env_or("MLXFORGE_HOST", c.host);
  c.port = static_cast<int>(env_long("MLXFORGE_PORT", c.port));
  c.max_ctx = static_cast<int>(env_long("MLXFORGE_MAX_CTX", c.max_ctx));
  c.max_waiting = static_cast<int>(env_long("MLXFORGE_MAX_WAITING", c.max_waiting));
  c.kv_budget_bytes = static_cast<std::size_t>(env_long("MLXFORGE_KV_BUDGET", 0));

  // Accept "--flag value" or "--flag=value"; the first positional is model_dir.
  auto value_of = [&](const std::string& a, size_t& i) -> std::string {
    auto eq = a.find('=');
    if (eq != std::string::npos) return a.substr(eq + 1);
    if (i + 1 >= args.size()) throw std::runtime_error("missing value for flag " + a);
    return args[++i];
  };
  auto name_of = [](const std::string& a) -> std::string {
    auto eq = a.find('=');
    return eq == std::string::npos ? a : a.substr(0, eq);
  };

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a.rfind("--", 0) != 0) {
      if (c.model_dir.empty()) {
        c.model_dir = a;
        continue;
      }
      throw std::runtime_error("unexpected positional argument: " + a);
    }
    const std::string flag = name_of(a);
    if (flag == "--host")
      c.host = value_of(a, i);
    else if (flag == "--port")
      c.port = std::stoi(value_of(a, i));
    else if (flag == "--max-ctx")
      c.max_ctx = std::stoi(value_of(a, i));
    else if (flag == "--max-waiting")
      c.max_waiting = std::stoi(value_of(a, i));
    else if (flag == "--kv-budget")
      c.kv_budget_bytes = static_cast<std::size_t>(std::stoll(value_of(a, i)));
    else
      throw std::runtime_error("unknown flag: " + flag);
  }
  return c;
}

}  // namespace mlxforge
