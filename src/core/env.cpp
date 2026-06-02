#include "core/env.h"

#include <cstdlib>
#include <string>

namespace mlxforge {

std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

long env_long(const char* key, long fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::stol(v) : fallback;
}

}  // namespace mlxforge
