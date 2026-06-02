#pragma once

// Tiny helpers for reading environment-variable overrides. An unset *or empty*
// value means "not provided" and falls back to the default — so e.g. `FOO=`
// behaves the same as `FOO` being unset. Used by the logger, the model-source
// resolver, and anywhere else honoring an MLXFORGE_*/HF_* knob.

#include <string>

namespace mlxforge {

// Returns the value of environment variable `key`, or `fallback` if it is unset
// or empty.
std::string env_or(const char* key, const std::string& fallback);

// Like env_or but parses the value as a long. Returns `fallback` if unset/empty;
// throws std::invalid_argument if set to a non-numeric string.
long env_long(const char* key, long fallback);

}  // namespace mlxforge
