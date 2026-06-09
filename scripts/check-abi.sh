#!/usr/bin/env bash
# ABI guard for libmlxforge. Two checks:
#   1) the (lean) library must not drag in cpp-httplib (server harness leak);
#   2) the exported C ABI is append-only — every symbol in the committed baseline
#      must still be present. New symbols are allowed (and reported); a missing
#      one is a breaking change and fails. See doc/abi-stability.md.
#
# Usage: scripts/check-abi.sh [path/to/libmlxforge.dylib]
#        UPDATE_BASELINE=1 scripts/check-abi.sh ...   # rewrite the baseline
set -euo pipefail

DYLIB="${1:-build/libmlxforge.dylib}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE="$ROOT/cmake/abi-baseline.txt"

[ -f "$DYLIB" ] || { echo "error: $DYLIB not found (build mlxforge_shared first)" >&2; exit 1; }

# 1) No cpp-httplib references (the product library must stay lean).
if nm -u "$DYLIB" 2>/dev/null | grep -qiE "httplib"; then
  echo "error: $DYLIB references cpp-httplib — the server harness leaked into the library" >&2
  exit 1
fi

# 2) Exported C ABI symbols (defined, external, mlxforge_*).
current="$(nm -gUj "$DYLIB" 2>/dev/null | grep -E '^_mlxforge_' | sed 's/^_//' | sort -u)"

if [ "${UPDATE_BASELINE:-0}" = "1" ] || [ ! -f "$BASELINE" ]; then
  printf '%s\n' "$current" > "$BASELINE"
  echo "wrote baseline ($(printf '%s\n' "$current" | grep -c . ) symbols) -> ${BASELINE#"$ROOT/"}"
  exit 0
fi

missing="$(comm -23 <(sort -u "$BASELINE") <(printf '%s\n' "$current") || true)"
if [ -n "$missing" ]; then
  echo "error: C ABI regression — baseline symbols missing from $DYLIB:" >&2
  printf '  %s\n' $missing >&2
  echo "If this removal is intentional it is a BREAKING change: bump the ABI and" >&2
  echo "update the baseline with UPDATE_BASELINE=1 (see doc/abi-stability.md)." >&2
  exit 1
fi

added="$(comm -13 <(sort -u "$BASELINE") <(printf '%s\n' "$current") || true)"
if [ -n "$added" ]; then
  echo "note: new exported symbols (append-only, OK — update the baseline to lock them in):"
  printf '  %s\n' $added
fi

echo "ABI OK: $(printf '%s\n' "$current" | grep -c .) mlxforge_* symbols, baseline intact, no httplib."
