#!/usr/bin/env bash
# Package the locally-built libmlxforge.dylib + the C ABI header into an
# XCFramework that a distributed Swift package could depend on as a binary
# target. (The framework itself is large and gitignored; regenerate as needed.)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD="$ROOT/build"
DYLIB="$BUILD/libmlxforge.dylib"
OUT="$ROOT/bindings/swift/MLXForge.xcframework"

if [[ ! -f "$DYLIB" ]]; then
  echo "error: $DYLIB not found — build it first:" >&2
  echo "  cmake -S \"$ROOT\" -B \"$BUILD\" && cmake --build \"$BUILD\" --target mlxforge_shared" >&2
  exit 1
fi

HDRS="$(mktemp -d)"
cp "$ROOT/src/capi/mlxforge.h" "$HDRS/"

rm -rf "$OUT"
xcodebuild -create-xcframework \
  -library "$DYLIB" -headers "$HDRS" \
  -output "$OUT"

echo "created $OUT"
