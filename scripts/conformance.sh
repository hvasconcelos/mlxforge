#!/usr/bin/env bash
# Conformance kit: validate a mlxforge build against the mlx-lm golden reference
# and the C-ABI contract. Run this to confirm a fork/port/packaging change still
# produces numerically-correct output (the failure mode here is silent garbage,
# not a crash — so this is the gate that matters).
#
# What it checks:
#   1. the build compiles;
#   2. the C ABI is intact and the library stays lean (scripts/check-abi.sh);
#   3. the doctest suite passes — including the golden-reference tests that
#      compare embeddings / post-norm / logits / argmax / greedy token streams
#      against committed .npy fixtures (reference/fixtures*), the tokenizer
#      byte-match vs HF golden ids, and the C-ABI behavior.
#
# Note: the numerically-sensitive integration tests SELF-SKIP without local
# weights. For full conformance, cache the reference model first:
#   huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-bf16
# A green run without weights has only exercised the pure-logic units.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

echo "==> [1/3] configure + build (tests + library)"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD" --parallel --target mlxforge_tests mlxforge_shared

echo "==> [2/3] ABI guard"
"$ROOT/scripts/check-abi.sh" "$BUILD/libmlxforge.dylib"

echo "==> [3/3] golden-reference + contract test suite"
if ctest --test-dir "$BUILD" --output-on-failure; then
  echo "==> CONFORMANCE OK"
else
  echo "==> CONFORMANCE FAILED" >&2
  exit 1
fi

# Surface whether the numerical paths actually ran (vs self-skipped).
if ! ls "$HOME"/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/* >/dev/null 2>&1; then
  echo "note: reference model not cached -> the golden-reference / scheduler tests" >&2
  echo "      self-skipped. Download it (see header) to exercise the numerical paths." >&2
fi
