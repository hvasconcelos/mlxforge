#!/usr/bin/env bash
# Smoke-test the Qwen3-VL image-to-text path through the CLI: build mlxforge-cli
# if needed, resolve the cached vision model, and run the `image` subcommand.
#
# Usage:
#   scripts/test-image.sh <image_file> ["prompt"] [max_tokens]
#
#   scripts/test-image.sh photo.jpg
#   scripts/test-image.sh photo.jpg "What objects are in this image?" 200
#
# Env overrides:
#   MLXFORGE_VL_MODEL   model spec (local dir or HF repo id); default = the cached
#                       mlx-community/Qwen3-VL-4B-Instruct-4bit snapshot
#   BUILD_DIR           build directory (default ./build)
#
# If no image is given, the committed test fixture (a tiny random image) is used —
# a quick "does the pipeline run" check, not a meaningful caption.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

IMAGE="${1:-$ROOT/reference/fixtures_qwen3_vl/image.png}"
PROMPT="${2:-Describe this image.}"
MAX_TOKENS="${3:-128}"

if [[ ! -f "$IMAGE" ]]; then
  echo "error: image file not found: $IMAGE" >&2
  exit 1
fi

# Resolve the model: explicit override, else the cached Qwen3-VL snapshot.
MODEL="${MLXFORGE_VL_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  HUB="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-4B-Instruct-4bit/snapshots"
  MODEL="$(ls -d "$HUB"/*/ 2>/dev/null | head -1 || true)"
  if [[ -z "$MODEL" ]]; then
    echo "error: Qwen3-VL model not found in the HF cache." >&2
    echo "       download it, or set MLXFORGE_VL_MODEL to a model spec:" >&2
    echo "       huggingface-cli download mlx-community/Qwen3-VL-4B-Instruct-4bit" >&2
    exit 1
  fi
fi

echo "==> building mlxforge-cli"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD" --parallel --target mlxforge-cli >/dev/null

echo "==> model:  $MODEL"
echo "==> image:  $IMAGE"
echo "==> prompt: $PROMPT"
echo "==> generating (max $MAX_TOKENS tokens):"
echo
exec "$BUILD/mlxforge-cli" image "$MODEL" "$IMAGE" "$PROMPT" "$MAX_TOKENS"
