# Distribution

How `libmlxforge` and its bindings are packaged so adopters don't have to build
MLX themselves. Everything is **Apple Silicon / macOS arm64**.

## Release artifacts (the source of truth)

`.github/workflows/release.yml` runs on a `v*` tag (Apple Silicon runner) and
attaches these to the GitHub Release — every downstream package consumes them:

| Artifact | Contents |
| --- | --- |
| `libmlxforge-<tag>-macos-arm64.tar.gz` | the lean `libmlxforge.dylib` (no `cpp-httplib`/`libcurl`) + `mlxforge.h` |
| `MLXForge-<tag>.xcframework.zip` | the XCFramework (Swift / SwiftPM binary target) |
| `mlxforge-cli-<tag>-macos-arm64.tar.gz` | the self-contained CLI harness |
| `SHA256SUMS-<tag>.txt` | checksums |

The workflow runs `scripts/check-abi.sh` against the dylib first, so a release
can't ship a library that leaked the server dependency or dropped a C-ABI symbol
(see [`abi-stability.md`](./abi-stability.md)).

## Homebrew (CLI + library)

`Formula/libmlxforge.rb` builds from source (Apple Silicon only). Via a tap:

```sh
brew tap hvasconcelos/libmlxforge https://github.com/hvasconcelos/libmlxforge
brew install libmlxforge         # mlxforge-cli + mlxforge-server + libmlxforge.dylib + mlxforge.h
# or track the branch:
brew install --HEAD libmlxforge
```

The harness binaries are self-contained (they link the engine statically); the
dylib + header are installed under `$(brew --prefix)/{lib,include}` for embedders.
A future step replaces source builds with prebuilt bottles from the release
artifacts above.

## Swift (SwiftPM)

The package in `bindings/swift` builds against the local CMake dylib for
development. For distribution, point a **binary target** at the released
XCFramework so adopters need no CMake/MLX build:

```swift
// Package.swift (distribution variant)
.binaryTarget(
  name: "CMLXForge",
  url: "https://github.com/hvasconcelos/libmlxforge/releases/download/v0.1.0/MLXForge-v0.1.0.xcframework.zip",
  checksum: "<swift package compute-checksum MLXForge-v0.1.0.xcframework.zip>"
)
```

Regenerate the XCFramework locally with `bindings/swift/scripts/make_xcframework.sh`.

## Node (npm)

`@mlxforge/node` builds from source today (`npm install` → node-gyp). For a
zero-toolchain install, the release workflow can run `npm run prebuild`
(prebuildify) to produce a prebuilt `.node`, which is bundled with the dylib and
loaded via `node-gyp-build` — so `npm install` ships a binary and needs no
compiler. The prebuilt package must ship `libmlxforge.dylib` alongside the addon
with an rpath into the package directory.

## Rust (crates)

`bindings/rust` links the locally-built dylib via `build.rs`. A published crate
would resolve a prebuilt dylib from the release artifacts (or a `MLXFORGE_LIB_DIR`
env override) instead of the repo's `build/` directory.
