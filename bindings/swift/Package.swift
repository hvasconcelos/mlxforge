// swift-tools-version:5.9
import PackageDescription
import Foundation

// Resolve the repo root (two levels up from this manifest) so the C ABI header
// and the libmlxforge.dylib from the CMake build are found regardless of the
// invocation directory. A distributed package would instead depend on a
// prebuilt MLXForge.xcframework (see scripts/make_xcframework.sh).
let manifestDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let repoRoot = manifestDir.deletingLastPathComponent().deletingLastPathComponent().path
let libDir = repoRoot + "/build"

let package = Package(
  name: "MLXForge",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "MLXForge", targets: ["MLXForge"]),
    .executable(name: "mlxforge-swift-example", targets: ["Example"]),
  ],
  targets: [
    // C shim exposing the libmlxforge C ABI as a Clang module.
    .target(name: "CMLXForge"),
    // Idiomatic Swift wrapper; links the locally-built dylib.
    .target(
      name: "MLXForge",
      dependencies: ["CMLXForge"],
      linkerSettings: [
        .unsafeFlags(["-L\(libDir)", "-lmlxforge", "-Xlinker", "-rpath", "-Xlinker", libDir])
      ]
    ),
    .executableTarget(name: "Example", dependencies: ["MLXForge"]),
    .testTarget(name: "MLXForgeTests", dependencies: ["MLXForge"]),
  ]
)
