# Homebrew formula for libmlxforge. Builds from source (Apple Silicon only); the
# CLI/server harnesses are self-contained (they link the engine statically) and
# the embeddable libmlxforge.dylib + C ABI header are installed for embedders.
#
# Use via a tap:
#   brew tap hvasconcelos/libmlxforge https://github.com/hvasconcelos/libmlxforge
#   brew install libmlxforge         # or: brew install --HEAD libmlxforge
class Libmlxforge < Formula
  desc "Embeddable batched MLX LLM engine for Apple Silicon (C ABI + CLI)"
  homepage "https://github.com/hvasconcelos/libmlxforge"
  url "https://github.com/hvasconcelos/libmlxforge/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # set per release
  license "MIT"
  head "https://github.com/hvasconcelos/libmlxforge.git", branch: "master"

  depends_on "cmake" => :build
  depends_on arch: :arm64
  depends_on :macos

  def install
    system "cmake", "-S", ".", "-B", "build",
           "-DCMAKE_BUILD_TYPE=Release",
           "-DMLXFORGE_BUILD_SERVER=ON",
           "-DMLXFORGE_BUILD_CLI=ON",
           *std_cmake_args
    system "cmake", "--build", "build",
           "--target", "mlxforge_shared", "mlxforge-cli", "mlxforge"

    # Self-contained harness binaries (the bare name is the library; the server
    # is namespaced to make the product hierarchy clear).
    bin.install "build/mlxforge-cli"
    bin.install "build/mlxforge" => "mlxforge-server"

    # The embeddable library + its single public header (for linking your own app).
    lib.install Dir["build/libmlxforge*.dylib"]
    include.install "src/capi/mlxforge.h"
  end

  test do
    # --help needs no Metal device, so it works in the sandboxed test runner.
    assert_match "mlxforge", shell_output("#{bin}/mlxforge-server --help")
  end
end
