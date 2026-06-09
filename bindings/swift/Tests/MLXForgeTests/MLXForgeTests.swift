import CMLXForge
import XCTest

@testable import MLXForge

final class MLXForgeTests: XCTestCase {
  func testAbiAndVersion() {
    XCTAssertEqual(mlxforge_abi_version(), MLXFORGE_ABI_VERSION)
    XCTAssertFalse(String(cString: mlxforge_version()).isEmpty)
  }

  func testBadSpecThrows() {
    XCTAssertThrowsError(try Engine("")) { error in
      XCTAssertTrue(error is MLXForgeError)
    }
  }

  func testGenerationIfModelPresent() async throws {
    guard let dir = ProcessInfo.processInfo.environment["MLXFORGE_MODEL_DIR"], !dir.isEmpty else {
      throw XCTSkip("MLXFORGE_MODEL_DIR not set; skipping model generation test")
    }
    let engine = try await Engine.load(dir)
    var s = Sampling.greedy
    s.maxTokens = 16
    let out = try await engine.complete(
      [ChatMessage(role: "user", content: "What is the capital of France?")], sampling: s)
    XCTAssertFalse(out.isEmpty)
  }

  func testConstrainedJsonIfModelPresent() async throws {
    guard let dir = ProcessInfo.processInfo.environment["MLXFORGE_MODEL_DIR"], !dir.isEmpty else {
      throw XCTSkip("MLXFORGE_MODEL_DIR not set; skipping constrained-JSON test")
    }
    let engine = try await Engine.load(dir)
    var s = Sampling.greedy
    s.maxTokens = 128
    s.jsonSchema = #"{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}}}"#
    let out = try await engine.complete(
      [ChatMessage(role: "user", content: "Tell me about Paris.")], sampling: s)
    // The output must be parseable JSON with the schema's keys.
    let data = Data(out.utf8)
    let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    XCTAssertNotNil(obj, "constrained output should be a JSON object: \(out)")
    XCTAssertNotNil(obj?["city"])
    XCTAssertTrue(obj?["population"] is Int, "population must be an integer: \(out)")
  }

  func testEmbeddingsIfModelPresent() async throws {
    guard let dir = ProcessInfo.processInfo.environment["MLXFORGE_MODEL_DIR"], !dir.isEmpty else {
      throw XCTSkip("MLXFORGE_MODEL_DIR not set; skipping embeddings test")
    }
    let engine = try await Engine.load(dir)
    func dot(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).reduce(0) { $0 + $1.0 * $1.1 } }

    let a = try await engine.embed("The cat sat on the warm mat.")
    let b = try await engine.embed("A kitten is resting on a soft rug.")
    let c = try await engine.embed("The stock market fell sharply amid economic fears.")
    XCTAssertFalse(a.isEmpty)
    XCTAssertEqual(a.count, b.count)
    XCTAssertEqual(dot(a, a), 1.0, accuracy: 0.02)   // unit-normalized
    XCTAssertGreaterThan(dot(a, b), dot(a, c))        // semantic ordering
  }

  // Vision-language: one image via image(), and several via images(). Gated on a
  // dedicated env var pointing at a Qwen3-VL checkpoint; reads the committed test
  // image from the repo's fixtures.
  func testImagesIfVLModelPresent() async throws {
    guard let dir = ProcessInfo.processInfo.environment["MLXFORGE_VL_MODEL"], !dir.isEmpty else {
      throw XCTSkip("MLXFORGE_VL_MODEL not set; skipping vision test")
    }
    let engine = try await Engine.load(dir)

    var repo = URL(fileURLWithPath: #filePath)  // …/bindings/swift/Tests/MLXForgeTests/<file>
    for _ in 0..<5 { repo.deleteLastPathComponent() }
    let imgURL = repo.appendingPathComponent("reference/fixtures_qwen3_vl/image.png")
    let bytes = [UInt8](try Data(contentsOf: imgURL))

    var s = Sampling.greedy
    s.maxTokens = 8

    var single = ""
    for try await chunk in try engine.image("What is in this image?", bytes, sampling: s) {
      single += chunk
    }
    XCTAssertFalse(single.isEmpty)

    var multi = ""
    for try await chunk in try engine.images("Compare these images.", [bytes, bytes], sampling: s) {
      multi += chunk
    }
    XCTAssertFalse(multi.isEmpty)
  }
}
