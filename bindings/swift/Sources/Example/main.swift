import Foundation
import MLXForge

// Minimal CLI that streams a chat reply, and then runs two concurrent prompts on
// one engine to show they share the batched scheduler.
//   swift run mlxforge-swift-example <model-spec> [prompt]

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

let args = CommandLine.arguments
guard args.count >= 2 else {
  err("usage: mlxforge-swift-example <model-spec> [prompt]")
  exit(2)
}
let spec = args[1]
let prompt = args.count >= 3 ? args[2] : "What is the capital of France?"

do {
  let engine = try await Engine.load(spec)
  err("loaded: \(engine.modelName)")

  var s = Sampling.greedy
  s.maxTokens = 64
  for try await chunk in try engine.chat([ChatMessage(role: "user", content: prompt)], sampling: s) {
    print(chunk, terminator: "")
  }
  print()

  // Two distinct prompts at once on one engine (continuous batching).
  async let color = engine.complete(
    [ChatMessage(role: "user", content: "Name one color.")], sampling: { var x = Sampling.greedy; x.maxTokens = 8; return x }())
  async let fruit = engine.complete(
    [ChatMessage(role: "user", content: "Name one fruit.")], sampling: { var x = Sampling.greedy; x.maxTokens = 8; return x }())
  err("concurrent -> color=\(try await color) fruit=\(try await fruit)")
} catch {
  err("error: \(error)")
  exit(1)
}
