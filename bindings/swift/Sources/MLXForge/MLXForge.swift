import CMLXForge
import Foundation

/// An error carrying the message reported by the C ABI's `char** err`.
public struct MLXForgeError: Error, CustomStringConvertible {
  public let message: String
  public var description: String { message }
}

/// Sampling parameters. The default is deterministic greedy decoding; the C ABI
/// normalizes the "disabled" sentinels, so a default value is always valid.
public struct Sampling {
  public var temperature: Float = 0
  public var topK: Int32 = 0
  public var topP: Float = 0
  public var minP: Float = 0
  public var repetitionPenalty: Float = 0
  public var frequencyPenalty: Float = 0
  public var presencePenalty: Float = 0
  public var seed: UInt64 = 0
  public var maxTokens: Int32 = 0
  /// Constrained decoding. "json" forces any valid JSON value; otherwise a
  /// JSON-Schema string (supported subset: a top-level object with ordered,
  /// required, scalar-typed properties). Output is masked to be well-formed JSON.
  public var jsonSchema: String? = nil
  /// OpenAI logprobs: 0 = off; N > 0 = the chosen token's log-prob plus (N - 1)
  /// alternatives (so 1 = chosen-only). Retrieved via `Engine.completeWithLogprobs`.
  public var logprobs: Int32 = 0

  public init() {}
  public static var greedy: Sampling { Sampling() }

  // The flat C struct, with json_schema left null; the submit path fills it in
  // (it must own the C string for the duration of the call).
  fileprivate var c: mlxforge_sampling {
    mlxforge_sampling(
      temperature: temperature, top_k: topK, top_p: topP, min_p: minP,
      repetition_penalty: repetitionPenalty, frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty, seed: seed, max_tokens: maxTokens,
      json_schema: nil, logprobs: logprobs)
  }
}

public struct ChatMessage {
  public let role: String
  public let content: String
  public init(role: String, content: String) {
    self.role = role
    self.content = content
  }
}

/// A batched MLX LLM engine. Many `chat()`/`text()` calls may run concurrently
/// on one engine — they share its continuous-batching scheduler.
public final class Engine {
  private let handle: OpaquePointer

  public init(_ spec: String, maxWaiting: Int32 = 0) throws {
    var opts = mlxforge_engine_opts(max_waiting: maxWaiting)
    var err: UnsafeMutablePointer<CChar>?
    guard let h = mlxforge_engine_create(spec, &opts, &err) else {
      let message = err.map { String(cString: $0) } ?? "engine create failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    handle = h
  }

  deinit { mlxforge_engine_free(handle) }

  public var ready: Bool { mlxforge_engine_ready(handle) != 0 }
  public var modelName: String { String(cString: mlxforge_engine_model_name(handle)) }

  /// Poll until the worker has finished loading the model.
  public func waitReady() async {
    while !ready { try? await Task.sleep(nanoseconds: 10_000_000) }
  }

  /// Construct an engine and resolve once the model has finished loading.
  public static func load(_ spec: String, maxWaiting: Int32 = 0) async throws -> Engine {
    let engine = try Engine(spec, maxWaiting: maxWaiting)
    await engine.waitReady()
    return engine
  }

  /// Stream a chat completion as decoded text chunks.
  public func chat(_ messages: [ChatMessage], sampling: Sampling = .greedy) throws
    -> AsyncThrowingStream<String, Error>
  {
    Engine.stream(try submitChat(messages, sampling))
  }

  /// Stream a raw-text completion (no chat template).
  public func text(_ prompt: String, sampling: Sampling = .greedy) throws
    -> AsyncThrowingStream<String, Error>
  {
    var s = sampling.c
    let schemaC = sampling.jsonSchema.map { strdup($0) } ?? nil
    if let p = schemaC { s.json_schema = UnsafePointer(p) }
    defer { if let p = schemaC { free(p) } }
    var err: UnsafeMutablePointer<CChar>?
    guard let req = mlxforge_submit_text(handle, prompt, &s, &err) else {
      let message = err.map { String(cString: $0) } ?? "submit failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    return Engine.stream(req)
  }

  /// Stream a vision-language completion: a text `prompt` about one image (raw
  /// encoded bytes — JPEG/PNG/…). The loaded model must be a vision-language
  /// checkpoint (e.g. Qwen3-VL).
  public func image(_ prompt: String, _ imageBytes: [UInt8], sampling: Sampling = .greedy) throws
    -> AsyncThrowingStream<String, Error>
  {
    var s = sampling.c
    let schemaC = sampling.jsonSchema.map { strdup($0) } ?? nil
    if let p = schemaC { s.json_schema = UnsafePointer(p) }
    defer { if let p = schemaC { free(p) } }
    var err: UnsafeMutablePointer<CChar>?
    let reqOpt = imageBytes.withUnsafeBufferPointer { buf in
      mlxforge_submit_image(handle, prompt, buf.baseAddress, buf.count, &s, &err)
    }
    guard let req = reqOpt else {
      let message = err.map { String(cString: $0) } ?? "submit failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    return Engine.stream(req)
  }

  /// Stream a vision-language completion over several images (each raw encoded
  /// bytes), expanded into the prompt in order. The loaded model must be a
  /// vision-language checkpoint (e.g. Qwen3-VL).
  public func images(_ prompt: String, _ imagesBytes: [[UInt8]], sampling: Sampling = .greedy)
    throws -> AsyncThrowingStream<String, Error>
  {
    var s = sampling.c
    let schemaC = sampling.jsonSchema.map { strdup($0) } ?? nil
    if let p = schemaC { s.json_schema = UnsafePointer(p) }
    defer { if let p = schemaC { free(p) } }
    var err: UnsafeMutablePointer<CChar>?
    // Flatten into one contiguous buffer so each mlxforge_image points into stable
    // storage for the call (the engine copies the bytes synchronously).
    let flat = imagesBytes.flatMap { $0 }
    let reqOpt = flat.withUnsafeBufferPointer { (fb) -> OpaquePointer? in
      var cImages: [mlxforge_image] = []
      var offset = 0
      for img in imagesBytes {
        cImages.append(mlxforge_image(data: fb.baseAddress.map { $0 + offset }, len: img.count))
        offset += img.count
      }
      return cImages.withUnsafeBufferPointer { cb in
        mlxforge_submit_images(handle, prompt, cb.baseAddress, cb.count, &s, &err)
      }
    }
    guard let req = reqOpt else {
      let message = err.map { String(cString: $0) } ?? "submit failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    return Engine.stream(req)
  }

  /// Run a chat to completion and return the full string.
  public func complete(_ messages: [ChatMessage], sampling: Sampling = .greedy) async throws
    -> String
  {
    var out = ""
    for try await chunk in try chat(messages, sampling: sampling) { out += chunk }
    return out
  }

  /// Run a chat to completion and return both the text and the per-token log-probs
  /// (the C ABI's OpenAI-shaped JSON `content` string, or nil when `logprobs` was
  /// not set on `sampling`). Logprobs are read once the stream is fully drained.
  public func completeWithLogprobs(_ messages: [ChatMessage], sampling: Sampling = .greedy)
    async throws -> (text: String, logprobs: String?)
  {
    let req = try submitChat(messages, sampling)
    return try await withCheckedThrowingContinuation { cont in
      DispatchQueue.global(qos: .userInitiated).async {
        var out = ""
        while true {
          var text: UnsafeMutablePointer<CChar>?
          let rc = mlxforge_request_next(req, &text)
          if rc == 0, let t = text {
            out += String(cString: t)
            mlxforge_string_free(t)
          } else {
            break
          }
        }
        var logprobs: String? = nil
        if let lp = mlxforge_request_logprobs(req) {
          logprobs = String(cString: lp)
          mlxforge_string_free(lp)
        }
        mlxforge_request_free(req)
        cont.resume(returning: (out, logprobs))
      }
    }
  }

  /// Embed text into a (by default unit-normalized) vector.
  ///
  /// With all defaults the model self-selects its convention: a Qwen3-Embedding
  /// checkpoint uses last-token pooling + a trailing EOS, a plain LLM uses mean
  /// pooling. Override explicitly when needed:
  /// - `pooling`: -1 = model default, 0 = mean, 1 = last token.
  /// - `addEos`: `nil` = model default; `true` appends the model's EOS id.
  /// - `instruction`: wraps text as "Instruct: {instruction}\nQuery: {text}"
  ///   (Qwen3-Embedding retrieval queries).
  /// - `normalize`: L2-normalize the pooled vector (default `true`).
  public func embed(_ text: String, pooling: Int32 = -1, addEos: Bool? = nil,
                    instruction: String? = nil, normalize: Bool = true) async throws -> [Float] {
    let handle = self.handle
    let addEosVal: Int32 = addEos.map { $0 ? 1 : 0 } ?? -1
    return try await withCheckedThrowingContinuation { cont in
      // The C call blocks until the worker runs the forward pass; run it off the
      // Swift-concurrency thread.
      DispatchQueue.global(qos: .userInitiated).async {
        var out: UnsafeMutablePointer<Float>?
        var len: Int = 0
        var err: UnsafeMutablePointer<CChar>?
        // Build opts; `instruction` must outlive the call, so bind it with
        // withCString (a NULL pointer when no instruction was given).
        let run: (UnsafePointer<CChar>?) -> Int32 = { instrPtr in
          var opts = mlxforge_embed_opts()
          opts.pooling = pooling
          opts.add_eos = addEosVal
          opts.skip_normalize = normalize ? 0 : 1
          opts.instruction = instrPtr
          return mlxforge_embed_ex(handle, text, &opts, &out, &len, &err)
        }
        let rc = instruction.map { $0.withCString { run($0) } } ?? run(nil)
        if rc == 0, let out = out {
          let vec = Array(UnsafeBufferPointer(start: out, count: len))
          mlxforge_floats_free(out)
          cont.resume(returning: vec)
        } else {
          let message = err.map { String(cString: $0) } ?? "embed failed"
          mlxforge_string_free(err)
          cont.resume(throwing: MLXForgeError(message: message))
        }
      }
    }
  }

  private func submitChat(_ messages: [ChatMessage], _ sampling: Sampling) throws -> OpaquePointer {
    // Own the C strings for the duration of the submit call.
    var owned: [UnsafeMutablePointer<CChar>] = []
    defer { owned.forEach { free($0) } }
    var msgs: [mlxforge_msg] = []
    msgs.reserveCapacity(messages.count)
    for m in messages {
      let role = strdup(m.role)!
      let content = strdup(m.content)!
      owned.append(role)
      owned.append(content)
      msgs.append(mlxforge_msg(role: role, content: content))
    }
    var s = sampling.c
    if let js = sampling.jsonSchema {
      let p = strdup(js)!
      owned.append(p)  // freed with the message strings in the defer above
      s.json_schema = UnsafePointer(p)
    }
    var err: UnsafeMutablePointer<CChar>?
    let req = msgs.withUnsafeBufferPointer { buf in
      mlxforge_submit_chat(handle, buf.baseAddress, buf.count, &s, &err)
    }
    guard let req else {
      let message = err.map { String(cString: $0) } ?? "submit failed"
      mlxforge_string_free(err)
      throw MLXForgeError(message: message)
    }
    return req
  }

  // Drive the blocking C-ABI poll on a background queue, yielding chunks; the
  // request is freed when the stream ends. (MLX work stays on the engine's own
  // worker thread — this thread only touches the request's token queue.)
  private static func stream(_ req: OpaquePointer) -> AsyncThrowingStream<String, Error> {
    AsyncThrowingStream { continuation in
      DispatchQueue.global(qos: .userInitiated).async {
        while true {
          var text: UnsafeMutablePointer<CChar>?
          let rc = mlxforge_request_next(req, &text)
          if rc == 0, let t = text {
            continuation.yield(String(cString: t))
            mlxforge_string_free(t)
          } else {
            break
          }
        }
        mlxforge_request_free(req)
        continuation.finish()
      }
    }
  }
}
