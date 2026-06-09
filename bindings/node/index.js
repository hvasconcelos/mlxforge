'use strict';

// Ergonomic JS wrapper over the native N-API addon. The addon exposes raw
// Engine/Request handles; this file adds async-iterable streaming and a couple
// of conveniences (load() that waits for readiness, complete() that collects).
const native = require('./build/Release/mlxforge_node.node');

// Wrap a native Request as an async-iterable of decoded text chunks. The
// request is disposed when iteration finishes (or the consumer breaks out).
function streamRequest(req) {
  return {
    // The underlying native handle, in case callers want cancel()/finishReason().
    request: req,
    cancel() {
      req.cancel();
    },
    get finishReason() {
      return req.finishReason();
    },
    async *[Symbol.asyncIterator]() {
      try {
        let chunk;
        while ((chunk = await req.next()) !== null) yield chunk;
      } finally {
        req.dispose();
      }
    },
  };
}

async function collect(iterable) {
  let out = '';
  for await (const chunk of iterable) out += chunk;
  return out;
}

class Engine {
  /**
   * @param {string} spec  local dir, HF repo id, or .gguf file
   * @param {{maxWaiting?: number}} [opts]
   */
  constructor(spec, opts = {}) {
    this._h = new native.Engine(spec, opts);
  }

  /** Construct an engine and resolve once the model has finished loading. */
  static async load(spec, opts = {}) {
    const engine = new Engine(spec, opts);
    await engine.waitReady();
    return engine;
  }

  get ready() {
    return this._h.ready();
  }

  get modelName() {
    return this._h.modelName();
  }

  async waitReady(pollMs = 10) {
    while (!this._h.ready()) await new Promise((r) => setTimeout(r, pollMs));
  }

  /**
   * Stream a chat completion. Many chat()/text() calls may run concurrently on
   * one engine — they share its continuous-batching scheduler.
   * @param {{role: string, content: string}[]} messages
   * @param {object} [sampling]
   * @returns an async-iterable of text chunks (+ .cancel()/.finishReason()).
   */
  chat(messages, sampling = {}) {
    return streamRequest(this._h.submitChat(messages, sampling));
  }

  /** Stream a raw-text completion (no chat template). */
  text(prompt, sampling = {}) {
    return streamRequest(this._h.submitText(prompt, sampling));
  }

  /** Run a chat to completion and return the full string. */
  complete(messages, sampling = {}) {
    return collect(this.chat(messages, sampling));
  }

  /**
   * Embed text into a (by default unit-normalized) vector (Float32Array).
   *
   * Call `embed(text)` and the model self-selects its convention: a
   * Qwen3-Embedding checkpoint uses last-token pooling + a trailing EOS, a plain
   * LLM uses mean pooling. The second argument is either a legacy pooling number
   * (0 = mean, 1 = last token) or an options object:
   *   { pooling?: 0|1, addEos?: boolean, instruction?: string, normalize?: boolean }
   * `instruction` wraps the text as "Instruct: {instruction}\nQuery: {text}"
   * (Qwen3-Embedding retrieval queries).
   * @returns {Promise<Float32Array>}
   */
  embed(text, opts) {
    return this._h.embed(text, opts);
  }

  dispose() {
    this._h.dispose();
  }
}

module.exports = { Engine, version: native.version, abiVersion: native.abiVersion };
