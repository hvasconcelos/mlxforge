// Type definitions for @mlxforge/node

export interface EngineOptions {
  /** Max queued requests before submit is rejected (default 256). */
  maxWaiting?: number;
}

export interface SamplingOptions {
  /** <= 0 (default) => greedy/argmax. */
  temperature?: number;
  /** <= 0 (default) => disabled. */
  topK?: number;
  /** <= 0 or >= 1 (default) => disabled. */
  topP?: number;
  /** <= 0 (default) => disabled. */
  minP?: number;
  /** 0/1 (default) => disabled. */
  repetitionPenalty?: number;
  /** 0 (default) => disabled. */
  frequencyPenalty?: number;
  /** 0 (default) => disabled. */
  presencePenalty?: number;
  /** RNG seed (used when temperature > 0). */
  seed?: number;
  /** Max new tokens (default 64). */
  maxTokens?: number;
  /**
   * Constrained decoding. "json" forces any valid JSON value; otherwise a
   * JSON-Schema string (supported subset: a top-level object with ordered,
   * required, scalar-typed properties). Output is masked to be well-formed JSON.
   */
  jsonSchema?: string;
  /** Alias for `jsonSchema` (e.g. "json"). */
  responseFormat?: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
}

/** An async-iterable stream of decoded text chunks for one request. */
export interface Stream extends AsyncIterable<string> {
  /** Ask the engine to stop generating this request. */
  cancel(): void;
  /** "stop" | "length" | "cancel" | "" (while running). */
  readonly finishReason: string;
}

export class Engine {
  constructor(spec: string, opts?: EngineOptions);
  /** Construct and resolve once the model has finished loading. */
  static load(spec: string, opts?: EngineOptions): Promise<Engine>;
  readonly ready: boolean;
  readonly modelName: string;
  waitReady(pollMs?: number): Promise<void>;
  /** Stream a chat completion (runs batched with other concurrent calls). */
  chat(messages: ChatMessage[], sampling?: SamplingOptions): Stream;
  /** Stream a raw-text completion (no chat template). */
  text(prompt: string, sampling?: SamplingOptions): Stream;
  /** Run a chat to completion and return the full string. */
  complete(messages: ChatMessage[], sampling?: SamplingOptions): Promise<string>;
  /**
   * Embed text into a unit-normalized vector. pooling: 0 = mean (default),
   * 1 = last token.
   */
  embed(text: string, pooling?: number): Promise<Float32Array>;
  dispose(): void;
}

export const version: string;
export const abiVersion: number;
