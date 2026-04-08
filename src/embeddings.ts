import { existsSync } from "node:fs";
import { getLlama, LlamaLogLevel, type LlamaEmbeddingContext, type LlamaModel } from "node-llama-cpp";
import type { EmbeddingConfig } from "./config.js";
import { swallow } from "./errors.js";

let model: LlamaModel;
let ctx: LlamaEmbeddingContext;
let ready = false;

export async function initEmbeddings(config: EmbeddingConfig): Promise<void> {
  if (!existsSync(config.modelPath)) {
    throw new Error(`Embedding model not found at: ${config.modelPath}\n  Download BGE-M3 GGUF or set EMBED_MODEL_PATH`);
  }
  const llama = await getLlama({
    logLevel: LlamaLogLevel.error,
    logger: (level, message) => {
      // Suppress benign vocab warning from embedding models
      if (message.includes("missing newline token")) return;
      if (level === LlamaLogLevel.error) console.error(`[llama] ${message}`);
      else if (level === LlamaLogLevel.warn) console.warn(`[llama] ${message}`);
    },
  });
  model = await llama.loadModel({ modelPath: config.modelPath });
  ctx = await model.createEmbeddingContext();
  ready = true;
}

let _embedCallCount = 0;
let _cacheHits = 0;

/** L2-normalize a vector to unit length.
 *  node-llama-cpp returns raw unnormalized embeddings from the GGUF model.
 *  sentence-transformers (the reference implementation) always normalizes.
 *  Without this, magnitude variation from the CLS token introduces ranking noise. */
function l2Normalize(vec: number[]): number[] {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
  if (sumSq === 0) return vec;
  const norm = Math.sqrt(sumSq);
  const result = new Array<number>(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i] / norm;
  return result;
}

// ── LRU Embedding Cache ──────────────────────────────────────────────────
// Caches up to 512 embeddings keyed by text. Saves ~16ms per repeated embed.
// Map insertion order = LRU order. On hit, delete + re-insert to move to end.
const MAX_CACHE_SIZE = 512;
const _cache = new Map<string, number[]>();

function cacheGet(text: string): number[] | undefined {
  const vec = _cache.get(text);
  if (vec) {
    // Move to end for LRU freshness
    _cache.delete(text);
    _cache.set(text, vec);
    _cacheHits++;
  }
  return vec;
}

function cachePut(text: string, vec: number[]): void {
  if (_cache.size >= MAX_CACHE_SIZE) {
    // Evict oldest (first key in Map)
    const oldest = _cache.keys().next().value;
    if (oldest !== undefined) _cache.delete(oldest);
  }
  _cache.set(text, vec);
}

export async function embed(text: string): Promise<number[]> {
  if (!ready) throw new Error("Embeddings not initialized");
  _embedCallCount++;
  const cached = cacheGet(text);
  if (cached) return cached;
  const result = await ctx.getEmbeddingFor(text);
  const vec = l2Normalize(Array.from(result.vector));
  cachePut(text, vec);
  return vec;
}

export async function embedBatch(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const results: number[][] = [];
  for (const text of texts) {
    results.push(await embed(text));
  }
  return results;
}

/** Returns cache hit count and resets. */
export function drainCacheHitCount(): number {
  const count = _cacheHits;
  _cacheHits = 0;
  return count;
}

/** Current cache size. */
export function getCacheSize(): number {
  return _cache.size;
}

/** Returns total embed() calls this session and resets the counter. */
export function drainEmbedCallCount(): number {
  const count = _embedCallCount;
  _embedCallCount = 0;
  return count;
}

/** Returns total embed() calls this session without resetting. */
export function getEmbedCallCount(): number {
  return _embedCallCount;
}

export function isEmbeddingsAvailable(): boolean {
  return ready;
}

export async function disposeEmbeddings(): Promise<void> {
  try {
    await ctx?.dispose();
    await model?.dispose();
    ready = false;
  } catch (e) {
    swallow("embeddings:silent", e);
    // ignore
  }
}
