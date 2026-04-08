/**
 * Tests for embedding system (src/embeddings.ts).
 *
 * Tests state management and error handling.
 * Full embedding tests require the GGUF model file.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";

let isEmbeddingsAvailable: typeof import("../src/embeddings.js")["isEmbeddingsAvailable"];
let embed: typeof import("../src/embeddings.js")["embed"];
let embedBatch: typeof import("../src/embeddings.js")["embedBatch"];
let initEmbeddings: typeof import("../src/embeddings.js")["initEmbeddings"];
let disposeEmbeddings: typeof import("../src/embeddings.js")["disposeEmbeddings"];
let embeddingsReady = false;

beforeAll(async () => {
  const mod = await import("../src/embeddings.js");
  isEmbeddingsAvailable = mod.isEmbeddingsAvailable;
  embed = mod.embed;
  embedBatch = mod.embedBatch;
  initEmbeddings = mod.initEmbeddings;
  disposeEmbeddings = mod.disposeEmbeddings;

  // Try to initialize embeddings with the real config
  try {
    const configMod = await import("../src/config.js");
    const config = configMod.loadConfig();
    await initEmbeddings(config.embedding);
    embeddingsReady = isEmbeddingsAvailable();
  } catch {
    embeddingsReady = false;
  }
});

afterAll(async () => {
  if (embeddingsReady && disposeEmbeddings) {
    await disposeEmbeddings();
  }
});

describe("isEmbeddingsAvailable", () => {
  it("returns a boolean", () => {
    const result = isEmbeddingsAvailable();
    expect(typeof result).toBe("boolean");
  });
});

describe("embedBatch", () => {
  it("returns empty array for empty input", async () => {
    const result = await embedBatch([]);
    expect(result).toEqual([]);
  });
});

describe("initEmbeddings", () => {
  it("throws when model path does not exist", async () => {
    await expect(
      initEmbeddings({
        modelPath: "/nonexistent/path/to/model.gguf",
        dimensions: 1024,
      }),
    ).rejects.toThrow("Embedding model not found");
  });
});

describe("embed (when initialized)", () => {
  it("returns a vector array", async () => {
    if (!embeddingsReady) {
      console.log("Embeddings not available — skipping");
      return;
    }
    const vec = await embed("test text");
    expect(Array.isArray(vec)).toBe(true);
    expect(vec.length).toBeGreaterThan(0);
    expect(typeof vec[0]).toBe("number");
  });
});
