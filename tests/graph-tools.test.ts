/**
 * Tests for graph-aware tools (src/graph-tools.ts).
 *
 * Unit tests for recall tool parameter handling.
 * Integration tests for actual memory search require SurrealDB + embeddings.
 */
import { describe, it, expect, beforeAll } from "vitest";

let surrealAvailable = false;
let embeddingsReady = false;
let createRecallTool: typeof import("../src/graph-tools.js")["createRecallTool"];

beforeAll(async () => {
  try {
    const configMod = await import("../src/config.js");
    const config = configMod.loadConfig();
    const surrealMod = await import("../src/surreal.js");
    await surrealMod.initSurreal(config.surreal);
    surrealAvailable = await surrealMod.isSurrealAvailable();

    const embMod = await import("../src/embeddings.js");
    if (!embMod.isEmbeddingsAvailable()) {
      await embMod.initEmbeddings(config.embedding);
    }
    embeddingsReady = embMod.isEmbeddingsAvailable();
  } catch {
    surrealAvailable = false;
    embeddingsReady = false;
  }

  const mod = await import("../src/graph-tools.js");
  createRecallTool = mod.createRecallTool;
});

describe("recall tool", () => {
  it("has correct name and label", () => {
    const tool = createRecallTool("test-session");
    expect(tool.name).toBe("recall");
    expect(tool.label).toBe("recall");
  });

  it("has a description mentioning memory", () => {
    const tool = createRecallTool("test-session");
    expect(tool.description).toContain("memory");
  });

  it("returns unavailable message when infra is down", async () => {
    if (surrealAvailable && embeddingsReady) return; // can't test this when both are up
    const tool = createRecallTool("test-session");
    const result = await tool.execute("call-1", { query: "test" });
    expect(result.content[0].text).toContain("unavailable");
  });

  it("returns results for a valid query", async () => {
    if (!surrealAvailable || !embeddingsReady) return;
    const tool = createRecallTool("test-session");
    const result = await tool.execute("call-2", { query: "authentication module" });
    // Should get a text response (may or may not have results depending on DB state)
    expect(result.content[0].text).toBeTruthy();
    expect(typeof result.content[0].text).toBe("string");
  });

  it("respects scope parameter", async () => {
    if (!surrealAvailable || !embeddingsReady) return;
    const tool = createRecallTool("test-session");
    const result = await tool.execute("call-3", { query: "test", scope: "concepts" });
    expect(result.content[0].text).toBeTruthy();
  });

  it("respects limit parameter", async () => {
    if (!surrealAvailable || !embeddingsReady) return;
    const tool = createRecallTool("test-session");
    const result = await tool.execute("call-4", { query: "code", limit: 2 });
    // Details should have count <= 2
    if (result.details?.count) {
      expect(result.details.count).toBeLessThanOrEqual(2);
    }
  });

  it("caps limit at 15", async () => {
    if (!surrealAvailable || !embeddingsReady) return;
    const tool = createRecallTool("test-session");
    const result = await tool.execute("call-5", { query: "code", limit: 100 });
    if (result.details?.count) {
      expect(result.details.count).toBeLessThanOrEqual(15);
    }
  });
});
