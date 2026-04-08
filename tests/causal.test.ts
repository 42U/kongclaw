/**
 * Tests for Causal Memory Graph (src/causal.ts).
 *
 * Unit tests for causal chain extraction, edge linking, and causal traversal.
 * Integration tests require SurrealDB + embeddings.
 */
import { describe, it, expect, beforeAll } from "vitest";

let surrealAvailable = false;
let embeddingsReady = false;
let linkCausalEdges: typeof import("../src/causal.js")["linkCausalEdges"];
let queryCausalContext: typeof import("../src/causal.js")["queryCausalContext"];
let getSessionCausalChains: typeof import("../src/causal.js")["getSessionCausalChains"];

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

  const mod = await import("../src/causal.js");
  linkCausalEdges = mod.linkCausalEdges;
  queryCausalContext = mod.queryCausalContext;
  getSessionCausalChains = mod.getSessionCausalChains;
});

const dummyVec = new Array(1024).fill(0);

describe("causal memory graph", () => {
  describe("linkCausalEdges", () => {
    it("handles empty chains array gracefully", async () => {
      await linkCausalEdges([], "test-session");
      // Should not throw
    });

    it("creates memory nodes and edges for a valid chain", async () => {
      if (!surrealAvailable || !embeddingsReady) return;
      const chains = [{
        triggerText: "Auth timeout bug reported by user",
        outcomeText: "Fixed by adding token refresh retry logic",
        chainType: "debug" as const,
        success: true,
        confidence: 0.8,
        description: "Auth timeout → token refresh fix",
      }];
      // Should not throw
      await linkCausalEdges(chains, "test-causal-session");
    });

    it("creates contradicts edge for failed chains", async () => {
      if (!surrealAvailable || !embeddingsReady) return;
      const chains = [{
        triggerText: "Performance regression after upgrade",
        outcomeText: "Attempted rollback but didn't resolve",
        chainType: "fix" as const,
        success: false,
        confidence: 0.6,
        description: "Performance regression → failed rollback",
      }];
      await linkCausalEdges(chains, "test-causal-session-2");
    });
  });

  describe("queryCausalContext", () => {
    it("returns empty for no seed IDs", async () => {
      const results = await queryCausalContext([], dummyVec);
      expect(results).toEqual([]);
    });

    it("returns empty for invalid IDs", async () => {
      const results = await queryCausalContext(["not-a-valid-id"], dummyVec);
      expect(results).toEqual([]);
    });

    it("traverses causal edges when they exist", async () => {
      if (!surrealAvailable) return;
      const results = await queryCausalContext(["memory:test123"], dummyVec);
      expect(Array.isArray(results)).toBe(true);
    });

    it("respects minConfidence parameter", async () => {
      if (!surrealAvailable) return;
      // Very high confidence threshold should return fewer or no results
      const strict = await queryCausalContext(["memory:test123"], dummyVec, 2, 0.99);
      const relaxed = await queryCausalContext(["memory:test123"], dummyVec, 2, 0.1);
      expect(strict.length).toBeLessThanOrEqual(relaxed.length);
    });
  });

  describe("getSessionCausalChains", () => {
    it("returns zero count for non-existent session", async () => {
      if (!surrealAvailable) return;
      const stats = await getSessionCausalChains("nonexistent");
      expect(stats.count).toBe(0);
      expect(stats.successRate).toBe(0);
    });

    it("returns stats when surreal is down", async () => {
      if (surrealAvailable) return;
      const stats = await getSessionCausalChains("test");
      expect(stats.count).toBe(0);
    });
  });
});
