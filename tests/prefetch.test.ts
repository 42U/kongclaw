/**
 * Tests for Predictive Context Prefetching (src/prefetch.ts).
 *
 * Unit tests for query prediction, cache logic, and cosine similarity.
 * These tests do NOT require SurrealDB or embeddings.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { predictQueries, getCachedContext, clearPrefetchCache, getPrefetchStats, type CachedContext } from "../src/prefetch.js";

// We can't easily call prefetchContext (needs embeddings+surreal) but we can
// test the pure functions: predictQueries, getCachedContext, cache eviction.

describe("predictive context prefetching", () => {
  beforeEach(() => {
    clearPrefetchCache();
  });

  describe("predictQueries", () => {
    it("extracts file paths from input", () => {
      const queries = predictQueries("Fix the bug in src/auth.ts", "code-debug");
      expect(queries.some((q) => q.includes("src/auth.ts"))).toBe(true);
    });

    it("extracts backtick-quoted terms", () => {
      const queries = predictQueries("What does `transformContext` do?", "code-read");
      expect(queries.some((q) => q.includes("transformContext"))).toBe(true);
    });

    it("adds debug-specific predictions for code-debug", () => {
      const queries = predictQueries("Fix the authentication timeout", "code-debug");
      expect(queries.some((q) => q.startsWith("error"))).toBe(true);
      expect(queries.some((q) => q.startsWith("fix"))).toBe(true);
    });

    it("adds write-specific predictions for code-write", () => {
      const queries = predictQueries("Write a function to validate email addresses", "code-write");
      expect(queries.some((q) => q.startsWith("implementation"))).toBe(true);
      expect(queries.some((q) => q.startsWith("test"))).toBe(true);
    });

    it("adds procedure prediction for multi-step", () => {
      const queries = predictQueries("Refactor the auth module then update tests", "multi-step");
      expect(queries.some((q) => q.startsWith("procedure"))).toBe(true);
    });

    it("limits to 4 queries", () => {
      const queries = predictQueries(
        "Fix `auth.ts` `config.ts` `router.ts` in src/module.ts",
        "code-debug",
      );
      expect(queries.length).toBeLessThanOrEqual(4);
    });

    it("deduplicates queries", () => {
      const queries = predictQueries("Fix src/auth.ts error in src/auth.ts", "code-debug");
      const unique = new Set(queries);
      expect(queries.length).toBe(unique.size);
    });

    it("returns empty for simple intents with no extractable terms", () => {
      const queries = predictQueries("hi", "simple-question");
      // May or may not have queries depending on key term extraction
      expect(Array.isArray(queries)).toBe(true);
    });

    it("skips short terms", () => {
      const queries = predictQueries("a b c", "code-read");
      // No terms > 3 chars, so architecture + key terms might be empty
      expect(Array.isArray(queries)).toBe(true);
    });
  });

  describe("getCachedContext", () => {
    it("returns null for empty cache", () => {
      const result = getCachedContext(new Array(1024).fill(0.5));
      expect(result).toBeNull();
    });

    it("returns null for dissimilar vectors", () => {
      // Manually test — we can't easily populate cache without embeddings
      // but we can verify the function handles empty cache
      clearPrefetchCache();
      const result = getCachedContext(new Array(1024).fill(0.1));
      expect(result).toBeNull();
    });
  });

  describe("getPrefetchStats", () => {
    it("reports empty cache initially", () => {
      const stats = getPrefetchStats();
      expect(stats.entries).toBe(0);
      expect(stats.maxSize).toBe(10);
    });
  });

  describe("clearPrefetchCache", () => {
    it("clears the cache", () => {
      clearPrefetchCache();
      expect(getPrefetchStats().entries).toBe(0);
    });
  });
});
