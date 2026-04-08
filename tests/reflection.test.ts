/**
 * Tests for Metacognitive Reflection (src/reflection.ts).
 *
 * Unit tests for threshold logic and formatting.
 * Integration tests require SurrealDB + embeddings.
 */
import { describe, it, expect, beforeAll } from "vitest";

let surrealAvailable = false;
let embeddingsReady = false;
let shouldReflect: typeof import("../src/reflection.js")["shouldReflect"];
let formatReflectionContext: typeof import("../src/reflection.js")["formatReflectionContext"];
let gatherSessionMetrics: typeof import("../src/reflection.js")["gatherSessionMetrics"];
let generateReflection: typeof import("../src/reflection.js")["generateReflection"];
let retrieveReflections: typeof import("../src/reflection.js")["retrieveReflections"];
let getReflectionCount: typeof import("../src/reflection.js")["getReflectionCount"];

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

  const mod = await import("../src/reflection.js");
  shouldReflect = mod.shouldReflect;
  formatReflectionContext = mod.formatReflectionContext;
  gatherSessionMetrics = mod.gatherSessionMetrics;
  generateReflection = mod.generateReflection;
  retrieveReflections = mod.retrieveReflections;
  getReflectionCount = mod.getReflectionCount;
});

describe("metacognitive reflection", () => {
  describe("shouldReflect", () => {
    it("returns false for a perfect session", () => {
      const result = shouldReflect({
        avgUtilization: 0.8,
        toolFailureRate: 0.0,
        steeringCandidates: 0,
        wastedTokens: 100,
        totalToolCalls: 10,
        totalTurns: 5,
      });
      expect(result.reflect).toBe(false);
      expect(result.reasons).toEqual([]);
    });

    it("triggers on low utilization", () => {
      const result = shouldReflect({
        avgUtilization: 0.15,
        toolFailureRate: 0.0,
        steeringCandidates: 0,
        wastedTokens: 100,
        totalToolCalls: 10,
        totalTurns: 3,
      });
      expect(result.reflect).toBe(true);
      expect(result.reasons.length).toBe(1);
      expect(result.reasons[0]).toContain("utilization");
    });

    it("triggers on high tool failure rate", () => {
      const result = shouldReflect({
        avgUtilization: 0.5,
        toolFailureRate: 0.35,
        steeringCandidates: 0,
        wastedTokens: 100,
        totalToolCalls: 10,
        totalTurns: 3,
      });
      expect(result.reflect).toBe(true);
      expect(result.reasons[0]).toContain("failure");
    });

    it("triggers on steering candidates", () => {
      const result = shouldReflect({
        avgUtilization: 0.5,
        toolFailureRate: 0.0,
        steeringCandidates: 2,
        wastedTokens: 100,
        totalToolCalls: 10,
        totalTurns: 3,
      });
      expect(result.reflect).toBe(true);
      expect(result.reasons[0]).toContain("steering");
    });

    it("triggers on excessive wasted tokens", () => {
      const result = shouldReflect({
        avgUtilization: 0.5,
        toolFailureRate: 0.0,
        steeringCandidates: 0,
        wastedTokens: 1500, // above 0.5% of 200k context window (threshold = 1000)
        totalToolCalls: 10,
        totalTurns: 3,
      });
      expect(result.reflect).toBe(true);
      expect(result.reasons[0]).toContain("wasted");
    });

    it("accumulates multiple reasons", () => {
      const result = shouldReflect({
        avgUtilization: 0.1,
        toolFailureRate: 0.4,
        steeringCandidates: 3,
        wastedTokens: 1500, // above 0.5% of 200k context window (threshold = 1000)
        totalToolCalls: 20,
        totalTurns: 5,
      });
      expect(result.reflect).toBe(true);
      expect(result.reasons.length).toBe(4);
    });

    it("does not trigger on first turn with low utilization", () => {
      const result = shouldReflect({
        avgUtilization: 0.1,
        toolFailureRate: 0.0,
        steeringCandidates: 0,
        wastedTokens: 100,
        totalToolCalls: 1,
        totalTurns: 1, // single turn → skip utilization check
      });
      expect(result.reflect).toBe(false);
    });
  });

  describe("formatReflectionContext", () => {
    it("returns empty string for no reflections", () => {
      expect(formatReflectionContext([])).toBe("");
    });

    it("formats reflections with tags", () => {
      const result = formatReflectionContext([{
        id: "reflection:abc",
        text: "Check error logs first before reading source code for timeout bugs.",
        category: "approach_strategy",
        severity: "moderate",
        importance: 7.0,
      }]);

      expect(result).toContain("<reflection_context>");
      expect(result).toContain("approach_strategy");
      expect(result).toContain("Check error logs first");
      expect(result).toContain("Lessons from past sessions");
    });

    it("formats multiple reflections", () => {
      const result = formatReflectionContext([
        {
          id: "reflection:1",
          text: "Lesson one",
          category: "failure_pattern",
          severity: "critical",
          importance: 7.0,
        },
        {
          id: "reflection:2",
          text: "Lesson two",
          category: "efficiency",
          severity: "minor",
          importance: 7.0,
        },
      ]);

      expect(result).toContain("Lesson one");
      expect(result).toContain("Lesson two");
    });
  });

  describe("gatherSessionMetrics", () => {
    it("returns metrics or null depending on surreal state", async () => {
      const metrics = await gatherSessionMetrics("nonexistent-session");
      // Either null (surreal down) or an object with numeric fields (surreal up)
      if (metrics === null) {
        expect(metrics).toBeNull();
      } else {
        expect(typeof metrics.totalToolCalls).toBe("number");
        expect(typeof metrics.totalTurns).toBe("number");
      }
    });
  });

  describe("retrieveReflections", () => {
    it("returns empty when infra is down", async () => {
      if (surrealAvailable && embeddingsReady) return;
      const results = await retrieveReflections(new Array(1024).fill(0), 3);
      expect(results).toEqual([]);
    });

    it("searches for reflections when available", async () => {
      if (!surrealAvailable || !embeddingsReady) return;
      const emb = await (await import("../src/embeddings.js")).embed("debugging timeout bugs");
      const results = await retrieveReflections(emb, 3);
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe("getReflectionCount", () => {
    it("returns a number", async () => {
      const count = await getReflectionCount();
      expect(typeof count).toBe("number");
      expect(count).toBeGreaterThanOrEqual(0);
    });
  });
});
