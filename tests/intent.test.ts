/**
 * Tests for zero-shot intent classification (src/intent.ts).
 *
 * Unit tests for estimateComplexity (pure function, no mocks needed).
 * Integration tests for classifyIntent require embeddings — skip if unavailable.
 */
import { describe, it, expect, vi, beforeAll } from "vitest";
import type { IntentResult, IntentCategory } from "../src/intent.js";

// --- Pure function tests (no embeddings needed) ---

// We can import estimateComplexity directly since it's a pure function
// But classifyIntent needs embeddings, so we test it conditionally.
describe("estimateComplexity", () => {
  // Dynamic import to avoid top-level side effects
  let estimateComplexity: typeof import("../src/intent.js")["estimateComplexity"];

  beforeAll(async () => {
    const mod = await import("../src/intent.js");
    estimateComplexity = mod.estimateComplexity;
  });

  const makeIntent = (category: IntentCategory, confidence = 0.8): IntentResult => ({
    category,
    confidence,
    scores: [{ category, score: confidence }],
  });

  it("simple-question → trivial complexity", () => {
    const result = estimateComplexity("What is 2+2?", makeIntent("simple-question"));
    expect(result.level).toBe("trivial");
    expect(result.suggestedThinking).toBe("low");
    expect(result.estimatedToolCalls).toBe(0);
  });

  it("code-write → moderate complexity", () => {
    const result = estimateComplexity("Write a sort function", makeIntent("code-write"));
    expect(result.level).toBe("moderate");
    expect(result.suggestedThinking).toBe("high");
    expect(result.estimatedToolCalls).toBeGreaterThanOrEqual(8);
  });

  it("deep-explore → deep complexity", () => {
    const result = estimateComplexity("Analyze every file", makeIntent("deep-explore"));
    expect(result.level).toBe("deep");
    expect(result.estimatedToolCalls).toBeGreaterThanOrEqual(12);
  });

  it("multi-step keywords escalate complexity", () => {
    const result = estimateComplexity(
      "First read the file, then refactor it, then update the tests",
      makeIntent("code-read"),
    );
    expect(result.level).toBe("complex");
    expect(result.suggestedThinking).toBe("high");
    expect(result.estimatedToolCalls).toBeGreaterThanOrEqual(12);
  });

  it("'every/all' keywords escalate to deep", () => {
    const result = estimateComplexity(
      "Read every file in this entire project",
      makeIntent("code-read"),
    );
    expect(result.level).toBe("deep");
    expect(result.estimatedToolCalls).toBeGreaterThanOrEqual(12);
  });

  it("long prompts (>100 words) increase tool estimate", () => {
    const longText = "word ".repeat(120) + "do something";
    const result = estimateComplexity(longText, makeIntent("simple-question"));
    expect(result.estimatedToolCalls).toBeGreaterThanOrEqual(12);
  });

  it("unknown intent gets moderate defaults", () => {
    const result = estimateComplexity("something unclear", makeIntent("unknown"));
    expect(result.level).toBe("moderate");
    expect(result.suggestedThinking).toBe("medium");
  });

  it("continuation inherits simple base", () => {
    const result = estimateComplexity("keep going", makeIntent("continuation"));
    expect(result.level).toBe("simple");
    expect(result.estimatedToolCalls).toBe(8);
  });
});

// --- Cosine similarity (internal but testable via classify behavior) ---

describe("classifyIntent", () => {
  let classifyIntent: typeof import("../src/intent.js")["classifyIntent"];
  let isEmbeddingsAvailable: typeof import("../src/embeddings.js")["isEmbeddingsAvailable"];
  let embeddingsReady = false;

  beforeAll(async () => {
    // Try to init embeddings — if they're not available, skip these tests
    try {
      const embMod = await import("../src/embeddings.js");
      isEmbeddingsAvailable = embMod.isEmbeddingsAvailable;
      if (!isEmbeddingsAvailable()) {
        // Try to initialize
        const configMod = await import("../src/config.js");
        const config = configMod.loadConfig();
        await embMod.initEmbeddings(config.embedding);
      }
      embeddingsReady = isEmbeddingsAvailable();
    } catch {
      embeddingsReady = false;
    }

    const intentMod = await import("../src/intent.js");
    classifyIntent = intentMod.classifyIntent;
  });

  it("returns unknown when embeddings unavailable", async () => {
    if (embeddingsReady) {
      // Can't easily test this when embeddings ARE available — skip
      return;
    }
    const result = await classifyIntent("What is 2+2?");
    expect(result.category).toBe("unknown");
    expect(result.confidence).toBe(0);
  });

  it("classifies simple question correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("What is the capital of France?");
    expect(result.category).toBe("simple-question");
    expect(result.confidence).toBeGreaterThan(0.6);
  });

  it("classifies code-read correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("Read the file src/agent.ts and explain what each function does");
    expect(result.category).toBe("code-read");
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it("classifies code-write correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("Implement a REST API endpoint for user registration");
    expect(result.category).toBe("code-write");
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it("classifies code-debug correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("Debug this TypeError: Cannot read property of undefined");
    expect(result.category).toBe("code-debug");
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it("classifies deep-explore correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("Analyze every single file in this entire codebase and map out the architecture");
    expect(result.category).toBe("deep-explore");
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it("classifies continuation correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("Yes keep going with that approach");
    expect(result.category).toBe("continuation");
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it("classifies multi-step correctly", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("First refactor the database module, then update all the tests, then write documentation");
    expect(result.category).toBe("multi-step");
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it("returns scores array with all categories", async () => {
    if (!embeddingsReady) return;
    const result = await classifyIntent("What is 2+2?");
    expect(result.scores.length).toBeGreaterThan(0);
    // Scores should be sorted descending
    for (let i = 1; i < result.scores.length; i++) {
      expect(result.scores[i - 1].score).toBeGreaterThanOrEqual(result.scores[i].score);
    }
  });

  it("low-confidence input returns unknown", async () => {
    if (!embeddingsReady) return;
    // Gibberish that shouldn't match any prototype well
    const result = await classifyIntent("xyzzy plugh abracadabra");
    // If confidence < 0.65, should be unknown
    if (result.confidence < 0.65) {
      expect(result.category).toBe("unknown");
    }
  });
});
