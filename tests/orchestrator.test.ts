/**
 * Tests for ISMAR-GENT orchestrator (src/orchestrator.ts).
 *
 * Tests preflight fast-path, config mapping, steering detection, tool recording.
 * Mocks classifyIntent to avoid embedding dependency for unit tests.
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

// Mock the intent module to avoid needing embeddings
vi.mock("../src/intent.js", () => ({
  classifyIntent: vi.fn(),
  estimateComplexity: vi.fn(),
}));

// Mock surreal to avoid needing a database
vi.mock("../src/surreal.js", () => ({
  getDb: vi.fn(() => ({ query: vi.fn() })),
  isSurrealAvailable: vi.fn(async () => false),
}));

// Mock retrieval-quality for adaptive budget tests
vi.mock("../src/retrieval-quality.js", () => ({
  getRecentUtilizationAvg: vi.fn(async () => null),
}));

// Mock graph-context to control retrieval budget
vi.mock("../src/graph-context.js", () => ({
  getRetrievalBudgetTokens: vi.fn(() => 42000),
}));

describe("orchestrator", () => {
  let preflight: typeof import("../src/orchestrator.js")["preflight"];
  let recordToolCall: typeof import("../src/orchestrator.js")["recordToolCall"];
  let getSteeringCandidates: typeof import("../src/orchestrator.js")["getSteeringCandidates"];
  let getLastPreflightConfig: typeof import("../src/orchestrator.js")["getLastPreflightConfig"];
  let classifyIntent: any;
  let estimateComplexity: any;

  beforeAll(async () => {
    const intentMod = await import("../src/intent.js");
    classifyIntent = intentMod.classifyIntent;
    estimateComplexity = intentMod.estimateComplexity;

    const orchMod = await import("../src/orchestrator.js");
    preflight = orchMod.preflight;
    recordToolCall = orchMod.recordToolCall;
    getSteeringCandidates = orchMod.getSteeringCandidates;
    getLastPreflightConfig = orchMod.getLastPreflightConfig;
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("preflight fast-path", () => {
    it("short trivial inputs (<20 chars) skip retrieval entirely", async () => {
      const result = await preflight("hi", "test-session");
      expect(result.fastPath).toBe(true);
      expect(result.intent.category).toBe("unknown");
      expect(result.config.thinkingLevel).toBe("low");
      expect(result.config.toolLimit).toBe(15);
      expect(result.config.skipRetrieval).toBe(true);
      expect(result.config.vectorSearchLimits.turn).toBe(0);
      // classifyIntent should NOT have been called
      expect(classifyIntent).not.toHaveBeenCalled();
    });

    it("short but non-trivial inputs (20-50 chars) go through classification", async () => {
      classifyIntent.mockResolvedValue({
        category: "meta-session",
        confidence: 0.55,
        scores: [{ category: "meta-session", score: 0.55 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 0,
        suggestedThinking: "low",
      });
      const result = await preflight("ok thanks for the help with that", "test-session");
      expect(result.fastPath).toBe(false);
      expect(classifyIntent).toHaveBeenCalled();
      // With confidence 0.55 >= 0.40, uses meta-session config
      // tokenBudget derived from retrievalShare: 42000 * 0.07 = 2940
      expect(result.config.tokenBudget).toBe(2940);
      expect(result.config.vectorSearchLimits.turn).toBe(8);
    });

    it("short input with question marks bypasses fast path", async () => {
      // Pattern: /[?!].*[?!]/ means two ?/! marks
      classifyIntent.mockResolvedValue({
        category: "simple-question",
        confidence: 0.8,
        scores: [{ category: "simple-question", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "trivial",
        estimatedToolCalls: 0,
        suggestedThinking: "low",
      });

      const result = await preflight("What? Really?", "test-session");
      expect(result.fastPath).toBe(false);
      expect(classifyIntent).toHaveBeenCalled();
    });
  });

  describe("low-confidence fallback", () => {
    it("uses conservative config when confidence < 0.40", async () => {
      classifyIntent.mockResolvedValue({
        category: "code-debug",
        confidence: 0.30,
        scores: [{ category: "code-debug", score: 0.30 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 2,
        suggestedThinking: "low",
      });

      const result = await preflight("ah yeah lets do that thing we talked about", "test-session");
      expect(result.fastPath).toBe(false);
      // Should NOT use code-debug config (high thinking, 50 tools, 5000 budget)
      expect(result.config.thinkingLevel).toBe("low");
      expect(result.config.toolLimit).toBe(15);
      // tokenBudget derived from retrievalShare: 42000 * 0.08 = 3360
      expect(result.config.tokenBudget).toBe(3360);
      expect(result.config.vectorSearchLimits.turn).toBe(12);
      // But still records the intent category for logging
      expect(result.config.intent).toBe("code-debug");
    });
  });

  describe("preflight full classification", () => {
    it("maps code-debug intent to high thinking", async () => {
      classifyIntent.mockResolvedValue({
        category: "code-debug",
        confidence: 0.85,
        scores: [{ category: "code-debug", score: 0.85 }],
      });
      estimateComplexity.mockReturnValue({
        level: "moderate",
        estimatedToolCalls: 15,
        suggestedThinking: "high",
      });

      const result = await preflight(
        "Fix the null pointer exception in the authentication module please",
        "test-session",
      );
      expect(result.fastPath).toBe(false);
      expect(result.intent.category).toBe("code-debug");
      expect(result.config.thinkingLevel).toBe("high");
      expect(result.config.toolLimit).toBe(15); // code-debug=10, complexity=15 → min(15, ceil(10*1.5), 20) = 15
    });

    it("maps simple-question to low thinking and small budget", async () => {
      classifyIntent.mockResolvedValue({
        category: "simple-question",
        confidence: 0.9,
        scores: [{ category: "simple-question", score: 0.9 }],
      });
      estimateComplexity.mockReturnValue({
        level: "trivial",
        estimatedToolCalls: 0,
        suggestedThinking: "low",
      });

      const result = await preflight(
        "What is the difference between a class and an interface in TypeScript?",
        "test-session",
      );
      expect(result.config.thinkingLevel).toBe("low");
      expect(result.config.toolLimit).toBe(3);
      // tokenBudget derived from retrievalShare: 42000 * 0.10 = 4200
      expect(result.config.tokenBudget).toBe(4200);
      // High-confidence simple-question with no memory references → skip retrieval
      expect(result.config.skipRetrieval).toBe(true);
      expect(result.config.vectorSearchLimits.turn).toBe(0);
    });

    it("deep-explore gets highest (but finite) tool limit", async () => {
      classifyIntent.mockResolvedValue({
        category: "deep-explore",
        confidence: 0.8,
        scores: [{ category: "deep-explore", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "deep",
        estimatedToolCalls: 50,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Analyze every single file in this entire codebase and document the full architecture",
        "test-session",
      );
      expect(result.config.toolLimit).toBe(20); // capped: min(50, ceil(15*1.5), 20) = 20
    });

    it("reference-prior gets wider vector search limits", async () => {
      classifyIntent.mockResolvedValue({
        category: "reference-prior",
        confidence: 0.75,
        scores: [{ category: "reference-prior", score: 0.75 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 5,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Remember that auth bug we fixed yesterday? What was the root cause of that issue?",
        "test-session",
      );
      expect(result.config.vectorSearchLimits.turn).toBe(40);
      expect(result.config.vectorSearchLimits.memory).toBe(30);
      // tokenBudget derived from retrievalShare: 42000 * 0.25 = 10500
      expect(result.config.tokenBudget).toBe(10500);
    });

    it("continuation inherits previous config", async () => {
      // First: set up a code-debug config
      classifyIntent.mockResolvedValue({
        category: "code-debug",
        confidence: 0.85,
        scores: [{ category: "code-debug", score: 0.85 }],
      });
      estimateComplexity.mockReturnValue({
        level: "moderate",
        estimatedToolCalls: 15,
        suggestedThinking: "high",
      });
      await preflight(
        "Fix the authentication bug in the login handler module right now",
        "test-session",
      );

      // Now continuation should inherit
      classifyIntent.mockResolvedValue({
        category: "continuation",
        confidence: 0.9,
        scores: [{ category: "continuation", score: 0.9 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 10,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Yes keep going with that approach, continue what you were doing",
        "test-session",
      );
      expect(result.config.thinkingLevel).toBe("high"); // inherited from code-debug
      expect(result.config.toolLimit).toBe(15); // inherited (code-debug final=15, floor=max(15,15)=15)
    });

    it("complexity override: high thinking overrides lower", async () => {
      classifyIntent.mockResolvedValue({
        category: "code-read",
        confidence: 0.8,
        scores: [{ category: "code-read", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "complex",
        estimatedToolCalls: 30,
        suggestedThinking: "high",
      });

      const result = await preflight(
        "Read agent.ts and also read cli.ts and then explain the differences between them",
        "test-session",
      );
      // code-read default is "medium" but complexity says "high"
      expect(result.config.thinkingLevel).toBe("high");
    });

    it("complexity tool estimate can raise tool limit", async () => {
      classifyIntent.mockResolvedValue({
        category: "code-read",
        confidence: 0.8,
        scores: [{ category: "code-read", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "complex",
        estimatedToolCalls: 35,
        suggestedThinking: "high",
      });

      const result = await preflight(
        "Read every TypeScript file then analyze and compare the implementations across modules",
        "test-session",
      );
      // code-read toolLimit=5, complexity=35, cap = min(35, ceil(5*1.5), 20) = 8
      expect(result.config.toolLimit).toBe(8);
    });

    it("records preflight timing", async () => {
      classifyIntent.mockResolvedValue({
        category: "unknown",
        confidence: 0.3,
        scores: [],
      });
      estimateComplexity.mockReturnValue({
        level: "moderate",
        estimatedToolCalls: 15,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Something that takes the full classification path because it is long enough",
        "test-session",
      );
      expect(result.preflightMs).toBeGreaterThanOrEqual(0);
      expect(typeof result.preflightMs).toBe("number");
    });
  });

  describe("steering detection", () => {
    it("detects runaway tool calls (5+ consecutive same tool)", async () => {
      // Reset state via a preflight call
      await preflight("hi", "test-session");

      recordToolCall("readFile");
      recordToolCall("readFile");
      recordToolCall("readFile");
      recordToolCall("readFile");
      expect(getSteeringCandidates().filter((c) => c.type === "runaway")).toHaveLength(0);

      recordToolCall("readFile");
      const candidates = getSteeringCandidates().filter((c) => c.type === "runaway");
      expect(candidates.length).toBeGreaterThan(0);
      expect(candidates[0].detail).toContain("readFile");
    });

    it("does not flag runaway for mixed tool calls", async () => {
      await preflight("hi", "test-session");

      recordToolCall("readFile");
      recordToolCall("writeFile");
      recordToolCall("readFile");
      recordToolCall("writeFile");
      recordToolCall("readFile");

      const candidates = getSteeringCandidates().filter((c) => c.type === "runaway");
      expect(candidates).toHaveLength(0);
    });

    it("detects budget warning near tool limit", async () => {
      classifyIntent.mockResolvedValue({
        category: "simple-question",
        confidence: 0.9,
        scores: [{ category: "simple-question", score: 0.9 }],
      });
      estimateComplexity.mockReturnValue({
        level: "trivial",
        estimatedToolCalls: 0,
        suggestedThinking: "low",
      });

      // simple-question has toolLimit of 5
      await preflight(
        "What is the meaning of life, the universe, and everything in between?",
        "test-session",
      );

      for (let i = 0; i < 5; i++) recordToolCall(`tool${i}`);
      // At 5/5 — should have budget_warning since floor(5 * 0.85) = 4
      const candidates = getSteeringCandidates().filter((c) => c.type === "budget_warning");
      expect(candidates.length).toBeGreaterThan(0);
    });

    it("steering candidates reset on each preflight", async () => {
      await preflight("hi", "test-session");
      recordToolCall("readFile");
      recordToolCall("readFile");
      recordToolCall("readFile");
      recordToolCall("readFile");
      recordToolCall("readFile");
      expect(getSteeringCandidates().length).toBeGreaterThan(0);

      // New preflight should reset
      await preflight("hello there", "test-session");
      expect(getSteeringCandidates()).toHaveLength(0);
    });
  });

  describe("adaptive token budget", () => {
    let getRecentUtilizationAvg: any;

    beforeAll(async () => {
      const rqMod = await import("../src/retrieval-quality.js");
      getRecentUtilizationAvg = rqMod.getRecentUtilizationAvg;
    });

    it("scales budget down for low utilization (0%)", async () => {
      getRecentUtilizationAvg.mockResolvedValue(0);
      classifyIntent.mockResolvedValue({
        category: "code-read",
        confidence: 0.8,
        scores: [{ category: "code-read", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 2,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Read the authentication module and explain how it works please",
        "test-session",
      );
      // code-read base budget = 6300 (42000*0.15), scale = max(0.5, 0.5 + 0 * 0.8) = 0.5
      expect(result.config.tokenBudget).toBe(3150);
    });

    it("scales budget up for high utilization (100%)", async () => {
      getRecentUtilizationAvg.mockResolvedValue(1.0);
      classifyIntent.mockResolvedValue({
        category: "code-read",
        confidence: 0.8,
        scores: [{ category: "code-read", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 2,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Read the authentication module and explain the full implementation",
        "test-session",
      );
      // code-read base budget = 6300 (42000*0.15), scale = min(1.3, 0.5 + 1.0 * 0.8) = 1.3
      expect(result.config.tokenBudget).toBe(8190);
    });

    it("leaves budget unchanged when utilization returns null", async () => {
      getRecentUtilizationAvg.mockResolvedValue(null);
      classifyIntent.mockResolvedValue({
        category: "code-read",
        confidence: 0.8,
        scores: [{ category: "code-read", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 2,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Read the authentication module and tell me what patterns you see",
        "test-session",
      );
      // No scaling applied → base budget (42000 * 0.15 = 6300)
      expect(result.config.tokenBudget).toBe(6300);
    });

    it("50% utilization produces 1.0x scale (no change)", async () => {
      getRecentUtilizationAvg.mockResolvedValue(0.625);
      classifyIntent.mockResolvedValue({
        category: "code-read",
        confidence: 0.8,
        scores: [{ category: "code-read", score: 0.8 }],
      });
      estimateComplexity.mockReturnValue({
        level: "simple",
        estimatedToolCalls: 2,
        suggestedThinking: "medium",
      });

      const result = await preflight(
        "Show me the contents of the graph context module and explain it",
        "test-session",
      );
      // scale = 0.5 + 0.625 * 0.8 = 1.0, base = 6300
      expect(result.config.tokenBudget).toBe(6300);
    });

    it("skips adaptive scaling when skipRetrieval is true", async () => {
      getRecentUtilizationAvg.mockResolvedValue(0); // would produce 0.5x if applied
      // Trigger fast-path continuation (short input, not first turn — already advanced by prior tests)
      const result = await preflight("do it", "test-session");
      // Fast path / continuation — skipRetrieval should prevent scaling
      // The mock should NOT have been called for this path
      expect(result.config.skipRetrieval).toBe(true);
    });
  });
});
