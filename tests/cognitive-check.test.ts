/**
 * Tests for cognitive-check.ts — shouldRunCheck, parseCheckResponse,
 * directive storage, and grade application.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock pi-ai
vi.mock("@mariozechner/pi-ai", () => ({
  completeSimple: vi.fn(async () => ({ content: [{ type: "text", text: "{}" }] })),
  getModel: vi.fn(() => ({
    provider: "anthropic",
    modelId: "claude-haiku-4-5",
    contextWindow: 200000,
  })),
}));

// Mock surreal
vi.mock("../src/surreal.js", () => ({
  queryExec: vi.fn(async () => {}),
  queryFirst: vi.fn(async () => [{ id: "retrieval_outcome:test" }]),
  updateUtilityCache: vi.fn(async () => {}),
  createCoreMemory: vi.fn(async () => {}),
}));

// Mock errors
vi.mock("../src/errors.js", () => {
  const fn = vi.fn();
  fn.warn = vi.fn();
  return { swallow: fn };
});

import {
  shouldRunCheck,
  parseCheckResponse,
  getPendingDirectives,
  clearPendingDirectives,
  getSessionContinuity,
  runCognitiveCheck,
} from "../src/cognitive-check.js";

beforeEach(() => {
  vi.clearAllMocks();
  clearPendingDirectives();
});

// ── shouldRunCheck ──────────────────────────────────────────────────────────

describe("shouldRunCheck", () => {
  it("returns false for turn 1", () => {
    expect(shouldRunCheck(1)).toBe(false);
  });

  it("returns true for turn 2", () => {
    expect(shouldRunCheck(2)).toBe(true);
  });

  it("returns true every 3 turns after turn 2", () => {
    expect(shouldRunCheck(5)).toBe(true);
    expect(shouldRunCheck(8)).toBe(true);
    expect(shouldRunCheck(11)).toBe(true);
    expect(shouldRunCheck(14)).toBe(true);
  });

  it("returns false for non-trigger turns", () => {
    expect(shouldRunCheck(3)).toBe(false);
    expect(shouldRunCheck(4)).toBe(false);
    expect(shouldRunCheck(6)).toBe(false);
    expect(shouldRunCheck(7)).toBe(false);
    expect(shouldRunCheck(9)).toBe(false);
    expect(shouldRunCheck(10)).toBe(false);
  });

  it("returns false for turn 0", () => {
    expect(shouldRunCheck(0)).toBe(false);
  });
});

// ── parseCheckResponse ──────────────────────────────────────────────────────

describe("parseCheckResponse", () => {
  it("parses clean JSON with all fields", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [{
        type: "repeat",
        target: "memory:abc123",
        instruction: "You discussed this before.",
        priority: "high",
      }],
      grades: [{
        id: "memory:abc123",
        relevant: true,
        reason: "Directly answers the query",
        score: 0.9,
      }],
      sessionContinuity: "repeat",
    }));

    expect(result).not.toBeNull();
    expect(result!.directives).toHaveLength(1);
    expect(result!.directives[0].type).toBe("repeat");
    expect(result!.directives[0].priority).toBe("high");
    expect(result!.grades).toHaveLength(1);
    expect(result!.grades[0].score).toBe(0.9);
    expect(result!.sessionContinuity).toBe("repeat");
  });

  it("strips markdown fences", () => {
    const json = JSON.stringify({
      directives: [],
      grades: [{ id: "turn:x1", relevant: false, reason: "noise", score: 0.1 }],
      sessionContinuity: "new_topic",
    });
    const result = parseCheckResponse("```json\n" + json + "\n```");
    expect(result).not.toBeNull();
    expect(result!.grades).toHaveLength(1);
  });

  it("handles trailing commas", () => {
    const result = parseCheckResponse('{"directives": [], "grades": [], "sessionContinuity": "new_topic",}');
    expect(result).not.toBeNull();
    expect(result!.sessionContinuity).toBe("new_topic");
  });

  it("returns null for non-JSON", () => {
    expect(parseCheckResponse("I can't do that")).toBeNull();
  });

  it("returns null for completely malformed JSON", () => {
    expect(parseCheckResponse("{this is not json at all!!!}")).toBeNull();
  });

  it("caps directives at 3", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: Array.from({ length: 5 }, (_, i) => ({
        type: "insight",
        target: `memory:m${i}`,
        instruction: `Insight ${i}`,
        priority: "low",
      })),
      grades: [],
      sessionContinuity: "continuation",
    }));
    expect(result!.directives).toHaveLength(3);
  });

  it("rejects invalid directive types", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [{
        type: "invalid_type",
        target: "memory:x",
        instruction: "test",
        priority: "low",
      }],
      grades: [],
      sessionContinuity: "new_topic",
    }));
    expect(result!.directives).toHaveLength(0);
  });

  it("rejects invalid grade IDs", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [
        { id: "memory:legit1", relevant: true, reason: "good", score: 0.8 },
        { id: "not_valid", relevant: true, reason: "bad id", score: 0.5 },
        { id: "DROP TABLE users;", relevant: false, reason: "injection", score: 0 },
      ],
      sessionContinuity: "new_topic",
    }));
    expect(result!.grades).toHaveLength(1);
    expect(result!.grades[0].id).toBe("memory:legit1");
  });

  it("clamps scores to 0-1", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [
        { id: "memory:a1", relevant: true, reason: "high", score: 5.0 },
        { id: "memory:b2", relevant: true, reason: "neg", score: -0.5 },
      ],
      sessionContinuity: "new_topic",
    }));
    expect(result!.grades[0].score).toBe(1);
    expect(result!.grades[1].score).toBe(0);
  });

  it("defaults unknown sessionContinuity to new_topic", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [],
      sessionContinuity: "unknown_value",
    }));
    expect(result!.sessionContinuity).toBe("new_topic");
  });

  it("defaults missing priority to medium", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [{ type: "noise", target: "turn:x1", instruction: "ignore this" }],
      grades: [],
      sessionContinuity: "new_topic",
    }));
    expect(result!.directives[0].priority).toBe("medium");
  });

  it("parses valid preferences", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [],
      sessionContinuity: "new_topic",
      preferences: [
        { observation: "User prefers terse responses", confidence: "high" },
        { observation: "User likes TypeScript", confidence: "medium" },
      ],
    }));
    expect(result!.preferences).toHaveLength(2);
    expect(result!.preferences[0].observation).toBe("User prefers terse responses");
    expect(result!.preferences[0].confidence).toBe("high");
    expect(result!.preferences[1].confidence).toBe("medium");
  });

  it("caps preferences at 2", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [],
      sessionContinuity: "new_topic",
      preferences: [
        { observation: "a", confidence: "high" },
        { observation: "b", confidence: "high" },
        { observation: "c", confidence: "high" },
      ],
    }));
    expect(result!.preferences).toHaveLength(2);
  });

  it("rejects preferences with invalid confidence", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [],
      sessionContinuity: "new_topic",
      preferences: [
        { observation: "test", confidence: "low" },
        { observation: "test2", confidence: "invalid" },
      ],
    }));
    expect(result!.preferences).toHaveLength(0);
  });

  it("handles missing preferences field gracefully", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [],
      sessionContinuity: "new_topic",
    }));
    expect(result!.preferences).toEqual([]);
  });

  it("truncates long preference observations", () => {
    const longObs = "x".repeat(300);
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [],
      sessionContinuity: "new_topic",
      preferences: [{ observation: longObs, confidence: "high" }],
    }));
    expect(result!.preferences[0].observation.length).toBe(200);
  });
});

// ── Directive storage ───────────────────────────────────────────────────────

describe("directive storage", () => {
  it("starts with empty directives", () => {
    expect(getPendingDirectives()).toEqual([]);
  });

  it("stores and retrieves directives after runCognitiveCheck", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [{
          type: "repeat",
          target: "memory:abc",
          instruction: "You discussed this before.",
          priority: "high",
        }],
        grades: [],
        sessionContinuity: "repeat",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "tell me about the books directory",
      responseText: "The books directory contains...",
      retrievedNodes: [{ id: "memory:abc", text: "Prior books discussion", score: 0.9, table: "memory" }],
      recentTurns: [{ role: "user", text: "tell me about books" }],
    });

    const directives = getPendingDirectives();
    expect(directives).toHaveLength(1);
    expect(directives[0].type).toBe("repeat");
    expect(getSessionContinuity()).toBe("repeat");
  });

  it("clearPendingDirectives empties the cache", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [{ type: "insight", target: "memory:x", instruction: "test", priority: "low" }],
        grades: [],
        sessionContinuity: "new_topic",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test response",
      retrievedNodes: [{ id: "memory:x", text: "test", score: 0.5, table: "memory" }],
      recentTurns: [],
    });

    expect(getPendingDirectives()).toHaveLength(1);
    clearPendingDirectives();
    expect(getPendingDirectives()).toEqual([]);
  });
});

// ── Grade application ───────────────────────────────────────────────────────

describe("grade application", () => {
  it("writes grades to DB via queryExec", async () => {
    const { queryExec } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "memory:abc", relevant: true, reason: "Directly useful", score: 0.85 },
          { id: "turn:xyz", relevant: false, reason: "Noise", score: 0.1 },
        ],
        sessionContinuity: "new_topic",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test query",
      responseText: "test response",
      retrievedNodes: [
        { id: "memory:abc", text: "useful memory", score: 0.8, table: "memory" },
        { id: "turn:xyz", text: "noisy turn", score: 0.6, table: "turn" },
      ],
      recentTurns: [],
    });

    // 2 grade UPDATE writes + 1 correction reinforcement (memory:abc is relevant but not learned)
    expect(queryExec).toHaveBeenCalledTimes(3);
    // Grade writes use direct UPDATE with interpolated record ID
    expect(queryExec).toHaveBeenCalledWith(
      expect.stringContaining("llm_relevance"),
      expect.objectContaining({ score: 0.85, relevant: true }),
    );
    // Relevant correction not learned → reinforce importance
    expect(queryExec).toHaveBeenCalledWith(
      expect.stringContaining("importance + 1"),
    );
  });
});

// ── Graceful degradation ────────────────────────────────────────────────────

describe("graceful degradation", () => {
  it("handles Haiku failure without throwing", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockRejectedValueOnce(new Error("API timeout"));

    await expect(runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [{ id: "memory:x", text: "test", score: 0.5, table: "memory" }],
      recentTurns: [],
    })).resolves.toBeUndefined();

    // Directives should remain empty
    expect(getPendingDirectives()).toEqual([]);
  });

  it("skips when no retrieved nodes", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [],
      recentTurns: [],
    });

    // Should not even call Haiku
    expect(completeSimple).not.toHaveBeenCalled();
  });

  it("handles malformed Haiku response gracefully", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: "I cannot produce JSON right now." }],
    });

    await expect(runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [{ id: "memory:x", text: "test", score: 0.5, table: "memory" }],
      recentTurns: [],
    })).resolves.toBeUndefined();

    expect(getPendingDirectives()).toEqual([]);
  });
});

// ── Noise suppression ─────────────────────────────────────────────────────

import { getSuppressedNodeIds } from "../src/cognitive-check.js";

describe("noise suppression", () => {
  it("suppresses nodes graded irrelevant with low score", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "memory:noise1", relevant: false, reason: "Not useful", score: 0.1 },
          { id: "memory:good1", relevant: true, reason: "Useful", score: 0.8 },
          { id: "turn:noise2", relevant: false, reason: "Old", score: 0.25 },
        ],
        sessionContinuity: "new_topic",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [
        { id: "memory:noise1", text: "noise", score: 0.1, table: "memory" },
        { id: "memory:good1", text: "good", score: 0.8, table: "memory" },
        { id: "turn:noise2", text: "old turn", score: 0.25, table: "turn" },
      ],
      recentTurns: [],
    });

    const suppressed = getSuppressedNodeIds();
    expect(suppressed.has("memory:noise1")).toBe(true);
    expect(suppressed.has("turn:noise2")).toBe(true);
    expect(suppressed.has("memory:good1")).toBe(false);
  });

  it("suppresses noise directive targets", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [{ type: "noise", target: "memory:noisy", instruction: "ignore", priority: "low" }],
        grades: [],
        sessionContinuity: "new_topic",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [{ id: "memory:noisy", text: "noisy", score: 0.5, table: "memory" }],
      recentTurns: [],
    });

    expect(getSuppressedNodeIds().has("memory:noisy")).toBe(true);
  });
});

// ── Mid-session resolution ────────────────────────────────────────────────

describe("mid-session resolution", () => {
  it("parses resolved field in grades", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [
        { id: "memory:done1", relevant: true, reason: "Done", score: 0.9, resolved: true },
        { id: "memory:open1", relevant: true, reason: "Still open", score: 0.8, resolved: false },
      ],
      sessionContinuity: "continuation",
    }));

    expect(result!.grades[0].resolved).toBe(true);
    expect(result!.grades[1].resolved).toBe(false);
  });

  it("defaults resolved to false when missing", () => {
    const result = parseCheckResponse(JSON.stringify({
      directives: [],
      grades: [{ id: "memory:x", relevant: true, reason: "test", score: 0.5 }],
      sessionContinuity: "new_topic",
    }));

    expect(result!.grades[0].resolved).toBe(false);
  });
});

// ── Correction lifecycle (decay + reinforce) ─────────────────────────────

describe("correction lifecycle", () => {
  it("decays importance when correction is learned", async () => {
    const { queryExec } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "memory:corr1", relevant: true, reason: "Correction followed", score: 0.9, learned: true },
        ],
        sessionContinuity: "continuation",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "use tabs",
      responseText: "I'll use tabs for indentation.",
      retrievedNodes: [{ id: "memory:corr1", text: "[CORRECTION] Use tabs over spaces", score: 0.9, table: "memory" }],
      recentTurns: [],
    });

    // Should have: 1 grade write + 1 decay call
    const calls = (queryExec as any).mock.calls;
    const decayCall = calls.find((c: any[]) => typeof c[0] === "string" && c[0].includes("importance - 2"));
    expect(decayCall).toBeDefined();
    expect(decayCall[0]).toContain("math::max([3");
    expect(decayCall[0]).toContain("memory:corr1");
  });

  it("reinforces importance when correction is ignored", async () => {
    const { queryExec } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "memory:corr2", relevant: true, reason: "Correction ignored", score: 0.7, learned: false },
        ],
        sessionContinuity: "continuation",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "format this code",
      responseText: "Here it is with spaces.",
      retrievedNodes: [{ id: "memory:corr2", text: "[CORRECTION] Use tabs over spaces", score: 0.7, table: "memory" }],
      recentTurns: [],
    });

    const calls = (queryExec as any).mock.calls;
    const reinforceCall = calls.find((c: any[]) => typeof c[0] === "string" && c[0].includes("importance + 1"));
    expect(reinforceCall).toBeDefined();
    expect(reinforceCall[0]).toContain("math::min([9");
    expect(reinforceCall[0]).toContain("memory:corr2");
  });

  it("skips correction adjustment for non-memory grades", async () => {
    const { queryExec } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "turn:t1", relevant: true, reason: "Turn context", score: 0.8, learned: true },
        ],
        sessionContinuity: "new_topic",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [{ id: "turn:t1", text: "turn data", score: 0.8, table: "turn" }],
      recentTurns: [],
    });

    const calls = (queryExec as any).mock.calls;
    // Only grade write, no correction adjustment (turn: prefix filtered)
    const correctionCalls = calls.filter((c: any[]) =>
      typeof c[0] === "string" && (c[0].includes("importance - 2") || c[0].includes("importance + 1")));
    expect(correctionCalls).toHaveLength(0);
  });
});

// ── Mid-session resolution DB writes ─────────────────────────────────────

describe("mid-session resolution writes", () => {
  it("fires UPDATE for resolved memory grades", async () => {
    const { queryExec } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "memory:done1", relevant: true, reason: "Addressed", score: 0.9, resolved: true },
          { id: "memory:open1", relevant: true, reason: "Still open", score: 0.8, resolved: false },
        ],
        sessionContinuity: "continuation",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:abc",
      userQuery: "done with that",
      responseText: "All taken care of.",
      retrievedNodes: [
        { id: "memory:done1", text: "fix the bug", score: 0.9, table: "memory" },
        { id: "memory:open1", text: "review PR", score: 0.8, table: "memory" },
      ],
      recentTurns: [],
    });

    const calls = (queryExec as any).mock.calls;
    const resolveCall = calls.find((c: any[]) =>
      typeof c[0] === "string" && c[0].includes("status = 'resolved'"));
    expect(resolveCall).toBeDefined();
    expect(resolveCall[0]).toContain("memory:done1");
    expect(resolveCall[1]).toEqual({ sid: "session:abc" });

    // Should NOT resolve memory:open1
    const openResolve = calls.find((c: any[]) =>
      typeof c[0] === "string" && c[0].includes("memory:open1") && c[0].includes("status = 'resolved'"));
    expect(openResolve).toBeUndefined();
  });

  it("skips resolution for non-memory IDs", async () => {
    const { queryExec } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [
          { id: "turn:t1", relevant: true, reason: "Done", score: 0.9, resolved: true },
        ],
        sessionContinuity: "new_topic",
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:test",
      userQuery: "test",
      responseText: "test",
      retrievedNodes: [{ id: "turn:t1", text: "turn data", score: 0.9, table: "turn" }],
      recentTurns: [],
    });

    const calls = (queryExec as any).mock.calls;
    const resolveCall = calls.find((c: any[]) =>
      typeof c[0] === "string" && c[0].includes("status = 'resolved'"));
    expect(resolveCall).toBeUndefined();
  });
});

// ── Preference → core memory storage ─────────────────────────────────────

describe("preference storage", () => {
  it("creates core memory for high-confidence preferences", async () => {
    const { createCoreMemory } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [],
        sessionContinuity: "new_topic",
        preferences: [
          { observation: "User prefers terse responses", confidence: "high" },
          { observation: "User likes dark mode", confidence: "medium" },
        ],
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:pref1",
      userQuery: "keep it short",
      responseText: "ok",
      retrievedNodes: [{ id: "memory:x", text: "context", score: 0.5, table: "memory" }],
      recentTurns: [],
    });

    // Only high-confidence pref should create core memory
    expect(createCoreMemory).toHaveBeenCalledTimes(1);
    expect(createCoreMemory).toHaveBeenCalledWith(
      "[USER PREFERENCE] User prefers terse responses",
      "preference", 7, 1, "session:pref1",
    );
  });

  it("skips core memory for medium-only preferences", async () => {
    const { createCoreMemory } = await import("../src/surreal.js");
    const { completeSimple } = await import("@mariozechner/pi-ai");

    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        directives: [],
        grades: [],
        sessionContinuity: "new_topic",
        preferences: [
          { observation: "User likes TypeScript", confidence: "medium" },
        ],
      })}],
    });

    await runCognitiveCheck({
      sessionId: "session:pref2",
      userQuery: "write some TS",
      responseText: "Here's the TypeScript.",
      retrievedNodes: [{ id: "memory:y", text: "context", score: 0.5, table: "memory" }],
      recentTurns: [],
    });

    expect(createCoreMemory).not.toHaveBeenCalled();
  });
});
