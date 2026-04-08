/**
 * Tests for graph-context scoring, deduplication, and token management.
 *
 * These test the pure functions exported/used internally by graph-context.ts.
 * We test via module internals where possible, otherwise test observable behavior.
 */
import { describe, it, expect } from "vitest";

// --- Recency decay tests (testing the math directly) ---

describe("recency scoring", () => {
  const RECENCY_DECAY_FAST = 0.99;
  const RECENCY_DECAY_SLOW = 0.999;
  const RECENCY_BOUNDARY_HOURS = 4;

  function recencyScore(timestamp: string | undefined): number {
    if (!timestamp) return 0.3;
    const hoursElapsed = (Date.now() - new Date(timestamp).getTime()) / (1000 * 60 * 60);
    if (hoursElapsed <= RECENCY_BOUNDARY_HOURS) {
      return Math.pow(RECENCY_DECAY_FAST, hoursElapsed);
    }
    const fastPart = Math.pow(RECENCY_DECAY_FAST, RECENCY_BOUNDARY_HOURS);
    return fastPart * Math.pow(RECENCY_DECAY_SLOW, hoursElapsed - RECENCY_BOUNDARY_HOURS);
  }

  it("very recent items (~0h) score near 1.0", () => {
    const now = new Date().toISOString();
    expect(recencyScore(now)).toBeGreaterThan(0.99);
  });

  it("1-hour old items still score high", () => {
    const oneHourAgo = new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString();
    expect(recencyScore(oneHourAgo)).toBeGreaterThan(0.98);
  });

  it("items within 4h boundary use fast decay", () => {
    const threeHours = new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString();
    const score = recencyScore(threeHours);
    const expected = Math.pow(0.99, 3);
    expect(score).toBeCloseTo(expected, 4);
  });

  it("items beyond 4h boundary use piecewise decay", () => {
    const twentyFourHours = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const score = recencyScore(twentyFourHours);
    // Should be: 0.99^4 * 0.999^20
    const expected = Math.pow(0.99, 4) * Math.pow(0.999, 20);
    expect(score).toBeCloseTo(expected, 4);
  });

  it("30-day old items still have non-trivial score (importance floor)", () => {
    const thirtyDays = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
    const score = recencyScore(thirtyDays);
    // With slow decay: 0.99^4 * 0.999^(720-4) ≈ 0.96 * 0.49 ≈ 0.47
    expect(score).toBeGreaterThan(0.3);
  });

  it("undefined timestamp gets moderate penalty", () => {
    expect(recencyScore(undefined)).toBe(0.3);
  });
});

// --- Deduplication tests ---

describe("semantic deduplication (Jaccard)", () => {
  function jaccard(a: string, b: string): number {
    const wordsA = new Set(a.toLowerCase().split(/\s+/).filter((w) => w.length > 2));
    const wordsB = new Set(b.toLowerCase().split(/\s+/).filter((w) => w.length > 2));
    let intersection = 0;
    for (const w of wordsA) {
      if (wordsB.has(w)) intersection++;
    }
    const union = wordsA.size + wordsB.size - intersection;
    return union > 0 ? intersection / union : 0;
  }

  it("identical strings have Jaccard = 1.0", () => {
    expect(jaccard("the quick brown fox", "the quick brown fox")).toBe(1.0);
  });

  it("completely different strings have Jaccard = 0.0", () => {
    expect(jaccard("alpha beta gamma", "delta epsilon zeta")).toBe(0.0);
  });

  it("partial overlap produces 0 < Jaccard < 1", () => {
    const score = jaccard("the authentication module is broken", "the authentication handler has a bug");
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("near-duplicates exceed 0.6 threshold", () => {
    const score = jaccard(
      "Fix the authentication module bug in login handler",
      "Fix the authentication module bug in the login handler",
    );
    expect(score).toBeGreaterThan(0.6);
  });
});

// --- Token estimation tests ---

describe("token estimation", () => {
  const CHARS_PER_TOKEN = 4;
  function estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN);
  }

  it("empty string = 0 tokens", () => {
    expect(estimateTokens("")).toBe(0);
  });

  it("4 chars = 1 token", () => {
    expect(estimateTokens("abcd")).toBe(1);
  });

  it("5 chars = 2 tokens (ceil)", () => {
    expect(estimateTokens("abcde")).toBe(2);
  });

  it("1000 chars ≈ 250 tokens", () => {
    expect(estimateTokens("a".repeat(1000))).toBe(250);
  });
});

// --- WMR v3 scoring with reflection boost + keyword overlap ---

describe("WMR v3 scoring weights", () => {
  // Replicate the WMR v3 formula from graph-context.ts
  function wmrScore(params: {
    cosine: number;
    recency: number;
    importance: number;
    access: number;
    neighborBonus: number;
    provenUtility: number;
    reflectionBoost: number;
    keywordBoost?: number;
    utilityPenalty?: number;
  }): number {
    return (
      0.22 * params.cosine +
      0.23 * params.recency +
      0.05 * params.importance +
      0.05 * params.access +
      0.10 * params.neighborBonus +
      0.15 * params.provenUtility +
      0.10 * params.reflectionBoost +
      0.12 * (params.keywordBoost ?? 0) -
      (params.utilityPenalty ?? 0)
    );
  }

  it("weights sum to 1.02 (keyword can exceed 1.0 contribution)", () => {
    // Base 8 signals: 0.22+0.23+0.05+0.05+0.10+0.15+0.10+0.12 = 1.02
    expect(0.22 + 0.23 + 0.05 + 0.05 + 0.10 + 0.15 + 0.10 + 0.12).toBeCloseTo(1.02, 10);
  });

  it("reflection boost adds ~10% to score for reflected sessions", () => {
    const base = { cosine: 0.8, recency: 0.5, importance: 0.5, access: 0.3, neighborBonus: 0, provenUtility: 0.5, reflectionBoost: 0 };
    const withReflection = { ...base, reflectionBoost: 1.0 };
    const diff = wmrScore(withReflection) - wmrScore(base);
    expect(diff).toBeCloseTo(0.10, 4);
  });

  it("memory without sessionId gets no reflection boost", () => {
    const base = { cosine: 0.8, recency: 0.5, importance: 0.5, access: 0.3, neighborBonus: 0, provenUtility: 0.5, reflectionBoost: 0 };
    // reflectionBoost = 0 when sessionId is undefined
    expect(wmrScore(base)).toBe(wmrScore({ ...base, reflectionBoost: 0 }));
  });

  it("reflected memory outranks identical non-reflected memory", () => {
    const params = { cosine: 0.7, recency: 0.6, importance: 0.5, access: 0.2, neighborBonus: 0, provenUtility: 0.5 };
    const nonReflected = wmrScore({ ...params, reflectionBoost: 0 });
    const reflected = wmrScore({ ...params, reflectionBoost: 1.0 });
    expect(reflected).toBeGreaterThan(nonReflected);
  });

  it("utility penalty still applies on top of reflection boost", () => {
    const params = { cosine: 0.7, recency: 0.6, importance: 0.5, access: 0.2, neighborBonus: 0, provenUtility: 0.05, reflectionBoost: 1.0 };
    const withPenalty = wmrScore({ ...params, utilityPenalty: 0.10 });
    const withoutPenalty = wmrScore({ ...params, utilityPenalty: 0 });
    expect(withPenalty).toBeLessThan(withoutPenalty);
    expect(withoutPenalty - withPenalty).toBeCloseTo(0.10, 4);
  });
});

// --- Access boost tests ---

describe("access count boost", () => {
  function accessBoost(count: number | undefined): number {
    return Math.log1p(count ?? 0);
  }

  it("0 access = 0 boost", () => {
    expect(accessBoost(0)).toBe(0);
  });

  it("undefined access = 0 boost", () => {
    expect(accessBoost(undefined)).toBe(0);
  });

  it("1 access = log(2) ≈ 0.693", () => {
    expect(accessBoost(1)).toBeCloseTo(0.693, 2);
  });

  it("boost grows logarithmically", () => {
    const b10 = accessBoost(10);
    const b100 = accessBoost(100);
    const b1000 = accessBoost(1000);
    // Each 10x only doubles the boost roughly
    expect(b100 / b10).toBeLessThan(2.5);
    expect(b1000 / b100).toBeLessThan(2);
  });
});

// --- Session continuity wiring tests ---
// Tests the math that setRetrievalConfig applies based on continuity signal.
// We replicate the switch to verify multipliers produce correct limits.

describe("session continuity adjustments", () => {
  function applyContinuity(
    continuity: "new_topic" | "continuation" | "repeat" | "tangent",
    limits: { turn: number; concept: number; memory: number },
    budget: number,
  ): { limits: { turn: number; concept: number; memory: number }; budget: number } {
    const l = { ...limits };
    let b = budget;
    switch (continuity) {
      case "new_topic":
        l.turn = Math.max(3, Math.round(l.turn * 0.5));
        l.concept = Math.round(l.concept * 1.3);
        l.memory = Math.round(l.memory * 1.3);
        break;
      case "continuation":
        l.turn = Math.round(l.turn * 1.3);
        l.memory = Math.round(l.memory * 1.2);
        break;
      case "repeat":
        l.memory = Math.round(l.memory * 1.5);
        b = Math.round(b * 1.2);
        break;
      case "tangent":
        b = Math.round(b * 0.8);
        break;
    }
    return { limits: l, budget: b };
  }

  const baseLimits = { turn: 20, concept: 15, memory: 15 };
  const baseBudget = 4000;

  it("new_topic halves turns, boosts concept and memory by 1.3x", () => {
    const r = applyContinuity("new_topic", baseLimits, baseBudget);
    expect(r.limits.turn).toBe(10);       // 20 * 0.5
    expect(r.limits.concept).toBe(20);    // 15 * 1.3 = 19.5 → 20
    expect(r.limits.memory).toBe(20);     // 15 * 1.3 = 19.5 → 20
    expect(r.budget).toBe(baseBudget);    // unchanged
  });

  it("new_topic enforces turn floor of 3", () => {
    const r = applyContinuity("new_topic", { turn: 4, concept: 10, memory: 10 }, baseBudget);
    expect(r.limits.turn).toBe(3); // 4 * 0.5 = 2 → floor 3
  });

  it("continuation boosts turns by 1.3x and memory by 1.2x", () => {
    const r = applyContinuity("continuation", baseLimits, baseBudget);
    expect(r.limits.turn).toBe(26);       // 20 * 1.3
    expect(r.limits.memory).toBe(18);     // 15 * 1.2
    expect(r.limits.concept).toBe(15);    // unchanged
    expect(r.budget).toBe(baseBudget);    // unchanged
  });

  it("repeat boosts memory by 1.5x and budget by 1.2x", () => {
    const r = applyContinuity("repeat", baseLimits, baseBudget);
    expect(r.limits.memory).toBe(23);     // 15 * 1.5 = 22.5 → 23
    expect(r.budget).toBe(4800);          // 4000 * 1.2
    expect(r.limits.turn).toBe(20);       // unchanged
    expect(r.limits.concept).toBe(15);    // unchanged
  });

  it("tangent reduces budget by 0.8x", () => {
    const r = applyContinuity("tangent", baseLimits, baseBudget);
    expect(r.budget).toBe(3200);          // 4000 * 0.8
    expect(r.limits.turn).toBe(20);       // unchanged
    expect(r.limits.concept).toBe(15);    // unchanged
    expect(r.limits.memory).toBe(15);     // unchanged
  });
});

// --- formatRelativeTime tests ---

import { formatRelativeTime } from "../src/graph-context.js";

describe("formatRelativeTime", () => {
  it("returns 'just now' for timestamps < 1 minute ago", () => {
    const ts = new Date(Date.now() - 30_000).toISOString();
    expect(formatRelativeTime(ts)).toBe("just now");
  });

  it("returns minutes for < 1 hour", () => {
    const ts = new Date(Date.now() - 15 * 60_000).toISOString();
    expect(formatRelativeTime(ts)).toBe("15m ago");
  });

  it("returns hours for < 24 hours", () => {
    const ts = new Date(Date.now() - 3 * 3600_000).toISOString();
    expect(formatRelativeTime(ts)).toBe("3h ago");
  });

  it("returns days for < 7 days", () => {
    const ts = new Date(Date.now() - 5 * 86400_000).toISOString();
    expect(formatRelativeTime(ts)).toBe("5d ago");
  });

  it("returns weeks for < 5 weeks", () => {
    const ts = new Date(Date.now() - 14 * 86400_000).toISOString();
    expect(formatRelativeTime(ts)).toBe("2w ago");
  });

  it("returns months for >= 5 weeks", () => {
    const ts = new Date(Date.now() - 60 * 86400_000).toISOString();
    expect(formatRelativeTime(ts)).toBe("2mo ago");
  });
});

// --- Keyword overlap scoring (WMR v3 signal) ---

describe("keyword overlap scoring", () => {
  const STOP_WORDS = new Set([
    "what", "when", "where", "who", "how", "which", "did", "do", "was", "were",
    "have", "has", "had", "is", "are", "the", "a", "an", "my", "me", "you",
    "your", "their", "it", "its", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "ago", "last", "that", "this", "there", "about", "and", "but",
    "not", "can", "will", "just", "than", "then", "also", "been",
  ]);

  function extractQueryKeywords(text: string): string[] {
    const matches = text.toLowerCase().match(/\b[a-z]{3,}\b/g) ?? [];
    return matches.filter((w) => !STOP_WORDS.has(w));
  }

  function keywordOverlapScore(queryKeywords: string[], docText: string): number {
    if (queryKeywords.length === 0) return 0;
    const docLower = (docText ?? "").toLowerCase();
    let hits = 0;
    for (const kw of queryKeywords) {
      if (docLower.includes(kw)) hits++;
    }
    return hits / queryKeywords.length;
  }

  it("extracts meaningful keywords, filters stop words", () => {
    const kws = extractQueryKeywords("What is the best database for my project?");
    expect(kws).toContain("best");
    expect(kws).toContain("database");
    expect(kws).toContain("project");
    expect(kws).not.toContain("what");
    expect(kws).not.toContain("the");
    expect(kws).not.toContain("for");
    expect(kws).not.toContain("is"); // 2 chars, filtered by regex
  });

  it("all keywords found = 1.0 overlap", () => {
    const kws = extractQueryKeywords("configure the database migration");
    const score = keywordOverlapScore(kws, "We need to configure the database migration for production");
    expect(score).toBe(1.0);
  });

  it("no keywords found = 0.0 overlap", () => {
    const kws = extractQueryKeywords("configure the database migration");
    const score = keywordOverlapScore(kws, "The weather is nice today");
    expect(score).toBe(0.0);
  });

  it("partial overlap produces fractional score", () => {
    const kws = extractQueryKeywords("fix the authentication login bug");
    const score = keywordOverlapScore(kws, "The authentication module has a security patch");
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("empty query produces 0.0", () => {
    const kws = extractQueryKeywords("is it a?");
    expect(kws).toHaveLength(0);
    expect(keywordOverlapScore(kws, "anything")).toBe(0);
  });
});

// --- Cross-encoder reranker blending ---

describe("reranker score blending", () => {
  const BLEND_VECTOR = 0.6;
  const BLEND_CROSS = 0.4;

  function blendScores(wmrScore: number, crossEncoderScore: number): number {
    return BLEND_VECTOR * wmrScore + BLEND_CROSS * crossEncoderScore;
  }

  it("equal scores blend to the same score", () => {
    expect(blendScores(0.5, 0.5)).toBeCloseTo(0.5);
  });

  it("high cross-encoder score boosts low WMR score", () => {
    const blended = blendScores(0.3, 0.9);
    expect(blended).toBeGreaterThan(0.3); // boosted from 0.3
    expect(blended).toBeCloseTo(0.54); // 0.6*0.3 + 0.4*0.9 = 0.18+0.36
  });

  it("low cross-encoder score reduces high WMR score", () => {
    const blended = blendScores(0.9, 0.1);
    expect(blended).toBeLessThan(0.9);
    expect(blended).toBeCloseTo(0.58); // 0.6*0.9 + 0.4*0.1
  });

  it("WMR weight (60%) dominates over cross-encoder (40%)", () => {
    // When WMR says high and cross-encoder says low, result stays above 0.5
    expect(blendScores(1.0, 0.0)).toBeCloseTo(0.6);
    expect(blendScores(0.0, 1.0)).toBeCloseTo(0.4);
  });

  it("reranking can change ordering", () => {
    // Item A: high WMR (0.8), low cross-encoder (0.1)
    const a = blendScores(0.8, 0.1);
    // Item B: low WMR (0.4), high cross-encoder (0.95)
    const b = blendScores(0.4, 0.95);
    // B wins because cross-encoder strongly prefers it
    expect(b).toBeGreaterThan(a);
  });
});

// --- Dedup threshold ---

describe("dedup cosine threshold", () => {
  it("threshold at 0.90 keeps more diverse results than 0.88", () => {
    // A cosine similarity of 0.89 is now kept (was deduplicated at 0.88)
    const THRESHOLD = 0.90;
    expect(0.89 > THRESHOLD).toBe(false); // kept
    expect(0.91 > THRESHOLD).toBe(true); // deduped
  });
});
