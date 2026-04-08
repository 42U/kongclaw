/**
 * Tests for wake-up synthesis (src/wakeup.ts).
 *
 * Tests handoff staleness annotations, graceful degradation, and synthesis flow.
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";

// Mock pi-ai
vi.mock("@mariozechner/pi-ai", () => ({
  completeSimple: vi.fn(async () => ({
    content: [{ type: "text", text: "I remember where we left off. The auth module was being refactored and we were halfway through the migration. Need to finish the token storage changes." }],
  })),
  getModel: vi.fn(() => ({
    provider: "anthropic",
    modelId: "claude-haiku-4-5",
    contextWindow: 200000,
  })),
}));

// Mock surreal
vi.mock("../src/surreal.js", () => ({
  isSurrealAvailable: vi.fn(async () => true),
  getLatestHandoff: vi.fn(async () => null),
  countResolvedSinceHandoff: vi.fn(async () => 0),
  getUnresolvedMemories: vi.fn(async () => []),
  getRecentFailedCausal: vi.fn(async () => []),
  getAllIdentityChunks: vi.fn(async () => []),
  getRecentMonologues: vi.fn(async () => []),
  getPreviousSessionTurns: vi.fn(async () => []),
  queryFirst: vi.fn(async () => []),
}));

// Mock errors
vi.mock("../src/errors.js", () => {
  const fn = vi.fn();
  fn.warn = vi.fn();
  return { swallow: fn };
});

let synthesizeWakeup: typeof import("../src/wakeup.js")["synthesizeWakeup"];
let synthesizeStartupCognition: typeof import("../src/wakeup.js")["synthesizeStartupCognition"];
let surrealMock: any;
let piAiMock: any;

beforeAll(async () => {
  const mod = await import("../src/wakeup.js");
  synthesizeWakeup = mod.synthesizeWakeup;
  synthesizeStartupCognition = mod.synthesizeStartupCognition;
  surrealMock = await import("../src/surreal.js");
  piAiMock = await import("@mariozechner/pi-ai");
});

beforeEach(() => {
  vi.clearAllMocks();
  // Reset defaults
  surrealMock.isSurrealAvailable.mockResolvedValue(true);
  surrealMock.getLatestHandoff.mockResolvedValue(null);
  surrealMock.countResolvedSinceHandoff.mockResolvedValue(0);
  surrealMock.getUnresolvedMemories.mockResolvedValue([]);
  surrealMock.getRecentFailedCausal.mockResolvedValue([]);
  surrealMock.getAllIdentityChunks.mockResolvedValue([]);
  surrealMock.getRecentMonologues.mockResolvedValue([]);
  surrealMock.getPreviousSessionTurns.mockResolvedValue([]);
  surrealMock.queryFirst.mockResolvedValue([]);
});

// ── Graceful degradation ────────────────────────────────────────────────

describe("synthesizeWakeup", () => {
  it("is a function", () => {
    expect(typeof synthesizeWakeup).toBe("function");
  });

  it("returns null when DB unavailable", async () => {
    surrealMock.isSurrealAvailable.mockResolvedValue(false);
    const result = await synthesizeWakeup();
    expect(result).toBeNull();
  });

  it("returns null on first boot (no handoff, no monologues, no identity, no turns)", async () => {
    const result = await synthesizeWakeup();
    expect(result).toBeNull();
  });

  it("returns null when only identity exists (no handoff or monologues)", async () => {
    surrealMock.getAllIdentityChunks.mockResolvedValue([
      { text: "I am Zera" },
    ]);
    const result = await synthesizeWakeup();
    expect(result).toBeNull();
  });
});

describe("synthesizeStartupCognition", () => {
  it("is a function", () => {
    expect(typeof synthesizeStartupCognition).toBe("function");
  });

  it("returns null when DB unavailable", async () => {
    surrealMock.isSurrealAvailable.mockResolvedValue(false);
    const result = await synthesizeStartupCognition();
    expect(result).toBeNull();
  });

  it("returns null on first boot", async () => {
    const result = await synthesizeStartupCognition();
    expect(result).toBeNull();
  });
});

// ── Handoff staleness annotations ───────────────────────────────────────

describe("handoff staleness in synthesizeWakeup", () => {
  it("annotates handoff with age and resolved count", async () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 3600_000).toISOString();
    surrealMock.getLatestHandoff.mockResolvedValue({
      text: "Working on auth refactor, token storage needs migration.",
      created_at: twoHoursAgo,
    });
    surrealMock.countResolvedSinceHandoff.mockResolvedValue(3);

    // LLM returns a long enough briefing (>= 100 chars)
    (piAiMock.completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: "I remember working on the auth refactor. Token storage migration was in progress. Three items have been resolved since my last handoff, so some of that work may already be complete. Let me check what's still open." }],
    });

    const result = await synthesizeWakeup();
    expect(result).not.toBeNull();

    // Verify countResolvedSinceHandoff was called with the handoff timestamp
    expect(surrealMock.countResolvedSinceHandoff).toHaveBeenCalledWith(twoHoursAgo);

    // Verify the LLM was called and the input contains the staleness annotation
    expect(piAiMock.completeSimple).toHaveBeenCalled();
    const llmCall = (piAiMock.completeSimple as any).mock.calls[0];
    const inputContent = llmCall[1].messages[0].content;
    expect(inputContent).toContain("[LAST HANDOFF]");
    expect(inputContent).toContain("2h old");
    expect(inputContent).toContain("3 memories resolved since");
    expect(inputContent).toContain("some items may already be done");
  });

  it("omits resolved annotation when count is 0", async () => {
    const oneHourAgo = new Date(Date.now() - 1 * 3600_000).toISOString();
    surrealMock.getLatestHandoff.mockResolvedValue({
      text: "Started new feature branch.",
      created_at: oneHourAgo,
    });
    surrealMock.countResolvedSinceHandoff.mockResolvedValue(0);

    (piAiMock.completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: "I just started a new feature branch about an hour ago. Nothing has been resolved yet so I should pick up right where I left off with that implementation work." }],
    });

    await synthesizeWakeup();

    const llmCall = (piAiMock.completeSimple as any).mock.calls[0];
    const inputContent = llmCall[1].messages[0].content;
    expect(inputContent).toContain("1h old");
    expect(inputContent).not.toContain("memories resolved since");
  });
});

describe("handoff staleness in synthesizeStartupCognition", () => {
  it("annotates handoff with age and resolved count", async () => {
    const fiveHoursAgo = new Date(Date.now() - 5 * 3600_000).toISOString();
    surrealMock.getLatestHandoff.mockResolvedValue({
      text: "Debugging the memory retrieval pipeline. ACAN scores seem off.",
      created_at: fiveHoursAgo,
    });
    surrealMock.countResolvedSinceHandoff.mockResolvedValue(2);

    // Startup cognition expects JSON
    (piAiMock.completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        greeting: "Back at the retrieval pipeline — 2 items resolved since last time.",
        proactive_thoughts: ["ACAN scores need review"],
        session_intent: "continue_prior",
      })}],
    });

    const result = await synthesizeStartupCognition();
    expect(result).not.toBeNull();
    expect(result!.greeting).toContain("retrieval pipeline");

    // Verify staleness annotation was passed to Haiku
    expect(surrealMock.countResolvedSinceHandoff).toHaveBeenCalledWith(fiveHoursAgo);
    const llmCall = (piAiMock.completeSimple as any).mock.calls[0];
    const inputContent = llmCall[1].messages[0].content;
    expect(inputContent).toContain("5h old");
    expect(inputContent).toContain("2 memories resolved since");
  });

  it("handles countResolvedSinceHandoff failure gracefully", async () => {
    const recentTime = new Date(Date.now() - 30 * 60_000).toISOString();
    surrealMock.getLatestHandoff.mockResolvedValue({
      text: "Quick session, just reviewed code.",
      created_at: recentTime,
    });
    surrealMock.countResolvedSinceHandoff.mockRejectedValue(new Error("DB error"));

    (piAiMock.completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        greeting: "Quick review session earlier.",
        proactive_thoughts: [],
        session_intent: "continue_prior",
      })}],
    });

    const result = await synthesizeStartupCognition();
    // Should still work — countResolvedSinceHandoff failure caught
    expect(result).not.toBeNull();
    expect(result!.greeting).toBeTruthy();
  });
});
