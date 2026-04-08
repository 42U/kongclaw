/**
 * Tests for agent.ts — hasSemantic, extractAndLinkConcepts, and createZeraAgent.
 *
 * Unit tests with heavy mocking — no DB or API calls.
 */
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";

// ── Mock all heavy dependencies ──────────────────────────────────────────────

vi.mock("../src/surreal.js", () => ({
  upsertTurn: vi.fn(async () => "turn:test"),
  createSession: vi.fn(async () => "session:test"),
  updateSessionStats: vi.fn(async () => {}),
  relate: vi.fn(async () => {}),
  ensureAgent: vi.fn(async () => "agent:test"),
  ensureProject: vi.fn(async () => "project:test"),
  createTask: vi.fn(async () => "task:test"),
  linkSessionToTask: vi.fn(async () => {}),
  linkTaskToProject: vi.fn(async () => {}),
  linkAgentToTask: vi.fn(async () => {}),
  linkAgentToProject: vi.fn(async () => {}),
  upsertConcept: vi.fn(async (name: string) => `concept:${name.toLowerCase().replace(/\s+/g, "_")}`),
  createMemory: vi.fn(async () => "memory:test"),
  createArtifact: vi.fn(async () => "artifact:test"),
  getSessionTurns: vi.fn(async () => []),
  runMemoryMaintenance: vi.fn(async () => {}),
  archiveOldTurns: vi.fn(async () => {}),
  consolidateMemories: vi.fn(async () => {}),
  garbageCollectMemories: vi.fn(async () => {}),
  createMonologue: vi.fn(async () => {}),
  queryFirst: vi.fn(async () => [{ id: "skill:test", totalTools: 5 }]),
  queryExec: vi.fn(async () => {}),
  getSessionRetrievedMemories: vi.fn(async () => []),
  getReflectionSessionIds: vi.fn(async () => new Set()),
  getLatestHandoff: vi.fn(async () => null),
  endSession: vi.fn(async () => {}),
  isSurrealAvailable: vi.fn(async () => true),
  getDueMemories: vi.fn(async () => []),
  advanceSurfaceFade: vi.fn(async () => {}),
  resolveSurfaceMemory: vi.fn(async () => {}),
}));

vi.mock("../src/embeddings.js", () => ({
  embed: vi.fn(async () => new Array(1024).fill(0.1)),
  getEmbedCallCount: vi.fn(() => 0),
}));

vi.mock("../src/graph-context.js", () => ({
  graphTransformContext: vi.fn(async (msg: any) => msg),
  setCurrentSessionId: vi.fn(),
  setRetrievalConfig: vi.fn(),
  notifyToolCall: vi.fn(),
  setToolBudgetState: vi.fn(),
  reportMessageTokens: vi.fn(),
  getLastRetrievedSkillIds: vi.fn(() => []),
}));

vi.mock("../src/retrieval-quality.js", () => ({
  evaluateRetrieval: vi.fn(async () => {}),
  recordToolOutcome: vi.fn(),
  getStagedItems: vi.fn(() => []),
}));

vi.mock("../src/cognitive-check.js", () => ({
  shouldRunCheck: vi.fn(() => false),
  runCognitiveCheck: vi.fn(async () => {}),
}));

vi.mock("../src/orchestrator.js", () => ({
  setContextWindow: vi.fn(),
}));

vi.mock("../src/reflection.js", () => ({
  setReflectionContextWindow: vi.fn(),
  gatherSessionMetrics: vi.fn(async () => ({ avgUtilization: 0.5, wastedTokens: 0, toolFailureRate: 0, steeringCandidates: 0 })),
  shouldReflect: vi.fn(() => false),
}));

vi.mock("../src/causal.js", () => ({
  linkCausalEdges: vi.fn(async () => {}),
}));

vi.mock("../src/acan.js", () => ({
  checkACANReadiness: vi.fn(async () => {}),
}));

const skillsMock = {
  graduateCausalToSkills: vi.fn(async () => 0),
  recordSkillOutcome: vi.fn(async () => {}),
};
vi.mock("../src/skills.js", () => skillsMock);

const mockDaemon = {
  sendTurnBatch: vi.fn(),
  getStatus: vi.fn(async () => ({ type: "status" as const, extractedTurns: 0, pendingBatches: 0, errors: 0 })),
  shutdown: vi.fn(async () => {}),
  getExtractedTurnCount: vi.fn(() => 0),
};
vi.mock("../src/daemon-manager.js", () => ({
  startMemoryDaemon: vi.fn(() => mockDaemon),
}));

vi.mock("../src/subagent.js", () => ({
  createSubagentTool: vi.fn(() => ({ name: "subagent", execute: vi.fn() })),
}));

vi.mock("../src/tools.js", () => ({
  createTools: vi.fn(() => []),
}));

vi.mock("../src/tools/index.js", () => ({
  createGraphTools: vi.fn(() => [
    { name: "recall", execute: vi.fn() },
    { name: "core_memory", execute: vi.fn() },
    { name: "introspect", execute: vi.fn() },
  ]),
}));

// Mock pi-ai
vi.mock("@mariozechner/pi-ai", () => ({
  streamSimple: vi.fn(),
  completeSimple: vi.fn(async () => ({ content: [{ type: "text", text: "done" }] })),
  getModel: vi.fn(() => ({
    provider: "anthropic",
    modelId: "claude-opus-4-6",
    contextWindow: 200000,
  })),
}));

// Mock pi-agent-core with a fake Agent class
const mockSubscribe = vi.fn(() => vi.fn());
const mockSetSystemPrompt = vi.fn();
const mockSetModel = vi.fn();
const mockSetThinkingLevel = vi.fn();
const mockSetTools = vi.fn();
const mockSetSteeringMode = vi.fn();
const mockPrompt = vi.fn(async () => {});
const mockSteer = vi.fn();
const mockFollowUp = vi.fn();

// We need to capture beforeToolCall and afterToolCall from the constructor
let capturedBeforeToolCall: any;
let capturedAfterToolCall: any;
let capturedHandleEvent: any;

vi.mock("@mariozechner/pi-agent-core", () => {
  // Must be a real function (not arrow) so it can be called with `new`
  function MockAgent(this: any, config: any) {
    capturedBeforeToolCall = config.beforeToolCall;
    capturedAfterToolCall = config.afterToolCall;
    this.subscribe = mockSubscribe.mockImplementation((fn: any) => {
      capturedHandleEvent = fn;
      return vi.fn();
    });
    this.setSystemPrompt = mockSetSystemPrompt;
    this.setModel = mockSetModel;
    this.setThinkingLevel = mockSetThinkingLevel;
    this.setTools = mockSetTools;
    this.setSteeringMode = mockSetSteeringMode;
    this.prompt = mockPrompt;
    this.steer = mockSteer;
    this.followUp = mockFollowUp;
  }
  return { Agent: MockAgent };
});

// ── Import after mocks ──────────────────────────────────────────────────────

let hasSemantic: typeof import("../src/agent.js")["hasSemantic"];
let extractAndLinkConcepts: typeof import("../src/agent.js")["extractAndLinkConcepts"];
let createZeraAgent: typeof import("../src/agent.js")["createZeraAgent"];

let surrealMock: any;
let embedMock: any;

beforeAll(async () => {
  const mod = await import("../src/agent.js");
  hasSemantic = mod.hasSemantic;
  extractAndLinkConcepts = mod.extractAndLinkConcepts;
  createZeraAgent = mod.createZeraAgent;

  surrealMock = await import("../src/surreal.js");
  const embMod = await import("../src/embeddings.js");
  embedMock = embMod.embed;
});

beforeEach(() => {
  vi.clearAllMocks();
  capturedBeforeToolCall = undefined;
  capturedAfterToolCall = undefined;
  capturedHandleEvent = undefined;
});

// ── hasSemantic ─────────────────────────────────────────────────────────────

describe("hasSemantic", () => {
  it("rejects short texts below MIN_EMBED_LENGTH (15)", () => {
    expect(hasSemantic("ok")).toBe(false);
    expect(hasSemantic("yes")).toBe(false);
    expect(hasSemantic("hi")).toBe(false);
    expect(hasSemantic("do it")).toBe(false);
    expect(hasSemantic("")).toBe(false);
    expect(hasSemantic("k")).toBe(false);
  });

  it("rejects trivial patterns even at sufficient length", () => {
    expect(hasSemantic("sounds good!!!!")).toBe(false);
    expect(hasSemantic("thank you!     ")).toBe(false); // with trailing spaces
    // "go ahead please" is NOT trivial — it has extra content beyond the trivial pattern
    // Only exact trivial phrases match (e.g., "go ahead" or "go ahead!")
    expect(hasSemantic("go ahead!!!!!!!")).toBe(false);
  });

  it("accepts meaningful text above the threshold", () => {
    expect(hasSemantic("fix the bug in agent.ts")).toBe(true);
    expect(hasSemantic("what is the status of the build?")).toBe(true);
    expect(hasSemantic("can you review this code for me")).toBe(true);
  });

  it("accepts text that starts with trivial words but has more content", () => {
    expect(hasSemantic("ok now let's fix the memory leak")).toBe(true);
    expect(hasSemantic("yes please complete the refactor")).toBe(true);
  });

  it("rejects trivial patterns case-insensitively", () => {
    expect(hasSemantic("OK sure thing.")).toBe(false);
    expect(hasSemantic("THANK YOU!!!")).toBe(false);
    expect(hasSemantic("Sounds good.")).toBe(false);
  });
});

// ── extractAndLinkConcepts ──────────────────────────────────────────────────

describe("extractAndLinkConcepts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("extracts backtick-quoted terms", async () => {
    await extractAndLinkConcepts("turn:1", "Check the `graph-context` and `agent.ts` files");

    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "graph-context",
      expect.any(Array),
      "auto-extract",
    );
    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "agent.ts",
      expect.any(Array),
      "auto-extract",
    );
  });

  it("extracts Capitalized Multi Word terms", async () => {
    await extractAndLinkConcepts("turn:2", "The Memory Retrieval system uses Vector Search");

    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "Memory Retrieval",
      expect.any(Array),
      "auto-extract",
    );
    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "Vector Search",
      expect.any(Array),
      "auto-extract",
    );
  });

  it("extracts acronyms", async () => {
    await extractAndLinkConcepts("turn:3", "ACAN handles BGE-M3 embeddings with HNSW indexing");

    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "ACAN",
      expect.any(Array),
      "auto-extract",
    );
    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "BGE-M3",
      expect.any(Array),
      "auto-extract",
    );
    expect(surrealMock.upsertConcept).toHaveBeenCalledWith(
      "HNSW",
      expect.any(Array),
      "auto-extract",
    );
  });

  it("creates relate edges from turn to concept", async () => {
    await extractAndLinkConcepts("turn:4", "Using `SurrealDB` for storage");

    expect(surrealMock.relate).toHaveBeenCalledWith(
      "turn:4",
      "mentions",
      expect.stringContaining("concept:"),
    );
  });

  it("does nothing for text with no concepts", async () => {
    await extractAndLinkConcepts("turn:5", "just a plain sentence with nothing special");

    expect(surrealMock.upsertConcept).not.toHaveBeenCalled();
    expect(surrealMock.relate).not.toHaveBeenCalled();
  });

  it("filters stop-prefix words from extracted terms", async () => {
    // "The" should be stripped from "The Config" if it appears as a concept
    await extractAndLinkConcepts("turn:6", "Check `the config file` here");
    // The stop prefix "the " should be stripped, leaving "config file"
    const calls = (surrealMock.upsertConcept as any).mock.calls;
    for (const call of calls) {
      expect(call[0]).not.toMatch(/^the\s+/i);
    }
  });

  it("deduplicates case-insensitive concepts", async () => {
    await extractAndLinkConcepts("turn:7", "Use `ACAN` to check `acan` status");

    // Should only create one concept, not two
    const acanCalls = (surrealMock.upsertConcept as any).mock.calls.filter(
      (c: any) => c[0].toLowerCase() === "acan",
    );
    expect(acanCalls.length).toBe(1);
  });

  it("handles embed failures gracefully", async () => {
    (embedMock as any).mockRejectedValueOnce(new Error("embedding down"));

    // Should not throw
    await extractAndLinkConcepts("turn:8", "Check the `SurrealDB` layer");

    // upsertConcept still called (with null embedding)
    expect(surrealMock.upsertConcept).toHaveBeenCalled();
  });
});

// ── createZeraAgent ─────────────────────────────────────────────────────────

describe("createZeraAgent", () => {
  it("returns a ZeraAgent with expected interface", async () => {
    const za = await createZeraAgent("/tmp/test", "Test system prompt");

    expect(za).toHaveProperty("agent");
    expect(za).toHaveProperty("sessionId", "session:test");
    expect(typeof za.subscribe).toBe("function");
    expect(typeof za.prompt).toBe("function");
    expect(typeof za.setToolLimit).toBe("function");
    expect(typeof za.configureForTurn).toBe("function");
    expect(typeof za.softInterrupt).toBe("function");
    expect(typeof za.isSoftInterrupted).toBe("function");
    expect(typeof za.cleanup).toBe("function");
    expect(typeof za.generateExitLine).toBe("function");
  });

  it("bootstraps the 5-pillar graph on creation", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    expect(surrealMock.ensureAgent).toHaveBeenCalledWith("kongclaw", "claude-opus-4-6");
    expect(surrealMock.ensureProject).toHaveBeenCalledWith("test");
    expect(surrealMock.linkAgentToProject).toHaveBeenCalledWith("agent:test", "project:test");
    expect(surrealMock.createTask).toHaveBeenCalledWith("Session in test");
    expect(surrealMock.linkAgentToTask).toHaveBeenCalledWith("agent:test", "task:test");
    expect(surrealMock.linkTaskToProject).toHaveBeenCalledWith("task:test", "project:test");
    expect(surrealMock.createSession).toHaveBeenCalled();
    expect(surrealMock.linkSessionToTask).toHaveBeenCalledWith("session:test", "task:test");
  });

  it("runs memory maintenance on creation", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    // These are called but we don't await them — just check they were invoked
    expect(surrealMock.runMemoryMaintenance).toHaveBeenCalled();
    expect(surrealMock.archiveOldTurns).toHaveBeenCalled();
    expect(surrealMock.consolidateMemories).toHaveBeenCalled();
  });
});

// ── beforeToolCall (tool limiting) ──────────────────────────────────────────

describe("beforeToolCall", () => {
  it("planning gate blocks first call when no text output yet", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    // First call triggers planning gate (turnTextLength === 0)
    const result = await capturedBeforeToolCall({}, undefined);
    expect(result).toHaveProperty("block", true);
    expect(result.reason).toContain("PLANNING GATE");
  });

  it("allows tool calls after model has output text", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    // First call hits planning gate
    await capturedBeforeToolCall({}, undefined);
    // Simulate model outputting text (via event handler)
    capturedHandleEvent({
      type: "message_end",
      message: { role: "assistant", content: [{ type: "text", text: "Here is my plan for this task." }] },
    });
    // Second call should pass — model has spoken
    const result = await capturedBeforeToolCall({}, undefined);
    expect(result).toBeUndefined();
  });

  it("blocks tool calls when soft interrupted", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    za.softInterrupt();

    const result = await capturedBeforeToolCall({}, undefined);
    expect(result).toHaveProperty("block", true);
    expect(result.reason).toContain("Ctrl+C");
  });

  it("blocks tool calls over the limit", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    za.setToolLimit(3);

    // First call hits planning gate
    const gate = await capturedBeforeToolCall({}, undefined);
    expect(gate).toHaveProperty("block", true);
    expect(gate.reason).toContain("PLANNING GATE");

    // Simulate model outputting text (satisfies planning gate)
    await capturedHandleEvent({
      type: "message_end",
      message: { role: "assistant", content: [{ type: "text", text: "Here is my plan." }] },
    });

    // Next two should pass (toolCallCount 2 and 3, limit is 3)
    const r2 = await capturedBeforeToolCall({}, undefined);
    expect(r2).toBeUndefined();
    const r3 = await capturedBeforeToolCall({}, undefined);
    expect(r3).toBeUndefined();

    // Fourth should be blocked by limit (toolCallCount 4 > 3)
    const r4 = await capturedBeforeToolCall({}, undefined);
    expect(r4).toHaveProperty("block", true);
    expect(r4.reason).toContain("Tool call limit reached");
  });

  it("resets tool count on prompt()", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    za.setToolLimit(2);

    // First call hits planning gate, second uses up limit
    await capturedBeforeToolCall({}, undefined);
    await capturedHandleEvent({
      type: "message_end",
      message: { role: "assistant", content: [{ type: "text", text: "Plan." }] },
    });
    await capturedBeforeToolCall({}, undefined);
    const blocked = await capturedBeforeToolCall({}, undefined);
    expect(blocked).toHaveProperty("block", true);

    // Call prompt to reset
    await za.prompt("new turn");
    // First call again hits planning gate (turnTextLength reset)
    const gate = await capturedBeforeToolCall({}, undefined);
    expect(gate).toHaveProperty("block", true);
    expect(gate.reason).toContain("PLANNING GATE");
  });

  it("resets soft interrupt on prompt()", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    za.softInterrupt();
    expect(za.isSoftInterrupted()).toBe(true);

    await za.prompt("new turn");
    expect(za.isSoftInterrupted()).toBe(false);
  });
});

// ── afterToolCall (truncation) ──────────────────────────────────────────────

describe("afterToolCall", () => {
  it("passes through small results unchanged", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    const result = await capturedAfterToolCall({
      result: {
        content: [{ type: "text", text: "short result" }],
      },
    });
    expect(result).toBeUndefined();
  });

  it("truncates large text results", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    // TOOL_RESULT_MAX for 200k context = 200000 * 0.05 = 10000
    const bigText = "x".repeat(15000);
    const result = await capturedAfterToolCall({
      result: {
        content: [{ type: "text", text: bigText }],
      },
    });

    expect(result).toBeDefined();
    expect(result.content[0].text.length).toBeLessThan(bigText.length);
    expect(result.content[0].text).toContain("truncated");
  });

  it("handles non-array content gracefully", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    const result = await capturedAfterToolCall({
      result: { content: "not an array" },
    });
    expect(result).toBeUndefined();
  });

  it("handles missing result gracefully", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    const result = await capturedAfterToolCall({});
    expect(result).toBeUndefined();
  });

  it("preserves non-text content blocks", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    const bigText = "x".repeat(15000);
    const result = await capturedAfterToolCall({
      result: {
        content: [
          { type: "image", data: "base64..." },
          { type: "text", text: bigText },
        ],
      },
    });

    expect(result).toBeDefined();
    // Image block should be passed through unchanged
    expect(result.content[0]).toEqual({ type: "image", data: "base64..." });
    // Text block should be truncated
    expect(result.content[1].text).toContain("truncated");
  });
});

// ── configureForTurn ────────────────────────────────────────────────────────

describe("configureForTurn", () => {
  it("applies adaptive config and resets counters", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    // Burn some tool calls first (first hits planning gate)
    await capturedBeforeToolCall({}, undefined);
    await capturedBeforeToolCall({}, undefined);

    za.configureForTurn({
      toolLimit: 5,
      thinkingLevel: "high",
      tokenBudget: 3000,
      vectorSearchLimits: { turn: 10, memory: 5, concept: 3, reflection: 2, skill: 2 },
    });

    // Tool count should be reset — next call hits planning gate again (turnTextLength reset implicitly)
    za.setToolLimit(5);
    const r = await capturedBeforeToolCall({}, undefined);
    // Planning gate fires because toolCallCount === 1 and turnTextLength === 0
    expect(r).toHaveProperty("block", true);
    expect(r.reason).toContain("PLANNING GATE");
  });
});

// ── Event handling (storeUserTurn, storeAssistantTurn) ───────────────────────

describe("event handling", () => {
  it("stores user turns on message_end", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "user",
        content: "fix the bug",
        timestamp: Date.now(),
      },
    });

    // Should call upsertTurn for the user message
    expect(surrealMock.upsertTurn).toHaveBeenCalled();
    const call = (surrealMock.upsertTurn as any).mock.calls[0][0];
    expect(call.session_id).toBe("session:test");
    expect(call.role).toBe("user");
  });

  it("stores assistant turns on message_end", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "I'll fix that for you" }],
        timestamp: Date.now(),
      },
    });

    expect(surrealMock.upsertTurn).toHaveBeenCalled();
    const call = (surrealMock.upsertTurn as any).mock.calls[0][0];
    expect(call.session_id).toBe("session:test");
    expect(call.role).toBe("assistant");
  });

  it("records tool outcomes on tool_execution_end", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    const { recordToolOutcome } = await import("../src/retrieval-quality.js");

    await capturedHandleEvent({
      type: "tool_execution_end",
      toolName: "Bash",
      toolCallId: "tc_1",
      isError: false,
      result: { content: [{ text: "ok" }] },
    });

    expect(recordToolOutcome).toHaveBeenCalledWith(true);
  });

  it("records tool failures on tool_execution_end with isError", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    const { recordToolOutcome } = await import("../src/retrieval-quality.js");

    await capturedHandleEvent({
      type: "tool_execution_end",
      toolName: "Bash",
      toolCallId: "tc_2",
      isError: true,
      result: { content: [{ text: "error" }] },
    });

    expect(recordToolOutcome).toHaveBeenCalledWith(false);
  });

  it("tracks tool call depth on tool_execution_start", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");
    const { notifyToolCall } = await import("../src/graph-context.js");

    await capturedHandleEvent({
      type: "tool_execution_start",
      toolCallId: "tc_3",
      toolName: "Read",
      args: { path: "/tmp/foo" },
    });

    expect(notifyToolCall).toHaveBeenCalled();
  });
});

// ── generateExitLine ────────────────────────────────────────────────────────

describe("generateExitLine", () => {
  it("generates an exit line from session stats", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    const line = await za.generateExitLine({
      cumFullTokens: 100000,
      cumSentTokens: 70000,
      turns: 5,
      toolCalls: 12,
    });

    expect(typeof line).toBe("string");
    expect(line!.length).toBeGreaterThan(0);
  });

  it("returns null when completeSimple throws", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (completeSimple as any).mockRejectedValueOnce(new Error("API down"));

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    const line = await za.generateExitLine({
      cumFullTokens: 100000,
      cumSentTokens: 70000,
      turns: 5,
      toolCalls: 12,
    });

    expect(line).toBeNull();
  });
});

// ── storeToolResult (via tool_execution_end event) ──────────────────────────

describe("storeToolResult", () => {
  it("stores tool result with embedding when text is long enough", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "tool_execution_end",
      toolName: "Bash",
      toolCallId: "tc_store_1",
      isError: false,
      result: { content: [{ text: "some meaningful output from the tool execution" }] },
    });

    // upsertTurn should be called with role: "tool"
    expect(surrealMock.upsertTurn).toHaveBeenCalled();
    const call = (surrealMock.upsertTurn as any).mock.calls[0][0];
    expect(call.role).toBe("tool");
    expect(call.tool_name).toBe("Bash");
    expect(call.text).toContain("some meaningful output");
    // Text > 10 chars should trigger embedding
    expect(call.embedding).toBeTruthy();
  });

  it("stores tool result without embedding when text is short", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "tool_execution_end",
      toolName: "Bash",
      toolCallId: "tc_store_2",
      isError: false,
      result: { content: [{ text: "ok" }] },
    });

    expect(surrealMock.upsertTurn).toHaveBeenCalled();
    const call = (surrealMock.upsertTurn as any).mock.calls[0][0];
    expect(call.role).toBe("tool");
    // Short text — no embedding
    expect(call.embedding).toBeNull();
  });
});

// ── storeAssistantTurn details ──────────────────────────────────────────────

describe("storeAssistantTurn", () => {
  it("updates session stats from usage data", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Here is a detailed explanation of the solution to your problem" }],
        model: "claude-opus-4-6",
        usage: { input: 500, output: 200, totalTokens: 700 },
        timestamp: Date.now(),
      },
    });

    expect(surrealMock.updateSessionStats).toHaveBeenCalledWith("session:test", 500, 200);
  });

  it("updates session stats even for tool-call-only messages (no text)", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "tool_use", id: "tc_x", name: "Bash", input: {} }],
        usage: { input: 300, output: 100, totalTokens: 400 },
        timestamp: Date.now(),
      },
    });

    // Token tracking should fire even without text content
    expect(surrealMock.updateSessionStats).toHaveBeenCalledWith("session:test", 300, 100);
    // But no turn should be stored (no text)
    expect(surrealMock.upsertTurn).not.toHaveBeenCalled();
  });

  it("creates responds_to relation to last user turn", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    // First store a user turn so lastUserTurnId is set
    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "user",
        content: "What is the meaning of life and everything in the universe?",
        timestamp: Date.now(),
      },
    });

    vi.clearAllMocks();

    // Now store assistant turn
    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "The answer is 42 according to the hitchhiker guide" }],
        usage: { input: 100, output: 50, totalTokens: 150 },
        timestamp: Date.now(),
      },
    });

    // Should create responds_to and part_of relations
    const relateCalls = (surrealMock.relate as any).mock.calls;
    const responds = relateCalls.find((c: any) => c[1] === "responds_to");
    const partOf = relateCalls.find((c: any) => c[1] === "part_of");
    expect(responds).toBeTruthy();
    expect(partOf).toBeTruthy();
  });

  it("calls evaluateRetrieval for text turns", async () => {
    const { evaluateRetrieval } = await import("../src/retrieval-quality.js");
    await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "A detailed response about the architecture of the system" }],
        usage: { input: 100, output: 50, totalTokens: 150 },
        timestamp: Date.now(),
      },
    });

    expect(evaluateRetrieval).toHaveBeenCalled();
  });
});

// ── softInterrupt / steer / followUp ────────────────────────────────────────

describe("control flow", () => {
  it("softInterrupt sets the interrupted flag", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    expect(za.isSoftInterrupted()).toBe(false);
    za.softInterrupt();
    expect(za.isSoftInterrupted()).toBe(true);
  });

  it("steer delegates to agent.steer with proper message shape", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    za.steer("You should try a different approach");

    expect(mockSteer).toHaveBeenCalledTimes(1);
    const arg = mockSteer.mock.calls[0][0];
    expect(arg.role).toBe("user");
    expect(arg.content).toBe("You should try a different approach");
    expect(arg.timestamp).toBeTypeOf("number");
  });

  it("followUp delegates to agent.followUp with proper message shape", async () => {
    const za = await createZeraAgent("/tmp/test", "Test prompt");

    za.followUp("Also check the error logs");

    expect(mockFollowUp).toHaveBeenCalledTimes(1);
    const arg = mockFollowUp.mock.calls[0][0];
    expect(arg.role).toBe("user");
    expect(arg.content).toBe("Also check the error logs");
  });
});

// ── cleanup / runCombinedExtraction ─────────────────────────────────────────

describe("cleanup", () => {
  it("calls runCombinedExtraction which processes transcript", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    // Mock getSessionTurns to return enough turns
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([
      { role: "user", text: "fix the bug in auth" },
      { role: "assistant", text: "I found and fixed the issue" },
      { role: "user", text: "great thanks" },
    ]);
    // Mock completeSimple to return valid extraction JSON
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        handoff: "Fixed an auth bug. The session went smoothly with clear problem identification and resolution.",
        causal: [{ triggerText: "auth bug report", outcomeText: "fixed auth", chainType: "fix", success: true, confidence: 0.9, description: "Bug fix" }],
        skill: null,
        monologue: [],
        reflection: null,
      })}],
    });

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    await za.cleanup();

    // Should have fetched session turns
    expect(surrealMock.getSessionTurns).toHaveBeenCalledWith("session:test", 500);
    // Should have called completeSimple for extraction
    expect(completeSimple).toHaveBeenCalled();
    // Should have created a handoff memory
    expect(surrealMock.createMemory).toHaveBeenCalled();
  });

  it("skips extraction when session has fewer than 2 turns", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([
      { role: "user", text: "hi" },
    ]);

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    // completeSimple is called once during createZeraAgent for exit line setup... reset
    (completeSimple as any).mockClear();

    await za.cleanup();

    // Should NOT call completeSimple since < 2 turns
    expect(completeSimple).not.toHaveBeenCalled();
  });

  it("handles extraction failure gracefully", async () => {
    (surrealMock.getSessionTurns as any).mockRejectedValueOnce(new Error("DB down"));

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    // Should not throw
    await expect(za.cleanup()).resolves.toBeUndefined();
  });

  it("marks resolved memories from extraction output", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([
      { role: "user", text: "fix the cleanup timeout" },
      { role: "assistant", text: "Done — reduced timeout and fixed the exit path" },
    ]);
    // Simulate retrieved memories from this session
    (surrealMock.getSessionRetrievedMemories as any).mockResolvedValueOnce([
      { id: "memory:abc123", text: "cleanup timeout during exit is blocking for 15 seconds" },
      { id: "memory:def456", text: "embeddings model fails silently on shutdown" },
    ]);
    // Mock extraction response with resolved IDs
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        handoff: "Fixed the cleanup timeout by reducing it and adding proper resource teardown.",
        causal: [],
        skill: null,
        monologue: [],
        reflection: null,
        resolved: ["memory:abc123"],
      })}],
    });

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    await za.cleanup();

    // Should have called queryExec to mark memory as resolved
    expect(surrealMock.queryExec).toHaveBeenCalledWith(
      expect.stringContaining("UPDATE memory:abc123 SET status = 'resolved'"),
      expect.objectContaining({ sid: "session:test" }),
    );
  });

  it("marks task as completed on cleanup", async () => {
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([]);

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    await za.cleanup();

    // Should mark task completed (inlined record ID, no params)
    expect(surrealMock.queryExec).toHaveBeenCalledWith(
      expect.stringContaining("status = 'completed'"),
    );
  });

  it("calls graduateCausalToSkills during cleanup", async () => {
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([]);

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    skillsMock.graduateCausalToSkills.mockClear();
    await za.cleanup();

    expect(skillsMock.graduateCausalToSkills).toHaveBeenCalledOnce();
  });

  it("cleanup continues even if graduation fails", async () => {
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([]);

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    skillsMock.graduateCausalToSkills.mockRejectedValueOnce(new Error("graduation failed"));
    await expect(za.cleanup()).resolves.toBeUndefined();
    // Task completion should still run despite graduation failure
    expect(surrealMock.queryExec).toHaveBeenCalledWith(
      expect.stringContaining("status = 'completed'"),
    );
  });

  it("ignores invalid memory IDs in resolved list", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce([
      { role: "user", text: "do something" },
      { role: "assistant", text: "done" },
    ]);
    (surrealMock.getSessionRetrievedMemories as any).mockResolvedValueOnce([
      { id: "memory:legit1", text: "a real memory" },
    ]);
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        handoff: "Did the thing.",
        causal: [],
        skill: null,
        monologue: [],
        reflection: null,
        resolved: ["memory:legit1", "not_a_memory", "turn:injection_attempt", 42],
      })}],
    });

    const za = await createZeraAgent("/tmp/test", "Test prompt");
    await za.cleanup();

    // Should only resolve the valid memory ID, not the malformed ones
    const resolveUpdateCalls = (surrealMock.queryExec as any).mock.calls.filter(
      (c: any[]) => typeof c[0] === "string" && c[0].includes("status = 'resolved'"),
    );
    expect(resolveUpdateCalls).toHaveLength(1);
    expect(resolveUpdateCalls[0][0]).toContain("memory:legit1");
  });
});

// ── artifact tracking via tool_execution_start ──────────────────────────────

describe("artifact tracking", () => {
  it("tracks file paths from tool args on tool_execution_start", async () => {
    const { notifyToolCall } = await import("../src/graph-context.js");
    await createZeraAgent("/tmp/test", "Test prompt");

    await capturedHandleEvent({
      type: "tool_execution_start",
      toolCallId: "tc_art_1",
      toolName: "Write",
      args: { path: "/home/zero/project/index.ts", content: "export {}" },
    });

    expect(notifyToolCall).toHaveBeenCalled();
  });

  it("handles tool_execution_start with no args gracefully", async () => {
    await createZeraAgent("/tmp/test", "Test prompt");

    // Should not throw even with missing args
    await expect(capturedHandleEvent({
      type: "tool_execution_start",
      toolCallId: "tc_art_2",
      toolName: "Bash",
    })).resolves.toBeUndefined();
  });
});

// ── Memory Daemon ───────────────────────────────────────────────────────────

describe("memory daemon", () => {
  const daemonOptions = {
    surrealConfig: { url: "ws://localhost:8000", namespace: "test", database: "test", username: "root", password: "root" },
    anthropicApiKey: "sk-ant-test-key",
    embeddingModelPath: "/tmp/model.gguf",
  };

  it("starts daemon when config is provided", async () => {
    const { startMemoryDaemon } = await import("../src/daemon-manager.js");
    await createZeraAgent("/tmp/test", "Test prompt", "claude-opus-4-6", daemonOptions);

    expect(startMemoryDaemon).toHaveBeenCalledWith(
      daemonOptions.surrealConfig,
      daemonOptions.anthropicApiKey,
      daemonOptions.embeddingModelPath,
      "session:test",
    );
  });

  it("sends turn batch after accumulating ~12K tokens of content", async () => {
    // Mock getSessionTurns for the daemon batch call
    (surrealMock.getSessionTurns as any).mockResolvedValue([
      { role: "user", text: "test input" },
      { role: "assistant", text: "test output" },
    ]);

    await createZeraAgent("/tmp/test", "Test prompt", "claude-opus-4-6", daemonOptions);
    mockDaemon.sendTurnBatch.mockClear();

    // Send multiple assistant messages that accumulate enough tokens
    // 12K tokens ≈ 48K chars. Each message needs worthEmbedding=true (>15 chars, not trivial)
    for (let i = 0; i < 5; i++) {
      await capturedHandleEvent({
        type: "message_end",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "x".repeat(10000) + ` meaningful content batch ${i}` }],
          usage: { input: 100, output: 50 },
          timestamp: Date.now(),
        },
      });
    }

    // 5 messages × ~10026 chars each ÷ 4 ≈ 12,532 tokens → should trigger daemon batch
    expect(mockDaemon.sendTurnBatch).toHaveBeenCalled();
  });

  it("does not send batch before threshold is reached", async () => {
    (surrealMock.getSessionTurns as any).mockResolvedValue([]);

    await createZeraAgent("/tmp/test", "Test prompt", "claude-opus-4-6", daemonOptions);
    mockDaemon.sendTurnBatch.mockClear();

    // Single short message — well under 12K tokens
    await capturedHandleEvent({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "Here is a brief answer to your question about the code" }],
        usage: { input: 100, output: 50 },
        timestamp: Date.now(),
      },
    });

    expect(mockDaemon.sendTurnBatch).not.toHaveBeenCalled();
  });

  it("shuts down daemon during cleanup before extraction", async () => {
    (surrealMock.getSessionTurns as any).mockResolvedValue([]);

    const za = await createZeraAgent("/tmp/test", "Test prompt", "claude-opus-4-6", daemonOptions);
    mockDaemon.shutdown.mockClear();

    await za.cleanup();

    expect(mockDaemon.shutdown).toHaveBeenCalledWith(10_000);
  });

  it("slims extraction when daemon has already processed turns", async () => {
    const { completeSimple } = await import("@mariozechner/pi-ai");

    // Daemon has extracted 8 turns, session has 10 — delta of 2, which is ≤ 4
    mockDaemon.getExtractedTurnCount.mockReturnValue(8);
    (surrealMock.getSessionTurns as any).mockResolvedValueOnce(
      Array.from({ length: 10 }, (_, i) => ({ role: i % 2 === 0 ? "user" : "assistant", text: `turn ${i}` })),
    );
    (completeSimple as any).mockResolvedValueOnce({
      content: [{ type: "text", text: JSON.stringify({
        handoff: "Session went well, all tasks completed.",
        causal: [],
        skill: null,
        monologue: [],
        reflection: null,
        resolved: [],
      })}],
    });

    const za = await createZeraAgent("/tmp/test", "Test prompt", "claude-opus-4-6", daemonOptions);
    await za.cleanup();

    // The system prompt should NOT contain causal extraction instructions (daemon handled it)
    const extractionCall = (completeSimple as any).mock.calls.find(
      (c: any) => c[1]?.systemPrompt?.includes("handoff"),
    );
    expect(extractionCall).toBeTruthy();
    // When !hasDelta, causal line should just be "causal: []"
    expect(extractionCall[1].systemPrompt).toContain("causal: []");
  });
});

// ── isCorrection ──────────────────────────────────────────────────────────────

describe("isCorrection", () => {
  let isCorrection: typeof import("../src/agent.js")["isCorrection"];

  beforeAll(async () => {
    const mod = await import("../src/agent.js");
    isCorrection = mod.isCorrection;
  });

  it("detects 'no, ...' corrections", () => {
    expect(isCorrection("no, I meant the other one")).toBe(true);
    expect(isCorrection("No! That's wrong")).toBe(true);
    expect(isCorrection("no. do it differently")).toBe(true);
  });

  it("detects 'wrong' corrections", () => {
    expect(isCorrection("wrong, try again")).toBe(true);
  });

  it("detects 'that's not correct' corrections", () => {
    expect(isCorrection("that's not correct, the file is elsewhere")).toBe(true);
    expect(isCorrection("thats incorrect actually")).toBe(true);
  });

  it("detects 'don't' corrections", () => {
    expect(isCorrection("don't do that, use the other approach")).toBe(true);
    expect(isCorrection("dont mock the database")).toBe(true);
  });

  it("detects 'stop' corrections", () => {
    expect(isCorrection("stop doing that please")).toBe(true);
  });

  it("detects 'actually' corrections", () => {
    expect(isCorrection("actually, I want it in TypeScript")).toBe(true);
    expect(isCorrection("actually I changed my mind")).toBe(true);
  });

  it("detects 'I said/meant/want' corrections", () => {
    expect(isCorrection("i said the blue one")).toBe(true);
    expect(isCorrection("i meant the src directory")).toBe(true);
    expect(isCorrection("i want it done differently")).toBe(true);
  });

  it("rejects short messages", () => {
    expect(isCorrection("no")).toBe(false);
    expect(isCorrection("wrong")).toBe(false);
    expect(isCorrection("nah")).toBe(false);
  });

  it("rejects non-correction messages", () => {
    expect(isCorrection("can you help me with this?")).toBe(false);
    expect(isCorrection("looks good, ship it")).toBe(false);
    expect(isCorrection("the tests are passing")).toBe(false);
    expect(isCorrection("now let's add the feature")).toBe(false);
  });
});
