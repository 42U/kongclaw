/**
 * Tests for src/tools/introspect.ts — the database introspection tool.
 *
 * Mock-based: no actual DB calls.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

// ── Mocks ────────────────────────────────────────────────────────────────────

const mockIsSurrealAvailable = vi.fn(async () => true);
const mockGetSurrealInfo = vi.fn(() => ({
  url: "ws://localhost:8042/rpc",
  ns: "kong",
  db: "memory",
  connected: true,
}));
const mockPingDb = vi.fn(async () => true);
const mockAssertRecordId = vi.fn((id: string) => {
  if (!/^[a-z_]+:[a-zA-Z0-9_]+$/.test(id)) throw new Error("invalid");
});
const mockQueryFirst = vi.fn(async () => []);
const mockGetDb = vi.fn();

vi.mock("../../src/surreal.js", () => ({
  isSurrealAvailable: (...a: any[]) => mockIsSurrealAvailable(...a),
  getSurrealInfo: (...a: any[]) => mockGetSurrealInfo(...a),
  pingDb: (...a: any[]) => mockPingDb(...a),
  assertRecordId: (...a: any[]) => mockAssertRecordId(...a),
  queryFirst: (...a: any[]) => mockQueryFirst(...a),
  getDb: (...a: any[]) => mockGetDb(...a),
}));

// ── Import after mocks ──────────────────────────────────────────────────────

import { createTool } from "../../src/tools/introspect.js";

const ctx = { sessionId: "session:test123", cwd: "/tmp" };
const tool = createTool(ctx);

// ── Helpers ──────────────────────────────────────────────────────────────────

function text(result: any): string {
  return result.content[0].text;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("introspect tool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsSurrealAvailable.mockResolvedValue(true);
    mockPingDb.mockResolvedValue(true);
    mockQueryFirst.mockResolvedValue([]);
  });

  // ── Metadata ─────────────────────────────────────────────────────────────

  it("has correct name and label", () => {
    expect(tool.name).toBe("introspect");
    expect(tool.label).toBe("introspect");
  });

  it("description warns against curl/bash DB access", () => {
    expect(tool.description).toContain("NEVER");
    expect(tool.description).toContain("curl");
    expect(tool.description).toContain("bash");
  });

  // ── DB unavailable ───────────────────────────────────────────────────────

  it("returns unavailable when DB is down", async () => {
    mockIsSurrealAvailable.mockResolvedValue(false);
    const result = await tool.execute("call1", { action: "status" });
    expect(text(result)).toBe("Database unavailable.");
  });

  // ── Status action ────────────────────────────────────────────────────────

  it("status action includes connection info and session", async () => {
    const result = await tool.execute("call2", { action: "status" });
    const t = text(result);
    expect(t).toContain("ws://localhost:8042/rpc");
    expect(t).toContain("zera");
    expect(t).toContain("memory");
    expect(t).toContain("Ping:        OK");
    expect(t).toContain("session:test123");
  });

  it("status action shows FAILED ping when DB ping fails", async () => {
    mockPingDb.mockResolvedValue(false);
    const result = await tool.execute("call3", { action: "status" });
    expect(text(result)).toContain("Ping:        FAILED");
  });

  it("status action shows table counts", async () => {
    // Return count 42 for all table queries
    mockQueryFirst.mockResolvedValue([{ count: 42 }]);
    const result = await tool.execute("call4", { action: "status" });
    const t = text(result);
    expect(t).toContain("42");
    expect(t).toContain("Total records:");
    expect(t).toContain("Total embeddings:");
  });

  // ── Count action ─────────────────────────────────────────────────────────

  it("count requires table param", async () => {
    const result = await tool.execute("call5", { action: "count" });
    expect(text(result)).toContain("'table' is required");
  });

  it("count rejects unknown table", async () => {
    const result = await tool.execute("call6", { action: "count", table: "hackers" });
    const t = text(result);
    expect(t).toContain("unknown table");
    expect(t).toContain("hackers");
  });

  it("count returns row count for valid table", async () => {
    mockQueryFirst.mockResolvedValue([{ count: 7 }]);
    const result = await tool.execute("call7", { action: "count", table: "memory" });
    expect(text(result)).toBe("memory: 7 rows");
  });

  it("count with filter returns filtered count", async () => {
    mockQueryFirst.mockResolvedValue([{ count: 3 }]);
    const result = await tool.execute("call8", { action: "count", table: "memory", filter: "active" });
    expect(text(result)).toContain("3 rows");
    expect(text(result)).toContain("filter: active");
  });

  it("count rejects unknown filter", async () => {
    const result = await tool.execute("call9", { action: "count", table: "memory", filter: "bogus" });
    expect(text(result)).toContain("unknown filter");
    expect(text(result)).toContain("bogus");
  });

  // ── Verify action ────────────────────────────────────────────────────────

  it("verify requires record_id", async () => {
    const result = await tool.execute("call10", { action: "verify" });
    expect(text(result)).toContain("'record_id' is required");
  });

  it("verify rejects invalid record ID", async () => {
    mockAssertRecordId.mockImplementation(() => { throw new Error("bad"); });
    const result = await tool.execute("call11", { action: "verify", record_id: "not-valid" });
    expect(text(result)).toContain("invalid record ID format");
  });

  it("verify reports not found when record missing", async () => {
    mockAssertRecordId.mockImplementation(() => {}); // valid
    mockQueryFirst.mockResolvedValue([]);
    const result = await tool.execute("call12", { action: "verify", record_id: "memory:abc123" });
    expect(text(result)).toContain("Record not found");
    expect(result.details.exists).toBe(false);
  });

  it("verify returns record data when found", async () => {
    mockAssertRecordId.mockImplementation(() => {});
    mockQueryFirst.mockResolvedValue([{ id: "memory:abc123", text: "hello world", status: "active" }]);
    const result = await tool.execute("call13", { action: "verify", record_id: "memory:abc123" });
    const t = text(result);
    expect(t).toContain("Record memory:abc123");
    expect(t).toContain("hello world");
    expect(result.details.exists).toBe(true);
  });

  it("verify strips large embedding arrays", async () => {
    mockAssertRecordId.mockImplementation(() => {});
    const bigArray = new Array(1536).fill(0.1);
    mockQueryFirst.mockResolvedValue([{ id: "memory:x", embedding: bigArray, text: "test" }]);
    const result = await tool.execute("call14", { action: "verify", record_id: "memory:x" });
    expect(text(result)).toContain("[1536 dims]");
    expect(text(result)).not.toContain("0.1");
  });

  // ── Query action ─────────────────────────────────────────────────────────

  it("query lists templates when no filter given", async () => {
    const result = await tool.execute("call15", { action: "query" });
    const t = text(result);
    expect(t).toContain("Available query templates");
    expect(t).toContain("recent");
    expect(t).toContain("sessions");
    expect(t).toContain("embedding_coverage");
  });

  it("query lists templates for unknown template name", async () => {
    const result = await tool.execute("call16", { action: "query", filter: "nonexistent" });
    expect(text(result)).toContain("Available query templates");
  });

  it("query 'recent' requires valid table", async () => {
    const result = await tool.execute("call17", { action: "query", filter: "recent" });
    expect(text(result)).toContain("requires a valid table");
  });

  it("query 'recent' with valid table returns results", async () => {
    mockQueryFirst.mockResolvedValue([
      { id: "memory:1", text: "first", created_at: "2026-01-01" },
      { id: "memory:2", text: "second", created_at: "2026-01-02" },
    ]);
    const result = await tool.execute("call18", { action: "query", table: "memory", filter: "recent" });
    const t = text(result);
    expect(t).toContain("recent (memory)");
    expect(t).toContain("first");
    expect(t).toContain("second");
  });

  it("query 'sessions' works without table", async () => {
    mockQueryFirst.mockResolvedValue([{ id: "session:1", turn_count: 5 }]);
    const result = await tool.execute("call19", { action: "query", filter: "sessions" });
    expect(text(result)).toContain("sessions");
    expect(result.details.count).toBe(1);
  });

  it("query returns 'no results' when empty", async () => {
    mockQueryFirst.mockResolvedValue([]);
    const result = await tool.execute("call20", { action: "query", filter: "sessions" });
    expect(text(result)).toContain("No results");
  });

  // ── Error handling ───────────────────────────────────────────────────────

  it("catches and reports execution errors", async () => {
    mockQueryFirst.mockRejectedValue(new Error("connection lost"));
    const result = await tool.execute("call21", { action: "count", table: "memory" });
    expect(text(result)).toContain("Introspect failed");
    expect(text(result)).toContain("connection lost");
  });
});
