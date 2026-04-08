/**
 * Tests for SurrealDB layer (src/surreal.ts).
 *
 * Integration tests — require a running SurrealDB instance.
 * Skipped automatically when DB is unavailable.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";

let surrealAvailable = false;
let initSurreal: typeof import("../src/surreal.js")["initSurreal"];
let isSurrealAvailable: typeof import("../src/surreal.js")["isSurrealAvailable"];
let getDb: typeof import("../src/surreal.js")["getDb"];
let createSession: typeof import("../src/surreal.js")["createSession"];
let upsertTurn: typeof import("../src/surreal.js")["upsertTurn"];
let getSessionTurns: typeof import("../src/surreal.js")["getSessionTurns"];
let ensureAgent: typeof import("../src/surreal.js")["ensureAgent"];
let upsertConcept: typeof import("../src/surreal.js")["upsertConcept"];
let createMemory: typeof import("../src/surreal.js")["createMemory"];
let relate: typeof import("../src/surreal.js")["relate"];
let vectorSearch: typeof import("../src/surreal.js")["vectorSearch"];
let bumpAccessCounts: typeof import("../src/surreal.js")["bumpAccessCounts"];
let closeSurreal: typeof import("../src/surreal.js")["closeSurreal"];

/** Runtime skip — `it.skipIf` evaluates too early (before beforeAll runs). */
function skipIfNoDb() {
  if (!surrealAvailable) {
    console.log("SurrealDB not available — skipping");
    return true;
  }
  return false;
}

beforeAll(async () => {
  try {
    const configMod = await import("../src/config.js");
    const config = configMod.loadConfig();
    const mod = await import("../src/surreal.js");
    initSurreal = mod.initSurreal;
    isSurrealAvailable = mod.isSurrealAvailable;
    getDb = mod.getDb;
    createSession = mod.createSession;
    upsertTurn = mod.upsertTurn;
    getSessionTurns = mod.getSessionTurns;
    ensureAgent = mod.ensureAgent;
    upsertConcept = mod.upsertConcept;
    createMemory = mod.createMemory;
    relate = mod.relate;
    vectorSearch = mod.vectorSearch;
    bumpAccessCounts = mod.bumpAccessCounts;
    closeSurreal = mod.closeSurreal;

    await initSurreal(config.surreal);
    surrealAvailable = await isSurrealAvailable();
  } catch (e) {
    console.log("SurrealDB not available, skipping integration tests:", e instanceof Error ? e.message : e);
    surrealAvailable = false;
  }
});

afterAll(async () => {
  if (surrealAvailable && closeSurreal) {
    await closeSurreal();
  }
});

describe("isSurrealAvailable", () => {
  it("returns a boolean", async () => {
    if (!isSurrealAvailable) return;
    const result = await isSurrealAvailable();
    expect(typeof result).toBe("boolean");
  });
});

describe("session management", () => {
  it("createSession returns a session ID", async () => {
    if (skipIfNoDb()) return;
    const id = await createSession("test-agent");
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });
});

describe("turn management", () => {
  it("upsertTurn stores and retrieves turns", async () => {
    if (skipIfNoDb()) return;
    const sessionId = await createSession("test-agent");
    const fakeEmbedding = Array(1024).fill(0).map((_, i) => Math.sin(i) * 0.1);

    await upsertTurn({
      session_id: sessionId,
      role: "user",
      text: "test turn for surreal test",
      embedding: fakeEmbedding,
    });

    const turns = await getSessionTurns(sessionId);
    expect(turns.length).toBeGreaterThanOrEqual(1);
    expect(turns.some((t) => t.text.includes("test turn for surreal test"))).toBe(true);
  });
});

describe("agent management", () => {
  it("ensureAgent creates or finds an agent", async () => {
    if (skipIfNoDb()) return;
    const id = await ensureAgent("test-surreal-agent");
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);

    // Idempotent — second call returns same or valid ID
    const id2 = await ensureAgent("test-surreal-agent");
    expect(typeof id2).toBe("string");
  });
});

describe("vectorSearch", () => {
  it("returns array (possibly empty)", async () => {
    if (skipIfNoDb()) return;
    const fakeVec = Array(1024).fill(0).map((_, i) => Math.cos(i) * 0.1);
    const results = await vectorSearch(fakeVec, "nonexistent-session");
    expect(Array.isArray(results)).toBe(true);
  });
});

describe("bumpAccessCounts", () => {
  it("does not throw for empty array", async () => {
    if (skipIfNoDb()) return;
    await expect(bumpAccessCounts([])).resolves.not.toThrow();
  });
});
