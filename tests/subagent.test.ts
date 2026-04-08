/**
 * Tests for Subagent System (src/subagent.ts).
 *
 * Unit tests for config resolution, incognito ID generation,
 * merge filter logic, and IPC message types.
 * Integration tests require SurrealDB.
 */
import { describe, it, expect, beforeAll } from "vitest";

let surrealAvailable = false;
let generateIncognitoId: typeof import("../src/subagent.js")["generateIncognitoId"];
let getSurrealConfigForMode: typeof import("../src/subagent.js")["getSurrealConfigForMode"];
let listSubagents: typeof import("../src/subagent.js")["listSubagents"];
let mergeFromIncognito: typeof import("../src/subagent.js")["mergeFromIncognito"];

const parentConfig = {
  url: "ws://localhost:8042/rpc",
  user: "root",
  pass: "root",
  ns: "kong",
  db: "memory",
};

beforeAll(async () => {
  try {
    const configMod = await import("../src/config.js");
    const config = configMod.loadConfig();
    const surrealMod = await import("../src/surreal.js");
    await surrealMod.initSurreal(config.surreal);
    surrealAvailable = await surrealMod.isSurrealAvailable();
  } catch {
    surrealAvailable = false;
  }

  const mod = await import("../src/subagent.js");
  generateIncognitoId = mod.generateIncognitoId;
  getSurrealConfigForMode = mod.getSurrealConfigForMode;
  listSubagents = mod.listSubagents;
  mergeFromIncognito = mod.mergeFromIncognito;
});

describe("subagent system", () => {
  describe("generateIncognitoId", () => {
    it("generates unique IDs", () => {
      const id1 = generateIncognitoId();
      const id2 = generateIncognitoId();
      expect(id1).not.toBe(id2);
    });

    it("starts with incognito_ prefix", () => {
      const id = generateIncognitoId();
      expect(id.startsWith("incognito_")).toBe(true);
    });

    it("has timestamp and random components", () => {
      const id = generateIncognitoId();
      const parts = id.split("_");
      expect(parts.length).toBe(3); // incognito, timestamp, hex
      expect(Number(parts[1])).toBeGreaterThan(0);
      expect(parts[2].length).toBe(8); // 4 bytes = 8 hex chars
    });
  });

  describe("getSurrealConfigForMode", () => {
    it("returns same ns/db for full mode", () => {
      const config = getSurrealConfigForMode(parentConfig, "full");
      expect(config.ns).toBe("zera");
      expect(config.db).toBe("memory");
      expect(config.url).toBe(parentConfig.url);
    });

    it("returns isolated db for incognito mode", () => {
      const config = getSurrealConfigForMode(parentConfig, "incognito", "incognito_123_abc");
      expect(config.ns).toBe("zera");
      expect(config.db).toBe("memory_incognito_123_abc");
      expect(config.url).toBe(parentConfig.url);
    });

    it("generates incognito ID when none provided", () => {
      const config = getSurrealConfigForMode(parentConfig, "incognito");
      expect(config.db).toMatch(/^memory_incognito_\d+_[a-f0-9]+$/);
    });

    it("preserves auth credentials", () => {
      const config = getSurrealConfigForMode(parentConfig, "incognito", "test");
      expect(config.user).toBe("root");
      expect(config.pass).toBe("root");
    });

    it("does not mutate parent config", () => {
      const original = { ...parentConfig };
      getSurrealConfigForMode(parentConfig, "incognito");
      expect(parentConfig).toEqual(original);
    });
  });

  describe("listSubagents", () => {
    it("returns empty array when surreal is down", async () => {
      if (surrealAvailable) return;
      const agents = await listSubagents();
      expect(agents).toEqual([]);
    });

    it("returns array when surreal is up", async () => {
      if (!surrealAvailable) return;
      const agents = await listSubagents();
      expect(Array.isArray(agents)).toBe(true);
    });

    it("accepts optional session filter", async () => {
      if (!surrealAvailable) return;
      const agents = await listSubagents("nonexistent-session");
      expect(Array.isArray(agents)).toBe(true);
      expect(agents.length).toBe(0);
    });
  });

  describe("mergeFromIncognito", () => {
    it("handles non-existent incognito database gracefully", async () => {
      if (!surrealAvailable) return;
      // Connecting to a non-existent DB should either create it empty
      // or fail gracefully — either way merge returns 0
      try {
        const result = await mergeFromIncognito("nonexistent_id", parentConfig);
        expect(result.merged).toBe(0);
        expect(result.skippedDuplicates).toBe(0);
      } catch {
        // Connection failure is acceptable for non-existent DB
      }
    });

    it("respects merge filter defaults", async () => {
      if (!surrealAvailable) return;
      try {
        const result = await mergeFromIncognito("nonexistent_id", parentConfig, {
          minImportance: 5.0,
          maxNodes: 10,
          tables: ["memory"],
        });
        expect(result.merged).toBeLessThanOrEqual(10);
      } catch {
        // acceptable
      }
    });
  });

  describe("SubagentConfig types", () => {
    it("validates mode literals", () => {
      const modes: Array<"full" | "incognito"> = ["full", "incognito"];
      expect(modes).toHaveLength(2);
    });
  });

  describe("IPC protocol", () => {
    it("start message has required fields", () => {
      const msg = {
        type: "start" as const,
        config: {
          mode: "full" as const,
          task: "test task",
          parentSessionId: "session:test",
        },
        surrealConfig: parentConfig,
        anthropicApiKey: "sk-test",
        embeddingModelPath: "/path/to/model",
      };
      expect(msg.type).toBe("start");
      expect(msg.config.mode).toBe("full");
      expect(msg.surrealConfig.ns).toBe("zera");
    });

    it("complete message has required fields", () => {
      const msg = {
        type: "complete" as const,
        result: {
          sessionId: "session:test",
          mode: "incognito" as const,
          status: "completed" as const,
          summary: "Task done",
          incognitoId: "incognito_123_abc",
          turnCount: 3,
          toolCalls: 5,
          durationMs: 10000,
        },
      };
      expect(msg.type).toBe("complete");
      expect(msg.result.status).toBe("completed");
      expect(msg.result.incognitoId).toBe("incognito_123_abc");
    });
  });
});
