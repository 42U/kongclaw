/**
 * Tests for identity system (src/identity.ts).
 *
 * Unit tests for pure functions (findWakeupFile, readWakeupFile, buildWakeupPrompt).
 * Integration tests for seedIdentity/saveUserIdentity require DB + embeddings.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { writeFileSync, unlinkSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

let findWakeupFile: typeof import("../src/identity.js")["findWakeupFile"];
let readWakeupFile: typeof import("../src/identity.js")["readWakeupFile"];
let deleteWakeupFile: typeof import("../src/identity.js")["deleteWakeupFile"];
let buildWakeupPrompt: typeof import("../src/identity.js")["buildWakeupPrompt"];
let hasUserIdentity: typeof import("../src/identity.js")["hasUserIdentity"];

beforeAll(async () => {
  const mod = await import("../src/identity.js");
  findWakeupFile = mod.findWakeupFile;
  readWakeupFile = mod.readWakeupFile;
  deleteWakeupFile = mod.deleteWakeupFile;
  buildWakeupPrompt = mod.buildWakeupPrompt;
  hasUserIdentity = mod.hasUserIdentity;
});

describe("findWakeupFile", () => {
  const testDir = join(tmpdir(), "kongclaw-test-identity-" + Date.now());

  beforeAll(() => {
    mkdirSync(testDir, { recursive: true });
  });

  afterAll(() => {
    try {
      unlinkSync(join(testDir, "WAKEUP.md"));
    } catch { /* ok */ }
  });

  it("returns null when WAKEUP.md does not exist", () => {
    expect(findWakeupFile(testDir)).toBeNull();
  });

  it("returns the path when WAKEUP.md exists", () => {
    const wakeupPath = join(testDir, "WAKEUP.md");
    writeFileSync(wakeupPath, "# Test Identity\nI am a test agent.");
    expect(findWakeupFile(testDir)).toBe(wakeupPath);
  });
});

describe("readWakeupFile", () => {
  const testDir = join(tmpdir(), "kongclaw-test-read-" + Date.now());
  const wakeupPath = join(testDir, "WAKEUP.md");

  beforeAll(() => {
    mkdirSync(testDir, { recursive: true });
    writeFileSync(wakeupPath, "  # My Identity\nI am helpful.  \n");
  });

  afterAll(() => {
    try { unlinkSync(wakeupPath); } catch { /* ok */ }
  });

  it("reads and trims file contents", () => {
    const content = readWakeupFile(wakeupPath);
    expect(content).toBe("# My Identity\nI am helpful.");
  });
});

describe("deleteWakeupFile", () => {
  it("deletes the file", () => {
    const testDir = join(tmpdir(), "kongclaw-test-del-" + Date.now());
    mkdirSync(testDir, { recursive: true });
    const path = join(testDir, "WAKEUP.md");
    writeFileSync(path, "temp");
    expect(existsSync(path)).toBe(true);
    deleteWakeupFile(path);
    expect(existsSync(path)).toBe(false);
  });

  it("does not throw for non-existent file", () => {
    expect(() => deleteWakeupFile("/tmp/nonexistent-kongclaw-wakeup-" + Date.now())).not.toThrow();
  });
});

describe("buildWakeupPrompt", () => {
  it("returns systemAddition and firstMessage", () => {
    const result = buildWakeupPrompt("I am Zera. I speak concisely.");
    expect(result).toHaveProperty("systemAddition");
    expect(result).toHaveProperty("firstMessage");
    expect(typeof result.systemAddition).toBe("string");
    expect(typeof result.firstMessage).toBe("string");
  });

  it("includes wakeup content in firstMessage", () => {
    const content = "I am a graph-backed agent named Zeraclaw.";
    const result = buildWakeupPrompt(content);
    expect(result.firstMessage).toContain(content);
  });

  it("includes IDENTITY ESTABLISHMENT in systemAddition", () => {
    const result = buildWakeupPrompt("test");
    expect(result.systemAddition).toContain("IDENTITY ESTABLISHMENT");
  });

  it("mentions WAKEUP.md in systemAddition", () => {
    const result = buildWakeupPrompt("test");
    expect(result.systemAddition).toContain("WAKEUP.md");
  });
});

describe("hasUserIdentity", () => {
  it("returns a boolean", async () => {
    const result = await hasUserIdentity();
    expect(typeof result).toBe("boolean");
  });
});
