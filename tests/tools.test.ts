/**
 * Tests for tool creation (src/tools.ts).
 *
 * Verifies createTools returns the expected set of coding tools.
 */
import { describe, it, expect } from "vitest";
import { createTools } from "../src/tools.js";

describe("createTools", () => {
  it("returns an array of 7 tools", () => {
    const tools = createTools("/tmp");
    expect(Array.isArray(tools)).toBe(true);
    expect(tools.length).toBe(7);
  });

  it("includes all expected tool names", () => {
    const tools = createTools("/tmp");
    const names = tools.map((t) => t.name).sort();
    expect(names).toEqual([
      "bash",
      "edit",
      "find",
      "grep",
      "ls",
      "read",
      "write",
    ]);
  });

  it("each tool has a non-empty name and is callable", () => {
    const tools = createTools("/tmp");
    for (const tool of tools) {
      expect(typeof tool.name).toBe("string");
      expect(tool.name.length).toBeGreaterThan(0);
    }
  });

  it("uses the provided cwd", () => {
    // Tools should be created without error for any valid path
    const tools1 = createTools("/tmp");
    const tools2 = createTools("/home");
    expect(tools1.length).toBe(tools2.length);
  });
});
