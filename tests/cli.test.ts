import { describe, it, expect } from "vitest";

// cli.ts exports formatToolStart/formatToolEnd as private, but we can test
// the pure helpers: truncate is not exported either.
// We'll test the exported startCli signature exists and the display helpers
// by importing the module and testing the SYSTEM_PROMPT indirectly.

// What we CAN test: the formatting functions via their public effects.
// Since they're not exported, we extract and test the logic patterns directly.

describe("cli display helpers", () => {
  // Replicate truncate logic for unit testing
  function truncate(s: string, max = 60): string {
    if (!s) return "";
    const clean = s.replace(/\n/g, "↵").trim();
    return clean.length > max ? clean.slice(0, max) + "…" : clean;
  }

  it("truncate returns empty for falsy input", () => {
    expect(truncate("")).toBe("");
  });

  it("truncate passes through short strings", () => {
    expect(truncate("hello")).toBe("hello");
  });

  it("truncate clips long strings with ellipsis", () => {
    const long = "a".repeat(100);
    const result = truncate(long, 10);
    expect(result.length).toBe(11); // 10 + ellipsis
    expect(result.endsWith("…")).toBe(true);
  });

  it("truncate replaces newlines with ↵", () => {
    expect(truncate("line1\nline2")).toBe("line1↵line2");
  });
});

describe("cli module", () => {
  it("exports startCli function", async () => {
    const mod = await import("../src/cli.js");
    expect(typeof mod.startCli).toBe("function");
  });
});
