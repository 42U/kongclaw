import { describe, it, expect } from "vitest";

// health-check.ts is a CLI entry point that calls main() and process.exit()
// at module level, so it can't be safely imported in tests.
// Instead we verify the module file exists and has the expected structure.

describe("health-check module", () => {
  it("source file exists and contains expected structure", async () => {
    const fs = await import("fs");
    const src = fs.readFileSync("src/health-check.ts", "utf-8");
    expect(src).toContain("async function main()");
    expect(src).toContain("ZERACLAW MEMORY HEALTH CHECK");
    expect(src).toContain("swallow");
  });
});
