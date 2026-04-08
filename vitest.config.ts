import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    testTimeout: 30000,
    hookTimeout: 60000,
    // Integration tests (skills, surreal, subagent, etc.) skip gracefully
    // when SurrealDB/embeddings aren't available (e.g., CI environment)
  },
});
