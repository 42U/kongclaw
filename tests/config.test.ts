/**
 * Tests for configuration loading (src/config.ts).
 *
 * Tests env var resolution, defaults, and config structure.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";

describe("loadConfig", () => {
  const origEnv = { ...process.env };

  beforeEach(() => {
    // Clear relevant env vars so defaults apply
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_OAUTH_TOKEN;
    delete process.env.SURREAL_URL;
    delete process.env.SURREAL_USER;
    delete process.env.SURREAL_PASS;
    delete process.env.SURREAL_NS;
    delete process.env.SURREAL_DB;
    delete process.env.EMBED_MODEL_PATH;
    delete process.env.ZERACLAW_MODEL;
  });

  afterEach(() => {
    // Restore original env
    for (const key of Object.keys(process.env)) {
      if (!(key in origEnv)) delete process.env[key];
    }
    Object.assign(process.env, origEnv);
  });

  it("returns a valid Config object with all required fields", async () => {
    const { loadConfig } = await import("../src/config.js");
    const config = loadConfig();

    expect(config).toHaveProperty("surreal");
    expect(config).toHaveProperty("embedding");
    expect(config).toHaveProperty("anthropicApiKey");
    expect(config).toHaveProperty("model");

    expect(config.surreal).toHaveProperty("url");
    expect(config.surreal).toHaveProperty("user");
    expect(config.surreal).toHaveProperty("pass");
    expect(config.surreal).toHaveProperty("ns");
    expect(config.surreal).toHaveProperty("db");

    expect(config.embedding).toHaveProperty("modelPath");
    expect(config.embedding).toHaveProperty("dimensions");
    expect(config.embedding.dimensions).toBe(1024);
  });

  it("uses surreal defaults when no env vars set", async () => {
    const { loadConfig } = await import("../src/config.js");
    const config = loadConfig();

    // Defaults should be present (either from env file or hardcoded)
    expect(typeof config.surreal.url).toBe("string");
    expect(typeof config.surreal.user).toBe("string");
    expect(typeof config.surreal.pass).toBe("string");
    expect(config.surreal.url).toMatch(/localhost|127\.0\.0\.1/);
  });

  it("respects SURREAL_URL env var", async () => {
    process.env.SURREAL_URL = "ws://custom:9999/rpc";
    // Re-import to pick up env change
    const mod = await import("../src/config.js");
    const config = mod.loadConfig();
    expect(config.surreal.url).toBe("ws://custom:9999/rpc");
  });

  it("respects ANTHROPIC_API_KEY env var", async () => {
    process.env.ANTHROPIC_API_KEY = "sk-test-key-12345";
    const { loadConfig } = await import("../src/config.js");
    const config = loadConfig();
    expect(config.anthropicApiKey).toBe("sk-test-key-12345");
  });

  it("respects ZERACLAW_MODEL env var", async () => {
    process.env.ZERACLAW_MODEL = "claude-sonnet-4-20250514";
    const { loadConfig } = await import("../src/config.js");
    const config = loadConfig();
    expect(config.model).toBe("claude-sonnet-4-20250514");
  });

  it("defaults model to claude-opus-4-6", async () => {
    const { loadConfig } = await import("../src/config.js");
    const config = loadConfig();
    expect(config.model).toBe("claude-opus-4-6");
  });

  it("embedding dimensions is always 1024", async () => {
    const { loadConfig } = await import("../src/config.js");
    const config = loadConfig();
    expect(config.embedding.dimensions).toBe(1024);
  });
});
