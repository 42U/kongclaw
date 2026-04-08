import { readFileSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { swallow } from "./errors.js";
import type { PersistedConfig } from "./setup.js";

export interface SurrealConfig {
  url: string;
  httpUrl: string;
  user: string;
  pass: string;
  ns: string;
  db: string;
}

export interface EmbeddingConfig {
  modelPath: string;
  dimensions: number;
}

export interface RerankerConfig {
  modelPath: string;
  enabled: boolean;
}

export interface Config {
  surreal: SurrealConfig;
  embedding: EmbeddingConfig;
  reranker: RerankerConfig;
  anthropicApiKey: string;
  model: string;
}

function loadLegacyKey(): string | undefined {
  // Read Anthropic API key from legacy auth profiles
  const paths = [
    join(homedir(), ".openclaw", "agents", "main", "agent", "auth-profiles.json"),
  ];
  for (const p of paths) {
    try {
      const data = JSON.parse(readFileSync(p, "utf-8"));
      const profiles = data.profiles ?? data;
      for (const [name, profile] of Object.entries(profiles)) {
        const prof = profile as Record<string, unknown>;
        if (name.startsWith("anthropic") && typeof prof.key === "string") {
          return prof.key;
        }
      }
    } catch (e) {
      swallow.error("config:silent", e);
      // not found, try next
    }
  }
  return undefined;
}

function loadSurrealEnv(): Record<string, string> {
  const envPath = join(homedir(), ".surreal_env");
  const vars: Record<string, string> = {};
  try {
    const content = readFileSync(envPath, "utf-8");
    for (const line of content.split("\n")) {
      const match = line.match(/^export\s+(\w+)="(.*)"/);
      if (match) vars[match[1]] = match[2];
    }
  } catch (e) {
    swallow.error("config:silent", e);
    // File not found — fall through to env vars
  }
  return vars;
}

/**
 * Load config with priority: env vars > ~/.surreal_env > persisted config > defaults.
 * The optional `persisted` param comes from bootstrap (first-run wizard or saved config).
 */
export function loadConfig(persisted?: PersistedConfig): Config {
  const surrealEnv = loadSurrealEnv();

  const anthropicApiKey =
    process.env.ANTHROPIC_API_KEY
    ?? process.env.ANTHROPIC_OAUTH_TOKEN
    ?? persisted?.anthropicApiKey
    ?? loadLegacyKey()
    ?? "";
  if (!anthropicApiKey) {
    console.warn("Warning: No Anthropic API key found (checked env, persisted config, legacy profiles)");
  }

  return {
    surreal: {
      url: process.env.SURREAL_URL ?? surrealEnv.SURREAL_URL ?? persisted?.surrealUrl ?? "ws://localhost:8042/rpc",
      get httpUrl() {
        const base = process.env.SURREAL_HTTP_URL ?? surrealEnv.SURREAL_HTTP_URL;
        if (base) return base;
        return this.url.replace("ws://", "http://").replace("wss://", "https://").replace("/rpc", "/sql");
      },
      user: process.env.SURREAL_USER ?? surrealEnv.SURREAL_USER ?? persisted?.surrealUser ?? "root",
      pass: process.env.SURREAL_PASS ?? surrealEnv.SURREAL_PASS ?? persisted?.surrealPass ?? "root",
      ns: process.env.SURREAL_NS ?? surrealEnv.SURREAL_NS ?? persisted?.surrealNs ?? "kong",
      db: process.env.SURREAL_DB ?? surrealEnv.SURREAL_DB ?? persisted?.surrealDb ?? "memory",
    },
    embedding: {
      modelPath: process.env.EMBED_MODEL_PATH
        ?? join(homedir(), ".node-llama-cpp", "models", "bge-m3-q4_k_m.gguf"),
      dimensions: 1024,
    },
    reranker: (() => {
      const modelPath = process.env.RERANKER_MODEL_PATH
        ?? join(homedir(), ".node-llama-cpp", "models", "bge-reranker-v2-m3-Q8_0.gguf");
      return { modelPath, enabled: existsSync(modelPath) };
    })(),
    anthropicApiKey,
    model: process.env.ZERACLAW_MODEL ?? persisted?.model ?? "claude-opus-4-6",
  };
}
