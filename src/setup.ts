/**
 * First-run setup wizard — creates ~/.kongclaw/config.json interactively.
 * Runs in the terminal before TUI or CLI boots, so uses raw readline.
 */

import { createInterface } from "node:readline";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const ZERACLAW_HOME = join(homedir(), ".kongclaw");
export const CONFIG_PATH = join(ZERACLAW_HOME, "config.json");

export interface PersistedConfig {
  anthropicApiKey: string;
  surrealUrl: string;
  surrealUser: string;
  surrealPass: string;
  surrealNs: string;
  surrealDb: string;
  model: string;
}

/** Load persisted config from disk, or return null if it doesn't exist. */
export function loadPersistedConfig(): PersistedConfig | null {
  try {
    if (!existsSync(CONFIG_PATH)) return null;
    const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
    // Minimal validation — must have an API key
    if (!raw.anthropicApiKey) return null;
    return raw as PersistedConfig;
  } catch {
    return null;
  }
}

/** Save config to disk. */
export function savePersistedConfig(config: PersistedConfig): void {
  mkdirSync(ZERACLAW_HOME, { recursive: true });
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n", { encoding: "utf-8", mode: 0o600 });
}

/** Prompt the user for a value with an optional default. */
function ask(
  rl: ReturnType<typeof createInterface>,
  question: string,
  defaultVal?: string,
  mask = false,
): Promise<string> {
  const suffix = defaultVal ? ` [${mask ? "****" : defaultVal}]` : "";
  return new Promise((resolve) => {
    rl.question(`  ${question}${suffix}: `, (answer) => {
      resolve(answer.trim() || defaultVal || "");
    });
  });
}

/** Run the interactive setup wizard. Returns the new config. */
export async function runSetup(): Promise<PersistedConfig> {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("");
  console.log("  ┌─────────────────────────────────────┐");
  console.log("  │     KongClaw — First-Time Setup      │");
  console.log("  └─────────────────────────────────────┘");
  console.log("");

  // API key — required
  let apiKey = "";
  while (!apiKey) {
    apiKey = await ask(rl, "Anthropic API key");
    if (!apiKey) {
      console.log("  API key is required. Get one at https://console.anthropic.com/\n");
    }
  }

  console.log("");
  console.log("  SurrealDB configuration");
  console.log("  (press Enter to accept defaults)\n");

  const surrealUrl = await ask(rl, "SurrealDB URL", "ws://localhost:8042/rpc");
  const surrealUser = await ask(rl, "SurrealDB user", "root");
  const surrealPass = await ask(rl, "SurrealDB password", "root");
  const surrealNs = await ask(rl, "SurrealDB namespace", "zera");
  const surrealDb = await ask(rl, "SurrealDB database", "memory");

  console.log("");
  const model = await ask(rl, "Default model", "claude-opus-4-6");

  rl.close();

  const config: PersistedConfig = {
    anthropicApiKey: apiKey,
    surrealUrl,
    surrealUser,
    surrealPass,
    surrealNs,
    surrealDb,
    model,
  };

  savePersistedConfig(config);
  console.log(`\n  Config saved to ${CONFIG_PATH}\n`);

  return config;
}
