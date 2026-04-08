/**
 * Preflight bootstrap — ensures all infrastructure is ready before the agent wakes up.
 *
 * 1. Config — load from ~/.kongclaw/config.json, or run first-time setup wizard
 * 2. API key — must be present and look valid
 * 3. Embedding model — must exist on disk, or auto-download via node-llama-cpp
 * 4. SurrealDB — must be reachable at the configured URL; if not, try to start a managed instance
 *
 * Env vars always override persisted config (power users).
 */

import { existsSync } from "node:fs";
import { execSync, spawn, type ChildProcess } from "node:child_process";
import { join } from "node:path";
import type { Config, SurrealConfig } from "./config.js";
import { loadConfig } from "./config.js";
import { loadPersistedConfig, runSetup, type PersistedConfig } from "./setup.js";

/** Child process we spawned — null if SurrealDB was already running externally. */
let managedSurreal: ChildProcess | null = null;

// ── Config resolution ──────────────────────────────────────────────────────

/**
 * Resolve config: env vars > persisted config > OpenClaw profiles > defaults.
 * Only runs the setup wizard if no API key can be found anywhere.
 */
export async function resolveConfig(): Promise<PersistedConfig> {
  // Try loading persisted config first
  const persisted = loadPersistedConfig();
  if (persisted) return persisted;

  // No persisted config — check if loadConfig can find an API key
  // from env vars, .env, ~/.surreal_env, or OpenClaw auth profiles
  const probed = loadConfig();
  if (probed.anthropicApiKey) {
    // Key found via existing sources — build a PersistedConfig from them
    // but don't persist it (user didn't go through wizard, may be transient env)
    return {
      anthropicApiKey: probed.anthropicApiKey,
      surrealUrl: probed.surreal.url,
      surrealUser: probed.surreal.user,
      surrealPass: probed.surreal.pass,
      surrealNs: probed.surreal.ns,
      surrealDb: probed.surreal.db,
      model: probed.model,
    };
  }

  // No config anywhere — run the wizard
  return runSetup();
}

// ── API key ────────────────────────────────────────────────────────────────

function checkApiKey(key: string): void {
  if (!key) {
    console.error(
      "\n  ✖ No Anthropic API key configured.\n" +
      "    Run: rm ~/.kongclaw/config.json && npm start   (to re-run setup)\n" +
      "    Or:  export ANTHROPIC_API_KEY=sk-ant-...\n",
    );
    process.exit(1);
  }
  if (!key.startsWith("sk-ant-")) {
    console.warn("  ⚠ API key doesn't start with sk-ant- — may not be a valid Anthropic key\n");
  }
}

// ── Embedding model ────────────────────────────────────────────────────────

const BGE_M3_URL = "https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-q4_k_m.gguf";

async function ensureEmbeddingModel(modelPath: string): Promise<void> {
  if (existsSync(modelPath)) {
    console.log("  ✔ Embedding model found");
    return;
  }

  console.log(`\n  Embedding model not found at ${modelPath}`);
  console.log("  Downloading BGE-M3 Q4_K_M (~420 MB)...\n");

  try {
    execSync(
      `npx node-llama-cpp pull --url "${BGE_M3_URL}"`,
      { stdio: "inherit", timeout: 600_000 },
    );
  } catch {
    console.error(
      "\n  ✖ Failed to download embedding model.\n" +
      "    Download it manually:\n" +
      `      npx node-llama-cpp pull --url "${BGE_M3_URL}"\n` +
      "    Or set EMBED_MODEL_PATH to an existing BGE-M3 GGUF file.\n",
    );
    process.exit(1);
  }

  if (!existsSync(modelPath)) {
    console.error(
      `\n  ✖ Download succeeded but model not found at expected path: ${modelPath}\n` +
      "    Set EMBED_MODEL_PATH to the downloaded file location.\n",
    );
    process.exit(1);
  }

  console.log("  ✔ Embedding model downloaded");
}

// ── SurrealDB ──────────────────────────────────────────────────────────────

/** Parse host and port from a SurrealDB WebSocket URL. */
function parseUrl(url: string): { host: string; port: number } {
  const stripped = url.replace(/^wss?:\/\//, "").replace(/\/.*$/, "");
  const [host, portStr] = stripped.split(":");
  return { host: host || "127.0.0.1", port: parseInt(portStr, 10) || 8000 };
}

/** Try a TCP connect to see if something is listening. */
async function isPortReachable(host: string, port: number, timeoutMs = 2000): Promise<boolean> {
  const net = await import("node:net");
  return new Promise((resolve) => {
    const sock = net.createConnection({ host, port });
    const timer = setTimeout(() => { sock.destroy(); resolve(false); }, timeoutMs);
    sock.on("connect", () => { clearTimeout(timer); sock.destroy(); resolve(true); });
    sock.on("error", () => { clearTimeout(timer); resolve(false); });
  });
}

/** Find the `surreal` binary on PATH, or return null. */
function findSurrealBinary(): string | null {
  try {
    return execSync("which surreal", { encoding: "utf-8" }).trim() || null;
  } catch {
    return null;
  }
}

/** Spawn a managed SurrealDB process and wait for it to accept connections. */
async function spawnSurreal(binary: string, config: SurrealConfig): Promise<void> {
  const { host, port } = parseUrl(config.url);
  const dataDir = join(process.cwd(), ".kongclaw-db");

  console.log(`  Starting SurrealDB on ${host}:${port} (data: ${dataDir})...`);

  managedSurreal = spawn(binary, [
    "start",
    "--user", config.user,
    "--pass", config.pass,
    "--bind", `${host}:${port}`,
    `file:${dataDir}`,
  ], {
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });

  managedSurreal.stderr?.on("data", (chunk: Buffer) => {
    const line = chunk.toString().trim();
    if (line && process.env.ZERACLAW_DEBUG === "1") {
      console.debug(`  [surreal] ${line}`);
    }
  });

  managedSurreal.on("error", (err) => {
    console.error(`  [surreal] Process error: ${err.message}`);
  });

  managedSurreal.on("exit", (code) => {
    if (code !== null && code !== 0) {
      console.error(`  [surreal] Exited with code ${code}`);
    }
    managedSurreal = null;
  });

  // Wait for it to start accepting connections
  const maxWaitMs = 15_000;
  const pollMs = 250;
  const deadline = Date.now() + maxWaitMs;

  while (Date.now() < deadline) {
    if (await isPortReachable(host, port, 500)) {
      return;
    }
    if (managedSurreal === null || managedSurreal.exitCode !== null) {
      throw new Error("SurrealDB process exited before becoming ready");
    }
    await new Promise((r) => setTimeout(r, pollMs));
  }

  throw new Error(`SurrealDB did not become ready within ${maxWaitMs / 1000}s`);
}

/** Fetch SurrealDB version from the HTTP /version endpoint and verify it's 3.x. */
async function checkSurrealVersion(config: SurrealConfig): Promise<void> {
  const httpUrl = config.url.replace("ws://", "http://").replace("wss://", "https://").replace("/rpc", "");
  try {
    const res = await fetch(`${httpUrl}/version`, { signal: AbortSignal.timeout(3000) });
    const text = (await res.text()).trim(); // e.g. "surrealdb-3.0.1"
    const match = text.match(/surrealdb-(\d+)\./);
    if (match && parseInt(match[1], 10) < 3) {
      console.error(
        `\n  ✖ SurrealDB ${text.replace("surrealdb-", "v")} detected — KongClaw requires v3.0+\n` +
        "    Upgrade: curl -sSf https://install.surrealdb.com | sh\n",
      );
      process.exit(1);
    }
    if (match) {
      console.log(`  ✔ SurrealDB ${text.replace("surrealdb-", "v")}`);
    }
  } catch {
    // Version endpoint unavailable — don't block startup, schema errors will surface later
  }
}

async function ensureSurreal(config: SurrealConfig): Promise<void> {
  const { host, port } = parseUrl(config.url);

  // Already listening? Great — don't care how it got there.
  if (await isPortReachable(host, port)) {
    console.log(`  ✔ SurrealDB reachable at ${host}:${port}`);
    await checkSurrealVersion(config);
    return;
  }

  // Not reachable — only try to auto-start if it's localhost
  const isLocal = host === "127.0.0.1" || host === "localhost" || host === "0.0.0.0";
  if (!isLocal) {
    console.error(
      `\n  ✖ SurrealDB not reachable at ${host}:${port}\n` +
      "    Check that the server is running and the URL is correct.\n" +
      "    To reconfigure: rm ~/.kongclaw/config.json && npm start\n",
    );
    process.exit(1);
  }

  // Local but not running — try to start it
  const binary = findSurrealBinary();
  if (!binary) {
    console.error(
      `\n  ✖ SurrealDB not running on ${host}:${port} and \`surreal\` binary not found.\n\n` +
      "    Install SurrealDB:\n" +
      "      curl -sSf https://install.surrealdb.com | sh\n\n" +
      "    Then restart KongClaw. Or start SurrealDB manually and set SURREAL_URL.\n",
    );
    process.exit(1);
  }

  console.log(`  Found surreal binary at ${binary}`);

  try {
    await spawnSurreal(binary, config);
    console.log(`  ✔ SurrealDB started on ${host}:${port}`);
    await checkSurrealVersion(config);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(
      `\n  ✖ Failed to start SurrealDB: ${msg}\n` +
      "    Start it manually:\n" +
      `      surreal start --user ${config.user} --pass ${config.pass} --bind ${host}:${port} file:.kongclaw-db\n`,
    );
    process.exit(1);
  }
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Run all preflight checks. Returns resolved config ready for use.
 * Exits the process with clear error messages if anything is wrong.
 */
export async function bootstrap(): Promise<PersistedConfig> {
  // Step 1: Resolve config (wizard if needed)
  const resolved = await resolveConfig();

  console.log("\n  Preflight...");

  // Step 2: Validate API key
  checkApiKey(resolved.anthropicApiKey);
  console.log("  ✔ API key configured");

  // Step 3: Check embedding model + SurrealDB (parallel — independent of each other)
  const embeddingModelPath = process.env.EMBED_MODEL_PATH
    ?? join((await import("node:os")).homedir(), ".node-llama-cpp", "models", "bge-m3-q4_k_m.gguf");

  const surrealConfig: SurrealConfig = {
    url: process.env.SURREAL_URL ?? resolved.surrealUrl,
    get httpUrl() {
      return this.url.replace("ws://", "http://").replace("wss://", "https://").replace("/rpc", "/sql");
    },
    user: process.env.SURREAL_USER ?? resolved.surrealUser,
    pass: process.env.SURREAL_PASS ?? resolved.surrealPass,
    ns: process.env.SURREAL_NS ?? resolved.surrealNs,
    db: process.env.SURREAL_DB ?? resolved.surrealDb,
  };

  await Promise.all([
    ensureEmbeddingModel(embeddingModelPath),
    ensureSurreal(surrealConfig),
  ]);

  console.log("  Preflight complete\n");

  return resolved;
}

/**
 * Kill the managed SurrealDB child process if we spawned one.
 * Call this during shutdown.
 */
export function killManagedSurreal(): void {
  if (!managedSurreal) return;
  try {
    managedSurreal.kill("SIGTERM");
    setTimeout(() => {
      if (managedSurreal && managedSurreal.exitCode === null) {
        managedSurreal.kill("SIGKILL");
      }
    }, 3000);
  } catch {
    // Already dead
  }
}

/** Returns true if we spawned the SurrealDB process ourselves. */
export function isManagedSurreal(): boolean {
  return managedSurreal !== null;
}
