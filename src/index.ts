#!/usr/bin/env node

import { loadConfig } from "./config.js";
import { initEmbeddings, disposeEmbeddings } from "./embeddings.js";
import { initSurreal, closeSurreal } from "./surreal.js";
import { initReranker, disposeReranker } from "./graph-context.js";
import { logCrash, logError, getLogPath } from "./logger.js";
import { seedIdentity } from "./identity.js";
import { seedCognitiveBootstrap } from "./cognitive-bootstrap.js";
import { startCli, shuttingDown as cliShuttingDown } from "./cli.js";
import { startTui, shuttingDown as tuiShuttingDown } from "./tui.js";
import { bootstrap, killManagedSurreal } from "./bootstrap.js";
import { writeHandoffFileSync, type HandoffFileData } from "./handoff-file.js";

// Process-global: the active session state for sync exit handler
let activeSessionState: { sessionId: string; lastUserText: string; lastAssistantText: string; unextractedTokens: number; cwd: string } | null = null;

/** Called by TUI/CLI to register session state for last-resort handoff. */
export function setActiveSessionState(state: typeof activeSessionState): void {
  activeSessionState = state;
}

/** Called by TUI/CLI when session ends gracefully. */
export function clearActiveSessionState(): void {
  activeSessionState = null;
}

// Catch unhandled rejections/exceptions — log and exit cleanly instead of silent crash
process.on("unhandledRejection", (reason) => {
  console.error(`[FATAL] Unhandled promise rejection:`, reason);
  logError("Unhandled promise rejection", reason);
  process.exit(1);
});
process.on("uncaughtException", (err) => {
  console.error(`[FATAL] Uncaught exception:`, err);
  logError("Uncaught exception", err);
  process.exit(1);
});

async function main(): Promise<void> {
  // Preflight: resolve config (wizard on first run), verify API key,
  // ensure embedding model exists, ensure SurrealDB is reachable.
  // Exits with clear errors if anything is missing.
  const resolved = await bootstrap();

  // Load full config — env vars override persisted values
  const config = loadConfig(resolved);

  // Inject API key into env so pi-ai's streamSimple picks it up
  if (config.anthropicApiKey) {
    if (config.anthropicApiKey.startsWith("sk-ant-oat")) {
      process.env.ANTHROPIC_OAUTH_TOKEN ??= config.anthropicApiKey;
    } else {
      process.env.ANTHROPIC_API_KEY ??= config.anthropicApiKey;
    }
  }

  // Init infrastructure — bootstrap already verified these are reachable,
  // so failures here are genuine errors, not missing dependencies.
  await Promise.all([
    initEmbeddings(config.embedding),
    initSurreal(config.surreal),
    config.reranker.enabled
      ? initReranker(config.reranker.modelPath)
      : Promise.resolve(),
  ]);

  // TUI is the default experience; --cli falls back to readline REPL
  const useCli = process.argv.includes("--cli") || process.env.ZERACLAW_TUI === "0";

  // Sync exit handler: write handoff file as last resort on abrupt exit
  process.on("exit", () => {
    if (!activeSessionState) return;
    writeHandoffFileSync({
      sessionId: activeSessionState.sessionId,
      timestamp: new Date().toISOString(),
      lastUserText: activeSessionState.lastUserText.slice(0, 500),
      lastAssistantText: activeSessionState.lastAssistantText.slice(0, 500),
      unextractedTokens: activeSessionState.unextractedTokens,
    }, activeSessionState.cwd);
  });

  // Cleanup on exit — defer to CLI/TUI's graceful shutdown if active
  process.on("SIGINT", async () => {
    if (cliShuttingDown || tuiShuttingDown) return;
    await Promise.all([closeSurreal(), disposeEmbeddings(), disposeReranker()]);
    killManagedSurreal();
    process.exit(0);
  });

  // Seed core identity chunks (idempotent — skips if already present)
  const seeded = await seedIdentity().catch(() => 0);
  if (seeded > 0) console.log(`   Seeded ${seeded} core identity chunks`);

  // Seed cognitive bootstrap — teaches the agent HOW to use its memory system
  const boot = await seedCognitiveBootstrap().catch(() => ({ identitySeeded: 0, coreSeeded: 0 }));
  if (boot.identitySeeded > 0 || boot.coreSeeded > 0)
    console.log(`   Cognitive bootstrap: ${boot.identitySeeded} identity + ${boot.coreSeeded} core`);

  console.log(`   Model: ${config.model}`);
  const startFn = useCli ? startCli : startTui;
  await startFn(process.cwd(), config.model, {
    surrealConfig: config.surreal,
    anthropicApiKey: config.anthropicApiKey,
    embeddingModelPath: config.embedding.modelPath,
  });
}

main().catch((err) => {
  console.error("Fatal:", err);
  killManagedSurreal();
  process.exit(1);
});
