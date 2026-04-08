/**
 * Lightweight error swallowing with severity levels.
 *
 * - swallow(ctx, e)       — SILENT: expected degradation (embeddings offline, non-critical telemetry).
 *                           Only visible with ZERACLAW_DEBUG=1. Logged to file when DEBUG.
 * - swallow.warn(ctx, e)  — WARN: unexpected but recoverable (DB query failure, compaction failure).
 *                           Always logged to stderr AND written to ~/.kongclaw/warnings.log.
 * - swallow.error(ctx, e) — ERROR: something is genuinely broken (cleanup failure, schema failure).
 *                           Always logged to stderr with stack trace AND written to ~/.kongclaw/warnings.log.
 *
 * Log file: ~/.kongclaw/warnings.log — append-only, timestamped, survives session restarts.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const DEBUG = process.env.ZERACLAW_DEBUG === "1";

// Ensure log directory exists once at module load
const LOG_DIR = join(homedir(), ".kongclaw");
try { mkdirSync(LOG_DIR, { recursive: true }); } catch { /* already exists */ }
const LOG_FILE = join(LOG_DIR, "warnings.log");

/** Append a line to the persistent log file. Non-blocking best-effort. */
function logToFile(level: string, context: string, msg: string, stack?: string): void {
  try {
    const ts = new Date().toISOString();
    const line = `[${ts}] [${level}] ${context}: ${msg}${stack ? "\n" + stack : ""}\n`;
    appendFileSync(LOG_FILE, line);
  } catch {
    // If logging itself fails, don't recurse or crash — just drop silently.
  }
}

/**
 * Swallow an error silently. Only visible with ZERACLAW_DEBUG=1.
 * Use for expected degradation (embeddings down, non-critical graph edges).
 */
function swallow(context: string, err?: unknown): void {
  const msg = err instanceof Error ? err.message : String(err ?? "unknown");
  pushToRing("debug", context, msg);
  if (!DEBUG) return;
  console.debug(`[swallow] ${context}: ${msg}`);
  logToFile("DEBUG", context, msg);
}

/**
 * Swallow an error but log a warning. Always visible.
 * Use for unexpected-but-recoverable issues (DB failures, compaction failures).
 */
swallow.warn = function swallowWarn(context: string, err?: unknown): void {
  const msg = err instanceof Error ? err.message : String(err ?? "unknown");
  pushToRing("warn", context, msg);
  console.warn(`[warn] ${context}: ${msg}`);
  logToFile("WARN", context, msg);
};

/**
 * Swallow an error but log an error. Always visible, includes stack.
 * Use for genuinely broken things (cleanup failure, schema failure).
 */
swallow.error = function swallowError(context: string, err?: unknown): void {
  const msg = err instanceof Error ? err.message : String(err ?? "unknown");
  const stack = err instanceof Error ? err.stack : undefined;
  pushToRing("error", context, msg);
  console.error(`[ERROR] ${context}: ${msg}${stack ? "\n" + stack : ""}`);
  logToFile("ERROR", context, msg, stack);
};

// ── Error Ring Buffer ──────────────────────────────────────────────────
// Keeps the last N swallowed errors in memory for /introspect errors.
const MAX_RING_SIZE = 30;
interface SwallowedError {
  ts: string;
  level: "debug" | "warn" | "error";
  context: string;
  message: string;
}
const _errorRing: SwallowedError[] = [];

function pushToRing(level: SwallowedError["level"], context: string, message: string): void {
  _errorRing.push({ ts: new Date().toISOString(), level, context, message });
  if (_errorRing.length > MAX_RING_SIZE) _errorRing.shift();
}

/** Get recent swallowed errors (newest first). */
export function getRecentErrors(): SwallowedError[] {
  return [..._errorRing].reverse();
}

export { swallow };
