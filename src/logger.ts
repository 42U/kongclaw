import * as fs from "fs";
import * as path from "path";
import { homedir } from "os";

const LOG_DIR = path.join(homedir(), ".kongclaw");
const LOG_FILE = path.join(LOG_DIR, "crash.log");

// Ensure log directory exists
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

/**
 * Write a line to the crash log file with timestamp.
 * Uses SYNC write — this only fires at crash time, so blocking is correct.
 * Async appendFile was racing with process.exit(1) and losing every time.
 */
export function logCrash(message: string): void {
  try {
    const timestamp = new Date().toISOString();
    const line = `[${timestamp}] ${message}\n`;
    fs.appendFileSync(LOG_FILE, line);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[LOGGER ERROR] Failed to write to ${LOG_FILE}: ${msg}`);
  }
}

/**
 * Write an error stack trace and context to the crash log.
 * Uses SYNC write — process.exit(1) follows immediately in error handlers,
 * so async writes never complete. Sync guarantees the crash is captured.
 */
export function logError(label: string, error: unknown): void {
  try {
    const timestamp = new Date().toISOString();
    const stack = error instanceof Error ? error.stack : String(error);
    const lines = [
      `[${timestamp}] [ERROR] ${label}`,
      `  ${stack}`,
      "",
    ];
    fs.appendFileSync(LOG_FILE, lines.join("\n"));
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[LOGGER ERROR] Failed to write to ${LOG_FILE}: ${msg}`);
  }
}

/**
 * Get the log file path for user reference.
 */
export function getLogPath(): string {
  return LOG_FILE;
}
