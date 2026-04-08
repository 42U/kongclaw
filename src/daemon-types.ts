/**
 * Shared types for the memory daemon system.
 * Imported by both the worker thread (memory-daemon.ts) and the
 * main thread manager (daemon-manager.ts).
 */
import type { SurrealConfig } from "./config.js";

export interface TurnData {
  role: string;
  text: string;
  tool_name?: string;        // For tool turns: which tool was called
  tool_result?: string;      // Truncated tool output (for richer extraction context)
  file_paths?: string[];     // File paths from tool args (for artifact extraction)
}

/** Data passed to the worker thread via workerData. */
export interface DaemonWorkerData {
  surrealConfig: SurrealConfig;
  anthropicApiKey: string;
  embeddingModelPath: string;
  sessionId: string;
}

/** Previously extracted item names — for dedup across daemon runs. */
export interface PriorExtractions {
  conceptNames: string[];
  artifactPaths: string[];
  skillNames: string[];
}

/** Messages from main thread → daemon worker. */
export type DaemonMessage =
  | {
      type: "turn_batch";
      turns: TurnData[];
      thinking: string[];
      retrievedMemories: { id: string; text: string }[];
      sessionId: string;
      priorExtractions?: PriorExtractions;
    }
  | { type: "shutdown" }
  | { type: "status_request" };

/** Messages from daemon worker → main thread. */
export type DaemonResponse =
  | {
      type: "extraction_complete";
      extractedTurnCount: number;
      causalCount: number;
      monologueCount: number;
      resolvedCount: number;
      conceptCount: number;
      correctionCount: number;
      preferenceCount: number;
      artifactCount: number;
      decisionCount: number;
      skillCount: number;
      /** Extracted names for dedup in subsequent batches */
      extractedNames?: PriorExtractions;
    }
  | { type: "status"; extractedTurns: number; pendingBatches: number; errors: number }
  | { type: "shutdown_complete" }
  | { type: "error"; message: string };
