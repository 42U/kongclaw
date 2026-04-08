/**
 * Subagent system — spawn isolated child Zeraclaw agents.
 *
 * Two modes:
 *   full     — shares parent's SurrealDB (same ns/db), full read/write
 *   incognito — completely isolated database, own persistent memory
 *
 * Subagents run as child processes (fork) for module-level singleton
 * isolation and future NemoClaw sandbox compatibility.
 */
import { fork, type ChildProcess } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { randomBytes } from "node:crypto";
import { Type, type Static } from "@sinclair/typebox";
import type { AgentTool, AgentToolResult } from "@mariozechner/pi-agent-core";
import type { SurrealConfig } from "./config.js";
import { getDb, isSurrealAvailable, createSurrealConnection, queryFirst, queryExec } from "./surreal.js";
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { vectorSearch, graphExpand } from "./surreal.js";
import { swallow } from "./errors.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Types ──

export type SubagentMode = "full" | "incognito";

export interface SubagentConfig {
  mode: SubagentMode;
  task: string;
  systemPrompt?: string;
  cwd?: string;
  modelId?: string;
  parentSessionId: string;
  incognitoId?: string;
}

export interface SubagentResult {
  sessionId: string;
  mode: SubagentMode;
  status: "completed" | "error";
  summary: string;
  incognitoId?: string;
  turnCount: number;
  toolCalls: number;
  durationMs: number;
}

// IPC message types
export interface IpcStartMessage {
  type: "start";
  config: SubagentConfig;
  surrealConfig: SurrealConfig;
  anthropicApiKey: string;
  embeddingModelPath: string;
}

export interface IpcOutputMessage {
  type: "output";
  text: string;
}

export interface IpcToolMessage {
  type: "tool";
  name: string;
  status: "start" | "end";
}

export interface IpcCompleteMessage {
  type: "complete";
  result: SubagentResult;
}

export interface IpcErrorMessage {
  type: "error";
  message: string;
}

export type IpcMessage =
  | IpcStartMessage
  | IpcOutputMessage
  | IpcToolMessage
  | IpcCompleteMessage
  | IpcErrorMessage;

// ── Config Resolution ──

export function generateIncognitoId(): string {
  return `incognito_${Date.now()}_${randomBytes(4).toString("hex")}`;
}

export function getSurrealConfigForMode(
  parentConfig: SurrealConfig,
  mode: SubagentMode,
  incognitoId?: string,
): SurrealConfig {
  switch (mode) {
    case "full":
      return { ...parentConfig }; // same ns/db as parent
    case "incognito": {
      const id = incognitoId ?? generateIncognitoId();
      return { ...parentConfig, db: `memory_${id}` };
    }
  }
}

// ── Spawn Logic ──

export async function spawnSubagent(
  config: SubagentConfig,
  parentSurrealConfig: SurrealConfig,
  anthropicApiKey: string,
  embeddingModelPath: string,
  onOutput?: (text: string) => void,
): Promise<SubagentResult> {
  const incognitoId = config.mode === "incognito"
    ? (config.incognitoId ?? generateIncognitoId())
    : undefined;

  const surrealConfig = getSurrealConfigForMode(
    parentSurrealConfig,
    config.mode,
    incognitoId,
  );

  const workerPath = join(__dirname, "subagent-worker.js");

  return new Promise<SubagentResult>((resolve, reject) => {
    const child: ChildProcess = fork(workerPath, [], {
      stdio: ["pipe", "pipe", "pipe", "ipc"],
      env: { ...process.env, NODE_NO_WARNINGS: "1" },
    });

    const startTime = Date.now();
    let result: SubagentResult | null = null;

    // Capture stderr for error reporting
    let stderrBuf = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderrBuf += chunk.toString();
    });

    child.on("message", (msg: IpcMessage) => {
      switch (msg.type) {
        case "output":
          onOutput?.(msg.text);
          break;
        case "complete":
          result = msg.result;
          break;
        case "error":
          reject(new Error(msg.message));
          child.kill();
          break;
      }
    });

    child.on("exit", (code) => {
      if (result) {
        // Auto-merge: if incognito subagent completed successfully, merge knowledge back
        if (config.mode === "incognito" && incognitoId && result.status === "completed") {
          mergeFromIncognito(incognitoId, parentSurrealConfig, { minImportance: 3.0, maxNodes: 50 })
            .then((mr) => { if (mr.merged > 0) console.error(`[subagent] Auto-merged ${mr.merged} nodes from incognito ${incognitoId}`); })
            .catch((e) => swallow.warn("subagent:autoMerge", e));
        }
        resolve(result);
      } else {
        reject(new Error(
          `Subagent exited with code ${code}${stderrBuf ? `: ${stderrBuf.slice(0, 200)}` : ""}`,
        ));
      }
    });

    child.on("error", (err) => {
      reject(new Error(`Failed to spawn subagent: ${err.message}`));
    });

    // Send start message
    const startMsg: IpcStartMessage = {
      type: "start",
      config: { ...config, incognitoId },
      surrealConfig,
      anthropicApiKey,
      embeddingModelPath,
    };
    child.send(startMsg);

    // Record the subagent spawn in parent DB
    recordSubagentSpawn(config, incognitoId).catch(e => swallow.warn("subagent:recordSpawn", e));
  });
}

// ── DB Tracking ──

async function recordSubagentSpawn(
  config: SubagentConfig,
  incognitoId?: string,
): Promise<void> {
  if (!(await isSurrealAvailable())) return;
  try {
    await queryExec(
      `CREATE subagent CONTENT $data`,
      {
        data: {
          parent_session_id: config.parentSessionId,
          mode: config.mode,
          task: config.task,
          status: "running",
          incognito_id: incognitoId ?? null,
        },
      },
    );
  } catch (e) {
    swallow("subagent:monitorWorker", e);
    // non-critical
  }
}

export async function updateSubagentRecord(
  parentSessionId: string,
  result: SubagentResult,
): Promise<void> {
  if (!(await isSurrealAvailable())) return;
  try {
    await queryExec(
      `UPDATE subagent SET
        child_session_id = $childSession,
        status = $status,
        summary = $summary
       WHERE parent_session_id = $parentSession
         AND status = "running"
       LIMIT 1`,
      {
        childSession: result.sessionId,
        status: result.status,
        summary: result.summary,
        parentSession: parentSessionId,
      },
    );
  } catch (e) {
    swallow("subagent:getWorkerResult", e);
    // non-critical
  }
}

export async function listSubagents(sessionId?: string): Promise<any[]> {
  if (!(await isSurrealAvailable())) return [];
  try {
    const query = sessionId
      ? `SELECT * FROM subagent WHERE parent_session_id = $sid ORDER BY created_at DESC LIMIT 20`
      : `SELECT * FROM subagent ORDER BY created_at DESC LIMIT 20`;
    return await queryFirst<any>(query, sessionId ? { sid: sessionId } : undefined);
  } catch (e) {
    swallow.warn("subagent:return [];", e);
    return [];
  }
}

// ── Merge-Back (Incognito → Parent) ──

export interface MergeFilter {
  minImportance?: number;
  tables?: string[];
  maxNodes?: number;
}

export interface MergeResult {
  merged: number;
  skippedDuplicates: number;
  tables: Record<string, number>;
}

export async function mergeFromIncognito(
  incognitoId: string,
  parentConfig: SurrealConfig,
  filter?: MergeFilter,
): Promise<MergeResult> {
  const minImportance = filter?.minImportance ?? 3.0;
  const tables = filter?.tables ?? ["memory", "concept", "skill", "reflection"];
  const maxNodes = filter?.maxNodes ?? 50;

  const incognitoConfig = getSurrealConfigForMode(parentConfig, "incognito", incognitoId);
  const incognitoDB = await createSurrealConnection(incognitoConfig);

  const result: MergeResult = { merged: 0, skippedDuplicates: 0, tables: {} };

  try {
    const parentDb = getDb();

    for (const table of tables) {
      const remaining = maxNodes - result.merged;
      if (remaining <= 0) break;

      // Query nodes from incognito DB
      let query: string;
      switch (table) {
        case "memory":
          query = `SELECT id, text, embedding, importance, category FROM memory WHERE importance >= ${minImportance} ORDER BY importance DESC LIMIT ${remaining}`;
          break;
        case "concept":
          query = `SELECT id, content AS text, embedding, stability AS importance FROM concept ORDER BY stability DESC LIMIT ${remaining}`;
          break;
        case "skill":
          query = `SELECT id, name, description, embedding, preconditions, steps, postconditions, success_count, failure_count FROM skill ORDER BY success_count DESC LIMIT ${remaining}`;
          break;
        case "reflection":
          query = `SELECT id, text, embedding, category, severity, importance FROM reflection WHERE importance >= ${minImportance} ORDER BY importance DESC LIMIT ${remaining}`;
          break;
        default:
          continue;
      }

      const nodes = await queryFirst<any>(query, undefined, incognitoDB);
      let tableCount = 0;

      for (const node of nodes) {
        // Dedup: check cosine similarity in parent
        const emb = node.embedding;
        if (emb?.length && isEmbeddingsAvailable()) {
          const dupes = await queryFirst<{ id: string; sim: number }>(
            `SELECT id, vector::similarity::cosine(embedding, $vec) AS sim
             FROM ${table}
             WHERE embedding != NONE AND array::len(embedding) > 0
             ORDER BY sim DESC LIMIT 1`,
            { vec: emb },
            parentDb,
          );
          if (dupes.length > 0 && dupes[0].sim > 0.9) {
            result.skippedDuplicates++;
            continue;
          }
        }

        // Copy to parent — strip id so SurrealDB generates a new one
        const { id: _id, ...data } = node;
        try {
          await queryExec(`CREATE ${table} CONTENT $data`, { data }, parentDb);
          tableCount++;
          result.merged++;
        } catch (e) {
          swallow("subagent:recordResult", e);
          // skip individual failures
        }
      }

      if (tableCount > 0) result.tables[table] = tableCount;
    }
  } finally {
    await incognitoDB.close();
  }

  return result;
}

// ── LLM Tool ──

const subagentSchema = Type.Object({
  task: Type.String({
    description: "The task for the subagent to accomplish. Be specific and clear.",
  }),
  mode: Type.Union([Type.Literal("full"), Type.Literal("incognito")], {
    description: "full: subagent shares your memory graph. incognito: isolated memory, no access to your knowledge.",
  }),
  incognito_id: Type.Optional(Type.String({
    description: "Reuse a persistent incognito agent by ID. Only for mode=incognito.",
  })),
});

type SubagentParams = Static<typeof subagentSchema>;

export function createSubagentTool(
  parentSessionId: string,
  cwd: string,
  modelId: string,
  parentSurrealConfig: SurrealConfig,
  anthropicApiKey: string,
  embeddingModelPath: string,
): AgentTool<typeof subagentSchema> {
  return {
    name: "subagent",
    label: "subagent",
    description: "Spawn an autonomous subagent to handle a task independently. The subagent is a full Zeraclaw instance running in an isolated process with its own tools (bash, read, edit, write, grep, find, ls). Use 'full' mode for tasks that benefit from your memory/knowledge. Use 'incognito' mode for sensitive or experimental tasks that should not access or pollute your memory graph.",
    parameters: subagentSchema,
    execute: async (_toolCallId, params: SubagentParams): Promise<AgentToolResult<any>> => {
      const config: SubagentConfig = {
        mode: params.mode,
        task: params.task,
        parentSessionId,
        cwd,
        modelId,
        incognitoId: params.incognito_id,
      };

      const outputLines: string[] = [];
      try {
        const result = await spawnSubagent(
          config,
          parentSurrealConfig,
          anthropicApiKey,
          embeddingModelPath,
          (text) => outputLines.push(text),
        );

        // Update tracking record
        await updateSubagentRecord(parentSessionId, result);

        const summary = [
          `Subagent (${result.mode}) completed: ${result.status}`,
          `Session: ${result.sessionId}`,
          `Turns: ${result.turnCount}, Tool calls: ${result.toolCalls}`,
          `Duration: ${(result.durationMs / 1000).toFixed(1)}s`,
          result.incognitoId ? `Incognito ID: ${result.incognitoId} (use for merge-back or re-attach)` : "",
          `\n--- Summary ---\n${result.summary}`,
        ].filter(Boolean).join("\n");

        return {
          content: [{ type: "text", text: summary }],
          details: result,
        };
      } catch (err: any) {
        return {
          content: [{ type: "text", text: `Subagent failed: ${err.message}` }],
          details: null,
        };
      }
    },
  };
}
