import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { AsyncLocalStorage } from "node:async_hooks";
import { Surreal } from "surrealdb";
import { logCrash, logError } from "./logger.js";
import type { SurrealConfig } from "./config.js";
import { swallow } from "./errors.js";
import { embed } from "./embeddings.js";

/** Record with a vector similarity score from SurrealDB search */
interface ScoredRecord { id?: string; score: number; [key: string]: unknown }

// AsyncLocalStorage allows subagents to use their own DB connection
// while the parent keeps the default. getDb() checks the store first.
const dbStorage = new AsyncLocalStorage<Surreal>();
let _defaultDb: Surreal;
let _defaultConfig: SurrealConfig | null = null;
let _reconnecting: Promise<void> | null = null;
let _shutdownFlag = false;

/** Signal that the process is shutting down — skip reconnection attempts. */
export function markShutdown(): void { _shutdownFlag = true; }

/** Returns the active DB connection (AsyncLocalStorage context or default). */
function db(): Surreal {
  return dbStorage.getStore() ?? _defaultDb;
}

/**
 * Ensure the default DB connection is alive. If disconnected, auto-reconnect.
 * Uses a singleton promise to prevent reconnection storms.
 */
async function ensureConnected(): Promise<void> {
  // Subagent connections manage their own lifecycle
  if (dbStorage.getStore()) return;
  if (!_defaultDb || !_defaultConfig) return;
  if (_shutdownFlag) return; // Don't attempt reconnection during shutdown
  if (_defaultDb.isConnected) return;

  // Coalesce concurrent reconnect attempts into one
  if (_reconnecting) return _reconnecting;

  _reconnecting = (async () => {
    const MAX_ATTEMPTS = 3;
    const BACKOFF_MS = [500, 1500, 4000];
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        console.warn(`[warn] SurrealDB disconnected — reconnecting (attempt ${attempt}/${MAX_ATTEMPTS})...`);
        logCrash(`SurrealDB disconnected — reconnecting (attempt ${attempt}/${MAX_ATTEMPTS})...`);
        _defaultDb = new Surreal();
        const CONNECT_TIMEOUT_MS = 5_000;
        await Promise.race([
          _defaultDb.connect(_defaultConfig.url, {
            namespace: _defaultConfig.ns,
            database: _defaultConfig.db,
            authentication: { username: _defaultConfig.user, password: _defaultConfig.pass },
          }),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`SurrealDB connect timed out after ${CONNECT_TIMEOUT_MS}ms`)), CONNECT_TIMEOUT_MS),
          ),
        ]);
        console.warn(`[warn] SurrealDB reconnected successfully.`);
        logCrash("SurrealDB reconnected successfully.");
        return;
      } catch (e) {
        if (attempt < MAX_ATTEMPTS) {
          await new Promise(r => setTimeout(r, BACKOFF_MS[attempt - 1]));
        } else {
          console.error(`[ERROR] SurrealDB reconnection failed after ${MAX_ATTEMPTS} attempts.`);
          logError(`SurrealDB reconnection failed after ${MAX_ATTEMPTS} attempts.`, e);
          throw e;
        }
      }
    }
  })().finally(() => { _reconnecting = null; });

  return _reconnecting;
}

const __dirname = dirname(fileURLToPath(import.meta.url));

export async function initSurreal(config: SurrealConfig): Promise<void> {
  _defaultConfig = config;
  _defaultDb = new Surreal();
  await _defaultDb.connect(config.url, {
    namespace: config.ns,
    database: config.db,
    authentication: { username: config.user, password: config.pass },
  });
  await runSchema(_defaultDb);
}

/** Create an independent Surreal connection (for subagents). */
export async function createSurrealConnection(config: SurrealConfig): Promise<Surreal> {
  const conn = new Surreal();
  await conn.connect(config.url, {
    namespace: config.ns,
    database: config.db,
    authentication: { username: config.user, password: config.pass },
  });
  await runSchema(conn);
  return conn;
}

/** Run a function with a specific DB connection as the active context. */
export function runWithDb<T>(dbInstance: Surreal, fn: () => T): T {
  return dbStorage.run(dbInstance, fn);
}

async function runSchema(conn?: Surreal): Promise<void> {
  // schema.surql is copied to dist/ — read from same dir as compiled JS
  // Fall back to src/ for ts-node/dev
  let schemaPath = join(__dirname, "schema.surql");
  let schema: string;
  try {
    schema = readFileSync(schemaPath, "utf-8");
  } catch (e) {
    swallow.error("surreal:schemaPath = join(__dirname, '..', 'src'", e);
    schemaPath = join(__dirname, "..", "src", "schema.surql");
    schema = readFileSync(schemaPath, "utf-8");
  }
  await (conn ?? _defaultDb).query(schema);
}

/**
 * Typed helper for SurrealDB queries. Unwraps the first statement result
 * so callers get `T[]` instead of dealing with `[T[]]` tuples.
 * Replaces all `(result as unknown as any[][]).flat()` patterns.
 */
/**
 * SurrealDB v2 requires that any field referenced in ORDER BY also appears in the SELECT
 * clause — it won't implicitly include the field just to sort by it. This sanitizer detects
 * the omission and patches the SELECT list automatically, so callers never have to think
 * about it. No-ops on wildcard SELECTs (`SELECT *`), non-SELECT statements, or when the
 * ORDER BY field is already selected.
 *
 * Example fix:
 *   SELECT id FROM retrieval_outcome ORDER BY created_at DESC
 *   → SELECT id, created_at FROM retrieval_outcome ORDER BY created_at DESC
 */
function patchOrderByFields(sql: string): string {
  const s = sql.trim();
  // Only patch SELECT statements that contain ORDER BY
  if (!/^\s*SELECT\b/i.test(s) || !/\bORDER\s+BY\b/i.test(s)) return sql;
  // Wildcard selects already include every field — nothing to patch
  if (/^\s*SELECT\s+\*/i.test(s)) return sql;

  // Extract the fields between SELECT and FROM
  const selectMatch = s.match(/^\s*SELECT\s+([\s\S]+?)\s+FROM\b/i);
  if (!selectMatch) return sql;
  const selectClause = selectMatch[1];

  // Extract ORDER BY fields; strip ASC/DESC and stop at LIMIT/GROUP/HAVING/end
  const orderMatch = s.match(/\bORDER\s+BY\s+([\s\S]+?)(?=\s+LIMIT\b|\s+GROUP\b|\s+HAVING\b|$)/i);
  if (!orderMatch) return sql;

  const orderFields = orderMatch[1]
    .split(',')
    .map(f => f.trim().replace(/\s+(ASC|DESC)\s*$/i, '').trim())
    .filter(Boolean);

  // Normalise currently selected field names (strip table prefixes and AS aliases)
  const selectedFields = selectClause
    .split(',')
    .map(f => f.trim().split(/\s+AS\s+/i)[0].trim())
    .map(f => f.split('.').pop()!)
    .filter(Boolean)
    .map(f => f.toLowerCase());

  // Find ORDER BY fields absent from SELECT
  const missing = orderFields.filter(f =>
    !selectedFields.includes(f.split('.').pop()!.toLowerCase()),
  );

  if (missing.length === 0) return sql;

  // Append missing fields to the SELECT clause
  return sql.replace(
    /(\bSELECT\s+)([\s\S]+?)(\s+FROM\b)/i,
    (_, pre, fields, post) => `${pre}${fields}, ${missing.join(', ')}${post}`,
  );
}

export async function queryFirst<T>(
  sql: string,
  bindings?: Record<string, unknown>,
  dbInstance?: ReturnType<typeof db>,
): Promise<T[]> {
  await ensureConnected();
  // Ensure namespace/database context is set inline (SurrealDB HTTP /sql endpoint requirement)
  // Also auto-patch any ORDER BY fields missing from the SELECT clause (SurrealDB v2 idiom requirement)
  const fullSql = `USE NS kong DB memory; ${patchOrderByFields(sql)}`;
  const result = await (dbInstance ?? db()).query<[T[]]>(fullSql, bindings);
  // USE NS..DB returns a result at index 0; actual query result is last element
  const rows = Array.isArray(result) ? result[result.length - 1] : result;
  return (Array.isArray(rows) ? rows : []).filter(Boolean);
}
/**
 * Run a multi-statement SurrealQL query and return the last statement's result.
 * For queries using LET + FOR + RETURN patterns where db.query() returns nested arrays.
 */
export async function queryMulti<T = unknown>(sql: string, bindings?: Record<string, unknown>): Promise<T | undefined> {
  await ensureConnected();
  // Patch ORDER BY fields (same as queryFirst) — all query paths must go through this
  const fullSql = `USE NS kong DB memory; ${patchOrderByFields(sql)}`;
  const raw = await db().query(fullSql, bindings);
  const flat = (raw as unknown[]).flat();
  return flat[flat.length - 1] as T | undefined;
}


/**
 * Fire-and-forget query helper for write operations (CREATE, UPDATE, RELATE, DELETE).
 * Discards the return value — use `queryFirst<T>` when you need results.
 */
export async function queryExec(
  sql: string,
  bindings?: Record<string, unknown>,
  dbInstance?: ReturnType<typeof db>,
): Promise<void> {
  await ensureConnected();
  // Patch ORDER BY fields (same as queryFirst) — all query paths must go through this
  const fullSql = `USE NS kong DB memory; ${patchOrderByFields(sql)}`;
  await (dbInstance ?? db()).query(fullSql, bindings);
}

export interface VectorSearchResult {
  id: string;
  text: string;
  score: number;
  role?: string;
  timestamp?: string;
  importance?: number;
  accessCount?: number;
  source?: string;
  sessionId?: string;
  table: string;
  embedding?: number[];
}

async function safeQuery(sql: string, bindings: Record<string, unknown>): Promise<VectorSearchResult[]> {
  try {
    return await queryFirst<VectorSearchResult>(sql, bindings);
  } catch (e) {
    swallow.warn("surreal:return [];", e);
    return []; // table doesn't exist or query failed — skip gracefully
  }
}

export interface TimeRange {
  before?: Date;
  after?: Date;
}

export async function vectorSearch(
  vec: number[],
  sessionId: string,
  limits: { turn?: number; identity?: number; concept?: number; memory?: number; artifact?: number; monologue?: number } = {},
  withEmbeddings = false,
  timeRange?: TimeRange,
): Promise<VectorSearchResult[]> {
  // Apply defaults
  const lim = {
    turn: limits.turn ?? 20,
    identity: limits.identity ?? 10,
    concept: limits.concept ?? 15,
    memory: limits.memory ?? 15,
    artifact: limits.artifact ?? 10,
    monologue: limits.monologue ?? 8,
  };
  // Split turn search: current session (higher relevance) + cross-session
  const sessionTurnLim = Math.ceil(lim.turn / 2);
  const crossTurnLim = lim.turn - sessionTurnLim;
  const emb = withEmbeddings ? ", embedding" : "";

  // Build optional time filter clause for temporal queries ("what happened last week?")
  const timeBindings: Record<string, unknown> = {};
  let timeFilter = "";
  if (timeRange?.before) {
    timeFilter += " AND created_at < $timeBefore";
    timeBindings.timeBefore = timeRange.before.toISOString();
  }
  if (timeRange?.after) {
    timeFilter += " AND created_at > $timeAfter";
    timeBindings.timeAfter = timeRange.after.toISOString();
  }

  // Run each table search independently so a missing table doesn't break everything
  // Note: identity_chunk removed from vector search — identity now comes through Tier 0 core_memory
  const [sessionTurns, crossTurns, concepts, memories, artifacts, monologues] = await Promise.all([
    safeQuery(
      `SELECT id, text, role, timestamp, 0 AS accessCount, 'turn' AS table,
              vector::similarity::cosine(embedding, $vec) AS score${emb}
       FROM turn
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND session_id = $sid${timeFilter}
       ORDER BY score DESC LIMIT $lim`,
      { vec, lim: sessionTurnLim, sid: sessionId, ...timeBindings },
    ),
    safeQuery(
      `SELECT id, text, role, timestamp, 0 AS accessCount, 'turn' AS table,
              vector::similarity::cosine(embedding, $vec) AS score${emb}
       FROM turn
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND session_id != $sid${timeFilter}
       ORDER BY score DESC LIMIT $lim`,
      { vec, lim: crossTurnLim, sid: sessionId, ...timeBindings },
    ),
    safeQuery(
      `SELECT id, content AS text, stability AS importance, access_count AS accessCount,
              created_at AS timestamp, 'concept' AS table,
              vector::similarity::cosine(embedding, $vec) AS score${emb}
       FROM concept
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND superseded_at IS NONE${timeFilter}
       ORDER BY score DESC LIMIT $lim`,
      { vec, lim: lim.concept, ...timeBindings },
    ),
    safeQuery(
      `SELECT id, text, importance, access_count AS accessCount,
              created_at AS timestamp, session_id AS sessionId, 'memory' AS table,
              vector::similarity::cosine(embedding, $vec) AS score${emb}
       FROM memory
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND (status = 'active' OR status IS NONE)${timeFilter}
       ORDER BY score DESC LIMIT $lim`,
      { vec, lim: lim.memory, ...timeBindings },
    ),
    safeQuery(
      `SELECT id, description AS text, 0 AS accessCount,
              created_at AS timestamp, 'artifact' AS table,
              vector::similarity::cosine(embedding, $vec) AS score${emb}
       FROM artifact
       WHERE embedding != NONE AND array::len(embedding) > 0
       ORDER BY score DESC LIMIT $lim`,
      { vec, lim: lim.artifact },
    ),
    safeQuery(
      `SELECT id, content AS text, category AS source, 0.5 AS importance, 0 AS accessCount,
              timestamp, 'monologue' AS table,
              vector::similarity::cosine(embedding, $vec) AS score${emb}
       FROM monologue
       WHERE embedding != NONE AND array::len(embedding) > 0
       ORDER BY score DESC LIMIT $lim`,
      { vec, lim: lim.monologue },
    ),
  ]);
  return [...sessionTurns, ...crossTurns, ...concepts, ...memories, ...artifacts, ...monologues];
}

/**
 * Tag-boosted concept retrieval: supplements vector search with keyword-based
 * concept lookup. Surfaces concepts that pure embeddings miss (e.g., query
 * mentions "memory architecture" → finds concepts tagged "episodic", "semantic-memory").
 * Runs in parallel with vectorSearch for zero additional latency.
 */
const TAG_STOP_WORDS = new Set(["the","a","an","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","can","shall","to","of","in","for","on","with","at","by","from","as","into","about","between","through","during","it","its","this","that","these","those","i","you","we","they","my","your","our","their","what","which","who","how","when","where","why","not","no","and","or","but","if","so","any","all","some","more","just","also","than","very","too","much","many"]);

export async function tagBoostedConcepts(
  queryText: string,
  queryVec: number[],
  limit = 10,
): Promise<VectorSearchResult[]> {
  const words = queryText.toLowerCase().replace(/[^a-z0-9\s-]/g, "").split(/\s+/)
    .filter(w => w.length > 2 && !TAG_STOP_WORDS.has(w));
  if (words.length === 0) return [];

  // Build tag match condition — match any tag that contains a query word
  // Sanitize: strip quotes to prevent injection
  const tagConditions = words.slice(0, 8)
    .map(w => `tags CONTAINS '${w.replace(/'/g, "")}'`)
    .join(" OR ");

  try {
    const rows = await queryFirst<VectorSearchResult>(
      `SELECT id, content AS text, stability AS importance, access_count AS accessCount,
              created_at AS timestamp, 'concept' AS table,
              vector::similarity::cosine(embedding, $vec) AS score
       FROM concept
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND superseded_at IS NONE
         AND (${tagConditions})
       ORDER BY score DESC
       LIMIT $limit`,
      { vec: queryVec, limit },
    );
    return rows;
  } catch (e) {
    swallow.warn("surreal:tagBoostedConcepts", e);
    return [];
  }
}

export interface TurnRecord {
  session_id: string;
  role: string;
  text: string;
  embedding: number[] | null;
  token_count?: number;
  tool_name?: string;
  model?: string;
  usage?: Record<string, unknown>;
}

export async function upsertTurn(turn: TurnRecord): Promise<string> {
  // Omit embedding entirely when empty — SurrealDB 3.0 rejects JS null for option<array<float>>
  const { embedding, ...rest } = turn;
  const record = embedding?.length ? { ...rest, embedding } : rest;
  const rows = await queryFirst<{ id: string }>(
    `CREATE turn CONTENT $turn RETURN id`,
    { turn: record },
  );
  return String(rows[0]?.id ?? "");
}

// Validate a SurrealDB record ID format "table:id" — reject anything else
const RECORD_ID_RE = /^[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_]+$/;
export function assertRecordId(id: string): void {
  if (!RECORD_ID_RE.test(id)) {
    throw new Error(`Invalid record ID format: ${id.slice(0, 40)}`);
  }
}

export async function relate(fromId: string, edge: string, toId: string): Promise<void> {
  assertRecordId(fromId);
  assertRecordId(toId);
  const safeName = edge.replace(/[^a-zA-Z0-9_]/g, "");
  // SurrealDB RELATE requires inline record IDs (not parameterizable)
  await queryExec(`RELATE ${fromId}->${safeName}->${toId}`);
}

export async function createSession(agentId = "default"): Promise<string> {
  const rows = await queryFirst<{ id: string }>(
    `CREATE session CONTENT { agent_id: $agent_id } RETURN id`,
    { agent_id: agentId },
  );
  return String(rows[0]?.id ?? "");
}

export async function updateSessionStats(
  sessionId: string,
  inputTokens: number,
  outputTokens: number,
): Promise<void> {
  assertRecordId(sessionId);
  await queryExec(
    `UPDATE ${sessionId} SET
      turn_count += 1,
      total_input_tokens += $input,
      total_output_tokens += $output,
      last_active = time::now()`,
    { input: inputTokens, output: outputTokens },
  );
}


/**
 * Close a session: set ended_at and optionally a summary.
 * Called during cleanup to ensure sessions don't stay open forever.
 */
export async function endSession(sessionId: string, summary?: string): Promise<void> {
  assertRecordId(sessionId);
  if (summary) {
    await queryExec(
      `UPDATE ${sessionId} SET ended_at = time::now(), summary = $summary`,
      { summary },
    );
  } else {
    await queryExec(
      `UPDATE ${sessionId} SET ended_at = time::now()`,
    );
  }
}

/**
 * Find orphaned sessions — started but never ended (no ended_at).
 * Excludes the current session and very recent sessions (< 2 min old).
 */
export async function getOrphanedSessions(
  limit = 10,
  excludeSessionId?: string,
): Promise<{ id: string; turn_count: number; started_at: string }[]> {
  const rows = await queryFirst<{ id: string; turn_count: number; started_at: string }>(
    `SELECT id, turn_count, started_at FROM session
     WHERE ended_at IS NONE
       AND started_at < time::now() - 2m
       ${excludeSessionId ? `AND id != $exclude` : ""}
     ORDER BY started_at DESC LIMIT $limit`,
    { limit, ...(excludeSessionId ? { exclude: excludeSessionId } : {}) },
  );
  return rows;
}

export async function isSurrealAvailable(): Promise<boolean> {
  try {
    const active = dbStorage.getStore() ?? _defaultDb;
    return active?.isConnected ?? false;
  } catch (e) {
    swallow.warn("surreal:return false;", e);
    return false;
  }
}

/** Connection info for introspection (no credentials). */
export function getSurrealInfo(): { url: string; ns: string; db: string; connected: boolean } | null {
  if (!_defaultConfig) return null;
  return {
    url: _defaultConfig.url,
    ns: _defaultConfig.ns,
    db: _defaultConfig.db,
    connected: _defaultDb?.isConnected ?? false,
  };
}

/** Real connectivity check — executes a query, not just .isConnected. */
export async function pingDb(): Promise<boolean> {
  try {
    await ensureConnected();
    await db().query("RETURN 'ok'");
    return true;
  } catch {
    return false;
  }
}

export async function closeSurreal(): Promise<void> {
  try {
    await _defaultDb?.close();
  } catch (e) {
    swallow("surreal:silent", e);
    // ignore
  }
}

// --- 5-Pillar entity operations ---

export async function ensureAgent(name: string, model?: string): Promise<string> {
  const rows = await queryFirst<{ id: string }>(
    `SELECT id FROM agent WHERE name = $name LIMIT 1`,
    { name },
  );
  if (rows.length > 0) return String(rows[0].id);

  const created = await queryFirst<{ id: string }>(
    `CREATE agent CONTENT { name: $name, model: $model } RETURN id`,
    { name, ...(model != null ? { model } : {}) },
  );
  return String(created[0]?.id ?? "");
}

export async function ensureProject(name: string): Promise<string> {
  const rows = await queryFirst<{ id: string }>(
    `SELECT id FROM project WHERE name = $name LIMIT 1`,
    { name },
  );
  if (rows.length > 0) return String(rows[0].id);

  const created = await queryFirst<{ id: string }>(
    `CREATE project CONTENT { name: $name } RETURN id`,
    { name },
  );
  return String(created[0]?.id ?? "");
}

export async function createTask(description: string): Promise<string> {
  const rows = await queryFirst<{ id: string }>(
    `CREATE task CONTENT { description: $desc, status: "in_progress" } RETURN id`,
    { desc: description },
  );
  return String(rows[0]?.id ?? "");
}

export async function upsertConcept(
  content: string,
  embedding: number[] | null,
  source?: string,
): Promise<string> {
  // Check for existing concept with similar content (case-insensitive dedup)
  const rows = await queryFirst<{ id: string }>(
    `SELECT id FROM concept WHERE string::lowercase(content) = string::lowercase($content) LIMIT 1`,
    { content },
  );
  if (rows.length > 0) {
    const id = String(rows[0].id);
    assertRecordId(id);
    await queryExec(
      `UPDATE ${id} SET access_count += 1, last_accessed = time::now()`,
    );
    return id;
  }

  const emb = embedding?.length ? embedding : undefined;
  const record: Record<string, unknown> = { content, source: source ?? undefined };
  if (emb) record.embedding = emb;
  const created = await queryFirst<{ id: string }>(
    `CREATE concept CONTENT $record RETURN id`,
    { record },
  );
  return String(created[0]?.id ?? "");
}

export async function createArtifact(
  path: string,
  type: string,
  description: string,
  embedding: number[] | null,
): Promise<string> {
  const record: Record<string, unknown> = { path, type, description };
  if (embedding?.length) record.embedding = embedding;
  const rows = await queryFirst<{ id: string }>(
    `CREATE artifact CONTENT $record RETURN id`,
    { record },
  );
  return String(rows[0]?.id ?? "");
}

export async function createMemory(
  text: string,
  embedding: number[] | null,
  importance: number,
  category?: string,
  sessionId?: string,
): Promise<string> {
  const source = category ?? "general";

  // Write-time dedup: if a semantically near-identical memory already exists,
  // bump its access_count + importance instead of creating a duplicate.
  if (embedding?.length) {
    const dupes = await queryFirst<{ id: string; importance: number; score: number }>(
      `SELECT id, importance,
              vector::similarity::cosine(embedding, $vec) AS score
       FROM memory
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND category = $cat
       ORDER BY score DESC
       LIMIT 1`,
      { vec: embedding, cat: source },
    );
    if (dupes.length > 0 && dupes[0].score > 0.92) {
      const existing = dupes[0];
      const newImp = Math.max(existing.importance ?? 0, importance);
      assertRecordId(String(existing.id));
      await queryExec(
        `UPDATE ${existing.id} SET access_count += 1, importance = $imp, last_accessed = time::now()`,
        { imp: newImp },
      );
      return String(existing.id);
    }
  }

  const record: Record<string, unknown> = { text, importance, category: source, source };
  if (embedding?.length) record.embedding = embedding;
  if (sessionId) record.session_id = sessionId;
  const rows = await queryFirst<{ id: string }>(
    `CREATE memory CONTENT $record RETURN id`,
    { record },
  );
  return String(rows[0]?.id ?? "");
}

export async function getSessionTurns(sessionId: string, limit = 50): Promise<{ role: string; text: string }[]> {
  return await queryFirst<{ role: string; text: string }>(
    `SELECT role, text, timestamp FROM turn WHERE session_id = $sid ORDER BY timestamp ASC LIMIT $lim`,
    { sid: sessionId, lim: limit },
  );
}

/** Richer turn data for daemon extraction — includes tool_name for artifact discovery. */
export async function getSessionTurnsRich(sessionId: string, limit = 20): Promise<{ role: string; text: string; tool_name?: string }[]> {
  return await queryFirst<{ role: string; text: string; tool_name?: string }>(
    `SELECT role, text, tool_name, timestamp FROM turn WHERE session_id = $sid ORDER BY timestamp ASC LIMIT $lim`,
    { sid: sessionId, lim: limit },
  );
}

export async function linkSessionToTask(sessionId: string, taskId: string): Promise<void> {
  assertRecordId(sessionId);
  assertRecordId(taskId);
  await queryExec(`RELATE ${sessionId}->session_task->${taskId}`);
}

export async function linkTaskToProject(taskId: string, projectId: string): Promise<void> {
  assertRecordId(taskId);
  assertRecordId(projectId);
  await queryExec(`RELATE ${taskId}->task_part_of->${projectId}`);
}

export async function linkAgentToTask(agentId: string, taskId: string): Promise<void> {
  assertRecordId(agentId);
  assertRecordId(taskId);
  await queryExec(`RELATE ${agentId}->performed->${taskId}`);
}

export async function linkAgentToProject(agentId: string, projectId: string): Promise<void> {
  assertRecordId(agentId);
  assertRecordId(projectId);
  await queryExec(`RELATE ${agentId}->owns->${projectId}`);
}

// --- Graph traversal for retrieval enhancement ---

/**
 * Multi-hop neighbor expansion: given seed node IDs, traverse typed edges
 * to find structurally related nodes. Returns unique neighbors with text.
 *
 * Phase 1A: Forward edges (responds_to, mentions, etc.) + reverse edges
 *   for reflects_on and skill_from_task (reflections/skills point AT
 *   sessions/tasks, so reverse traversal finds them from seed nodes).
 * Phase 1B: Optional multi-hop — when hops > 1, takes top frontier nodes
 *   by score and runs another round (same pattern as queryCausalContext).
 */
export async function graphExpand(
  nodeIds: string[],
  queryVec: number[],
  hops = 1,
): Promise<VectorSearchResult[]> {
  if (nodeIds.length === 0) return [];

  // Forward edges: seed → edge → neighbor
  const forwardEdges = [
    "responds_to", "mentions", "related_to", "narrower", "broader",
    "about_concept", "reflects_on", "skill_from_task",
    "contradicts", "supports", "supersedes",
  ];
  // Reverse edges: neighbor → edge → seed (finds reflections/skills pointing at sessions/tasks)
  const reverseEdges = ["reflects_on", "skill_from_task"];

  const scoreExpr = ", IF embedding != NONE AND array::len(embedding) > 0 THEN vector::similarity::cosine(embedding, $vec) ELSE 0 END AS score";
  const bindings = { vec: queryVec };
  const selectFields = `SELECT id, text, content, description, importance, stability,
                access_count AS accessCount, created_at AS timestamp,
                meta::tb(id) AS table${scoreExpr}`;

  const seen = new Set<string>(nodeIds);
  const allNeighbors: VectorSearchResult[] = [];
  let frontier = nodeIds.slice(0, 5).filter((id) => RECORD_ID_RE.test(id));

  for (let hop = 0; hop < hops && frontier.length > 0; hop++) {
    // Forward traversal: seed → edge → ?
    const forwardQueries = frontier.flatMap((id) =>
      forwardEdges.map((edge) =>
        queryFirst<any>(
          `${selectFields} FROM ${id}->${edge}->? LIMIT 3`,
          bindings,
        ).catch(e => { swallow.warn("surreal:graphExpand", e); return [] as Record<string, unknown>[]; })
      )
    );

    // Reverse traversal: ? → edge → seed (for edges where the interesting node points at the seed)
    const reverseQueries = frontier.flatMap((id) =>
      reverseEdges.map((edge) =>
        queryFirst<any>(
          `${selectFields} FROM ${id}<-${edge}<-? LIMIT 3`,
          bindings,
        ).catch(e => { swallow.warn("surreal:graphExpand", e); return [] as Record<string, unknown>[]; })
      )
    );

    const queryResults = await Promise.all([...forwardQueries, ...reverseQueries]);

    const nextFrontier: { id: string; score: number }[] = [];

    for (const rows of queryResults) {
      for (const row of rows) {
        const nodeId = String(row.id);
        if (seen.has(nodeId)) continue;
        seen.add(nodeId);

        const text = row.text ?? row.content ?? row.description ?? null;
        if (text) {
          const score = row.score ?? 0;
          allNeighbors.push({
            text,
            importance: row.importance ?? row.stability,
            accessCount: row.accessCount,
            timestamp: row.timestamp,
            table: String(row.table ?? "unknown"),
            id: nodeId,
            score,
          });
          if (RECORD_ID_RE.test(nodeId)) {
            nextFrontier.push({ id: nodeId, score });
          }
        }
      }
    }

    // Next hop frontier: top 3 by score to control fanout
    frontier = nextFrontier
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((n) => n.id);
  }

  return allNeighbors;
}

/**
 * Increment access_count for retrieved concept/memory nodes.
 * Called after retrieval to track which memories are frequently useful.
 */
export async function bumpAccessCounts(ids: string[]): Promise<void> {
  for (const id of ids) {
    try {
      assertRecordId(id);
      await queryExec(
        `UPDATE ${id} SET access_count += 1, last_accessed = time::now()`
      );
    } catch (e) {
      swallow.warn("surreal:assertRecordId", e);
      // skip — node might not have access_count field or invalid ID
    }
  }
}

export function getDb(): Surreal {
  return dbStorage.getStore() ?? _defaultDb;
}

// --- Constitutive Memory: Previous Session Continuity ---

/**
 * Fetch the last N turns from the most recent *previous* session.
 * Used at wakeup to give the agent literal conversation context from where
 * it left off, not just a summarized handoff note.
 * @param currentSessionId - The current session ID to exclude (so we get the *previous* one)
 * @param limit - Number of turns to fetch (default 10, from the end of the session)
 */
export async function getPreviousSessionTurns(
  currentSessionId?: string,
  limit = 10,
): Promise<{ role: string; text: string; tool_name?: string; timestamp: string }[]> {
  try {
    // Find the most recent session that isn't the current one
    let prevSessionQuery: string;
    let bindings: Record<string, unknown> = { lim: limit };

    if (currentSessionId) {
      prevSessionQuery = `SELECT id, started_at FROM session WHERE id != $current ORDER BY started_at DESC LIMIT 1`;
      bindings.current = currentSessionId;
    } else {
      // No current session yet — just get the latest
      prevSessionQuery = `SELECT id, started_at FROM session ORDER BY started_at DESC LIMIT 1`;
    }

    const sessionRows = await queryFirst<{ id: string }>(prevSessionQuery, bindings);
    if (sessionRows.length === 0) return [];

    const prevSessionId = String(sessionRows[0].id);

    // Get the last N turns from that session, ordered chronologically
    // We query more than needed in reverse order, then reverse to get chronological
    const turns = await queryFirst<{ role: string; text: string; tool_name?: string; timestamp: string }>(
      `SELECT role, text, tool_name, timestamp FROM turn
       WHERE session_id = $sid AND text != NONE AND text != ""
       ORDER BY timestamp DESC LIMIT $lim`,
      { sid: prevSessionId, lim: limit },
    );

    // Reverse so they're in chronological order (oldest first)
    return turns.reverse();
  } catch (e) {
    swallow.warn("surreal:return [];", e);
    return [];
  }
}

// --- Constitutive Memory: Monologue & Handoff ---

export async function getLatestHandoff(): Promise<{ text: string; created_at: string } | null> {
  try {
    const rows = await queryFirst<{ text: string; created_at: string }>(
      `SELECT text, created_at FROM memory WHERE category = "handoff" ORDER BY created_at DESC LIMIT 1`,
    );
    return rows[0] ?? null;
  } catch (e) {
    swallow.warn("surreal:return null;", e);
    return null;
  }
}

export async function countResolvedSinceHandoff(handoffCreatedAt: string): Promise<number> {
  try {
    const rows = await queryFirst<{ count: number }>(
      `SELECT count() AS count FROM memory WHERE status = 'resolved' AND resolved_at > $ts GROUP ALL`,
      { ts: handoffCreatedAt },
    );
    return rows[0]?.count ?? 0;
  } catch (e) {
    swallow.warn("surreal:countResolvedSinceHandoff", e);
    return 0;
  }
}

export async function getUnresolvedMemories(limit = 5): Promise<{ id: string; text: string; importance: number; category: string }[]> {
  try {
    // Exclude handoff/monologue/reflection — those are narrative, not actionable items.
    // Apply age decay: memories older than 7 days lose 1 importance per week (capped at -3).
    // This prevents stale issues from dominating the wakeup context forever.
    return await queryFirst<{ id: string; text: string; importance: number; category: string }>(
      `SELECT id, text,
              math::max([importance - math::min([math::floor(duration::days(time::now() - created_at) / 7), 3]), 0]) AS importance,
              category
       FROM memory
       WHERE (status IS NONE OR status != 'resolved')
         AND category NOT IN ['handoff', 'monologue', 'reflection', 'compaction', 'consolidation']
         AND importance >= 6
       ORDER BY importance DESC
       LIMIT $lim`,
      { lim: limit },
    );
  } catch (e) {
    swallow.warn("surreal:getUnresolvedMemories", e);
    return [];
  }
}

/**
 * Resolve a memory by ID — marks it as done so it stops surfacing.
 * Can be called directly when we know an issue is fixed, without needing
 * the fragile retrieval-match mechanism.
 */
export async function resolveMemory(memoryId: string): Promise<boolean> {
  try {
    assertRecordId(memoryId);
    await queryFirst(
      `UPDATE ${memoryId} SET status = 'resolved', resolved_at = time::now()`,
    );
    return true;
  } catch (e) {
    swallow.warn("surreal:resolveMemory", e);
    return false;
  }
}

/**
 * Resolve memories matching a text pattern — useful when we know an issue
 * was fixed but don't have the exact memory ID.
 */
export async function resolveMemoriesByPattern(pattern: string): Promise<number> {
  try {
    const rows = await queryFirst<{ id: string }>(
      `SELECT id FROM memory WHERE (status IS NONE OR status != 'resolved') AND text CONTAINS $pat`,
      { pat: pattern },
    );
    if (rows.length === 0) return 0;
    for (const row of rows) {
      const rid = String(row.id);
      assertRecordId(rid);
      await queryFirst(
        `UPDATE ${rid} SET status = 'resolved', resolved_at = time::now()`,
      );
    }
    return rows.length;
  } catch (e) {
    swallow.warn("surreal:resolveMemoriesByPattern", e);
    return 0;
  }
}

export async function getRecentFailedCausal(limit = 3): Promise<{ description: string; chain_type: string }[]> {
  try {
    return await queryFirst<{ description: string; chain_type: string }>(
      `SELECT description, chain_type, created_at FROM causal_chain WHERE success = false ORDER BY created_at DESC LIMIT $lim`,
      { lim: limit },
    );
  } catch (e) {
    swallow.warn("surreal:getRecentFailedCausal", e);
    return [];
  }
}

export async function getAllIdentityChunks(): Promise<{ text: string }[]> {
  try {
    return await queryFirst<{ text: string }>(
      `SELECT text, chunk_index FROM identity_chunk ORDER BY chunk_index ASC`,
    );
  } catch (e) {
    swallow.warn("surreal:return [];", e);
    return [];
  }
}

export async function getRecentMonologues(limit = 5): Promise<{ category: string; content: string; timestamp: string }[]> {
  try {
    return await queryFirst<{ category: string; content: string; timestamp: string }>(
      `SELECT category, content, timestamp FROM monologue ORDER BY timestamp DESC LIMIT $lim`,
      { lim: limit },
    );
  } catch (e) {
    swallow.warn("surreal:return [];", e);
    return [];
  }
}

export async function createMonologue(
  sessionId: string,
  category: string,
  content: string,
  embedding: number[] | null,
): Promise<string> {
  const record: Record<string, unknown> = { session_id: sessionId, category, content };
  if (embedding?.length) record.embedding = embedding;
  const rows = await queryFirst<{ id: string }>(
    `CREATE monologue CONTENT $record RETURN id`,
    { record },
  );
  return String(rows[0]?.id ?? "");
}

// --- Tiered Memory: Core Memory CRUD ---

export interface CoreMemoryEntry {
  id: string;
  text: string;
  category: string;
  priority: number;
  tier: number;
  active: boolean;
  session_id?: string;
  created_at?: string;
  updated_at?: string;
}

export async function getAllCoreMemory(tier?: number): Promise<CoreMemoryEntry[]> {
  try {
    if (tier != null) {
      return await queryFirst<CoreMemoryEntry>(
        `SELECT * FROM core_memory WHERE active = true AND tier = $tier ORDER BY priority DESC`,
        { tier },
      );
    }
    return await queryFirst<CoreMemoryEntry>(
      `SELECT * FROM core_memory WHERE active = true ORDER BY tier ASC, priority DESC`,
    );
  } catch (e) {
    swallow.warn("surreal:return [];", e);
    return [];
  }
}

export async function createCoreMemory(
  text: string,
  category: string,
  priority: number,
  tier: number,
  sessionId?: string,
): Promise<string> {
  const record: Record<string, unknown> = { text, category, priority, tier, active: true };
  if (sessionId) record.session_id = sessionId;
  const rows = await queryFirst<{ id: string }>(
    `CREATE core_memory CONTENT $record RETURN id`,
    { record },
  );
  const id = String(rows[0]?.id ?? "");
  if (!id) throw new Error("createCoreMemory: CREATE returned no ID — write may have failed");
  return id;
}

export async function updateCoreMemory(
  id: string,
  fields: Partial<Pick<CoreMemoryEntry, "text" | "category" | "priority" | "tier" | "active">>,
): Promise<boolean> {
  assertRecordId(id);
  const sets: string[] = [];
  const bindings: Record<string, unknown> = {};
  for (const [key, val] of Object.entries(fields)) {
    if (val !== undefined) {
      sets.push(`${key} = $${key}`);
      bindings[key] = val;
    }
  }
  if (sets.length === 0) return false;
  sets.push("updated_at = time::now()");
  const rows = await queryFirst<{ id: string }>(
    `UPDATE ${id} SET ${sets.join(", ")} RETURN id`,
    bindings,
  );
  return rows.length > 0;
}

export async function deleteCoreMemory(id: string): Promise<void> {
  assertRecordId(id);
  await queryExec(`UPDATE ${id} SET active = false, updated_at = time::now()`);
}

export async function deactivateSessionMemories(sessionId: string): Promise<void> {
  try {
    await queryExec(
      `UPDATE core_memory SET active = false, updated_at = time::now() WHERE session_id = $sid AND tier = 1`,
      { sid: sessionId },
    );
  } catch (e) {
    swallow.warn("surreal:deactivateSessionMemories", e);
  }
}

// --- Memory Lifecycle: Session-Retrieved Memories ---

export async function getSessionRetrievedMemories(sessionId: string): Promise<{ id: string; text: string }[]> {
  try {
    // Get distinct memory IDs that were surfaced during this session
    const rows = await queryFirst<{ memory_id: string }>(
      `SELECT memory_id FROM retrieval_outcome WHERE session_id = $sid AND memory_table = 'memory' GROUP BY memory_id`,
      { sid: sessionId },
    );
    if (rows.length === 0) return [];

    const ids = rows.map(r => r.memory_id).filter(Boolean);
    if (ids.length === 0) return [];

    // Fetch the actual memory text for each ID
    // Use direct interpolation — parameterized $ids doesn't resolve as record references
    const validIds = ids.filter(id => { try { assertRecordId(id); return true; } catch { return false; } });
    if (validIds.length === 0) return [];
    const idList = validIds.map(id => `${id}`).join(", ");
    return await queryFirst<{ id: string; text: string }>(
      `SELECT id, text FROM memory WHERE id IN [${idList}] AND (status = 'active' OR status IS NONE)`,
    );
  } catch (e) {
    swallow.warn("surreal:getSessionRetrievedMemories", e);
    return [];
  }
}

// --- Memory Substrate: Compaction Checkpoints ---

export async function createCompactionCheckpoint(
  sessionId: string,
  rangeStart: number,
  rangeEnd: number,
): Promise<string> {
  const rows = await queryFirst<{ id: string }>(
    `CREATE compaction_checkpoint CONTENT $data RETURN id`,
    { data: { session_id: sessionId, msg_range_start: rangeStart, msg_range_end: rangeEnd, status: "pending" } },
  );
  return String(rows[0]?.id ?? "");
}

export async function completeCompactionCheckpoint(checkpointId: string, memoryId: string): Promise<void> {
  assertRecordId(checkpointId);
  await queryExec(`UPDATE ${checkpointId} SET status = "complete", memory_id = $mid`, { mid: memoryId });
}

export async function failCompactionCheckpoint(checkpointId: string): Promise<void> {
  assertRecordId(checkpointId);
  await queryExec(`UPDATE ${checkpointId} SET status = "failed"`);
}

export async function getPendingCheckpoints(sessionId: string): Promise<{ id: string; msg_range_start: number; msg_range_end: number }[]> {
  return await queryFirst<{ id: string; msg_range_start: number; msg_range_end: number }>(
    `SELECT id, msg_range_start, msg_range_end FROM compaction_checkpoint WHERE session_id = $sid AND (status = "pending" OR status = "failed")`,
    { sid: sessionId },
  );
}

// --- Memory Substrate: Utility Cache ---

export async function updateUtilityCache(memoryId: string, utilization: number): Promise<void> {
  try {
    // Upsert: increment count, running average
    await queryExec(
      `UPSERT memory_utility_cache SET
        memory_id = $mid,
        retrieval_count += 1,
        avg_utilization = IF retrieval_count > 1
          THEN (avg_utilization * (retrieval_count - 1) + $util) / retrieval_count
          ELSE $util
        END,
        last_updated = time::now()
       WHERE memory_id = $mid`,
      { mid: memoryId, util: utilization },
    );
  } catch (e) {
    swallow.warn("surreal:relate", e);
    // non-critical
  }
}

export interface UtilityCacheEntry {
  avg_utilization: number;
  retrieval_count: number;
}

export async function getUtilityFromCache(ids: string[]): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  if (ids.length === 0) return result;
  try {
    const rows = await queryFirst<{ memory_id: string; avg_utilization: number }>(
      `SELECT memory_id, avg_utilization FROM memory_utility_cache WHERE memory_id IN $ids`,
      { ids },
    );
    for (const row of rows) {
      if (row.avg_utilization != null) result.set(String(row.memory_id), row.avg_utilization);
    }
  } catch (e) { swallow.warn("surreal:non-critical", e); }
  return result;
}

/** Extended utility cache lookup that includes retrieval count for pre-filtering */
export async function getUtilityCacheEntries(ids: string[]): Promise<Map<string, UtilityCacheEntry>> {
  const result = new Map<string, UtilityCacheEntry>();
  if (ids.length === 0) return result;
  try {
    const rows = await queryFirst<{ memory_id: string; avg_utilization: number; retrieval_count: number }>(
      `SELECT memory_id, avg_utilization, retrieval_count FROM memory_utility_cache WHERE memory_id IN $ids`,
      { ids },
    );
    for (const row of rows) {
      if (row.avg_utilization != null) {
        result.set(String(row.memory_id), {
          avg_utilization: row.avg_utilization,
          retrieval_count: row.retrieval_count ?? 0,
        });
      }
    }
  } catch (e) { swallow.warn("surreal:non-critical", e); }
  return result;
}

// --- Memory Substrate: Importance Maintenance ---

export async function runMemoryMaintenance(): Promise<void> {
  try {
    // Importance floor: decay but never below 2.0
    await queryExec(`UPDATE memory SET importance = math::max([importance * 0.95, 2.0]) WHERE importance > 2.0`);
    // Boost memories with proven utility
    await queryExec(
      `UPDATE memory SET importance = math::max([importance, 3 + ((
        SELECT VALUE avg_utilization FROM memory_utility_cache WHERE memory_id = string::concat(meta::tb(id), ":", meta::id(id)) LIMIT 1
      )[0] ?? 0) * 4]) WHERE importance < 7`,
    );
  } catch (e) {
    swallow.warn("surreal:bumpAccessCounts", e);
    // maintenance failures are non-critical
  }
}

// --- Memory Substrate: Garbage Collection ---
// Prune old memories that are low-importance, never accessed, and have no proven utility.
// These accumulate from compaction/causal extraction and dilute retrieval quality.

export async function garbageCollectMemories(): Promise<number> {
  try {
    const countRows = await queryFirst<{ count: number }>(`SELECT count() AS count FROM memory GROUP ALL`);
    const count = countRows[0]?.count ?? 0;
    // Only GC when we have significant volume
    if (count <= 200) return 0;

    // Tier 1: Never-accessed stale memories (>14 days, low importance, no utility)
    const pruned = await db().query(
      `LET $stale = (
        SELECT id FROM memory
        WHERE created_at < time::now() - 14d
          AND importance <= 2.0
          AND (access_count = 0 OR access_count IS NONE)
          AND string::concat("memory:", id) NOT IN (
            SELECT VALUE memory_id FROM (
              SELECT memory_id FROM retrieval_outcome
              WHERE utilization > 0.2
              GROUP BY memory_id
            )
          )
        LIMIT 50
      );
      -- Cascade: clean graph edges before deleting nodes
      FOR $m IN $stale {
        DELETE FROM caused_by WHERE in = $m.id OR out = $m.id;
        DELETE FROM supports WHERE in = $m.id OR out = $m.id;
        DELETE FROM contradicts WHERE in = $m.id OR out = $m.id;
        DELETE FROM about_concept WHERE in = $m.id;
        DELETE FROM supersedes WHERE in = $m.id;
        DELETE FROM describes WHERE in = $m.id OR out = $m.id;
        DELETE $m.id;
      };
      RETURN array::len($stale);`,
    );
    const tier1 = Number(pruned ?? 0);

    // Tier 2: Proven-useless memories (retrieved 5+ times, <2% utilization)
    const provenBad = await db().query(
      `LET $dead = (
        SELECT VALUE memory_id FROM memory_utility_cache
        WHERE retrieval_count >= 5 AND avg_utilization < 0.02
        LIMIT 30
      );
      FOR $mid IN $dead {
        DELETE FROM caused_by WHERE in = type::thing("memory", $mid) OR out = type::thing("memory", $mid);
        DELETE FROM supports WHERE in = type::thing("memory", $mid) OR out = type::thing("memory", $mid);
        DELETE FROM contradicts WHERE in = type::thing("memory", $mid) OR out = type::thing("memory", $mid);
        DELETE FROM about_concept WHERE in = type::thing("memory", $mid);
        DELETE FROM supersedes WHERE in = type::thing("memory", $mid);
        DELETE type::thing("memory", $mid);
        DELETE FROM memory_utility_cache WHERE memory_id = $mid;
      };
      RETURN array::len($dead);`,
    );
    const tier2 = Number(provenBad ?? 0);

    return tier1 + tier2;
  } catch (e) {
    swallow.warn("surreal:return 0;", e);
    return 0;
  }
}

/** Prune superseded concepts that are at stability floor and never accessed. */
export async function garbageCollectConcepts(): Promise<number> {
  try {
    const pruned = await db().query(
      `LET $dead = (
        SELECT id FROM concept
        WHERE superseded_at IS NOT NONE
          AND stability <= 0.15
          AND (access_count = 0 OR access_count IS NONE)
          AND created_at < time::now() - 30d
        LIMIT 20
      );
      FOR $c IN $dead {
        DELETE FROM narrower WHERE in = $c.id OR out = $c.id;
        DELETE FROM broader WHERE in = $c.id OR out = $c.id;
        DELETE FROM related_to WHERE in = $c.id OR out = $c.id;
        DELETE FROM about_concept WHERE out = $c.id;
        DELETE FROM mentions WHERE out = $c.id;
        DELETE FROM supersedes WHERE out = $c.id;
        DELETE FROM artifact_mentions WHERE out = $c.id;
        DELETE $c.id;
      };
      RETURN array::len($dead);`,
    );
    return Number(pruned ?? 0);
  } catch (e) {
    swallow.warn("surreal:garbageCollectConcepts", e);
    return 0;
  }
}

// --- Memory Substrate: Turn Archival ---

export async function archiveOldTurns(): Promise<number> {
  try {
    const countRows = await queryFirst<{ count: number }>(`SELECT count() AS count FROM turn GROUP ALL`);
    const count = countRows[0]?.count ?? 0;
    if (count <= 2000) return 0;

    // Archive turns older than 7 days that were never retrieved
    const archived = await queryMulti<number>(
      `LET $stale = (SELECT id FROM turn WHERE timestamp < time::now() - 7d AND id NOT IN (SELECT VALUE memory_id FROM retrieval_outcome WHERE memory_table = 'turn'));
       FOR $t IN $stale {
         INSERT INTO turn_archive (SELECT * FROM ONLY $t.id);
         DELETE $t.id;
       };
       RETURN array::len($stale);`,
    );
    return Number(archived ?? 0);
  } catch (e) {
    swallow.warn("surreal:return 0;", e);
    return 0;
  }
}

// --- Memory Substrate: Consolidation ---

export async function consolidateMemories(): Promise<number> {
  try {
    const countRows = await queryFirst<{ count: number }>(`SELECT count() AS count FROM memory GROUP ALL`);
    const count = countRows[0]?.count ?? 0;
    if (count <= 50) return 0;

    let merged = 0;

    // Pass 1: Vector similarity dedup (accurate, catches paraphrases)
    // For each memory with an embedding, find its nearest neighbor in the same category
    const embMemories = await queryFirst<{
      id: string; text: string; importance: number; category: string;
      access_count: number; embedding: number[];
    }>(
      `SELECT id, text, importance, category, access_count, embedding, created_at
       FROM memory
       WHERE embedding != NONE AND array::len(embedding) > 0
       ORDER BY created_at ASC
       LIMIT 50`,
    );

    const seen = new Set<string>();
    for (const mem of embMemories) {
      if (seen.has(String(mem.id))) continue;

      // Find near-duplicates via vector similarity
      const dupes = await queryFirst<{
        id: string; importance: number; access_count: number; score: number;
      }>(
        `SELECT id, importance, access_count,
                vector::similarity::cosine(embedding, $vec) AS score
         FROM memory
         WHERE id != $mid
           AND category = $cat
           AND embedding != NONE AND array::len(embedding) > 0
         ORDER BY score DESC
         LIMIT 3`,
        { vec: mem.embedding, mid: mem.id, cat: mem.category },
      );

      for (const dupe of dupes) {
        if (dupe.score < 0.88) break; // below threshold, stop
        if (seen.has(String(dupe.id))) continue;

        // Keep the one with higher importance, break ties by access count
        const keepMem = (mem.importance > dupe.importance) ||
          (mem.importance === dupe.importance && (mem.access_count ?? 0) >= (dupe.access_count ?? 0));
        const [keep, drop] = keepMem ? [mem.id, dupe.id] : [dupe.id, mem.id];
        assertRecordId(String(keep));
        assertRecordId(String(drop));
        await queryExec(`UPDATE ${keep} SET access_count += 1, importance = math::max([importance, $imp])`, { imp: dupe.importance });
        await queryExec(`DELETE ${drop}`);
        seen.add(String(drop));
        merged++;
      }
    }

    // Pass 2: Backfill embeddings for memories missing them, then vector-dedup
    const unembedded = await queryFirst<{ id: string; text: string; importance: number; category: string; access_count: number }>(
      `SELECT id, text, importance, category, access_count
       FROM memory
       WHERE embedding IS NONE OR array::len(embedding) = 0
       LIMIT 20`,
    );
    for (const mem of unembedded) {
      if (seen.has(String(mem.id))) continue;
      try {
        const emb = await embed(mem.text);
        if (!emb) continue;
        assertRecordId(String(mem.id));
        await queryExec(`UPDATE ${mem.id} SET embedding = $emb`, { emb });

        // Now find near-duplicates via vector similarity (same as Pass 1)
        const dupes = await queryFirst<{ id: string; importance: number; access_count: number; score: number }>(
          `SELECT id, importance, access_count,
                  vector::similarity::cosine(embedding, $vec) AS score
           FROM memory
           WHERE id != $mid
             AND category = $cat
             AND embedding != NONE AND array::len(embedding) > 0
           ORDER BY score DESC
           LIMIT 3`,
          { vec: emb, mid: mem.id, cat: mem.category },
        );
        for (const dupe of dupes) {
          if (dupe.score < 0.88) break;
          if (seen.has(String(dupe.id))) continue;
          const keepMem = (mem.importance > dupe.importance) ||
            (mem.importance === dupe.importance && (mem.access_count ?? 0) >= (dupe.access_count ?? 0));
          const [keep, drop] = keepMem ? [mem.id, dupe.id] : [dupe.id, mem.id];
          assertRecordId(String(keep));
          assertRecordId(String(drop));
          await queryExec(`UPDATE ${keep} SET access_count += 1, importance = math::max([importance, $imp])`, { imp: dupe.importance });
          await queryExec(`DELETE ${drop}`);
          seen.add(String(drop));
          merged++;
        }
      } catch (e) { swallow.warn("surreal:consolidate-backfill", e); }
    }

    return merged;
  } catch (e) {
    swallow.warn("surreal:return 0;", e);
    return 0;
  }
}

// ── Reflection session lookup (cached per process) ────────────────────────
let _reflectionSessions: Set<string> | null = null;

/** Returns session IDs that have reflections. Cached per process lifetime. */
export async function getReflectionSessionIds(): Promise<Set<string>> {
  if (_reflectionSessions) return _reflectionSessions;
  try {
    const rows = await queryFirst<{ session_id: string }>(
      `SELECT session_id FROM reflection GROUP BY session_id`,
    );
    _reflectionSessions = new Set(rows.map(r => r.session_id).filter(Boolean));
  } catch (e) {
    swallow.warn("surreal:getReflectionSessionIds", e);
    _reflectionSessions = new Set();
  }
  return _reflectionSessions;
}

// ── Fibonacci Resurfacing ──────────────────────────────────────────────
// Memories flagged surfaceable get proactively injected when "due".
// Fibonacci intervals: 1,1,2,3,5,8,13,21,34,55,89 days between resurfaces.
// User engagement resets the index; ignoring it stretches the gap.

const FIB_DAYS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];

function fibDays(index: number): number {
  return index < FIB_DAYS.length ? FIB_DAYS[index] : FIB_DAYS[FIB_DAYS.length - 1];
}

/** Flag a memory for Fibonacci resurfacing. Sets next_surface_at to now + 1 day. */
export async function markSurfaceable(memoryId: string): Promise<void> {
  await queryExec(
    `UPDATE $id SET surfaceable = true, fib_index = 0, surface_count = 0, next_surface_at = time::now() + 1d`,
    { id: memoryId },
  );
}

/** Get memories that are due for resurfacing (next_surface_at <= now). */
export async function getDueMemories(limit = 5): Promise<{ id: string; text: string; importance: number; fib_index: number; surface_count: number; created_at: string }[]> {
  return (await queryFirst<any>(
    `SELECT id, text, importance, fib_index, surface_count, created_at
     FROM memory
     WHERE surfaceable = true
       AND next_surface_at <= time::now()
       AND status = 'active'
     ORDER BY importance DESC
     LIMIT $lim`,
    { lim: limit },
  )) ?? [];
}

/** Record that a surfaced memory was shown but user did NOT engage. Advance Fibonacci. */
export async function advanceSurfaceFade(memoryId: string): Promise<void> {
  // Get current fib_index, advance it, calculate next surface date
  const current = await queryFirst<{ fib_index: number }[]>(
    `SELECT fib_index FROM $id`, { id: memoryId },
  );
  const rows = current as any; const idx = (Array.isArray(rows) ? rows[0]?.fib_index : (rows as any)?.fib_index) ?? 0;
  const nextIdx = Math.min(idx + 1, FIB_DAYS.length - 1);
  const days = fibDays(nextIdx);
  await queryExec(
    `UPDATE $id SET fib_index = $nextIdx, surface_count += 1, last_surfaced = time::now(), next_surface_at = time::now() + type::duration($dur)`,
    { id: memoryId, nextIdx, dur: `${days}d` },
  );
}

/** User engaged with a surfaced memory — resolve it. It served its purpose.
 *  The memory stays in the graph for retrieval but exits the resurface queue permanently.
 *  Whatever emerges from the conversation becomes its own thing through normal extraction. */
export async function resolveSurfaceMemory(memoryId: string, outcome: "engaged" | "dismissed"): Promise<void> {
  await queryExec(
    `UPDATE $id SET surfaceable = false, last_engaged = time::now(), surface_outcome = $outcome`,
    { id: memoryId, outcome },
  );
}
