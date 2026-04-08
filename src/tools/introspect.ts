import { Type, type Static } from "@sinclair/typebox";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";
import type { ToolFactory } from "./types.js";
import {
  isSurrealAvailable, getSurrealInfo, pingDb, assertRecordId,
  queryFirst, getDb,
} from "../surreal.js";
import { getRecentErrors } from "../errors.js";

// Tables the agent is allowed to inspect
const ALLOWED_TABLES = new Set([
  "agent", "project", "task", "artifact", "concept",
  "turn", "identity_chunk", "session", "memory",
  "core_memory", "monologue", "skill", "reflection",
  "retrieval_outcome", "orchestrator_metrics",
  "causal_chain", "compaction_checkpoint", "subagent",
  "memory_utility_cache",
]);

// Tables that have HNSW vector indexes
const VECTOR_TABLES = new Set([
  "concept", "memory", "artifact", "identity_chunk", "turn", "monologue", "skill", "reflection",
]);

// Predefined filters for count action
const COUNT_FILTERS: Record<string, string> = {
  active: "WHERE active = true",
  inactive: "WHERE active = false",
  recent_24h: "WHERE created_at > time::now() - 24h",
  with_embedding: "WHERE embedding != NONE AND array::len(embedding) > 0",
  unresolved: "WHERE status != 'resolved' OR status IS NONE",
};

// Predefined safe query templates
const QUERY_TEMPLATES: Record<string, { sql: string; description: string; needsTable?: boolean }> = {
  recent: {
    sql: "SELECT id, text, content, description, created_at FROM type::table($t) ORDER BY created_at DESC LIMIT 5",
    description: "Last 5 records by creation time",
    needsTable: true,
  },
  sessions: {
    sql: "SELECT id, started_at, turn_count, total_input_tokens, total_output_tokens, last_active FROM session ORDER BY started_at DESC LIMIT 10",
    description: "Last 10 sessions with stats",
  },
  core_by_category: {
    sql: "SELECT category, count() AS count FROM core_memory WHERE active = true GROUP BY category",
    description: "Core memory entries grouped by category",
  },
  memory_status: {
    sql: "SELECT status, count() AS count FROM memory GROUP BY status",
    description: "Memory counts grouped by status",
  },
  embedding_coverage: {
    sql: "", // handled specially
    description: "Per-table embedding vs total counts",
  },
};

const introspectSchema = Type.Object({
  action: Type.Union([
    Type.Literal("status"),
    Type.Literal("count"),
    Type.Literal("verify"),
    Type.Literal("query"),
    Type.Literal("errors"),
  ], { description: "Action: status (health overview), count (row counts), verify (confirm record), query (predefined reports), errors (recent swallowed errors)." }),
  table: Type.Optional(Type.String({ description: "Table name for count/query actions." })),
  filter: Type.Optional(Type.String({ description: "For count: active, inactive, recent_24h, with_embedding, unresolved. For query: template name." })),
  record_id: Type.Optional(Type.String({ description: "Record ID for verify action (e.g. memory:abc123)." })),
});

type IntrospectParams = Static<typeof introspectSchema>;

export const createTool: ToolFactory = (ctx) => ({
  name: "introspect",
  label: "introspect",
  description: "Inspect your memory database. Use this for ALL database queries — NEVER use curl, bash, or HTTP to access SurrealDB directly. Your DB runs on a managed connection. Actions: status (health + table counts), count (filtered row counts), verify (confirm record exists), query (predefined reports).",
  parameters: introspectSchema,
  execute: async (_toolCallId: string, params: IntrospectParams): Promise<AgentToolResult<any>> => {
    if (!(await isSurrealAvailable())) {
      return { content: [{ type: "text", text: "Database unavailable." }], details: null };
    }

    try {
      switch (params.action) {
        case "status": return await statusAction(ctx.sessionId);
        case "count": return await countAction(params.table, params.filter);
        case "verify": return await verifyAction(params.record_id);
        case "query": return await queryAction(params.table, params.filter);
        case "errors": return errorsAction();
      }
    } catch (err) {
      return { content: [{ type: "text", text: `Introspect failed: ${err}` }], details: null };
    }
  },
});

// ── Actions ──────────────────────────────────────────────────────────────

async function statusAction(sessionId: string): Promise<AgentToolResult<any>> {
  const info = getSurrealInfo();
  const alive = await pingDb();

  const lines: string[] = [];
  lines.push("MEMORY DATABASE STATUS");
  lines.push("═══════════════════════════════════");
  lines.push(`Connection:  ${info?.url ?? "unknown"}`);
  lines.push(`Namespace:   ${info?.ns ?? "unknown"}`);
  lines.push(`Database:    ${info?.db ?? "unknown"}`);
  lines.push(`Ping:        ${alive ? "OK" : "FAILED"}`);
  lines.push(`Session:     ${sessionId}`);
  lines.push("");

  // Table counts
  const counts: Record<string, number> = {};
  const embCounts: Record<string, number> = {};

  for (const t of ALLOWED_TABLES) {
    try {
      const rows = await queryFirst<{ count: number }>(
        `SELECT count() AS count FROM type::table($t) GROUP ALL`,
        { t },
      );
      counts[t] = rows[0]?.count ?? 0;
    } catch {
      counts[t] = -1; // table may not exist
    }
  }

  for (const t of VECTOR_TABLES) {
    try {
      const rows = await queryFirst<{ count: number }>(
        `SELECT count() AS count FROM type::table($t) WHERE embedding != NONE AND array::len(embedding) > 0 GROUP ALL`,
        { t },
      );
      embCounts[t] = rows[0]?.count ?? 0;
    } catch {
      embCounts[t] = 0;
    }
  }

  for (const t of ALLOWED_TABLES) {
    const c = counts[t];
    const label = (t + ":").padEnd(28);
    const countStr = c === -1 ? "error" : String(c).padStart(5);
    const embStr = VECTOR_TABLES.has(t) ? `  (${embCounts[t] ?? 0} embedded)` : "";
    lines.push(`  ${label}${countStr}${embStr}`);
  }

  const totalNodes = Object.values(counts).filter(c => c >= 0).reduce((a, b) => a + b, 0);
  const totalEmb = Object.values(embCounts).reduce((a, b) => a + b, 0);
  lines.push("");
  lines.push(`Total records:     ${totalNodes}`);
  lines.push(`Total embeddings:  ${totalEmb}`);

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    details: { counts, embCounts, alive, totalNodes, totalEmb },
  };
}

async function countAction(table?: string, filter?: string): Promise<AgentToolResult<any>> {
  if (!table) {
    return {
      content: [{ type: "text", text: `Error: 'table' is required. Available tables: ${[...ALLOWED_TABLES].sort().join(", ")}` }],
      details: null,
    };
  }

  if (!ALLOWED_TABLES.has(table)) {
    return {
      content: [{ type: "text", text: `Error: unknown table "${table}". Available: ${[...ALLOWED_TABLES].sort().join(", ")}` }],
      details: null,
    };
  }

  let whereClause = "";
  if (filter) {
    if (!COUNT_FILTERS[filter]) {
      return {
        content: [{ type: "text", text: `Error: unknown filter "${filter}". Available: ${Object.keys(COUNT_FILTERS).join(", ")}` }],
        details: null,
      };
    }
    whereClause = " " + COUNT_FILTERS[filter];
  }

  const rows = await queryFirst<{ count: number }>(
    `SELECT count() AS count FROM type::table($t)${whereClause} GROUP ALL`,
    { t: table },
  );
  const count = rows[0]?.count ?? 0;

  return {
    content: [{ type: "text", text: `${table}: ${count} rows${filter ? ` (filter: ${filter})` : ""}` }],
    details: { table, count, filter },
  };
}

async function verifyAction(recordId?: string): Promise<AgentToolResult<any>> {
  if (!recordId) {
    return { content: [{ type: "text", text: "Error: 'record_id' is required (e.g. memory:abc123)." }], details: null };
  }

  try {
    assertRecordId(recordId);
  } catch {
    return { content: [{ type: "text", text: `Error: invalid record ID format "${recordId}". Expected table:id pattern.` }], details: null };
  }

  // Safe: recordId is validated by assertRecordId regex
  const rows = await queryFirst<Record<string, unknown>>(
    `SELECT * FROM ${recordId}`,
  );

  if (rows.length === 0) {
    return {
      content: [{ type: "text", text: `Record not found: ${recordId}` }],
      details: { exists: false, id: recordId },
    };
  }

  const record = rows[0];
  // Strip embedding arrays to avoid bloating context
  const cleaned: Record<string, unknown> = {};
  for (const [key, val] of Object.entries(record)) {
    if (Array.isArray(val) && val.length > 100 && typeof val[0] === "number") {
      cleaned[key] = `[${val.length} dims]`;
    } else {
      cleaned[key] = val;
    }
  }

  const lines = Object.entries(cleaned)
    .map(([k, v]) => `  ${k}: ${typeof v === "string" ? v.slice(0, 300) : JSON.stringify(v)}`)
    .join("\n");

  return {
    content: [{ type: "text", text: `Record ${recordId}:\n${lines}` }],
    details: { exists: true, id: recordId, record: cleaned },
  };
}

async function queryAction(table?: string, template?: string): Promise<AgentToolResult<any>> {
  const tmpl = template ?? "";

  if (!QUERY_TEMPLATES[tmpl]) {
    const available = Object.entries(QUERY_TEMPLATES)
      .map(([k, v]) => `  ${k}${v.needsTable ? " (requires table)" : ""}: ${v.description}`)
      .join("\n");
    return {
      content: [{ type: "text", text: `Available query templates:\n${available}` }],
      details: { templates: Object.keys(QUERY_TEMPLATES) },
    };
  }

  const spec = QUERY_TEMPLATES[tmpl];

  if (spec.needsTable) {
    if (!table || !ALLOWED_TABLES.has(table)) {
      return {
        content: [{ type: "text", text: `Error: "${tmpl}" requires a valid table. Available: ${[...ALLOWED_TABLES].sort().join(", ")}` }],
        details: null,
      };
    }
  }

  // Special case: embedding_coverage
  if (tmpl === "embedding_coverage") {
    const lines: string[] = [];
    for (const t of VECTOR_TABLES) {
      try {
        const totalRows = await queryFirst<{ count: number }>(
          `SELECT count() AS count FROM type::table($t) GROUP ALL`, { t },
        );
        const embRows = await queryFirst<{ count: number }>(
          `SELECT count() AS count FROM type::table($t) WHERE embedding != NONE AND array::len(embedding) > 0 GROUP ALL`, { t },
        );
        const total = totalRows[0]?.count ?? 0;
        const emb = embRows[0]?.count ?? 0;
        const pct = total > 0 ? Math.round((emb / total) * 100) : 0;
        lines.push(`  ${(t + ":").padEnd(20)} ${emb}/${total} (${pct}%)`);
      } catch { /* skip */ }
    }
    return {
      content: [{ type: "text", text: `Embedding coverage:\n${lines.join("\n")}` }],
      details: null,
    };
  }

  const rows = await queryFirst<Record<string, unknown>>(
    spec.sql,
    table ? { t: table } : undefined,
  );

  if (rows.length === 0) {
    return { content: [{ type: "text", text: `No results for "${tmpl}"${table ? ` on ${table}` : ""}.` }], details: null };
  }

  // Format results — strip embeddings, truncate text
  const formatted = rows.map((r, i) => {
    const fields = Object.entries(r)
      .filter(([k]) => k !== "embedding")
      .map(([k, v]) => {
        if (typeof v === "string" && v.length > 200) return `${k}: ${v.slice(0, 200)}...`;
        return `${k}: ${JSON.stringify(v)}`;
      })
      .join(", ");
    return `${i + 1}. ${fields}`;
  }).join("\n");

  return {
    content: [{ type: "text", text: `${tmpl}${table ? ` (${table})` : ""}:\n${formatted}` }],
    details: { count: rows.length },
  };
}

function errorsAction(): AgentToolResult<any> {
  const errors = getRecentErrors();
  if (errors.length === 0) {
    return {
      content: [{ type: "text", text: "No swallowed errors this session." }],
      details: { count: 0 },
    };
  }

  const lines = errors.map((e) =>
    `[${e.ts.slice(11, 19)}] [${e.level.toUpperCase()}] ${e.context}: ${e.message.slice(0, 200)}`,
  );

  return {
    content: [{ type: "text", text: `Recent swallowed errors (${errors.length}):\n${lines.join("\n")}` }],
    details: { count: errors.length, errors },
  };
}
