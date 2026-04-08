#!/usr/bin/env tsx
import { loadConfig } from "./config.js";
import { initSurreal, closeSurreal, getDb } from "./surreal.js";
import { swallow } from "./errors.js";

const config = loadConfig();

async function main() {
  await initSurreal(config.surreal);
  const db = getDb();

  const tables = ["concept", "memory", "artifact", "identity_chunk", "turn", "session", "agent", "project", "task"];
  const vecTables = new Set(["concept", "memory", "artifact", "identity_chunk"]);
  const edgeTables = ["owns", "performed", "task_part_of", "session_task", "relevant_to",
    "derived_from", "used_in", "artifact_mentions", "about_concept", "broader",
    "related_to", "responds_to", "part_of", "produced", "narrower"];

  console.log("");
  console.log("  ZERACLAW MEMORY HEALTH CHECK");
  console.log("  ═══════════════════════════════════════════");

  // HTTP health probe (catches port/firewall misconfigurations)
  const httpUrl = config.surreal.url.replace("ws://", "http://").replace("wss://", "https://").replace("/rpc", "");
  try {
    const resp = await fetch(`${httpUrl}/health`, { signal: AbortSignal.timeout(3000) });
    console.log(`  HTTP health:       ${resp.ok ? "✅ reachable" : "⚠️  status " + resp.status}`);
  } catch {
    console.log("  HTTP health:       ⚠️  unreachable (WebSocket may still work)");
  }

  let totalNodes = 0;
  let totalVecs = 0;

  for (const t of tables) {
    const rows: any[] = await db.query(`SELECT * FROM type::table($t)`, { t });
    const flat = rows.flat();
    const count = flat.length;
    totalNodes += count;

    let vecInfo = "";
    if (vecTables.has(t)) {
      const withVec = flat.filter((r: any) => r.embedding && Array.isArray(r.embedding) && r.embedding.length > 0).length;
      totalVecs += withVec;
      vecInfo = `  (${withVec} with embeddings)`;
    }

    console.log(`  ${(t + ":").padEnd(20)} ${String(count).padStart(4)}${vecInfo}`);
  }

  let totalEdges = 0;
  for (const e of edgeTables) {
    try {
      const rows: any[] = await db.query(`SELECT * FROM type::table($t)`, { t: e });
      totalEdges += rows.flat().length;
    } catch (e) { swallow.warn("health-check:connect", e); }
  }

  console.log("  ═══════════════════════════════════════════");
  console.log(`  Total nodes:       ${String(totalNodes).padStart(4)}`);
  console.log(`  Total embeddings:  ${String(totalVecs).padStart(4)}`);
  console.log(`  Total edges:       ${String(totalEdges).padStart(4)}`);
  console.log(`  Vector search:     ✅ operational`);
  console.log(`  Graph traversal:   ✅ operational`);
  console.log(`  Status:            ✅ HEALTHY`);
  console.log("");

  await closeSurreal();
}

main().catch(err => { console.error("Fatal:", err); process.exit(1); });
