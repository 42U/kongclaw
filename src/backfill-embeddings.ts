#!/usr/bin/env tsx
/**
 * One-time backfill: compute BGE-M3 embeddings for all records that have text but no vector.
 * Targets: concept, memory, artifact, identity_chunk, turn, skill, reflection, monologue
 */

import { initEmbeddings, embed, embedBatch, disposeEmbeddings } from "./embeddings.js";
import { loadConfig } from "./config.js";

const config = loadConfig();

const SURREAL_URL = config.surreal.httpUrl;
const SURREAL_AUTH = Buffer.from(`${config.surreal.user}:${config.surreal.pass}`).toString("base64");
const HEADERS = {
  "Accept": "application/json",
  "Content-Type": "text/plain",
  "surreal-ns": config.surreal.ns,
  "surreal-db": config.surreal.db,
  "Authorization": `Basic ${SURREAL_AUTH}`,
};

async function query(sql: string): Promise<any[]> {
  const res = await fetch(`${SURREAL_URL}`, {
    method: "POST",
    headers: HEADERS,
    body: sql,
  });
  return res.json() as Promise<any[]>;
}

interface BackfillTarget {
  table: string;
  textField: string;  // field containing the text to embed
  vectorField: string; // field to store the embedding in
}

const TARGETS: BackfillTarget[] = [
  { table: "concept",        textField: "content",     vectorField: "embedding" },
  { table: "memory",         textField: "text",        vectorField: "embedding" },
  { table: "artifact",       textField: "description", vectorField: "embedding" },
  { table: "identity_chunk", textField: "content",     vectorField: "embedding" },
  { table: "turn",           textField: "content",     vectorField: "embedding" },
  { table: "skill",          textField: "description", vectorField: "embedding" },
  { table: "reflection",     textField: "content",     vectorField: "embedding" },
  { table: "monologue",      textField: "content",     vectorField: "embedding" },
];

async function backfillTable(target: BackfillTarget): Promise<number> {
  // Find records with text but no embedding
  const results = await query(
    `SELECT id, ${target.textField} FROM ${target.table} WHERE ${target.textField} IS NOT NONE AND (${target.vectorField} IS NONE OR array::len(${target.vectorField}) = 0);`
  );
  
  const rows = results[0]?.result ?? [];
  if (rows.length === 0) {
    console.log(`  ${target.table}: no records need embeddings`);
    return 0;
  }
  
  console.log(`  ${target.table}: ${rows.length} records to embed...`);
  
  // Extract texts, truncating to ~8k chars for safety
  const texts = rows.map((r: any) => {
    const text = r[target.textField] ?? "";
    return typeof text === "string" ? text.slice(0, 8000) : String(text).slice(0, 8000);
  });
  
  // Embed in batches of 32
  const BATCH_SIZE = 32;
  let updated = 0;
  
  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batchRows = rows.slice(i, i + BATCH_SIZE);
    const batchTexts = texts.slice(i, i + BATCH_SIZE);
    
    const vectors = await embedBatch(batchTexts);
    
    // Build UPDATE statements
    const updates = batchRows.map((row: any, j: number) => {
      const vecStr = `[${vectors[j].join(",")}]`;
      return `UPDATE ${row.id} SET ${target.vectorField} = ${vecStr};`;
    }).join("\n");
    
    const updateResults = await query(updates);
    const okCount = updateResults.filter((r: any) => r.status === "OK").length;
    updated += okCount;
    
    if (i + BATCH_SIZE < rows.length) {
      process.stdout.write(`    batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(rows.length / BATCH_SIZE)} (${updated} updated)\r`);
    }
  }
  
  console.log(`  ${target.table}: ✅ ${updated} embeddings written`);
  return updated;
}

async function main() {
  console.log("Initializing BGE-M3 embeddings...");
  await initEmbeddings(config.embedding);
  
  // Quick sanity check
  const testVec = await embed("test");
  console.log(`Model loaded. Vector dimension: ${testVec.length}`);
  
  if (testVec.length !== 1024) {
    console.error(`❌ Expected 1024d, got ${testVec.length}d — aborting`);
    process.exit(1);
  }
  
  console.log("\nBackfilling embeddings...");
  let total = 0;
  
  for (const target of TARGETS) {
    try {
      total += await backfillTable(target);
    } catch (err) {
      console.error(`  ${target.table}: ❌ ${err}`);
    }
  }
  
  console.log(`\n✅ Done. ${total} total embeddings written.`);
  
  await disposeEmbeddings();
}

main().catch(err => {
  console.error("Fatal:", err);
  process.exit(1);
});
