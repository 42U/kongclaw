#!/usr/bin/env tsx
/**
 * Seed core identity chunks into the graph.
 * These are the hardcoded self-knowledge chunks that prevent Zeraclaw
 * from falling back to "I'm a stateless AI" patterns.
 */
import { initEmbeddings, disposeEmbeddings } from "./embeddings.js";
import { loadConfig } from "./config.js";
import { initSurreal, closeSurreal } from "./surreal.js";
import { seedIdentity } from "./identity.js";

async function main() {
  const config = loadConfig();

  console.log("Connecting to SurrealDB...");
  await initSurreal(config.surreal);

  console.log("Initializing BGE-M3 embeddings...");
  await initEmbeddings(config.embedding);

  console.log("Seeding identity chunks...");
  const count = await seedIdentity();

  if (count > 0) {
    console.log(`✅ ${count} identity chunks seeded with embeddings`);
  } else {
    console.log("ℹ️  Identity already seeded (idempotent — no duplicates created)");
  }

  await disposeEmbeddings();
  await closeSurreal();
}

main().catch(err => {
  console.error("Fatal:", err);
  process.exit(1);
});
