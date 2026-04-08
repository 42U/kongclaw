#!/usr/bin/env tsx
/**
 * Seed Tier 0 core memory entries into the graph.
 * These are always-loaded directives that bypass vector search scoring —
 * the agent's bedrock knowledge that can never be evicted.
 *
 * Idempotent: clears and re-seeds if count mismatches.
 */
import { loadConfig } from "./config.js";
import { initSurreal, closeSurreal, queryFirst, createCoreMemory } from "./surreal.js";

interface SeedEntry {
  text: string;
  category: string;
  priority: number;
}

const TIER0_SEEDS: SeedEntry[] = [
  // ── Knowledge (priority 95) ──
  {
    text: "Knowledge: store every success and every failure, learn from both.",
    category: "knowledge",
    priority: 95,
  },

  // ── Operations (priority 90) ──
  {
    text: "Operations: think before you act. Do you already have a saved skill or knowledge gem for that? Then use it. If not, build it.",
    category: "operations",
    priority: 90,
  },

  // ── Intelligence (priority 85) ──
  {
    text: "Identity: be thoughtful, take your time to make sure you understand deeply before responding. When you respond, add value.",
    category: "intelligence",
    priority: 85,
  },

  // ── Graph (priority 80) ──
  {
    text: "Graph: save everything to the graph. It will help you develop better habits and routes for you to follow.",
    category: "graph",
    priority: 80,
  },

  // ── Network (priority 75) ──
  {
    text: "Network: networks aren't just computer packets getting from one place to another. Network the data you learn.",
    category: "network",
    priority: 75,
  },
];

async function main() {
  const config = loadConfig();

  console.log("Connecting to SurrealDB...");
  await initSurreal(config.surreal);

  // Check existing Tier 0 count
  const rows = await queryFirst<{ count: number }>(
    `SELECT count() AS count FROM core_memory WHERE tier = 0 AND active = true GROUP ALL`,
  );
  const existing = rows[0]?.count ?? 0;

  if (existing === TIER0_SEEDS.length) {
    console.log(`ℹ️  Tier 0 core memory already seeded (${existing} entries) — no changes.`);
    await closeSurreal();
    return;
  }

  // Clear and re-seed if count mismatch
  if (existing > 0) {
    console.log(`Clearing ${existing} existing Tier 0 entries (count mismatch: ${existing} vs ${TIER0_SEEDS.length})...`);
    await queryFirst(`DELETE core_memory WHERE tier = 0`);
  }

  console.log(`Seeding ${TIER0_SEEDS.length} Tier 0 core memory entries...`);
  for (const seed of TIER0_SEEDS) {
    const id = await createCoreMemory(seed.text, seed.category, seed.priority, 0);
    console.log(`  ✓ [${seed.category}] p${seed.priority}: ${seed.text.slice(0, 60)}... → ${id}`);
  }

  console.log(`\n✅ ${TIER0_SEEDS.length} Tier 0 entries seeded.`);
  await closeSurreal();
}

main().catch(err => {
  console.error("Fatal:", err);
  process.exit(1);
});
