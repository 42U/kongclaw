/**
 * Predictive Context Prefetching — Phase 7d
 *
 * After preflight classifies intent, predict 2-4 follow-up queries and fire
 * vector searches in the background during LLM thinking. Results are cached
 * in an LRU with 5-min TTL. On the next turn, graph-context checks the cache
 * first — if a hit matches (cosine > 0.85), skip the SurrealDB round-trip.
 */

import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { vectorSearch, graphExpand, isSurrealAvailable, type VectorSearchResult } from "./surreal.js";
import { findRelevantSkills, type Skill } from "./skills.js";
import { retrieveReflections, type Reflection } from "./reflection.js";
import type { IntentCategory } from "./intent.js";
import { swallow } from "./errors.js";

// --- Types ---

interface CacheEntry {
  queryVec: number[];
  results: VectorSearchResult[];
  skills: Skill[];
  reflections: Reflection[];
  timestamp: number;
}

// --- LRU Cache ---

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const MAX_CACHE_SIZE = 10;
const CACHE_HIT_THRESHOLD = 0.85; // cosine similarity threshold for cache hit

const warmCache = new Map<string, CacheEntry>();

// --- Hit rate telemetry ---
let _prefetchHits = 0;
let _prefetchMisses = 0;

export function recordPrefetchHit(): void { _prefetchHits++; }
export function recordPrefetchMiss(): void { _prefetchMisses++; }

export function getPrefetchHitRate(): { hits: number; misses: number; attempts: number; hitRate: number } {
  const attempts = _prefetchHits + _prefetchMisses;
  return { hits: _prefetchHits, misses: _prefetchMisses, attempts, hitRate: attempts > 0 ? _prefetchHits / attempts : 0 };
}

function evictStale(): void {
  const now = Date.now();
  for (const [key, entry] of warmCache) {
    if (now - entry.timestamp > CACHE_TTL_MS) {
      warmCache.delete(key);
    }
  }
  // LRU eviction if still over capacity
  while (warmCache.size > MAX_CACHE_SIZE) {
    const oldest = warmCache.keys().next().value;
    if (oldest) warmCache.delete(oldest);
  }
}

// --- Query Prediction ---

/**
 * Given user input and classified intent, generate 2-4 predicted follow-up queries.
 * Pattern-based — no LLM call.
 */
export function predictQueries(input: string, intent: IntentCategory): string[] {
  const queries: string[] = [];

  // Extract file paths from input
  const filePaths = input.match(/[\w./\\-]+\.\w{1,10}/g) ?? [];
  for (const fp of filePaths.slice(0, 2)) {
    queries.push(fp);
  }

  // Extract quoted terms or backtick terms
  const quoted = input.match(/[`"']([^`"']{3,60})[`"']/g) ?? [];
  for (const q of quoted.slice(0, 2)) {
    queries.push(q.replace(/[`"']/g, ""));
  }

  // Intent-specific predictions
  switch (intent) {
    case "code-debug":
      queries.push(`error ${extractKeyTerms(input)}`);
      queries.push(`fix ${extractKeyTerms(input)}`);
      break;
    case "code-write":
      queries.push(`implementation pattern ${extractKeyTerms(input)}`);
      queries.push(`test ${extractKeyTerms(input)}`);
      break;
    case "code-read":
      queries.push(`architecture ${extractKeyTerms(input)}`);
      break;
    case "multi-step":
      queries.push(`procedure ${extractKeyTerms(input)}`);
      queries.push(`workflow ${extractKeyTerms(input)}`);
      break;
    case "reference-prior":
      queries.push(extractKeyTerms(input));
      break;
    default:
      // No additional predictions for simple/meta/continuation
      break;
  }

  // Deduplicate and limit
  const unique = [...new Set(queries.filter((q) => q.length > 3))];
  return unique.slice(0, 4);
}

/**
 * Extract key terms from input (capitalized words, technical terms).
 */
function extractKeyTerms(input: string): string {
  const words = input.split(/\s+/).filter((w) => {
    if (w.length < 3) return false;
    // Keep capitalized words, technical terms, not common words
    const STOP = new Set(["the", "and", "for", "with", "from", "this", "that", "have", "will", "can", "not", "are", "was", "but"]);
    return !STOP.has(w.toLowerCase());
  });
  return words.slice(0, 6).join(" ");
}

// --- Prefetching ---

/**
 * Fire vector searches in background for predicted queries.
 * Called from cli.ts after preflight, non-blocking.
 */
export async function prefetchContext(
  queries: string[],
  sessionId: string,
): Promise<void> {
  if (!isEmbeddingsAvailable() || !(await isSurrealAvailable())) return;
  if (queries.length === 0) return;

  evictStale();

  for (const query of queries) {
    try {
      const queryVec = await embed(query);

      // Vector search
      const results = await vectorSearch(queryVec, sessionId, {
        turn: 5, identity: 2, concept: 3, memory: 3, artifact: 2,
      });

      // Graph expand on top results
      const topIds = results
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
        .slice(0, 5)
        .map((r) => r.id);

      let neighbors: VectorSearchResult[] = [];
      if (topIds.length > 0) {
        try {
          const expanded = await graphExpand(topIds, queryVec);
          const existingIds = new Set(results.map((r) => r.id));
          neighbors = expanded.filter((n) => !existingIds.has(n.id));
        } catch (e) { swallow("prefetch:ok", e); }
      }

      // Also prefetch skills and reflections
      const [skills, reflections] = await Promise.all([
        findRelevantSkills(queryVec, 2).catch(() => [] as Skill[]),
        retrieveReflections(queryVec, 2).catch(() => [] as Reflection[]),
      ]);

      const entry: CacheEntry = {
        queryVec,
        results: [...results, ...neighbors],
        skills,
        reflections,
        timestamp: Date.now(),
      };

      warmCache.set(query, entry);
    } catch (e) {
      swallow("prefetch:silent", e);
      // Individual prefetch failure is non-critical
    }
  }
}

// --- Cache Lookup ---

/**
 * Cosine similarity between two vectors.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

export interface CachedContext {
  results: VectorSearchResult[];
  skills: Skill[];
  reflections: Reflection[];
}

/**
 * Check if any cached prefetch result is close enough (cosine > threshold)
 * to the actual query vector. Returns cached results or null.
 */
export function getCachedContext(queryVec: number[]): CachedContext | null {
  evictStale();

  let bestMatch: CacheEntry | null = null;
  let bestSim = 0;

  for (const [, entry] of warmCache) {
    const sim = cosineSimilarity(queryVec, entry.queryVec);
    if (sim > bestSim) {
      bestSim = sim;
      bestMatch = entry;
    }
  }

  if (bestMatch && bestSim >= CACHE_HIT_THRESHOLD) {
    return {
      results: bestMatch.results,
      skills: bestMatch.skills,
      reflections: bestMatch.reflections,
    };
  }

  return null;
}

/**
 * Get current cache stats for /stats display.
 */
export function getPrefetchStats(): { entries: number; maxSize: number } {
  evictStale();
  return { entries: warmCache.size, maxSize: MAX_CACHE_SIZE };
}

/**
 * Clear the prefetch cache (useful for testing).
 */
export function clearPrefetchCache(): void {
  warmCache.clear();
}
