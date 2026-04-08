/**
 * Retrieval Quality Tracker
 *
 * Measures whether retrieved context was actually useful, not just relevant.
 * Tracks 6 signals from research:
 *
 * 1. Referenced in response — text overlap between injected context and model output
 * 2. Task success — tool executions after retrieval succeeded
 * 3. Retrieval stability — same context produces same retrievals (tracked over time)
 * 4. Access pattern — repeated retrieval across sessions (handled by access_count)
 * 5. Context waste — tokens injected but not referenced in response
 * 6. Contradiction — model response contradicts retrieved context
 */

import { getDb, updateUtilityCache, queryFirst, queryExec } from "./surreal.js";
import { swallow } from "./errors.js";

interface RetrievedItem {
  id: string;
  text: string;
  table: string;
  finalScore: number;
  fromNeighbor?: boolean;
  importance?: number;     // 0-10 scale raw importance
  accessCount?: number;    // raw access count
  timestamp?: string;      // created_at for recency calculation
}

interface QualitySignals {
  utilization: number;     // 0-1: how much of this memory appeared in the response
  toolSuccess: boolean | null; // true/false/null if no tools ran
  contextTokens: number;  // tokens this item consumed in context window
  wasNeighbor: boolean;    // came from graph expansion, not vector search
  recency: number;         // 0-1: how recent the memory is (for ACAN training)
}

// State for current turn — populated during retrieval, evaluated after response
let _pendingRetrieval: {
  sessionId: string;
  turnId: string;
  items: RetrievedItem[];
  toolResults: { success: boolean }[];
  queryEmbedding?: number[];
} | null = null;

/**
 * Snapshot of staged retrieval items for the cognitive check.
 * Must be called before evaluateRetrieval clears _pendingRetrieval.
 */
export function getStagedItems(): RetrievedItem[] {
  return _pendingRetrieval?.items ? [..._pendingRetrieval.items] : [];
}

/**
 * Called by graphTransformContext after selecting context nodes.
 * Stages the retrieval for quality evaluation once the response arrives.
 */
export function stageRetrieval(
  sessionId: string,
  items: RetrievedItem[],
  queryEmbedding?: number[],
): void {
  _pendingRetrieval = {
    sessionId,
    turnId: "",
    items,
    toolResults: [],
    queryEmbedding,
  };
}

/**
 * Called by agent event handler when a tool execution completes.
 * Records whether tool succeeded or failed (signal #2).
 */
export function recordToolOutcome(success: boolean): void {
  if (_pendingRetrieval) {
    _pendingRetrieval.toolResults.push({ success });
  }
}

/**
 * Called by agent event handler when assistant response is complete.
 * Computes all quality signals and persists retrieval_outcome records.
 */
export async function evaluateRetrieval(
  responseTurnId: string,
  responseText: string,
): Promise<void> {
  if (!_pendingRetrieval || _pendingRetrieval.items.length === 0) {
    _pendingRetrieval = null;
    return;
  }

  const { sessionId, items, toolResults, queryEmbedding } = _pendingRetrieval;
  _pendingRetrieval = null;

  // Signal #2: Overall tool success for this turn
  // Use majority-based success: mark as successful if >= 50% of tool calls
  // succeeded. The previous `every()` logic caused near-100% failure rates because
  // a single exploratory failure (e.g. file-not-found) would tank the whole turn.
  const toolSuccess = toolResults.length > 0
    ? toolResults.filter((r) => r.success).length / toolResults.length >= 0.5
    : null; // null = no tools were called

  const responseLower = responseText.toLowerCase();

  const db = getDb();

  for (const item of items) {
    const signals = computeSignals(item, responseLower, toolSuccess);

    try {
      // SurrealDB 3.0: option<bool> rejects JS null — omit the field entirely when null
      const record: Record<string, unknown> = {
        session_id: sessionId,
        turn_id: responseTurnId,
        memory_id: String(item.id),
        memory_table: item.table,
        retrieval_score: item.finalScore,
        utilization: signals.utilization,
        context_tokens: signals.contextTokens,
        was_neighbor: signals.wasNeighbor,
        // Store real feature values so ACAN trains on actual signals, not hardcoded defaults
        importance: (item.importance ?? 5) / 10,
        access_count: Math.min((item.accessCount ?? 0) / 50, 1),
        recency: signals.recency,
      };
      if (signals.toolSuccess != null) {
        record.tool_success = signals.toolSuccess;
      }
      if (queryEmbedding) {
        record.query_embedding = queryEmbedding;
      }
      await queryExec(`CREATE retrieval_outcome CONTENT $data`, { data: record }, db);
      // Update utility cache incrementally
      updateUtilityCache(String(item.id), signals.utilization).catch(e => swallow.warn("retrieval-quality:updateUtilityCache", e));
    } catch (err) {
      // silently drop — non-critical telemetry
    }
  }
}

/**
 * Compute quality signals for a single retrieved item.
 *
 * Signal #1 — Utilization: Three-tier overlap detection.
 * 1. Key terms — capitalized words, backtick-quoted, acronyms, camelCase,
 *    hyphenated terms. Robust to paraphrasing.
 * 2. Trigram overlap — catches broader content reuse (punctuation-stripped).
 * 3. Unigram overlap — fallback for short texts or heavy paraphrasing.
 *    Weighted at 0.5× to reflect weaker signal.
 * Combined: max(keyTermScore, trigramScore, unigramScore × 0.5).
 */
function computeSignals(
  item: RetrievedItem,
  responseLower: string,
  toolSuccess: boolean | null,
): QualitySignals {
  const rawText = item.text ?? "";
  const memText = rawText.toLowerCase();
  const contextTokens = Math.ceil(rawText.length / 4);

  const keyTermScore = keyTermOverlap(rawText, responseLower);
  const trigramScore = trigramOverlap(memText, responseLower);
  const unigramScore = unigramOverlap(memText, responseLower);
  const utilization = Math.max(keyTermScore, trigramScore, unigramScore * 0.5);

  // Recency: exponential decay from timestamp (matches graph-context.ts recencyScore)
  let recency = 0.5; // default if no timestamp
  if (item.timestamp) {
    const ageMs = Date.now() - new Date(item.timestamp).getTime();
    const ageHours = ageMs / 3_600_000;
    recency = Math.exp(-ageHours / 168); // half-life ~1 week
  }

  return {
    utilization,
    toolSuccess,
    contextTokens,
    wasNeighbor: item.fromNeighbor ?? false,
    recency,
  };
}

/** Strip punctuation that would prevent word matching across text boundaries */
function stripPunctuation(text: string): string {
  return text.replace(/[.,;:!?()"'\[\]{}<>—–…]/g, " ");
}

// Patterns for extracting key terms
const KEY_TERM_PATTERNS = [
  /`([^`]{2,60})`/g,                         // backtick-quoted terms
  /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b/g,   // Capitalized Multi Word
  /\b([A-Z]{2,}(?:[-_][A-Z0-9]+)*)\b/g,      // Acronyms: GCN, BGE-M3
  /\b([A-Z][a-z]*[A-Z]\w*)\b/g,              // camelCase/PascalCase: SurrealDB, graphContext
  /\b([A-Z][a-z]{2,})\b/g,                   // Single capitalized words: Zeraclaw, Python
  /\b(\w+(?:[-_]\w+){1,3})\b/g,              // hyphenated/underscored: graph-context, tool_result
];
const STOP_WORDS = new Set([
  "the", "a", "an", "but", "and", "or", "if", "when", "this", "that",
  "for", "with", "from", "into", "not", "are", "was", "were", "has",
  "have", "been", "its", "can", "will", "may", "also", "just", "then",
  "than", "too", "very", "such", "each", "all", "any", "most", "more",
  "some", "other", "about", "over", "only", "new", "used", "how", "where",
  "what", "which", "who", "whom", "does", "did", "had", "could", "would",
  "should", "shall", "let", "get", "got", "set", "put", "run", "see",
  "try", "use", "one", "two", "now", "way", "own", "same", "here",
  "there", "still", "yet", "both", "few", "many", "much", "well",
]);

function extractKeyTerms(text: string): Set<string> {
  const terms = new Set<string>();
  for (const pattern of KEY_TERM_PATTERNS) {
    for (const match of text.matchAll(pattern)) {
      const term = match[1].trim().toLowerCase();
      if (term.length >= 3 && !STOP_WORDS.has(term)) {
        terms.add(term);
      }
    }
  }
  return terms;
}

function keyTermOverlap(source: string, targetLower: string): number {
  const terms = extractKeyTerms(source);
  if (terms.size === 0) return 0;
  const cleanTarget = stripPunctuation(targetLower);
  let found = 0;
  for (const term of terms) {
    if (cleanTarget.includes(term)) found++;
  }
  return found / terms.size;
}

function trigramOverlap(source: string, target: string): number {
  const srcGrams = extractNgrams(stripPunctuation(source));
  if (srcGrams.size === 0) return 0;
  const tgtGrams = extractNgrams(stripPunctuation(target));
  let matches = 0;
  for (const gram of srcGrams) {
    if (tgtGrams.has(gram)) matches++;
  }
  return matches / srcGrams.size;
}

function extractNgrams(text: string): Set<string> {
  const words = text.split(/\s+/).filter((w) => w.length > 2);
  const grams = new Set<string>();
  if (words.length >= 3) {
    for (let i = 0; i <= words.length - 3; i++) {
      grams.add(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
    }
  } else if (words.length === 2) {
    grams.add(`${words[0]} ${words[1]}`);
  } else if (words.length === 1) {
    grams.add(words[0]);
  }
  return grams;
}

/**
 * Unigram (single-word) overlap — fallback for short texts or heavy paraphrasing.
 * Only considers words >= 5 chars to filter noise. Result is weighted down by
 * the caller (×0.5) since individual word matches are a weaker signal.
 */
function unigramOverlap(source: string, target: string): number {
  const srcWords = new Set(
    stripPunctuation(source).split(/\s+/)
      .filter((w) => w.length >= 5 && !STOP_WORDS.has(w)),
  );
  if (srcWords.size === 0) return 0;
  const cleanTarget = " " + stripPunctuation(target) + " ";
  let found = 0;
  for (const word of srcWords) {
    // Word boundary check via space padding to avoid partial matches
    if (cleanTarget.includes(` ${word} `) || cleanTarget.includes(` ${word}s `)) found++;
  }
  return found / srcWords.size;
}

// --- Query historical quality for scoring boost ---

/**
 * Get the average utilization score for a memory across past retrievals.
 * Used to boost memories with proven utility in the scoring function.
 * Returns null if no history exists.
 */
export async function getHistoricalUtility(memoryId: string): Promise<number | null> {
  const batch = await getHistoricalUtilityBatch([memoryId]);
  return batch.get(memoryId) ?? null;
}

/**
 * Batch-fetch historical utility for multiple memory IDs in a single query.
 * Returns a Map of memoryId → average utilization (only entries with data).
 */
export async function getHistoricalUtilityBatch(ids: string[]): Promise<Map<string, number>> {
  const result = new Map<string, number>();
  if (ids.length === 0) return result;
  try {
    const db = getDb();
    const flat = await queryFirst<{ memory_id: string; avg: number }>(
      `SELECT memory_id,
        math::mean(IF llm_relevance != NONE THEN llm_relevance ELSE utilization END) AS avg
       FROM retrieval_outcome
       WHERE memory_id IN $ids AND (utilization > 0 OR llm_relevance != NONE)
       GROUP BY memory_id`,
      { ids },
    );
    for (const row of flat) {
      if (row.avg != null) {
        result.set(String(row.memory_id), row.avg);
      }
    }
  } catch (e) {
    swallow("retrieval-quality:silent", e);
    // non-critical — return empty map
  }
  return result;
}

/**
 * Get aggregate quality stats for the current session.
 * Useful for /stats display.
 */
export async function getSessionQualityStats(sessionId: string): Promise<{
  totalRetrievals: number;
  avgUtilization: number;
  wastedTokens: number;
  toolSuccessRate: number | null;
} | null> {
  try {
    const rows = await queryFirst<any>(
      `SELECT
        count() AS total,
        math::mean(utilization) AS avg_util,
        math::sum(IF utilization < 0.15 THEN context_tokens ELSE 0 END) AS wasted,
        math::mean(IF tool_success != NONE THEN IF tool_success THEN 1 ELSE 0 END ELSE NONE END) AS tool_rate
       FROM retrieval_outcome
       WHERE session_id = $sid
       GROUP ALL`,
      { sid: sessionId },
    );
    if (rows.length === 0 || !rows[0].total) return null;
    const r = rows[0];
    return {
      totalRetrievals: r.total ?? 0,
      avgUtilization: r.avg_util ?? 0,
      wastedTokens: r.wasted ?? 0,
      toolSuccessRate: Number.isFinite(r.tool_rate) ? r.tool_rate : null,
    };
  } catch (e) {
    swallow.warn("retrieval-quality:return null;", e);
    return null;
  }
}

/** Rolling average utilization from recent retrieval outcomes in this session. */
export async function getRecentUtilizationAvg(
  sessionId: string,
  windowSize = 10,
): Promise<number | null> {
  try {
    const rows = await queryFirst<{ avg: number }>(
      `SELECT math::mean(utilization) AS avg FROM (SELECT utilization, created_at FROM retrieval_outcome WHERE session_id = $sid ORDER BY created_at DESC LIMIT $lim)`,
      { sid: sessionId, lim: windowSize },
    );
    return rows[0]?.avg ?? null;
  } catch {
    return null;
  }
}
