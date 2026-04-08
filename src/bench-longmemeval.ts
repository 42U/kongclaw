#!/usr/bin/env npx tsx
/**
 * bench-longmemeval.ts — LongMemEval benchmark harness for Zeraclaw
 *
 * Evaluates Zeraclaw's retrieval quality against the LongMemEval benchmark
 * (500 questions, 6 types) and compares head-to-head with MemPalace's results.
 *
 * No SurrealDB required — uses only BGE-M3 embeddings + pure math.
 *
 * Usage:
 *   npx tsx src/bench-longmemeval.ts /tmp/longmemeval-data/longmemeval_s_cleaned.json
 *   npx tsx src/bench-longmemeval.ts data.json --mode hybrid --limit 20
 *   npx tsx src/bench-longmemeval.ts data.json --all-modes
 *
 * Data:
 *   mkdir -p /tmp/longmemeval-data
 *   curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
 *     https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { parseArgs } from "node:util";
import { createHash } from "node:crypto";
import { initEmbeddings, embed, disposeEmbeddings } from "./embeddings.js";
import { getLlama, LlamaLogLevel, type LlamaRankingContext } from "node-llama-cpp";

// ── Types ──────────────────────────────────────────────────────────────────

interface LongMemEvalEntry {
  question_id: string;
  question_type: string;
  question: string;
  answer: string;
  question_date?: string;
  haystack_sessions: { role: string; content: string }[][];
  haystack_session_ids: string[];
  haystack_dates: string[];
  answer_session_ids: string[];
}

interface CorpusDoc {
  text: string;
  sessionId: string;
  timestamp: string;
  embedding: number[];
}

interface RankedResult {
  corpusIndex: number;
  score: number;
}

interface QuestionMetrics {
  [key: string]: number; // "recall_any@1", "ndcg@5", etc.
}

interface QuestionResult {
  question_id: string;
  question_type: string;
  question: string;
  answer: string;
  metrics: { session: QuestionMetrics };
}

type ScoringMode = "raw" | "wmr" | "hybrid" | "full" | "rerank" | "hybrid-rerank" | "fusion-rerank";
type Granularity = "session" | "turn" | "session-full";

interface BenchConfig {
  dataFile: string;
  mode: ScoringMode;
  granularity: Granularity;
  limit: number;
  skip: number;
  hybridWeight: number;
  outFile: string;
  cacheDir: string;
  allModes: boolean;
  rerankModelPath: string;
  rerankTopN: number;
}

// ── Metrics (ported from MemPalace longmemeval_bench.py:53-80) ─────────

function dcg(relevances: number[], k: number): number {
  let score = 0;
  for (let i = 0; i < Math.min(relevances.length, k); i++) {
    score += relevances[i] / Math.log2(i + 2);
  }
  return score;
}

function ndcg(
  rankings: number[],
  correctIds: Set<string>,
  corpusIds: string[],
  k: number,
): number {
  const relevances = rankings
    .slice(0, k)
    .map((idx) => (correctIds.has(corpusIds[idx]) ? 1.0 : 0.0));
  const ideal = [...relevances].sort((a, b) => b - a);
  const idcg = dcg(ideal, k);
  if (idcg === 0) return 0;
  return dcg(relevances, k) / idcg;
}

function evaluateRetrieval(
  rankings: number[],
  correctIds: Set<string>,
  corpusIds: string[],
  k: number,
): { recallAny: number; ndcgScore: number } {
  const topKIds = new Set(rankings.slice(0, k).map((idx) => corpusIds[idx]));
  const recallAny = [...correctIds].some((id) => topKIds.has(id)) ? 1.0 : 0.0;
  const ndcgScore = ndcg(rankings, correctIds, corpusIds, k);
  return { recallAny, ndcgScore };
}

// For session-level evaluation: extract session ID from corpus ID
function sessionIdFromCorpusId(corpusId: string): string {
  if (corpusId.includes("_turn_")) {
    return corpusId.split("_turn_")[0];
  }
  return corpusId;
}

// ── Corpus Builder ─────────────────────────────────────────────────────

function buildCorpus(
  entry: LongMemEvalEntry,
  granularity: Granularity,
): { texts: string[]; ids: string[]; timestamps: string[] } {
  const texts: string[] = [];
  const ids: string[] = [];
  const timestamps: string[] = [];

  const sessions = entry.haystack_sessions;
  const sessionIds = entry.haystack_session_ids;
  const dates = entry.haystack_dates;

  for (let i = 0; i < sessions.length; i++) {
    const session = sessions[i];
    const sessId = sessionIds[i];
    const date = dates[i];

    if (granularity === "session" || granularity === "session-full") {
      // "session" = user turns only (matches MemPalace baseline)
      // "session-full" = user doc + separate assistant doc, both map to same session ID
      //   This avoids diluting user signal with verbose assistant text
      const userTurns = session.filter((t) => t.role === "user").map((t) => t.content);
      if (userTurns.length > 0) {
        texts.push(userTurns.join("\n"));
        ids.push(sessId);
        timestamps.push(date);
      }
      if (granularity === "session-full") {
        const asstTurns = session.filter((t) => t.role === "assistant").map((t) => t.content);
        if (asstTurns.length > 0) {
          texts.push(asstTurns.join("\n"));
          ids.push(sessId); // same session ID — both docs count as a hit
          timestamps.push(date);
        }
      }
    } else {
      let turnNum = 0;
      for (const turn of session) {
        if (turn.role === "user") {
          texts.push(turn.content);
          ids.push(`${sessId}_turn_${turnNum}`);
          timestamps.push(date);
          turnNum++;
        }
      }
    }
  }

  return { texts, ids, timestamps };
}

// ── Embedding Cache ────────────────────────────────────────────────────

class EmbeddingCache {
  private cache = new Map<string, number[]>();
  private cacheDir: string;

  constructor(cacheDir: string) {
    this.cacheDir = cacheDir;
  }

  private hash(text: string): string {
    return createHash("sha256").update(text).digest("hex").slice(0, 16);
  }

  async embedWithCache(text: string): Promise<number[]> {
    const truncated = truncateForEmbed(text);
    const key = this.hash(truncated);
    const cached = this.cache.get(key);
    if (cached) return cached;
    const vec = await embed(truncated);
    this.cache.set(key, vec);
    return vec;
  }

  async embedBatchWithCache(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (const text of texts) {
      results.push(await this.embedWithCache(text));
    }
    return results;
  }

  loadFromDisk(questionId: string): boolean {
    if (!this.cacheDir) return false;
    const path = join(this.cacheDir, `${questionId}.json`);
    if (!existsSync(path)) return false;
    try {
      const data = JSON.parse(readFileSync(path, "utf-8"));
      for (const [k, v] of Object.entries(data)) {
        this.cache.set(k, v as number[]);
      }
      return true;
    } catch {
      return false;
    }
  }

  saveToDisk(questionId: string): void {
    if (!this.cacheDir) return;
    if (!existsSync(this.cacheDir)) mkdirSync(this.cacheDir, { recursive: true });
    const path = join(this.cacheDir, `${questionId}.json`);
    const data: Record<string, number[]> = {};
    for (const [k, v] of this.cache) {
      data[k] = v;
    }
    writeFileSync(path, JSON.stringify(data));
  }

  clear(): void {
    this.cache.clear();
  }
}

// ── Text Truncation ────────────────────────────────────────────────────
// BGE-M3 GGUF has a finite context window. Truncate long documents to
// avoid crashes and keep embeddings focused on key content.
// 2000 chars ≈ 500 tokens — generous enough to capture context,
// short enough to stay within model limits.
const MAX_EMBED_CHARS = 2000;

function truncateForEmbed(text: string): string {
  if (text.length <= MAX_EMBED_CHARS) return text;
  return text.slice(0, MAX_EMBED_CHARS);
}

// ── Query Prefix ───────────────────────────────────────────────────────
// Most retrieval embedding models use asymmetric encoding:
// queries get a prefix instruction, documents do not.
// Set via --query-prefix flag. Common prefixes:
//   BGE-M3:     "Represent this sentence for searching relevant passages: "
//   Snowflake:  "search_query: "   (documents get "search_document: ")
//   nomic:      "search_query: "
//   empty:      "" (no prefix — test baseline first)
let QUERY_PREFIX = "";
let DOC_PREFIX = "";

// ── Cosine Similarity ──────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ── Recency Score (from graph-context.ts:494-503) ──────────────────────

const RECENCY_DECAY_FAST = 0.99;
const RECENCY_DECAY_SLOW = 0.995;
const RECENCY_BOUNDARY_HOURS = 4;

function recencyScore(timestamp: string, referenceTime: number): number {
  if (!timestamp) return 0.3;
  const hoursElapsed =
    (referenceTime - new Date(timestamp).getTime()) / (1000 * 60 * 60);
  if (hoursElapsed <= 0) return 1.0; // future or same time
  if (hoursElapsed <= RECENCY_BOUNDARY_HOURS) {
    return Math.pow(RECENCY_DECAY_FAST, hoursElapsed);
  }
  const fastPart = Math.pow(RECENCY_DECAY_FAST, RECENCY_BOUNDARY_HOURS);
  return (
    fastPart *
    Math.pow(RECENCY_DECAY_SLOW, hoursElapsed - RECENCY_BOUNDARY_HOURS)
  );
}

// ── Keyword Extraction (ported from MemPalace) ─────────────────────────

const STOP_WORDS = new Set([
  "what", "when", "where", "who", "how", "which", "did", "do", "was", "were",
  "have", "has", "had", "is", "are", "the", "a", "an", "my", "me", "i", "you",
  "your", "their", "it", "its", "in", "on", "at", "to", "for", "of", "with",
  "by", "from", "ago", "last", "that", "this", "there", "about", "get", "got",
  "give", "gave", "buy", "bought", "made", "make", "and", "but", "not", "can",
  "will", "just", "than", "then", "also", "been", "being", "would", "could",
  "should", "may", "might", "shall", "into", "over", "after", "before",
]);

function extractKeywords(text: string): string[] {
  const matches = text.toLowerCase().match(/\b[a-z]{3,}\b/g) ?? [];
  return matches.filter((w) => !STOP_WORDS.has(w));
}

function keywordOverlap(queryKeywords: string[], docText: string): number {
  if (queryKeywords.length === 0) return 0;
  const docLower = docText.toLowerCase();
  let hits = 0;
  for (const kw of queryKeywords) {
    if (docLower.includes(kw)) hits++;
  }
  return hits / queryKeywords.length;
}

// ── Scoring Modes ──────────────────────────────────────────────────────

function scoreRaw(
  queryEmbedding: number[],
  corpus: CorpusDoc[],
): RankedResult[] {
  return corpus
    .map((doc, i) => ({
      corpusIndex: i,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
    }))
    .sort((a, b) => b.score - a.score);
}

function scoreWMR(
  queryEmbedding: number[],
  corpus: CorpusDoc[],
  referenceTime: number,
): RankedResult[] {
  // WMR from graph-context.ts:684-696 with benchmark defaults
  return corpus
    .map((doc, i) => {
      const cosine = cosineSimilarity(queryEmbedding, doc.embedding);
      const recency = recencyScore(doc.timestamp, referenceTime);
      const importance = 0.05; // default: 0.5/10
      const access = 0; // no history
      const neighborBonus = 0; // no graph
      const provenUtility = 0.35; // default neutral
      const reflectionBoost = 0; // no reflections

      const finalScore =
        0.27 * cosine +
        0.28 * recency +
        0.05 * importance +
        0.05 * access +
        0.10 * neighborBonus +
        0.15 * provenUtility +
        0.10 * reflectionBoost;

      return { corpusIndex: i, score: finalScore };
    })
    .sort((a, b) => b.score - a.score);
}

function scoreHybrid(
  queryEmbedding: number[],
  queryText: string,
  corpus: CorpusDoc[],
  hybridWeight: number,
): RankedResult[] {
  const queryKeywords = extractKeywords(queryText);

  return corpus
    .map((doc, i) => {
      const cosine = cosineSimilarity(queryEmbedding, doc.embedding);
      const overlap = keywordOverlap(queryKeywords, doc.text);
      // Boost similarity for keyword matches (inverse of MemPalace's distance reduction)
      const fusedScore = cosine * (1.0 + hybridWeight * overlap);
      return { corpusIndex: i, score: fusedScore };
    })
    .sort((a, b) => b.score - a.score);
}

function scoreFull(
  queryEmbedding: number[],
  queryText: string,
  corpus: CorpusDoc[],
  referenceTime: number,
  hybridWeight: number,
): RankedResult[] {
  const queryKeywords = extractKeywords(queryText);

  // First pass: WMR scoring to find top-5
  const wmrScored = corpus.map((doc, i) => {
    const cosine = cosineSimilarity(queryEmbedding, doc.embedding);
    const recency = recencyScore(doc.timestamp, referenceTime);
    const overlap = keywordOverlap(queryKeywords, doc.text);
    return { index: i, cosine, recency, overlap, timestamp: doc.timestamp };
  });
  wmrScored.sort((a, b) => b.cosine - a.cosine);

  // Simulate graph expansion: temporal neighbors of top-5 get neighborBonus
  const top5Timestamps = wmrScored.slice(0, 5).map((r) => new Date(r.timestamp).getTime());
  const TWO_DAYS_MS = 2 * 24 * 60 * 60 * 1000;
  const neighborSet = new Set<number>();
  for (const scored of wmrScored.slice(5)) {
    const ts = new Date(scored.timestamp).getTime();
    for (const topTs of top5Timestamps) {
      if (Math.abs(ts - topTs) <= TWO_DAYS_MS) {
        neighborSet.add(scored.index);
        break;
      }
    }
  }

  // Final scoring: WMR + neighbor bonus + keyword overlap
  return corpus
    .map((doc, i) => {
      const cosine = cosineSimilarity(queryEmbedding, doc.embedding);
      const recency = recencyScore(doc.timestamp, referenceTime);
      const overlap = keywordOverlap(queryKeywords, doc.text);
      const neighborBonus = neighborSet.has(i) ? 1.0 : 0;

      const finalScore =
        0.27 * cosine * (1.0 + hybridWeight * overlap) +
        0.28 * recency +
        0.05 * 0.05 + // importance default
        0.05 * 0 + // access default
        0.10 * neighborBonus +
        0.15 * 0.35 + // utility default
        0.10 * 0; // reflection default

      return { corpusIndex: i, score: finalScore };
    })
    .sort((a, b) => b.score - a.score);
}

// ── Cross-Encoder Reranking ─────────────────────────────────────────────
// Reranker context window is 8192 tokens. Query + document must fit together.
// Reserve ~200 tokens for query, leave ~8000 for document (~24000 chars).
const MAX_RERANK_DOC_CHARS = 24000;

async function scoreRerank(
  queryEmbedding: number[],
  queryText: string,
  corpus: CorpusDoc[],
  topN: number,
  rankCtx: LlamaRankingContext,
): Promise<RankedResult[]> {
  // Stage 1: cosine similarity
  const stage1 = scoreRaw(queryEmbedding, corpus);
  const candidates = stage1.slice(0, topN);

  // Stage 2: cross-encoder reranking on top-N (truncate long docs for context window)
  const candidateTexts = candidates.map((r) => {
    const text = corpus[r.corpusIndex].text;
    return text.length > MAX_RERANK_DOC_CHARS ? text.slice(0, MAX_RERANK_DOC_CHARS) : text;
  });
  const rerankScores = await rankCtx.rankAll(queryText, candidateTexts);

  // Re-sort by rerank scores
  const reranked = candidates
    .map((r, i) => ({ corpusIndex: r.corpusIndex, score: rerankScores[i] }))
    .sort((a, b) => b.score - a.score);

  // Append remaining documents beyond top-N in original order
  const rerankedSet = new Set(reranked.map((r) => r.corpusIndex));
  const tail = stage1.filter((r) => !rerankedSet.has(r.corpusIndex));
  return [...reranked, ...tail];
}

async function scoreHybridRerank(
  queryEmbedding: number[],
  queryText: string,
  corpus: CorpusDoc[],
  hybridWeight: number,
  topN: number,
  rankCtx: LlamaRankingContext,
): Promise<RankedResult[]> {
  // Stage 1: hybrid scoring (cosine + keyword overlap)
  const stage1 = scoreHybrid(queryEmbedding, queryText, corpus, hybridWeight);
  const candidates = stage1.slice(0, topN);

  // Stage 2: cross-encoder reranking (truncate long docs)
  const candidateTexts = candidates.map((r) => {
    const text = corpus[r.corpusIndex].text;
    return text.length > MAX_RERANK_DOC_CHARS ? text.slice(0, MAX_RERANK_DOC_CHARS) : text;
  });
  const rerankScores = await rankCtx.rankAll(queryText, candidateTexts);

  const reranked = candidates
    .map((r, i) => ({ corpusIndex: r.corpusIndex, score: rerankScores[i] }))
    .sort((a, b) => b.score - a.score);

  const rerankedSet = new Set(reranked.map((r) => r.corpusIndex));
  const tail = stage1.filter((r) => !rerankedSet.has(r.corpusIndex));
  return [...reranked, ...tail];
}

async function scoreFusionRerank(
  queryEmbedding: number[],
  queryText: string,
  corpus: CorpusDoc[],
  hybridWeight: number,
  topN: number,
  rankCtx: LlamaRankingContext,
): Promise<RankedResult[]> {
  // Stage 1: get top-N from BOTH raw cosine AND hybrid scoring
  const rawRanked = scoreRaw(queryEmbedding, corpus);
  const hybridRanked = scoreHybrid(queryEmbedding, queryText, corpus, hybridWeight);

  // Merge: union of top-N from each, deduplicated
  const seen = new Set<number>();
  const candidates: RankedResult[] = [];
  for (const r of rawRanked.slice(0, topN)) {
    if (!seen.has(r.corpusIndex)) {
      seen.add(r.corpusIndex);
      candidates.push(r);
    }
  }
  for (const r of hybridRanked.slice(0, topN)) {
    if (!seen.has(r.corpusIndex)) {
      seen.add(r.corpusIndex);
      candidates.push(r);
    }
  }

  // Stage 2: cross-encoder reranking on the merged set
  const candidateTexts = candidates.map((r) => {
    const text = corpus[r.corpusIndex].text;
    return text.length > MAX_RERANK_DOC_CHARS ? text.slice(0, MAX_RERANK_DOC_CHARS) : text;
  });
  const rerankScores = await rankCtx.rankAll(queryText, candidateTexts);

  const reranked = candidates
    .map((r, i) => ({ corpusIndex: r.corpusIndex, score: rerankScores[i] }))
    .sort((a, b) => b.score - a.score);

  // Append tail from raw ranking
  const rerankedSet = new Set(reranked.map((r) => r.corpusIndex));
  const tail = rawRanked.filter((r) => !rerankedSet.has(r.corpusIndex));
  return [...reranked, ...tail];
}

// ── Parse question_date to epoch ms ────────────────────────────────────

function parseQuestionDate(dateStr?: string): number {
  if (!dateStr) return Date.now();
  // Format: "2023/01/15 (Sun) 10:20" or similar
  const cleaned = dateStr.replace(/\s*\([^)]*\)\s*/, " ").trim();
  const parsed = new Date(cleaned.replace(/\//g, "-"));
  return isNaN(parsed.getTime()) ? Date.now() : parsed.getTime();
}

// ── Benchmark Runner ───────────────────────────────────────────────────

const K_VALUES = [1, 3, 5, 10, 30, 50];

// MemPalace reference scores (from BENCHMARKS.md)
const MEMPALACE_SCORES: Record<string, Record<string, number>> = {
  raw: {
    "recall_any@1": 0.844, "recall_any@3": 0.938, "recall_any@5": 0.966,
    "recall_any@10": 0.982, "recall_any@30": 0.998, "recall_any@50": 1.0,
  },
  hybrid: {
    "recall_any@1": 0.912, "recall_any@3": 0.974, "recall_any@5": 0.984,
    "recall_any@10": 0.992, "recall_any@30": 1.0, "recall_any@50": 1.0,
  },
};

async function runSingleMode(
  entries: LongMemEvalEntry[],
  config: BenchConfig,
  mode: ScoringMode,
  cache: EmbeddingCache,
  rankCtx?: LlamaRankingContext | null,
): Promise<{
  results: QuestionResult[];
  sessionMetrics: Map<string, number[]>;
  perType: Map<string, Map<string, number[]>>;
  elapsed: number;
}> {
  const sessionMetrics = new Map<string, number[]>();
  const perType = new Map<string, Map<string, number[]>>();
  const results: QuestionResult[] = [];

  // Initialize metric accumulators
  for (const k of K_VALUES) {
    sessionMetrics.set(`recall_any@${k}`, []);
    sessionMetrics.set(`ndcg@${k}`, []);
  }

  const start = Date.now();
  let embedCount = 0;

  for (let qi = 0; qi < entries.length; qi++) {
    const entry = entries[qi];
    const correctIds = new Set(entry.answer_session_ids);

    // Build corpus
    const { texts, ids, timestamps } = buildCorpus(entry, config.granularity);
    if (texts.length === 0) continue;

    // Map corpus IDs to session-level IDs for evaluation
    const sessionCorpusIds =
      config.granularity === "turn"
        ? ids.map(sessionIdFromCorpusId)
        : ids;

    // Load cached embeddings or compute fresh
    const hadCache = cache.loadFromDisk(entry.question_id);

    // Embed corpus (with optional doc prefix for asymmetric models)
    const prefixedTexts = DOC_PREFIX ? texts.map((t) => DOC_PREFIX + t) : texts;
    const corpusEmbeddings = await cache.embedBatchWithCache(prefixedTexts);
    embedCount += hadCache ? 0 : texts.length;

    // Embed query
    // Asymmetric encoding: prefix the query (not documents) for retrieval models
    const queryEmbedding = await cache.embedWithCache(QUERY_PREFIX + entry.question);
    if (!hadCache) embedCount++;

    // Build corpus docs
    const corpus: CorpusDoc[] = texts.map((text, i) => ({
      text,
      sessionId: ids[i],
      timestamp: timestamps[i],
      embedding: corpusEmbeddings[i],
    }));

    // Score
    const refTime = parseQuestionDate(entry.question_date);
    let ranked: RankedResult[];

    switch (mode) {
      case "raw":
        ranked = scoreRaw(queryEmbedding, corpus);
        break;
      case "wmr":
        ranked = scoreWMR(queryEmbedding, corpus, refTime);
        break;
      case "hybrid":
        ranked = scoreHybrid(queryEmbedding, entry.question, corpus, config.hybridWeight);
        break;
      case "full":
        ranked = scoreFull(queryEmbedding, entry.question, corpus, refTime, config.hybridWeight);
        break;
      case "rerank":
        if (!rankCtx) throw new Error("Reranker not initialized");
        ranked = await scoreRerank(queryEmbedding, entry.question, corpus, config.rerankTopN, rankCtx);
        break;
      case "hybrid-rerank":
        if (!rankCtx) throw new Error("Reranker not initialized");
        ranked = await scoreHybridRerank(queryEmbedding, entry.question, corpus, config.hybridWeight, config.rerankTopN, rankCtx);
        break;
      case "fusion-rerank":
        if (!rankCtx) throw new Error("Reranker not initialized");
        ranked = await scoreFusionRerank(queryEmbedding, entry.question, corpus, config.hybridWeight, config.rerankTopN, rankCtx);
        break;
    }

    // Convert to ranking indices
    const rankings = ranked.map((r) => r.corpusIndex);

    // Evaluate metrics
    const metrics: QuestionMetrics = {};
    for (const k of K_VALUES) {
      const { recallAny, ndcgScore } = evaluateRetrieval(
        rankings,
        correctIds,
        config.granularity === "turn" ? sessionCorpusIds : ids,
        k,
      );
      metrics[`recall_any@${k}`] = recallAny;
      metrics[`ndcg@${k}`] = ndcgScore;

      sessionMetrics.get(`recall_any@${k}`)!.push(recallAny);
      sessionMetrics.get(`ndcg@${k}`)!.push(ndcgScore);
    }

    // Per-type breakdown
    if (!perType.has(entry.question_type)) {
      perType.set(entry.question_type, new Map());
      for (const k of K_VALUES) {
        perType.get(entry.question_type)!.set(`recall_any@${k}`, []);
        perType.get(entry.question_type)!.set(`ndcg@${k}`, []);
      }
    }
    for (const k of K_VALUES) {
      perType.get(entry.question_type)!.get(`recall_any@${k}`)!.push(metrics[`recall_any@${k}`]);
      perType.get(entry.question_type)!.get(`ndcg@${k}`)!.push(metrics[`ndcg@${k}`]);
    }

    results.push({
      question_id: entry.question_id,
      question_type: entry.question_type,
      question: entry.question,
      answer: entry.answer,
      metrics: { session: metrics },
    });

    // Save cache after each question (disabled by default — 1024-dim embeddings are ~8KB each in JSON)
    // if (!hadCache && config.cacheDir) {
    //   cache.saveToDisk(entry.question_id);
    // }

    // Progress
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const r5 = metrics["recall_any@5"];
    const avgR5 = mean(sessionMetrics.get("recall_any@5")!);
    process.stderr.write(
      `\r  [${qi + 1}/${entries.length}] ${elapsed}s | R@5=${r5.toFixed(0)} | avg R@5=${(avgR5 * 100).toFixed(1)}% | embeds=${embedCount}`,
    );

    // Clear corpus embeddings from cache to limit memory (keep query embeddings)
    // Actually, since each question has unique sessions, the cache naturally doesn't bloat
  }

  process.stderr.write("\n");

  const elapsed = Date.now() - start;
  return { results, sessionMetrics, perType, elapsed };
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// ── Report Generator ───────────────────────────────────────────────────

function printReport(
  mode: string,
  granularity: string,
  sessionMetrics: Map<string, number[]>,
  perType: Map<string, Map<string, number[]>>,
  elapsed: number,
  questionCount: number,
): void {
  const perQ = (elapsed / 1000 / questionCount).toFixed(2);

  console.log();
  console.log("=".repeat(64));
  console.log(`  RESULTS — Zeraclaw (${mode} mode, ${granularity} granularity)`);
  console.log("=".repeat(64));
  console.log(`  Questions: ${questionCount}`);
  console.log(`  Time: ${(elapsed / 1000).toFixed(1)}s (${perQ}s per question)`);
  console.log();
  console.log("  SESSION-LEVEL METRICS:");

  for (const k of K_VALUES) {
    const recall = mean(sessionMetrics.get(`recall_any@${k}`)!);
    const ndcgVal = mean(sessionMetrics.get(`ndcg@${k}`)!);
    const kPad = String(k).padStart(2);
    console.log(
      `    Recall@${kPad}: ${(recall * 100).toFixed(1).padStart(5)}%    NDCG@${kPad}: ${(ndcgVal * 100).toFixed(1)}%`,
    );
  }

  // Per-type breakdown
  console.log();
  console.log("  PER-TYPE BREAKDOWN (R@5 / R@10):");
  for (const [qtype, metrics] of [...perType.entries()].sort()) {
    const r5 = mean(metrics.get("recall_any@5")!);
    const r10 = mean(metrics.get("recall_any@10")!);
    const n = metrics.get("recall_any@5")!.length;
    console.log(
      `    ${qtype.padEnd(36)} R@5=${(r5 * 100).toFixed(1).padStart(5)}%  R@10=${(r10 * 100).toFixed(1).padStart(5)}%  (n=${n})`,
    );
  }

  // Head-to-head vs MemPalace
  const mpRef = (mode === "hybrid" || mode === "hybrid-rerank" || mode === "fusion-rerank")
    ? MEMPALACE_SCORES.hybrid : MEMPALACE_SCORES.raw;
  if (mpRef && questionCount >= 50) {
    console.log();
    console.log("─".repeat(64));
    console.log("  HEAD-TO-HEAD vs MemPalace");
    console.log("─".repeat(64));
    console.log(
      `  ${"Metric".padEnd(16)} ${"Zeraclaw".padEnd(14)} ${"MemPalace".padEnd(14)} ${"Delta"}`,
    );

    for (const k of K_VALUES) {
      const key = `recall_any@${k}`;
      const ours = mean(sessionMetrics.get(key)!);
      const theirs = mpRef[key];
      if (theirs !== undefined) {
        const delta = ours - theirs;
        const sign = delta >= 0 ? "+" : "";
        console.log(
          `  ${`R@${k}`.padEnd(16)} ${(ours * 100).toFixed(1).padStart(5)}%${" ".repeat(8)} ${(theirs * 100).toFixed(1).padStart(5)}%${" ".repeat(8)} ${sign}${(delta * 100).toFixed(1)}%`,
        );
      }
    }

    const ourR5 = mean(sessionMetrics.get("recall_any@5")!);
    const theirR5 = mpRef["recall_any@5"];
    if (theirR5 !== undefined) {
      console.log();
      if (ourR5 > theirR5) {
        console.log(`  >>> Zeraclaw BGE-M3 wins by +${((ourR5 - theirR5) * 100).toFixed(1)}% R@5 <<<`);
      } else if (ourR5 === theirR5) {
        console.log(`  >>> Tied at ${(ourR5 * 100).toFixed(1)}% R@5 <<<`);
      } else {
        console.log(`  >>> MemPalace leads by +${((theirR5 - ourR5) * 100).toFixed(1)}% R@5 — investigate failures <<<`);
      }
    }
  }

  console.log("=".repeat(64));
  console.log();
}

function printComparisonTable(
  modeResults: Map<string, Map<string, number[]>>,
): void {
  console.log();
  console.log("=".repeat(72));
  console.log("  ALL MODES COMPARISON — Zeraclaw vs MemPalace");
  console.log("=".repeat(72));

  // Header
  const modes = [...modeResults.keys()];
  const header = "  " + "Metric".padEnd(12) + modes.map((m) => m.padStart(10)).join("") + "  MP-raw  MP-hyb";
  console.log(header);
  console.log("  " + "─".repeat(header.length - 2));

  for (const k of K_VALUES) {
    const key = `recall_any@${k}`;
    let line = `  R@${String(k).padStart(2)}`.padEnd(14);
    for (const mode of modes) {
      const vals = modeResults.get(mode)!.get(key)!;
      line += `${(mean(vals) * 100).toFixed(1).padStart(8)}%`;
    }
    const mpRaw = MEMPALACE_SCORES.raw[key];
    const mpHyb = MEMPALACE_SCORES.hybrid[key];
    if (mpRaw !== undefined) line += `  ${(mpRaw * 100).toFixed(1).padStart(5)}%`;
    if (mpHyb !== undefined) line += `  ${(mpHyb * 100).toFixed(1).padStart(5)}%`;
    console.log(line);
  }

  console.log("=".repeat(72));
  console.log();
}

// ── CLI & Main ─────────────────────────────────────────────────────────

function parseCliArgs(): BenchConfig {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      mode: { type: "string", default: "raw" },
      granularity: { type: "string", default: "session" },
      limit: { type: "string", default: "0" },
      skip: { type: "string", default: "0" },
      "hybrid-weight": { type: "string", default: "0.30" },
      "query-prefix": { type: "string", default: "" },
      "doc-prefix": { type: "string", default: "" },
      out: { type: "string", default: "" },
      "cache-dir": { type: "string", default: "" },
      "reranker-model": { type: "string", default: "" },
      "rerank-top-n": { type: "string", default: "30" },
      "all-modes": { type: "boolean", default: false },
      help: { type: "boolean", short: "h", default: false },
    },
  });

  if (values.help || positionals.length === 0) {
    console.log(`
Zeraclaw LongMemEval Benchmark
================================

Usage:
  npx tsx src/bench-longmemeval.ts <data_file> [options]

Options:
  --mode MODE                   raw|wmr|hybrid|full|rerank|hybrid-rerank (default: raw)
  --granularity session|turn    Corpus granularity (default: session)
  --limit N                     Max questions (0 = all, default: 0)
  --skip N                      Skip first N questions (default: 0)
  --hybrid-weight 0.30          Keyword overlap weight (default: 0.30)
  --reranker-model PATH         Path to bge-reranker GGUF (for rerank modes)
  --rerank-top-n N              Candidates to send to reranker (default: 30)
  --out results.jsonl           Output file (default: auto)
  --cache-dir ./embed-cache     Cache embeddings to disk
  --all-modes                   Run all modes and compare
  -h, --help                    Show this help

Data:
  mkdir -p /tmp/longmemeval-data
  curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \\
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
`);
    process.exit(0);
  }

  const dataFile = positionals[0];
  const mode = (values.mode ?? "raw") as ScoringMode;
  const granularity = (values.granularity ?? "session") as Granularity;
  const limit = parseInt(values.limit ?? "0", 10);
  const skip = parseInt(values.skip ?? "0", 10);
  const hybridWeight = parseFloat(values["hybrid-weight"] ?? "0.30");
  const allModes = values["all-modes"] ?? false;

  // Set global prefixes
  QUERY_PREFIX = values["query-prefix"] ?? "";
  DOC_PREFIX = values["doc-prefix"] ?? "";

  const outFile =
    values.out ||
    `kongclaw_longmemeval_${mode}_${granularity}_${new Date().toISOString().slice(0, 10)}.jsonl`;

  const rerankModelPath =
    values["reranker-model"] ||
    process.env.RERANKER_MODEL_PATH ||
    join(homedir(), ".node-llama-cpp", "models", "bge-reranker-v2-m3-Q8_0.gguf");
  const rerankTopN = parseInt(values["rerank-top-n"] ?? "30", 10);

  return {
    dataFile,
    mode,
    granularity,
    limit,
    skip,
    hybridWeight,
    outFile,
    cacheDir: values["cache-dir"] ?? "",
    allModes,
    rerankModelPath,
    rerankTopN,
  };
}

async function main(): Promise<void> {
  const config = parseCliArgs();

  // Validate data file
  if (!existsSync(config.dataFile)) {
    console.error(`Error: Data file not found: ${config.dataFile}`);
    console.error();
    console.error("Download it:");
    console.error("  mkdir -p /tmp/longmemeval-data");
    console.error(
      "  curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \\",
    );
    console.error(
      "    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    );
    process.exit(1);
  }

  // Load data
  console.log(`Loading data from ${config.dataFile}...`);
  const rawData = JSON.parse(readFileSync(config.dataFile, "utf-8"));
  let entries: LongMemEvalEntry[] = Array.isArray(rawData) ? rawData : [rawData];

  // Apply skip/limit
  if (config.skip > 0) entries = entries.slice(config.skip);
  if (config.limit > 0) entries = entries.slice(0, config.limit);

  console.log(`Loaded ${entries.length} questions`);

  // Find question types
  const types = new Set(entries.map((e) => e.question_type));
  console.log(`Question types: ${[...types].join(", ")}`);

  // Initialize embeddings
  const modelPath =
    process.env.EMBED_MODEL_PATH ??
    join(homedir(), ".node-llama-cpp", "models", "bge-m3-q4_k_m.gguf");

  if (!existsSync(modelPath)) {
    console.error(`Error: Embedding model not found at: ${modelPath}`);
    console.error("Set EMBED_MODEL_PATH or download BGE-M3 GGUF");
    process.exit(1);
  }

  const modelName = modelPath.split("/").pop() ?? "unknown";
  console.log(`Initializing embeddings (1024-dim): ${modelName}`);
  if (QUERY_PREFIX) console.log(`  Query prefix: "${QUERY_PREFIX}"`);
  if (DOC_PREFIX) console.log(`  Doc prefix: "${DOC_PREFIX}"`);
  await initEmbeddings({ modelPath, dimensions: 1024 });
  console.log("Embeddings ready.");

  const cache = new EmbeddingCache(config.cacheDir);

  // Initialize cross-encoder reranker if needed
  let rankCtx: LlamaRankingContext | null = null;
  const needsReranker = config.mode === "rerank" || config.mode === "hybrid-rerank" || config.mode === "fusion-rerank" || config.allModes;

  if (needsReranker) {
    if (!existsSync(config.rerankModelPath)) {
      console.error(`Reranker model not found at: ${config.rerankModelPath}`);
      console.error("Download it:");
      console.error(`  curl -fSL -o ~/.node-llama-cpp/models/bge-reranker-v2-m3-Q8_0.gguf \\`);
      console.error(`    "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf"`);
      if (!config.allModes) process.exit(1);
    } else {
      const rerankName = config.rerankModelPath.split("/").pop() ?? "unknown";
      console.log(`Initializing reranker: ${rerankName} (top-${config.rerankTopN})`);
      const llama = await getLlama({
        logLevel: LlamaLogLevel.error,
        logger: (level, message) => {
          if (message.includes("missing newline token")) return;
          if (level === LlamaLogLevel.error) console.error(`[rerank] ${message}`);
        },
      });
      const rerankModel = await llama.loadModel({ modelPath: config.rerankModelPath });
      rankCtx = await rerankModel.createRankingContext();
      console.log("Reranker ready.");
    }
  }

  try {
    if (config.allModes) {
      // Run all modes: base modes always, rerank modes if reranker available
      const modes: ScoringMode[] = ["raw", "wmr", "hybrid", "full"];
      if (rankCtx) modes.push("rerank", "hybrid-rerank", "fusion-rerank");
      const allResults = new Map<string, Map<string, number[]>>();

      for (const mode of modes) {
        console.log();
        console.log(`${"─".repeat(64)}`);
        console.log(`  Running mode: ${mode}`);
        console.log(`${"─".repeat(64)}`);

        const { results, sessionMetrics, perType, elapsed } =
          await runSingleMode(entries, config, mode, cache, rankCtx);

        allResults.set(mode, sessionMetrics);

        printReport(
          mode,
          config.granularity,
          sessionMetrics,
          perType,
          elapsed,
          entries.length,
        );

        // Write JSONL for this mode
        const modeOutFile = config.outFile.replace(".jsonl", `_${mode}.jsonl`);
        writeFileSync(
          modeOutFile,
          results.map((r) => JSON.stringify(r)).join("\n") + "\n",
        );
        console.log(`  Results written to ${modeOutFile}`);
      }

      printComparisonTable(allResults);
    } else {
      // Single mode
      console.log(`Mode: ${config.mode} | Granularity: ${config.granularity}`);

      const { results, sessionMetrics, perType, elapsed } =
        await runSingleMode(entries, config, config.mode, cache, rankCtx);

      printReport(
        config.mode,
        config.granularity,
        sessionMetrics,
        perType,
        elapsed,
        entries.length,
      );

      // Write JSONL
      writeFileSync(
        config.outFile,
        results.map((r) => JSON.stringify(r)).join("\n") + "\n",
      );
      console.log(`Results written to ${config.outFile}`);
    }
  } finally {
    await disposeEmbeddings();
    if (rankCtx) {
      try { await rankCtx.dispose(); } catch { /* ignore */ }
    }
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
