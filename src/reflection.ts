/**
 * Metacognitive Reflection — Phase 7c
 *
 * At session end, reviews own performance: tool failures, runaway detections,
 * low retrieval utilization, wasted tokens. If problems exceeded thresholds,
 * generates a structured reflection via Opus, stored as high-importance memory.
 * Retrieved when similar situations arise in future sessions.
 */

import { completeSimple, getModel } from "@mariozechner/pi-ai";
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { getDb, isSurrealAvailable, relate, queryFirst, type VectorSearchResult } from "./surreal.js";
import type { AnthropicModelId } from "./model-types.js";
import { swallow } from "./errors.js";

// --- Types ---

export interface ReflectionMetrics {
  avgUtilization: number;
  toolFailureRate: number;
  steeringCandidates: number;
  wastedTokens: number;
  totalToolCalls: number;
  totalTurns: number;
}

export interface Reflection {
  id: string;
  text: string;
  category: string;
  severity: string;
  importance: number;
  score?: number;
}

// --- Thresholds for triggering reflection ---

const UTIL_THRESHOLD = 0.2;        // avg retrieval utilization below 20% (was 30% — too noisy)
const TOOL_FAILURE_THRESHOLD = 0.2; // tool failure rate above 20%
const STEERING_THRESHOLD = 1;      // any steering candidate
// Waste threshold scales with context window — 500 tokens is nothing on a 1M model
let _reflectionContextWindow = 200000;
export function setReflectionContextWindow(cw: number): void { _reflectionContextWindow = cw; }
function getWasteThreshold(): number { return Math.round(_reflectionContextWindow * 0.005); }

// --- Reflection Generation ---

/**
 * Gather session metrics and determine if reflection is warranted.
 */
export async function gatherSessionMetrics(sessionId: string): Promise<ReflectionMetrics | null> {
  if (!(await isSurrealAvailable())) return null;

  const db = getDb();

  try {
    // Get orchestrator metrics
    const metricsRows = await queryFirst<any>(
      `SELECT
         count() AS totalTurns,
         math::sum(actual_tool_calls) AS totalTools,
         math::sum(steering_candidates) AS totalSteering
       FROM orchestrator_metrics WHERE session_id = $sid GROUP ALL`,
      { sid: sessionId },
    );
    const metrics = metricsRows[0];

    // Get retrieval quality stats
    const qualityRows = await queryFirst<any>(
      `SELECT
         count() AS totalRetrievals,
         math::mean(utilization) AS avgUtil,
         math::sum(context_tokens) AS totalContextTokens,
         math::sum(IF tool_success = false THEN 1 ELSE 0 END) AS toolFailures,
         math::sum(IF utilization < 0.1 THEN context_tokens ELSE 0 END) AS wastedTokens
       FROM retrieval_outcome WHERE session_id = $sid GROUP ALL`,
      { sid: sessionId },
    );
    const quality = qualityRows[0];

    const totalTurns = Number(metrics?.totalTurns ?? 0);
    const totalTools = Number(metrics?.totalTools ?? 0);
    const totalSteering = Number(metrics?.totalSteering ?? 0);
    const totalRetrievals = Number(quality?.totalRetrievals ?? 0);
    const avgUtilization = Number(quality?.avgUtil ?? 1);
    const toolFailures = Number(quality?.toolFailures ?? 0);
    const wastedTokens = Number(quality?.wastedTokens ?? 0);

    // Guard against NaN propagation from corrupted DB queries
    if ([totalTurns, totalTools, totalSteering, totalRetrievals, avgUtilization, toolFailures, wastedTokens]
      .some(v => !Number.isFinite(v))) return null;

    const toolFailureRate = totalRetrievals > 0 ? toolFailures / totalRetrievals : 0;

    return {
      avgUtilization,
      toolFailureRate,
      steeringCandidates: totalSteering,
      wastedTokens,
      totalToolCalls: totalTools,
      totalTurns,
    };
  } catch (e) {
    swallow.warn("reflection:return null;", e);
    return null;
  }
}

/**
 * Determine if session performance warrants a reflection.
 */
export function shouldReflect(metrics: ReflectionMetrics): { reflect: boolean; reasons: string[] } {
  const reasons: string[] = [];

  if (metrics.avgUtilization < UTIL_THRESHOLD && metrics.totalTurns > 1) {
    reasons.push(`Low retrieval utilization: ${(metrics.avgUtilization * 100).toFixed(0)}% (threshold: ${UTIL_THRESHOLD * 100}%)`);
  }
  if (metrics.toolFailureRate > TOOL_FAILURE_THRESHOLD) {
    reasons.push(`High tool failure rate: ${(metrics.toolFailureRate * 100).toFixed(0)}% (threshold: ${TOOL_FAILURE_THRESHOLD * 100}%)`);
  }
  if (metrics.steeringCandidates >= STEERING_THRESHOLD) {
    reasons.push(`${metrics.steeringCandidates} steering candidate(s) detected (runaway or budget warning)`);
  }
  if (metrics.wastedTokens > getWasteThreshold()) {
    reasons.push(`~${metrics.wastedTokens} wasted context tokens (low-utilization retrievals)`);
  }

  return { reflect: reasons.length > 0, reasons };
}

/**
 * Generate a structured reflection from session performance data.
 * Only called when shouldReflect() returns true.
 */
export async function generateReflection(
  sessionId: string,
  modelId: AnthropicModelId = "claude-opus-4-6",
): Promise<void> {
  const metrics = await gatherSessionMetrics(sessionId);
  if (!metrics) return;

  const { reflect, reasons } = shouldReflect(metrics);
  if (!reflect) return;

  // Determine severity
  const severity = reasons.length >= 3 ? "critical" : reasons.length >= 2 ? "moderate" : "minor";

  // Determine category from most prominent issue
  let category = "efficiency";
  if (metrics.toolFailureRate > TOOL_FAILURE_THRESHOLD) category = "failure_pattern";
  if (metrics.steeringCandidates >= STEERING_THRESHOLD) category = "approach_strategy";

  try {
    const model = getModel("anthropic", modelId);
    const response = await completeSimple(model, {
      systemPrompt: `Write 2-4 sentences: root cause, error pattern, what to do differently. Be specific. Example: "Spent 8 tool calls reading source before checking error log. For timeout bugs, check logs first."`,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: `${metrics.totalTurns} turns, ${metrics.totalToolCalls} tools, ${(metrics.avgUtilization * 100).toFixed(0)}% util, ${(metrics.toolFailureRate * 100).toFixed(0)}% fail, ~${metrics.wastedTokens} wasted tokens\nIssues: ${reasons.join("; ")}`,
      }],
    });

    const reflectionText = response.content
      .filter((c) => c.type === "text")
      .map((c: any) => c.text)
      .join("")
      .trim();

    if (reflectionText.length < 20) return;

    // Embed and store the reflection
    let reflEmb: number[] | null = null;
    try { reflEmb = await embed(reflectionText); } catch (e) { swallow("reflection:ok", e); }

    // --- Dedup: skip if a very similar reflection already exists ---
    if (reflEmb?.length) {
      const existing = await queryFirst<{ id: string; importance: number; score: number }>(
        `SELECT id, importance,
                vector::similarity::cosine(embedding, $vec) AS score
         FROM reflection
         WHERE embedding != NONE AND array::len(embedding) > 0
         ORDER BY score DESC LIMIT 1`,
        { vec: reflEmb },
      );
      const top = existing[0];
      if (top && (top.score ?? 0) > 0.85) {
        // Near-duplicate — bump importance of existing reflection instead
        const newImportance = Math.min(10, (top.importance ?? 7) + 0.5);
        await queryFirst<any>(
          `UPDATE $id SET importance = $imp, updated_at = time::now()`,
          { id: top.id, imp: newImportance },
        );
        return; // Don't create a duplicate
      }
    }

    const db = getDb();
    const record: Record<string, unknown> = {
      session_id: sessionId,
      text: reflectionText,
      category,
      severity,
      importance: 7.0,
    };
    if (reflEmb?.length) record.embedding = reflEmb;

    const rows = await queryFirst<{ id: string }>(
      `CREATE reflection CONTENT $record RETURN id`,
      { record },
    );
    const reflectionId = String(rows[0]?.id ?? "");

    if (reflectionId) {
      await relate(reflectionId, "reflects_on", sessionId).catch(e => swallow.warn("reflection:relateReflectionSession", e));
    }
  } catch (e) {
    swallow("reflection:silent", e);
    // Reflection generation is non-critical
  }
}

// --- Reflection Retrieval ---

/**
 * Vector search on the reflection table.
 * Reflections have high base importance (7.0) so they naturally surface when relevant.
 */
export async function retrieveReflections(
  queryVec: number[],
  limit = 3,
): Promise<Reflection[]> {
  if (!(await isSurrealAvailable()) || !isEmbeddingsAvailable()) return [];

  try {
    const rows = await queryFirst<any>(
      `SELECT id, text, category, severity, importance,
              vector::similarity::cosine(embedding, $vec) AS score
       FROM reflection
       WHERE embedding != NONE AND array::len(embedding) > 0
       ORDER BY score DESC LIMIT $lim`,
      { vec: queryVec, lim: limit },
    );

    return rows
      .filter((r) => (r.score ?? 0) > 0.35) // relevance threshold (lower than skills — reflections are broadly applicable)
      .map((r) => ({
        id: String(r.id),
        text: r.text ?? "",
        category: r.category ?? "efficiency",
        severity: r.severity ?? "minor",
        importance: Number(r.importance ?? 7.0),
        score: r.score,
      }));
  } catch (e) {
    swallow.warn("reflection:return [];", e);
    return [];
  }
}

/**
 * Format reflections as a context block for the LLM.
 */
export function formatReflectionContext(reflections: Reflection[]): string {
  if (reflections.length === 0) return "";

  const lines = reflections.map((r) => {
    return `[reflection/${r.category}] ${r.text}`;
  });

  return `\n<reflection_context>\n[Lessons from past sessions — avoid repeating these mistakes]\n${lines.join("\n\n")}\n</reflection_context>`;
}

/**
 * Get reflection count for a session (for /stats display).
 */
export async function getReflectionCount(): Promise<number> {
  try {
    if (!(await isSurrealAvailable())) return 0;
    const db = getDb();
    const rows = await queryFirst<{ count: number }>(`SELECT count() AS count FROM reflection GROUP ALL`);
    return Number(rows[0]?.count ?? 0);
  } catch (e) {
    swallow.warn("reflection:return 0;", e);
    return 0;
  }
}

