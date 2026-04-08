/**
 * Procedural Memory (Skill Library) — Phase 7b
 *
 * When the agent successfully completes a multi-step task, extract the procedure
 * as a reusable skill (preconditions, steps, postconditions, outcome).
 * Next time a similar task is requested, inject the proven procedure as context.
 * Skills earn success/failure counts from outcomes — RL-like reinforcement.
 */

import { completeSimple, getModel } from "@mariozechner/pi-ai";
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import type { AnthropicModelId } from "./model-types.js";
import {
  getDb, isSurrealAvailable, relate, getSessionTurns, queryFirst, queryExec, type VectorSearchResult,
} from "./surreal.js";
import { swallow } from "./errors.js";

// --- Types ---

export interface SkillStep {
  tool: string;
  description: string;
  argsPattern?: string;
}

export interface Skill {
  id: string;
  name: string;
  description: string;
  preconditions?: string;
  steps: SkillStep[];
  postconditions?: string;
  successCount: number;
  failureCount: number;
  avgDurationMs: number;
  confidence: number;
  active: boolean;
  score?: number;
}

export interface ExtractedSkill {
  name: string;
  description: string;
  preconditions: string;
  steps: SkillStep[];
  postconditions: string;
}

// --- Skill Extraction ---

/**
 * Run at session end. If the session had 5+ tool calls and final outcomes succeeded,
 * extract the procedure as a reusable skill.
 */
export async function extractSkill(
  sessionId: string,
  taskId: string,
  modelId: AnthropicModelId = "claude-opus-4-6",
): Promise<string | null> {
  if (!(await isSurrealAvailable())) return null;

  const db = getDb();

  // Check if session had enough tool activity
  const metricsRows = await queryFirst<{ totalTools: number }>(
    `SELECT math::sum(actual_tool_calls) AS totalTools
     FROM orchestrator_metrics WHERE session_id = $sid GROUP ALL`,
    { sid: sessionId },
  ).catch(() => [] as { totalTools: number }[]);
  const totalTools = Number(metricsRows[0]?.totalTools ?? 0);
  if (totalTools < 3) return null; // not enough activity to extract a skill

  const turns = await getSessionTurns(sessionId, 50);
  if (turns.length < 4) return null;

  const transcript = turns
    .map((t) => `[${t.role}] ${(t.text ?? "").slice(0, 300)}`)
    .join("\n");

  try {
    const model = getModel("anthropic", modelId);
    const response = await completeSimple(model, {
      systemPrompt: `Return JSON or null. Fields: {name, description, preconditions, steps: [{tool, description}] (max 8), postconditions}. Generic patterns only (no specific paths). null if no clear multi-step workflow.`,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: `${totalTools} tool calls:\n${transcript.slice(0, 20000)}`,
      }],
    });

    const text = response.content
      .filter((c) => c.type === "text")
      .map((c: any) => c.text)
      .join("");

    // Handle "null" response
    if (text.trim() === "null" || text.trim() === "None") return null;

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    const parsed = JSON.parse(jsonMatch[0]) as ExtractedSkill;
    if (!parsed.name || !parsed.description || !Array.isArray(parsed.steps) || parsed.steps.length === 0) {
      return null;
    }

    // Embed the skill description for vector search
    let skillEmb: number[] | null = null;
    try { skillEmb = await embed(`${parsed.name}: ${parsed.description}`); } catch (e) { swallow("skills:ok", e); }

    const record: Record<string, unknown> = {
      name: String(parsed.name).slice(0, 100),
      description: String(parsed.description).slice(0, 200),
      preconditions: parsed.preconditions ? String(parsed.preconditions).slice(0, 200) : undefined,
      steps: parsed.steps.slice(0, 8).map((s) => ({
        tool: String(s.tool ?? "unknown"),
        description: String(s.description ?? "").slice(0, 200),
      })),
      postconditions: parsed.postconditions ? String(parsed.postconditions).slice(0, 200) : undefined,
    };
    if (skillEmb?.length) record.embedding = skillEmb;
    record.confidence = 1.0;
    record.active = true;

    const rows = await queryFirst<{ id: string }>(
      `CREATE skill CONTENT $record RETURN id`,
      { record },
    );
    const skillId = String(rows[0]?.id ?? "");

    if (skillId && taskId) {
      await relate(skillId, "skill_from_task", taskId).catch(e => swallow.warn("skills:relateSkillTask", e));
    }
    if (skillId) await supersedeOldSkills(skillId, skillEmb ?? []);

    return skillId || null;
  } catch (e) {
    swallow.warn("skills:return null;", e);
    return null;
  }
}


// --- Supersession ---

/**
 * After saving a new skill, fade similar existing skills above similarity threshold.
 * Policy: save on success, let old methods fade when a better one covers the same ground.
 */
export async function supersedeOldSkills(newSkillId: string, newEmb: number[]): Promise<void> {
  if (!newEmb.length) return;
  try {
    const rows = await queryFirst<{ id: string; score: number }>(
      `SELECT id, vector::similarity::cosine(embedding, $vec) AS score
       FROM skill
       WHERE id != $sid
         AND (active = NONE OR active = true)
         AND embedding != NONE AND array::len(embedding) > 0
       ORDER BY score DESC LIMIT 5`,
      { vec: newEmb, sid: newSkillId },
    );
    for (const row of rows) {
      if ((row.score ?? 0) >= 0.82) {
        await queryExec(
          `UPDATE $id SET active = false, superseded_by = $newId`,
          { id: row.id, newId: newSkillId },
        );
      }
    }
  } catch (e) { swallow("skills:supersedeOld", e); }
}

// --- Skill Retrieval ---

/**
 * Vector search on the skill table. Called from graphTransformContext
 * when the intent is code-write, code-debug, or multi-step.
 */
export async function findRelevantSkills(
  queryVec: number[],
  limit = 3,
): Promise<Skill[]> {
  if (!(await isSurrealAvailable()) || !isEmbeddingsAvailable()) return [];

  try {
    const db = getDb();
    const rows = await queryFirst<any>(
      `SELECT id, name, description, preconditions, steps, postconditions,
              success_count AS successCount, failure_count AS failureCount,
              avg_duration_ms AS avgDurationMs,
              vector::similarity::cosine(embedding, $vec) AS score
       FROM skill
       WHERE embedding != NONE AND array::len(embedding) > 0 AND (active = NONE OR active = true)
       ORDER BY score DESC LIMIT $lim`,
      { vec: queryVec, lim: limit },
    );

    return rows
      .filter((r) => (r.score ?? 0) > 0.4) // relevance threshold
      .map((r) => ({
        id: String(r.id),
        name: r.name ?? "",
        description: r.description ?? "",
        preconditions: r.preconditions,
        steps: Array.isArray(r.steps) ? r.steps : [],
        postconditions: r.postconditions,
        successCount: Number(r.successCount ?? 1),
        failureCount: Number(r.failureCount ?? 0),
        avgDurationMs: Number(r.avgDurationMs ?? 0),
        confidence: Number(r.confidence ?? 1.0),
        active: r.active !== false,
        score: r.score,
      }));
  } catch (e) {
    swallow.warn("skills:return [];", e);
    return [];
  }
}

/**
 * Format matched skills as a structured context block for the LLM.
 */
export function formatSkillContext(skills: Skill[]): string {
  if (skills.length === 0) return "";

  const lines = skills.map((s) => {
    const total = s.successCount + s.failureCount;
    const rate = total > 0 ? `${s.successCount}/${total} successful` : "new";
    const stepsStr = s.steps
      .map((step, i) => `  ${i + 1}. [${step.tool}] ${step.description}`)
      .join("\n");
    return `### ${s.name} (${rate})\n${s.description}\n${s.preconditions ? `Pre: ${s.preconditions}\n` : ""}Steps:\n${stepsStr}${s.postconditions ? `\nPost: ${s.postconditions}` : ""}`;
  });

  // If the top skill has high confidence (>= 0.7 cosine, >= 80% success rate),
  // frame it as a directive rather than a suggestion — skill auto-invocation.
  const topSkill = skills[0];
  const topTotal = topSkill.successCount + topSkill.failureCount;
  const topRate = topTotal > 0 ? topSkill.successCount / topTotal : 0;
  const isHighConfidence = (topSkill.score ?? 0) >= 0.7 && topRate >= 0.8 && topTotal >= 3;

  const header = isHighConfidence
    ? `[PROVEN PROCEDURE — "${topSkill.name}" matched with ${(topRate * 100).toFixed(0)}% success rate across ${topTotal} uses. Follow these steps unless the situation clearly differs.]`
    : "[Previously successful procedures — adapt as needed, don't follow blindly]";

  return `\n<skill_context>\n${header}\n${lines.join("\n\n")}\n</skill_context>`;
}

/**
 * Record skill outcome when a retrieved skill is used in a turn.
 * Updates success/failure count for RL-like reinforcement.
 */
export async function recordSkillOutcome(
  skillId: string,
  success: boolean,
  durationMs: number,
): Promise<void> {
  if (!(await isSurrealAvailable())) return;
  const RECORD_ID_RE = /^[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_]+$/;
  if (!RECORD_ID_RE.test(skillId)) return;

  try {
    const field = success ? "success_count" : "failure_count";
    await queryExec(
      `UPDATE ${skillId} SET
        ${field} += 1,
        avg_duration_ms = (avg_duration_ms * (success_count + failure_count - 1) + $dur) / (success_count + failure_count),
        last_used = time::now()`,
      { dur: durationMs },
    );
  } catch (e) { swallow("skills:non-critical", e); }
}

// --- Causal Chain → Skill Graduation ---

/**
 * Promote recurring successful causal chains into reusable skills.
 * Runs during memory maintenance. When 3+ successful chains of the same type
 * exist, synthesize a skill via Haiku.
 */
export async function graduateCausalToSkills(): Promise<number> {
  if (!(await isSurrealAvailable())) return 0;

  try {
    // Find chain types with 3+ successful, high-confidence chains
    const groups = await queryFirst<{ chain_type: string; cnt: number; descriptions: string[] }>(
      `SELECT chain_type, count() AS cnt, array::group(description) AS descriptions
       FROM causal_chain
       WHERE success = true AND confidence >= 0.7
       GROUP BY chain_type`,
    );

    let created = 0;

    for (const group of groups) {
      if (group.cnt < 3) continue;

      // Check if a skill already covers this chain type
      const existing = await queryFirst<{ id: string }>(
        `SELECT id FROM skill WHERE string::lowercase(name) CONTAINS string::lowercase($ct) LIMIT 1`,
        { ct: group.chain_type },
      );
      if (existing.length > 0) continue;

      // Synthesize a skill from the chain descriptions — Opus for quality
      const opus = getModel("anthropic", "claude-opus-4-6");
      const resp = await completeSimple(opus, {
        systemPrompt: `Return JSON: {name, description, preconditions, steps: [{tool, description}] (max 6), postconditions}. Synthesize a reusable procedure from these recurring patterns. Generic — no specific file paths or variable names.`,
        messages: [{
          role: "user",
          timestamp: Date.now(),
          content: `${group.cnt} successful "${group.chain_type}" patterns:\n${group.descriptions.slice(0, 8).join("\n")}`,
        }],
      });

      const text = resp.content.filter((c) => c.type === "text").map((c: any) => c.text).join("");
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (!jsonMatch) continue;

      let parsed: ExtractedSkill;
      try { parsed = JSON.parse(jsonMatch[0]); } catch { continue; }
      if (!parsed.name || !Array.isArray(parsed.steps) || parsed.steps.length === 0) continue;

      let skillEmb: number[] | null = null;
      try { skillEmb = await embed(`${parsed.name}: ${parsed.description}`); } catch (e) { swallow("skills:ok", e); }

      const record: Record<string, unknown> = {
        name: String(parsed.name).slice(0, 100),
        description: String(parsed.description).slice(0, 200),
        preconditions: parsed.preconditions ? String(parsed.preconditions).slice(0, 200) : undefined,
        steps: parsed.steps.slice(0, 6).map((s) => ({
          tool: String(s.tool ?? "unknown"),
          description: String(s.description ?? "").slice(0, 200),
        })),
        postconditions: parsed.postconditions ? String(parsed.postconditions).slice(0, 200) : undefined,
        graduated_from: group.chain_type,
        confidence: 1.0,
        active: true,
      };
      if (skillEmb?.length) record.embedding = skillEmb;

      const rows = await queryFirst<{ id: string }>(
        `CREATE skill CONTENT $record RETURN id`,
        { record },
      );
      if (rows[0]?.id) {
        await supersedeOldSkills(String(rows[0].id), skillEmb ?? []);
        created++;
      }
    }

    return created;
  } catch (e) {
    swallow.warn("skills:graduateCausal", e);
    return 0;
  }
}
