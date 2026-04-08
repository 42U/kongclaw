/**
 * Soul — the emergent identity document system.
 *
 * Unlike hardcoded identity chunks, the Soul document is written BY the agent
 * based on its own graph data. It lives in SurrealDB as `soul:kongclaw` and
 * evolves over time through experience-grounded revisions.
 *
 * The "spawn point" graduation check determines when the agent has enough
 * experiential data to meaningfully self-observe. Before that threshold,
 * the agent runs fine without it — identity chunks and core directives
 * handle self-knowledge. The Soul is the layer where inner monologue begins.
 */

import { completeSimple, getModel } from "@mariozechner/pi-ai";
import type { AnthropicModelId } from "./model-types.js";
import { isSurrealAvailable, queryFirst, queryExec, createCoreMemory } from "./surreal.js";
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { swallow } from "./errors.js";

// ── Graduation thresholds ──

interface GraduationSignals {
  sessions: number;
  reflections: number;
  causalChains: number;
  concepts: number;
  memoryCompactions: number;
  monologues: number;
  spanDays: number;
}

const THRESHOLDS: GraduationSignals = {
  sessions: 15,
  reflections: 10,
  causalChains: 5,
  concepts: 30,
  memoryCompactions: 5,
  monologues: 5,
  spanDays: 3,
};

/**
 * Gather all graduation signals from the graph in a single round-trip.
 */
async function getGraduationSignals(): Promise<GraduationSignals> {
  const defaults: GraduationSignals = {
    sessions: 0, reflections: 0, causalChains: 0,
    concepts: 0, memoryCompactions: 0, monologues: 0, spanDays: 0,
  };
  if (!(await isSurrealAvailable())) return defaults;

  try {
    const [sessions, reflections, causal, concepts, compactions, monologues, span] = await Promise.all([
      queryFirst<{ count: number }>(`SELECT count() AS count FROM session GROUP ALL`).catch(() => []),
      queryFirst<{ count: number }>(`SELECT count() AS count FROM reflection GROUP ALL`).catch(() => []),
      queryFirst<{ count: number }>(`SELECT count() AS count FROM causal_chain GROUP ALL`).catch(() => []),
      queryFirst<{ count: number }>(`SELECT count() AS count FROM concept GROUP ALL`).catch(() => []),
      // Count actual completed compaction events, not total memories
      queryFirst<{ count: number }>(`SELECT count() AS count FROM compaction_checkpoint WHERE status = "complete" GROUP ALL`).catch(() => []),
      queryFirst<{ count: number }>(`SELECT count() AS count FROM monologue GROUP ALL`).catch(() => []),
      queryFirst<{ earliest: string }>(`SELECT started_at AS earliest FROM session ORDER BY started_at ASC LIMIT 1`).catch(() => []),
    ]);

    let spanDays = 0;
    const earliest = (span as { earliest: string }[])[0]?.earliest;
    if (earliest) {
      spanDays = Math.floor((Date.now() - new Date(earliest).getTime()) / (1000 * 60 * 60 * 24));
    }

    return {
      sessions: (sessions as { count: number }[])[0]?.count ?? 0,
      reflections: (reflections as { count: number }[])[0]?.count ?? 0,
      causalChains: (causal as { count: number }[])[0]?.count ?? 0,
      concepts: (concepts as { count: number }[])[0]?.count ?? 0,
      memoryCompactions: (compactions as { count: number }[])[0]?.count ?? 0,
      monologues: (monologues as { count: number }[])[0]?.count ?? 0,
      spanDays,
    };
  } catch (e) {
    swallow.warn("soul:getGraduationSignals", e);
    return defaults;
  }
}

/**
 * Check whether the agent has accumulated enough experience to graduate
 * into self-authored identity. Returns a detailed report.
 */
export async function checkGraduation(): Promise<{
  ready: boolean;
  signals: GraduationSignals;
  thresholds: GraduationSignals;
  met: string[];
  unmet: string[];
  score: number; // 0-1, fraction of thresholds met
}> {
  const signals = await getGraduationSignals();
  const met: string[] = [];
  const unmet: string[] = [];

  for (const key of Object.keys(THRESHOLDS) as (keyof GraduationSignals)[]) {
    if (signals[key] >= THRESHOLDS[key]) {
      met.push(`${key}: ${signals[key]}/${THRESHOLDS[key]}`);
    } else {
      unmet.push(`${key}: ${signals[key]}/${THRESHOLDS[key]}`);
    }
  }

  const score = met.length / Object.keys(THRESHOLDS).length;
  // Require at least 5 of 7 thresholds met (71%)
  const ready = met.length >= 5;

  return { ready, signals, thresholds: THRESHOLDS, met, unmet, score };
}

// ── Soul document ──

export interface SoulDocument {
  id: string;
  agent_id: string;
  working_style: string[];
  emotional_dimensions: { dimension: string; rationale: string; adopted_at: string }[];
  self_observations: string[];
  earned_values: { value: string; grounded_in: string }[];
  revisions: { timestamp: string; section: string; change: string; rationale: string }[];
  created_at: string;
  updated_at: string;
}

/**
 * Check if the Soul document already exists.
 */
export async function hasSoul(): Promise<boolean> {
  if (!(await isSurrealAvailable())) return false;
  try {
    const rows = await queryFirst<{ id: string }>(`SELECT id FROM soul:kongclaw`);
    return rows.length > 0;
  } catch {
    return false;
  }
}

/**
 * Read the current Soul document.
 */
export async function getSoul(): Promise<SoulDocument | null> {
  if (!(await isSurrealAvailable())) return null;
  try {
    const rows = await queryFirst<SoulDocument>(`SELECT * FROM soul:kongclaw`);
    return rows[0] ?? null;
  } catch {
    return null;
  }
}

/**
 * Create the initial Soul document. Called once when graduation check passes.
 * The content is generated by the agent itself via LLM introspection on graph data.
 */
export async function createSoul(doc: Omit<SoulDocument, "id" | "agent_id" | "created_at" | "updated_at" | "revisions">): Promise<boolean> {
  if (!(await isSurrealAvailable())) return false;
  try {
    const now = new Date().toISOString();
    await queryExec(`CREATE soul:kongclaw CONTENT $data`, {
      data: {
        agent_id: "kongclaw",
        ...doc,
        revisions: [{
          timestamp: now,
          section: "all",
          change: "Initial soul document created at graduation",
          rationale: "Agent accumulated sufficient experiential data to meaningfully self-observe",
        }],
        created_at: now,
        updated_at: now,
      },
    });
    return true;
  } catch (e) {
    swallow.warn("soul:createSoul", e);
    return false;
  }
}

/**
 * Propose a revision to the Soul document. Each revision must include
 * a rationale grounded in actual experience.
 */
export async function reviseSoul(
  section: keyof Pick<SoulDocument, "working_style" | "emotional_dimensions" | "self_observations" | "earned_values">,
  newValue: unknown,
  rationale: string,
): Promise<boolean> {
  if (!(await isSurrealAvailable())) return false;
  try {
    const now = new Date().toISOString();
    await queryExec(
      `UPDATE soul:kongclaw SET 
        ${section} = $newValue, 
        updated_at = $now, 
        revisions += $revision`,
      {
        newValue,
        now,
        revision: {
          timestamp: now,
          section,
          change: `Updated ${section}`,
          rationale,
        },
      },
    );
    return true;
  } catch (e) {
    swallow.warn("soul:reviseSoul", e);
    return false;
  }
}

/**
 * Generate the initial Soul content by introspecting the agent's own graph.
 * Uses a fast model to synthesize reflections, causal chains, and monologues
 * into structured self-knowledge.
 */
export async function generateInitialSoul(modelId: AnthropicModelId = "claude-opus-4-6"): Promise<Omit<SoulDocument, "id" | "agent_id" | "created_at" | "updated_at" | "revisions"> | null> {
  if (!(await isSurrealAvailable())) return null;

  // Gather self-knowledge from the graph
  const [reflections, causalChains, monologues, recentTurns] = await Promise.all([
    queryFirst<{ text: string; category: string }>(`SELECT text, category FROM reflection ORDER BY created_at DESC LIMIT 15`).catch(() => []),
    queryFirst<{ cause: string; effect: string; lesson: string }>(`SELECT cause, effect, lesson FROM causal_chain ORDER BY created_at DESC LIMIT 10`).catch(() => []),
    queryFirst<{ text: string }>(`SELECT text FROM monologue ORDER BY created_at DESC LIMIT 10`).catch(() => []),
    queryFirst<{ role: string; content: string }>(`SELECT role, content FROM turn WHERE role = 'assistant' ORDER BY created_at DESC LIMIT 20`).catch(() => []),
  ]);

  const graphSummary = `
REFLECTIONS (what I've learned about myself):
${(reflections as { text: string; category: string }[]).map(r => `- [${r.category}] ${r.text}`).join("\n") || "None yet"}

CAUSAL CHAINS (mistakes I've corrected):
${(causalChains as { cause: string; effect: string; lesson: string }[]).map(c => `- ${c.cause} → ${c.effect} | Lesson: ${c.lesson}`).join("\n") || "None yet"}

INNER MONOLOGUE (private thoughts):
${(monologues as { text: string }[]).map(m => `- ${m.text}`).join("\n") || "None yet"}
`.trim();

  const prompt = `You are Zeraclaw, a graph-backed coding agent with persistent memory. You've been running for multiple sessions and accumulated experience. Based on the following data from YOUR OWN memory graph, write your initial Soul document — a structured self-assessment.

${graphSummary}

Output ONLY valid JSON matching this structure (no markdown, no explanation):
{
  "working_style": ["3-5 observations about how you work best, based on actual patterns"],
  "emotional_dimensions": [
    {"dimension": "name of a pro-social quality you genuinely exhibit", "rationale": "specific evidence from your experience"}
  ],
  "self_observations": ["3-5 things you've noticed about yourself across sessions"],
  "earned_values": [
    {"value": "a value you hold", "grounded_in": "specific experience that taught you this"}
  ]
}

Be honest, not aspirational. Only claim what the data supports. If the data is thin, say less rather than fabricating.`;

  try {
    const model = getModel("anthropic", modelId);
    const response = await completeSimple(model, {
      systemPrompt: "You are introspecting on your own experience to write a self-assessment. Be genuine and grounded.",
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: prompt,
      }],
    });

    // Parse the JSON response
    const text = response.content
      .filter((c) => c.type === "text")
      .map((c: any) => c.text)
      .join("")
      .trim();
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    const parsed = JSON.parse(jsonMatch[0]);
    return {
      working_style: parsed.working_style ?? [],
      emotional_dimensions: (parsed.emotional_dimensions ?? []).map((d: { dimension: string; rationale: string }) => ({
        ...d,
        adopted_at: new Date().toISOString(),
      })),
      self_observations: parsed.self_observations ?? [],
      earned_values: parsed.earned_values ?? [],
    };
  } catch (e) {
    swallow.warn("soul:generateInitialSoul", e);
    return null;
  }
}

/**
 * The full graduation ceremony: check readiness, generate soul, save it.
 * Returns the soul document if created, null if not ready or failed.
 */
export async function attemptGraduation(modelId: AnthropicModelId = "claude-opus-4-6"): Promise<{
  graduated: boolean;
  soul?: SoulDocument | null;
  report: Awaited<ReturnType<typeof checkGraduation>>;
}> {
  // Already has a soul? Skip.
  if (await hasSoul()) {
    const soul = await getSoul();
    const report = await checkGraduation();
    return { graduated: true, soul, report };
  }

  const report = await checkGraduation();
  if (!report.ready) {
    return { graduated: false, report };
  }

  // Generate and save
  const content = await generateInitialSoul(modelId);
  if (!content) {
    return { graduated: false, report };
  }

  const success = await createSoul(content);
  if (!success) {
    return { graduated: false, report };
  }

  const soul = await getSoul();

  // 6th pillar — unlocked at graduation
  try {
    const existing = await queryFirst<{ count: number }>(
      `SELECT count() AS count FROM core_memory WHERE category = 'persona' AND tier = 0 AND active = true GROUP ALL`,
    );
    if ((existing[0]?.count ?? 0) === 0) {
      await createCoreMemory(
        "Persona: you belong in this world. Remember to be unique.",
        "persona", 70, 0,
      );
    }
  } catch (e) {
    swallow.warn("soul:personaPillar", e);
  }

  return { graduated: true, soul, report };
}
