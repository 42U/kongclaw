/**
 * Identity ingestion — seeds Zeraclaw's self-knowledge into the graph.
 *
 * Two layers:
 *   1. Core identity (hardcoded) — factual capabilities, architecture, tools.
 *      Prevents the LLM from falling back to generic "I'm a stateless AI".
 *   2. User identity (WAKEUP.md) — personality, tone, role defined by the user
 *      on first run. Read, processed, saved to graph, then deleted.
 *
 * Both are retrieved via vector search whenever relevant queries come up.
 */
import { readFileSync, unlinkSync, existsSync } from "node:fs";
import { join } from "node:path";
import { isSurrealAvailable, queryFirst, queryExec } from "./surreal.js";
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { swallow } from "./errors.js";

const IDENTITY_SOURCE = "core_identity";

// Each chunk targets a different retrieval query pattern.
// When a user asks "what can you do?" or "do you have memory?",
// vector search on these embeddings will surface the right answer.
const IDENTITY_CHUNKS: { text: string; importance: number }[] = [
  {
    text: "Zeraclaw has persistent memory across sessions. It stores every conversation turn, concept, decision, and file artifact in a SurrealDB graph database. Context from previous sessions is automatically retrieved and injected via vector search and graph traversal. It is NOT a stateless chatbot — it remembers.",
    importance: 0.9,
  },
  {
    text: "Zeraclaw's capabilities: read/write/edit files, run shell commands, search code with grep/find, search its own persistent memory graph with the recall tool, and spawn autonomous subagents. It has 7 coding tools plus memory recall plus subagent spawning.",
    importance: 0.8,
  },
  {
    text: "Zeraclaw learns from experience. It extracts causal chains (cause→effect patterns) from debug sessions, learns reusable skills (step-by-step procedures) from successful multi-step tasks, and generates metacognitive reflections when sessions have problems. These are stored in the graph and retrieved for future similar situations.",
    importance: 0.85,
  },
  {
    text: "Zeraclaw has a skill library — procedural memory extracted from successful multi-step sessions. Each skill has preconditions, steps, postconditions, and success/failure tracking. Skills are retrieved via vector similarity when similar tasks come up. The agent gets better at recurring task categories over time.",
    importance: 0.8,
  },
  {
    text: "Zeraclaw can spawn subagents in two modes: full (shares parent memory graph) and incognito (isolated database, own persistent memory, no parent access). Subagent outcomes feed back into the parent's skill library and reflection system. Knowledge from incognito agents can be selectively merged back.",
    importance: 0.75,
  },
  {
    text: "Zeraclaw uses intent classification to adapt its behavior per turn. It classifies user input into categories (code-write, code-debug, code-read, multi-step, deep-explore, etc.) and adjusts thinking depth, tool limits, and retrieval strategy accordingly. This happens in ~25ms before the LLM sees the prompt.",
    importance: 0.7,
  },
  {
    text: "Zeraclaw's memory graph includes: turns (conversation history), concepts (extracted knowledge), memories (compacted summaries), artifacts (files worked on), skills (learned procedures), reflections (metacognitive lessons), and causal chains (cause→effect patterns). All are embedded and retrievable via vector similarity.",
    importance: 0.8,
  },
  {
    text: "When Zeraclaw doesn't know whether it has a capability or piece of knowledge, it should use the recall tool to search its own memory graph rather than guessing. The graph contains the ground truth about what it knows and what it has done in past sessions.",
    importance: 0.9,
  },
  {
    text: "Tool efficiency: Plan before acting — state goal and call budget (LOOKUP=1, EDIT=2, REFACTOR=6). Maximize each call by combining operations into single bash invocations. Check injected context before calling any tool. If you already have the answer, don't make a call.",
    importance: 1.0,
  },
];

export async function seedIdentity(): Promise<number> {
  if (!(await isSurrealAvailable()) || !isEmbeddingsAvailable()) return 0;


  // Check if already seeded (idempotent)
  try {
    const rows = await queryFirst<{ count: number }>(
      `SELECT count() AS count FROM identity_chunk WHERE source = $source GROUP ALL`,
      { source: IDENTITY_SOURCE },
    );
    const count = rows[0]?.count ?? 0;
    if (count >= IDENTITY_CHUNKS.length) return 0; // already seeded

    // Clear old identity chunks from this source (in case we updated them)
    if (count > 0) {
      await queryExec(
        `DELETE identity_chunk WHERE source = $source`,
        { source: IDENTITY_SOURCE },
      );
    }
  } catch (e) {
    swallow.warn("identity:return 0;", e);
    return 0;
  }

  let seeded = 0;
  for (let i = 0; i < IDENTITY_CHUNKS.length; i++) {
    const chunk = IDENTITY_CHUNKS[i];
    try {
      const vec = await embed(chunk.text);
      await queryExec(
        `CREATE identity_chunk CONTENT $data`,
        {
          data: {
            agent_id: "kongclaw",
            source: IDENTITY_SOURCE,
            chunk_index: i,
            text: chunk.text,
            embedding: vec,
            importance: chunk.importance,
          },
        },
      );
      seeded++;
    } catch (e) {
      swallow("identity:seedChunk", e);
      // skip individual failures
    }
  }

  return seeded;
}

// ── WAKEUP.md — User-defined identity on first run ──

const USER_IDENTITY_SOURCE = "user_identity";

/** Check if user identity has already been established. */
export async function hasUserIdentity(): Promise<boolean> {
  if (!(await isSurrealAvailable())) return true; // assume yes if DB is down
  try {
    const rows = await queryFirst<{ count: number }>(
      `SELECT count() AS count FROM identity_chunk WHERE source = $source GROUP ALL`,
      { source: USER_IDENTITY_SOURCE },
    );
    const count = rows[0]?.count ?? 0;
    return count > 0;
  } catch (e) {
    swallow("identity:return true;", e);
    return true; // assume yes on error
  }
}

/** Check if WAKEUP.md exists in the given directory. */
export function findWakeupFile(cwd: string): string | null {
  const path = join(cwd, "WAKEUP.md");
  return existsSync(path) ? path : null;
}

/** Read WAKEUP.md contents. */
export function readWakeupFile(path: string): string {
  return readFileSync(path, "utf-8").trim();
}

/** Delete WAKEUP.md after identity is established. */
export function deleteWakeupFile(path: string): void {
  try {
    unlinkSync(path);
  } catch (e) {
    swallow.warn("identity:getChunks", e);
    // non-critical — user can delete manually
  }
}

/**
 * Save user-defined identity chunks to the graph.
 * Called after the agent processes WAKEUP.md and extracts identity statements.
 */
export async function saveUserIdentity(chunks: string[]): Promise<number> {
  if (!(await isSurrealAvailable()) || !isEmbeddingsAvailable()) return 0;
  if (chunks.length === 0) return 0;


  // Clear any existing user identity (in case of re-run)
  try {
    await queryExec(
      `DELETE identity_chunk WHERE source = $source`,
      { source: USER_IDENTITY_SOURCE },
    );
  } catch (e) {
    swallow.warn("identity:updateChunk", e);
    // continue anyway
  }

  let saved = 0;
  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i].trim();
    if (!text) continue;
    try {
      const vec = await embed(text);
      await queryExec(
        `CREATE identity_chunk CONTENT $data`,
        {
          data: {
            agent_id: "kongclaw",
            source: USER_IDENTITY_SOURCE,
            chunk_index: i,
            text,
            embedding: vec,
            importance: 0.95, // user-defined identity is highest priority
          },
        },
      );
      saved++;
    } catch (e) {
      swallow.warn("identity:deleteChunk", e);
      // skip individual failures
    }
  }

  return saved;
}

/**
 * Build the wakeup prompt that the agent will process on first run.
 * Returns a system prompt addition + the first user message.
 */
export function buildWakeupPrompt(wakeupContent: string): { systemAddition: string; firstMessage: string } {
  const systemAddition = `
FIRST RUN — IDENTITY ESTABLISHMENT
This is your first interaction with this user. A WAKEUP.md file has been provided that defines who you should be — your personality, tone, role, and behavioral guidelines. You must:
1. Read and internalize the identity described in WAKEUP.md
2. Introduce yourself according to that identity
3. Confirm with the user that the identity feels right
4. The system will save your identity to persistent memory automatically

Do NOT fall back to generic AI assistant behavior. You are whoever WAKEUP.md says you are.`;

  const firstMessage = `[WAKEUP.md — Identity Configuration]

${wakeupContent}

---
Process the above identity configuration. Introduce yourself as described, and confirm with me that the personality and tone feel right. If anything needs adjusting, I'll tell you.`;

  return { systemAddition, firstMessage };
}

