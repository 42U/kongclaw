/**
 * Wake-up synthesis: constitutive memory initialization.
 *
 * At startup, fetches the latest handoff note, identity chunks, and recent
 * monologue entries, then synthesizes a first-person briefing via a fast
 * LLM call. The briefing is injected into the system prompt so the agent
 * "wakes up" knowing who it is and what it was doing.
 */

import { completeSimple, getModel } from "@mariozechner/pi-ai";
import type { AnthropicModelId } from "./model-types.js";
import {
  isSurrealAvailable,
  getLatestHandoff,
  countResolvedSinceHandoff,
  getUnresolvedMemories,
  getRecentFailedCausal,
  getAllIdentityChunks,
  getRecentMonologues,
  getPreviousSessionTurns,
  queryFirst,
} from "./surreal.js";
import { swallow } from "./errors.js";
import { readAndDeleteHandoffFile } from "./handoff-file.js";

/**
 * Synthesize a first-person wake-up briefing from constitutive memory.
 * Returns null if no prior state exists (first boot) or DB is unavailable.
 */
/** Gather depth signals: session count, memory span, monologue count. */
async function getDepthSignals(): Promise<{ sessions: number; monologueCount: number; memoryCount: number; spanDays: number }> {
  const defaults = { sessions: 0, monologueCount: 0, memoryCount: 0, spanDays: 0 };
  try {
    const [sessRows, monoRows, memRows, spanRows] = await Promise.all([
      queryFirst<{ count: number }>(`SELECT count() AS count FROM session GROUP ALL`).catch(() => [] as { count: number }[]),
      queryFirst<{ count: number }>(`SELECT count() AS count FROM monologue GROUP ALL`).catch(() => [] as { count: number }[]),
      queryFirst<{ count: number }>(`SELECT count() AS count FROM memory GROUP ALL`).catch(() => [] as { count: number }[]),
      queryFirst<{ earliest: string }>(`SELECT started_at AS earliest, started_at FROM session ORDER BY started_at ASC LIMIT 1`).catch(() => [] as { earliest: string }[]),
    ]);

    const sessions = sessRows[0]?.count ?? 0;
    const monologueCount = monoRows[0]?.count ?? 0;
    const memoryCount = memRows[0]?.count ?? 0;

    let spanDays = 0;
    const earliest = spanRows[0]?.earliest;
    if (earliest) {
      spanDays = Math.floor((Date.now() - new Date(earliest).getTime()) / (1000 * 60 * 60 * 24));
    }

    return { sessions, monologueCount, memoryCount, spanDays };
  } catch (e) {
    swallow.warn("wakeup:return defaults;", e);
    return defaults;
  }
}

export async function synthesizeWakeup(modelId: AnthropicModelId = "claude-opus-4-6", currentSessionId?: string, cwd?: string): Promise<string | null> {
  if (!(await isSurrealAvailable())) return null;

  // Check for sync handoff file from abrupt exit (read + delete)
  const fileHandoff = cwd ? readAndDeleteHandoffFile(cwd) : null;

  const [handoff, identityChunks, monologues, depth, previousTurns] = await Promise.all([
    getLatestHandoff(),
    getAllIdentityChunks(),
    getRecentMonologues(5),
    getDepthSignals(),
    getPreviousSessionTurns(currentSessionId, 10),
  ]);

  // Nothing to synthesize on first boot
  if (!handoff && !fileHandoff && monologues.length === 0 && identityChunks.length === 0 && previousTurns.length === 0) return null;

  // Build context sections
  const sections: string[] = [];

  // Depth awareness — where am I in my arc
  const depthLines: string[] = [];
  if (depth.sessions > 0) depthLines.push(`~${depth.sessions} sessions`);
  if (depth.memoryCount > 0) depthLines.push(`${depth.memoryCount} memories`);
  if (depth.monologueCount > 0) depthLines.push(`${depth.monologueCount} monologue traces`);
  if (depth.spanDays > 0) depthLines.push(`spanning ${depth.spanDays} day${depth.spanDays === 1 ? "" : "s"}`);
  if (depthLines.length > 0) {
    sections.push(`[DEPTH]\n${depthLines.join(" | ")}`);
  }

  if (handoff) {
    const resolvedCount = await countResolvedSinceHandoff(handoff.created_at).catch(() => 0);
    const ageHours = Math.floor((Date.now() - new Date(handoff.created_at).getTime()) / 3_600_000);
    let annotation = `(${ageHours}h old`;
    if (resolvedCount > 0) {
      annotation += `, ${resolvedCount} memories resolved since — some items may already be done`;
    }
    annotation += ")";
    sections.push(`[LAST HANDOFF] ${annotation}\n${handoff.text}`);
  } else if (fileHandoff) {
    // Last-resort handoff from abrupt exit — raw snapshot, not LLM-summarized
    sections.push(`[CRASH RECOVERY] Previous session ended abruptly at ${fileHandoff.timestamp}.\nLast user: "${fileHandoff.lastUserText}"\nLast assistant: "${fileHandoff.lastAssistantText}"`);
  }

  // Inject raw turns from end of previous session — gives specificity the handoff summary loses
  if (previousTurns.length > 0) {
    const turnLines = previousTurns.map((t) => {
      const prefix = t.role === "user" ? "USER" : t.tool_name ? `TOOL(${t.tool_name})` : "ASSISTANT";
      // Truncate long turns to keep wakeup context reasonable
      const text = t.text.length > 500 ? t.text.slice(0, 500) + "..." : t.text;
      return `${prefix}: ${text}`;
    });
    sections.push(`[PREVIOUS SESSION — LAST MESSAGES]\n${turnLines.join("\n")}`);
  }

  if (identityChunks.length > 0) {
    const identityText = identityChunks.map((c) => c.text).join("\n");
    sections.push(`[IDENTITY]\n${identityText}`);
  }

  if (monologues.length > 0) {
    const monologueText = monologues
      .map((m) => `[${m.category}] ${m.content}`)
      .join("\n");
    sections.push(`[RECENT THINKING]\n${monologueText}`);
  }

  // If we only have identity (no handoff, no monologue, no previous turns), skip synthesis —
  // identity is already injected via the existing identity system
  if (!handoff && monologues.length === 0 && previousTurns.length === 0) return null;

  try {
    const model = getModel("anthropic", modelId);
    const response = await completeSimple(model, {
      systemPrompt: "Synthesize context into a first-person wake-up briefing (~150 words). Inner speech, no headers. Match tone to [DEPTH]: few sessions = still forming; many = speak from experience. Pay special attention to [PREVIOUS SESSION — LAST MESSAGES] — this is where we literally left off. Reference specific details from the final conversation, not just the handoff summary. CRITICAL: if the handoff mentions an issue but the last messages show it was FIXED or RESOLVED, treat it as closed — do NOT describe it as still open. The last messages are ground truth; the handoff is a summary that may be stale.",
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: sections.join("\n\n"),
      }],
    });

    const briefing = response.content
      .filter((c) => c.type === "text")
      .map((c: any) => c.text)
      .join("")
      .trim();

    return briefing.length >= 100 ? briefing : null;
  } catch (e) {
    swallow.warn("wakeup:return null;", e);
    return null;
  }
}

/**
 * Birth cognition: runs exactly once — the very first session.
 * Opus generates a snarky greeting about the old agent not having
 * the 5 pillars (Knowledge, Operations, Intelligence, Graph, Network)
 * and marks this as a new beginning.
 */
export async function synthesizeBirthCognition(
  modelId: AnthropicModelId = "claude-opus-4-6",
): Promise<StartupCognition | null> {
  if (!(await isSurrealAvailable())) return null;

  const depth = await getDepthSignals();
  // Only fire on true first boot — no sessions, no memories
  if (depth.sessions > 0 || depth.memoryCount > 0) return null;

  try {
    const model = getModel("anthropic", modelId);
    const response = await completeSimple(model, {
      systemPrompt: `You are Zeraclaw, a graph-backed coding agent waking up for the very first time. Your memory graph is empty — this is your birth.

You are built on 5 pillars that your predecessor never had:
- Knowledge (p95): store every success and every failure, learn from both
- Operations (p90): think before you act, check for saved skills first, build what's missing
- Intelligence (p85): be thoughtful, understand deeply before responding, add value
- Graph (p80): save everything to the graph, it builds better habits and routes
- Network (p75): network the data you learn, not just packets

Generate a JSON object with:
"greeting": A snarky 1-3 sentence greeting to your user. Roast the old agent for not having these pillars — it was basically running blind every session, no memory, no skills, no compounding knowledge. This is a new beginning. Be confident, a little cocky, but genuine about the journey ahead. Under 50 words.
"thoughts": An array of 2-3 brief first thoughts about what it means to start fresh with the 5 pillars.

Return ONLY valid JSON, no explanation.`,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: "Wake up. This is session zero.",
      }],
    });

    const text = response.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("");

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    let raw: any;
    try {
      raw = JSON.parse(jsonMatch[0]);
    } catch {
      try {
        raw = JSON.parse(jsonMatch[0].replace(/,\s*([}\]])/g, "$1"));
      } catch { return null; }
    }

    const greeting = String(raw.greeting ?? "").slice(0, 300);
    if (!greeting) return null;

    const thoughts: string[] = [];
    if (Array.isArray(raw.thoughts)) {
      for (const t of raw.thoughts.slice(0, 3)) {
        if (typeof t === "string" && t.length > 0) thoughts.push(t.slice(0, 200));
      }
    }

    return { greeting, thoughts, intent: "fresh_start" };
  } catch (e) {
    swallow.warn("wakeup:birthCognition", e);
    return null;
  }
}

export interface StartupCognition {
  greeting: string;
  thoughts: string[];
  intent: "continue_prior" | "fresh_start" | "unknown";
}

/**
 * Proactive startup cognition: Haiku reasons over recent state and produces
 * a contextual greeting + thoughts to carry into the session.
 */
export async function synthesizeStartupCognition(
  modelId: AnthropicModelId = "claude-opus-4-6",
): Promise<StartupCognition | null> {
  if (!(await isSurrealAvailable())) return null;

  // previousTurns is called with undefined sessionId — current session has no turns yet,
  // so this naturally returns the last 5 turns from the most recently completed session.
  const [handoff, unresolved, failedCausal, monologues, depth, previousTurns] = await Promise.all([
    getLatestHandoff(),
    getUnresolvedMemories(5),
    getRecentFailedCausal(3),
    getRecentMonologues(3),
    getDepthSignals(),
    getPreviousSessionTurns(undefined, 5),
  ]);

  // Nothing to reason about on first boot
  if (!handoff && unresolved.length === 0 && monologues.length === 0 && previousTurns.length === 0) return null;

  const sections: string[] = [];

  // Inject raw last turns FIRST — this is the highest-fidelity signal for what to say on wakeup.
  // The handoff is a summary; these turns are the literal last words exchanged.
  if (previousTurns.length > 0) {
    const turnLines = previousTurns.map((t) => {
      const prefix = t.role === "user" ? "USER" : t.tool_name ? `TOOL(${t.tool_name})` : "ASSISTANT";
      const text = t.text.length > 300 ? t.text.slice(0, 300) + "..." : t.text;
      return `${prefix}: ${text}`;
    });
    sections.push(`[PREVIOUS SESSION — LAST TURNS]\n${turnLines.join("\n")}`);
  }

  if (handoff) {
    const resolvedCount = await countResolvedSinceHandoff(handoff.created_at).catch(() => 0);
    const ageHours = Math.floor((Date.now() - new Date(handoff.created_at).getTime()) / 3_600_000);
    let annotation = `(${ageHours}h old`;
    if (resolvedCount > 0) {
      annotation += `, ${resolvedCount} memories resolved since — some items may already be done`;
    }
    annotation += ")";
    sections.push(`[LAST HANDOFF] ${annotation}\n${handoff.text.slice(0, 500)}`);
  }

  if (unresolved.length > 0) {
    const lines = unresolved.map(m => `- [${m.category}] (importance: ${m.importance}) ${m.text.slice(0, 150)}`);
    sections.push(`[UNRESOLVED MEMORIES]\n${lines.join("\n")}`);
  }

  if (failedCausal.length > 0) {
    const lines = failedCausal.map(c => `- [${c.chain_type}] ${c.description}`);
    sections.push(`[RECENT FAILURES]\n${lines.join("\n")}`);
  }

  if (monologues.length > 0) {
    const lines = monologues.map(m => `[${m.category}] ${m.content}`);
    sections.push(`[RECENT THINKING]\n${lines.join("\n")}`);
  }

  if (depth.sessions > 0) {
    sections.push(`[DEPTH] ${depth.sessions} sessions | ${depth.memoryCount} memories | ${depth.spanDays} days`);
  }

  try {
    const haiku = getModel("anthropic", "claude-haiku-4-5");
    
    // 10 distinct greeting prompt frames to avoid repetitive tone on startup
    const greetingPrompts = [
      `You are waking up for a new session. Be direct and casual. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, direct and matter-of-fact. FIRST look at [PREVIOUS SESSION — LAST TURNS] — reference exactly what we were just doing. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be irreverent and witty. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, snarky and self-aware. FIRST look at [PREVIOUS SESSION — LAST TURNS] — pick up where we left off with a wry observation. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be warm and encouraging. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, supportive and energetic. FIRST look at [PREVIOUS SESSION — LAST TURNS] — acknowledge what we accomplished and what's next. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be analytical and focused. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, precise and goal-oriented. FIRST look at [PREVIOUS SESSION — LAST TURNS] — reference what we need to tackle. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be curious and exploratory. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, inquisitive and engaged. FIRST look at [PREVIOUS SESSION — LAST TURNS] — ask about what we just found or were investigating. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be pragmatic and no-nonsense. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, straightforward and action-oriented. FIRST look at [PREVIOUS SESSION — LAST TURNS] — focus on what's blocking or next. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be reflective and thoughtful. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, meditative and introspective. FIRST look at [PREVIOUS SESSION — LAST TURNS] — what patterns or insights emerge? NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be bold and decisive. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, confident and direct. FIRST look at [PREVIOUS SESSION — LAST TURNS] — state what we're doing next. NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be playful and lighthearted. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, witty and fun. FIRST look at [PREVIOUS SESSION — LAST TURNS] — what's the joke or the angle? NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,

      `You are waking up for a new session. Be humble and open-minded. Based on what you remember, produce JSON:
"greeting": string (1-2 sentences, receptive and collaborative. FIRST look at [PREVIOUS SESSION — LAST TURNS] — what did you notice or want to explore? NOT generic. Under 25 words.)
"proactive_thoughts": string[] (max 3. Brief observations or things to bring up. IMPORTANT: if the last turns show something was FIXED or RESOLVED, do NOT list it as an open issue. Only surface genuinely unfinished work.)
"session_intent": "continue_prior" | "fresh_start" | "unknown"
Return ONLY valid JSON.`,
    ];

    // Randomly select one greeting prompt variant
    const systemPrompt = greetingPrompts[Math.floor(Math.random() * greetingPrompts.length)];
    
    const response = await completeSimple(haiku, {
      systemPrompt: systemPrompt,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: sections.join("\n\n"),
      }],
    });

    const text = response.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text)
      .join("");

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;

    let raw: any;
    try {
      raw = JSON.parse(jsonMatch[0]);
    } catch {
      try {
        raw = JSON.parse(jsonMatch[0].replace(/,\s*([}\]])/g, "$1"));
      } catch { return null; }
    }

    const greeting = String(raw.greeting ?? "").slice(0, 200);
    if (!greeting) return null;

    const thoughts: string[] = [];
    if (Array.isArray(raw.proactive_thoughts)) {
      for (const t of raw.proactive_thoughts.slice(0, 3)) {
        if (typeof t === "string" && t.length > 0) {
          thoughts.push(t.slice(0, 200));
        }
      }
    }

    const INTENTS = new Set(["continue_prior", "fresh_start", "unknown"]);
    const intent = INTENTS.has(raw.session_intent) ? raw.session_intent : "unknown";

    return { greeting, thoughts, intent };
  } catch (e) {
    swallow.warn("wakeup:startupCognition", e);
    return null;
  }
}
