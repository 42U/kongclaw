/**
 * Deferred Cleanup — extract knowledge from orphaned sessions.
 *
 * When the process dies abruptly (Ctrl+C×2), session cleanup never runs.
 * On next startup, this module finds orphaned sessions (started but
 * never ended), loads their turns, generates a handoff note, and marks
 * them ended. Turns are already persisted — this just processes them.
 */
import { getOrphanedSessions, getSessionTurns, endSession, createMemory, garbageCollectMemories, garbageCollectConcepts } from "./surreal.js";
import { embed } from "./embeddings.js";
import { completeSimple, getModel } from "@mariozechner/pi-ai";
import { generateReflection } from "./reflection.js";
import { graduateCausalToSkills } from "./skills.js";
import { swallow } from "./errors.js";

let ran = false;

/**
 * Find and process orphaned sessions. Fire-and-forget from startup.
 * Only runs once per process lifetime.
 */
export async function runDeferredCleanup(
  currentSessionId: string,
): Promise<number> {
  if (ran) return 0;
  ran = true;

  try {
    const orphaned = await getOrphanedSessions(10, currentSessionId);
    if (orphaned.length === 0) return 0;

    let processed = 0;

    for (const session of orphaned) {
      try {
        await processOrphanedSession(session.id);
        processed++;
      } catch (e) {
        swallow.warn("deferredCleanup:session", e);
      }
    }

    // Run tasks that normally happen at shutdown but were interrupted:
    // reflection, causal graduation, and garbage collection.
    // These are global (not per-session) so run once after all sessions processed.
    if (processed > 0) {
      await Promise.all([
        // Reflection for each orphaned session
        ...orphaned.map(s => generateReflection(s.id).catch(e => swallow("deferred:reflection", e))),
        // Causal chain graduation
        graduateCausalToSkills().catch(e => swallow("deferred:graduate", e)),
        // Active forgetting
        garbageCollectMemories().catch(e => swallow("deferred:gc", e)),
        garbageCollectConcepts().catch(e => swallow("deferred:gcConcepts", e)),
      ]);
    }

    return processed;
  } catch (e) {
    swallow.warn("deferredCleanup:outer", e);
    return 0;
  }
}

async function processOrphanedSession(surrealSessionId: string): Promise<void> {
  // Mark ended immediately so no concurrent run picks it up
  await endSession(surrealSessionId).catch(e => swallow("deferred:endSession", e));

  const turns = await getSessionTurns(surrealSessionId, 50);
  if (turns.length < 2) return;

  // Generate handoff note from the orphaned session's turns
  try {
    const turnSummary = turns.slice(-15)
      .map(t => `[${t.role}] ${(t.text ?? "").slice(0, 200)}`)
      .join("\n");

    const opus = getModel("anthropic", "claude-opus-4-6");
    const response = await Promise.race([
      completeSimple(opus, {
        systemPrompt: "Summarize this session for handoff to your next self. What was worked on, what's unfinished, what to remember. 2-3 sentences. Write in first person.",
        messages: [{ role: "user", timestamp: Date.now(), content: turnSummary }],
      }),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Deferred handoff timed out")), 30_000),
      ),
    ]);

    const handoffText = response.content
      .filter((c: any) => c.type === "text")
      .map((c: any) => c.text).join("").trim();

    if (handoffText.length > 20) {
      let emb: number[] | null = null;
      try { emb = await embed(handoffText.slice(0, 2000)); } catch { /* ok */ }
      await createMemory(handoffText, emb, 8, "handoff", surrealSessionId);
    }
  } catch (e) {
    swallow.warn("deferredCleanup:handoff", e);
  }
}
