/**
 * Supersedes — concept evolution tracking.
 *
 * When the daemon extracts a correction (user correcting the assistant),
 * this module finds the concept(s) that contained the stale knowledge
 * and creates `supersedes` edges from the correction memory to those
 * concepts, decaying their stability so they lose priority in recall.
 *
 * Edge direction: correction_memory -> supersedes -> stale_concept
 *
 * This ensures that:
 * 1. Stale knowledge doesn't win over corrections in retrieval
 * 2. The graph records *why* a concept was deprecated
 * 3. Stability decay is proportional to correction confidence
 *
 * Ported from kongcode/src/engine/supersedes.ts — adapted for Zeraclaw's
 * module-level DB functions instead of SurrealStore class.
 */

import { queryFirst, queryExec, relate } from "./surreal.js";
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { swallow } from "./errors.js";

/** Minimum cosine similarity to consider a concept as the target of a correction. */
const SUPERSEDE_THRESHOLD = 0.70;

/** How much to decay stability of superseded concepts (multiplicative). */
const STABILITY_DECAY_FACTOR = 0.4;

/** Floor — don't decay below this so the concept remains discoverable. */
const STABILITY_FLOOR = 0.15;

/**
 * Find concepts that match the "original" (wrong) statement in a correction,
 * create supersedes edges, and decay their stability.
 *
 * @param correctionMemId - The memory:xxx record ID of the correction
 * @param originalText    - The "original" (incorrect) text from the correction
 * @param correctionText  - The "corrected" (right) text
 * @param precomputedVec  - Optional pre-computed embedding of the full correction text
 */
export async function linkSupersedesEdges(
  correctionMemId: string,
  originalText: string,
  correctionText: string,
  precomputedVec?: number[] | null,
): Promise<number> {
  if (!isEmbeddingsAvailable() || !originalText) return 0;

  let supersededCount = 0;

  try {
    // Embed the *original* (wrong) text — that's what we're looking for in the graph
    const originalVec = await embed(originalText);
    if (!originalVec?.length) return 0;

    // Find concepts whose content is semantically similar to the wrong statement
    const candidates = await queryFirst<{ id: string; score: number; stability: number }>(
      `SELECT id, vector::similarity::cosine(embedding, $vec) AS score, stability
       FROM concept
       WHERE embedding != NONE AND array::len(embedding) > 0
         AND superseded_at IS NONE
         AND stability > $floor
       ORDER BY score DESC
       LIMIT 5`,
      { vec: originalVec, floor: STABILITY_FLOOR },
    );

    for (const candidate of candidates) {
      if (candidate.score < SUPERSEDE_THRESHOLD) break;

      const conceptId = String(candidate.id);

      // Create supersedes edge: correction -> supersedes -> stale concept
      await relate(correctionMemId, "supersedes", conceptId)
        .catch((e) => swallow("supersedes:relate", e));

      // Decay stability of the stale concept
      const currentStability = candidate.stability ?? 1.0;
      const newStability = Math.max(
        STABILITY_FLOOR,
        currentStability * STABILITY_DECAY_FACTOR,
      );

      await queryExec(
        `UPDATE $conceptId SET stability = $newStability, superseded_at = time::now(), superseded_by = $correctionId`,
        { conceptId, newStability, correctionId: correctionMemId },
      ).catch((e) => swallow("supersedes:decay", e));

      supersededCount++;
    }
  } catch (e) {
    swallow("supersedes:link", e);
  }

  return supersededCount;
}
