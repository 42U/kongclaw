import { Type, type Static } from "@sinclair/typebox";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";
import type { ToolFactory } from "./types.js";
import { embed, isEmbeddingsAvailable } from "../embeddings.js";
import {
  vectorSearch, graphExpand, isSurrealAvailable,
  type VectorSearchResult,
} from "../surreal.js";
import { findRelevantSkills, formatSkillContext } from "../skills.js";
import { swallow } from "../errors.js";

const recallSchema = Type.Object({
  query: Type.String({ description: "What to search for in memory. Can be a concept, topic, decision, file path, or natural language description of what you're trying to recall." }),
  scope: Type.Optional(Type.Union([
    Type.Literal("all"),
    Type.Literal("memories"),
    Type.Literal("concepts"),
    Type.Literal("turns"),
    Type.Literal("artifacts"),
    Type.Literal("skills"),
  ], { description: "Limit search to a specific memory type. Default: all." })),
  limit: Type.Optional(Type.Number({ description: "Max results to return. Default: 5, max: 15." })),
});

type RecallParams = Static<typeof recallSchema>;

export const createTool: ToolFactory = (ctx) => ({
  name: "recall",
  label: "recall",
  description: "Search your persistent memory graph for past conversations, decisions, concepts, files, and context from previous sessions. Use this when you need to remember something specific not in your current context. IMPORTANT: Context from past sessions is already auto-injected — check what you have before calling this. Call ONCE with a targeted query, not repeatedly. For checking if files/code still exist, use filesystem tools (read, grep, find) instead.",
  parameters: recallSchema,
  execute: async (_toolCallId: string, params: RecallParams): Promise<AgentToolResult<any>> => {
    if (!isEmbeddingsAvailable() || !(await isSurrealAvailable())) {
      return { content: [{ type: "text", text: "Memory system unavailable." }], details: null };
    }

    const maxResults = Math.min(params.limit ?? 5, 15);

    try {
      const queryVec = await embed(params.query);

      // Skills scope: use dedicated skill search
      const scope = params.scope ?? "all";
      if (scope === "skills") {
        const skills = await findRelevantSkills(queryVec, maxResults);
        if (skills.length === 0) {
          return { content: [{ type: "text", text: `No skills found matching "${params.query}".` }], details: null };
        }
        const formatted = formatSkillContext(skills);
        return {
          content: [{ type: "text", text: `Found ${skills.length} relevant skills:\n${formatted}` }],
          details: { count: skills.length, ids: skills.map((s) => s.id) },
        };
      }

      // Build scope-specific limits
      const limits = {
        turn: scope === "all" || scope === "turns" ? maxResults : 0,
        identity: 0,
        concept: scope === "all" || scope === "concepts" ? maxResults : 0,
        memory: scope === "all" || scope === "memories" ? maxResults : 0,
        artifact: scope === "all" || scope === "artifacts" ? maxResults : 0,
      };

      const results = await vectorSearch(queryVec, ctx.sessionId, limits);

      // Graph expand on top results for richer context
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
        } catch (e) { swallow("recall:skip", e); }
      }

      // Combine and sort by score
      const all = [...results, ...neighbors]
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
        .slice(0, maxResults);

      if (all.length === 0) {
        return { content: [{ type: "text", text: `No memories found matching "${params.query}".` }], details: null };
      }

      const formatted = all.map((r, i) => {
        const tag = r.table === "turn" ? `[${r.role ?? "turn"}]` : `[${r.table}]`;
        const time = r.timestamp ? ` (${new Date(r.timestamp).toLocaleDateString()})` : "";
        const score = r.score ? ` score:${r.score.toFixed(2)}` : "";
        return `${i + 1}. ${tag}${time}${score}\n   ${(r.text ?? "").slice(0, 500)}`;
      }).join("\n\n");

      return {
        content: [{ type: "text", text: `Found ${all.length} results for "${params.query}":\n\n${formatted}` }],
        details: { count: all.length, ids: all.map((r) => r.id) },
      };
    } catch (err) {
      return { content: [{ type: "text", text: `Memory search failed: ${err}` }], details: null };
    }
  },
});
