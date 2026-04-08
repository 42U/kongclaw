import { Type, type Static } from "@sinclair/typebox";
import type { AgentToolResult } from "@mariozechner/pi-agent-core";
import type { ToolFactory } from "./types.js";
import {
  isSurrealAvailable,
  getAllCoreMemory, createCoreMemory, updateCoreMemory, deleteCoreMemory,
} from "../surreal.js";
import { invalidateCoreMemoryCache } from "../graph-context.js";

const coreMemorySchema = Type.Object({
  action: Type.Union([
    Type.Literal("list"),
    Type.Literal("add"),
    Type.Literal("update"),
    Type.Literal("deactivate"),
  ], { description: "Action to perform on core memory." }),
  tier: Type.Optional(Type.Number({ description: "Filter by tier (0=always loaded, 1=session-pinned). Default: list all." })),
  category: Type.Optional(Type.String({ description: "Category filter or category for new entry (identity/rules/tools/operations/general)." })),
  text: Type.Optional(Type.String({ description: "Text content for add/update actions." })),
  priority: Type.Optional(Type.Number({ description: "Priority for add/update (higher=loaded first). Default: 50." })),
  id: Type.Optional(Type.String({ description: "Record ID for update/deactivate actions (e.g. core_memory:abc123)." })),
  session_id: Type.Optional(Type.String({ description: "Session ID for Tier 1 entries — auto-deactivated when session ends." })),
});

type CoreMemoryParams = Static<typeof coreMemorySchema>;

export const createTool: ToolFactory = (ctx) => ({
  name: "core_memory",
  label: "core_memory",
  description: "Manage your always-loaded core directives (Tier 0) and session-pinned context (Tier 1). Tier 0 entries are present in EVERY turn — use for identity, rules, tool patterns, operational knowledge. Tier 1 entries are pinned for the current session — use for active task context, user preferences, ongoing work notes.",
  parameters: coreMemorySchema,
  execute: async (_toolCallId: string, params: CoreMemoryParams): Promise<AgentToolResult<any>> => {
    if (!(await isSurrealAvailable())) {
      return { content: [{ type: "text", text: "Database unavailable." }], details: null };
    }

    try {
      switch (params.action) {
        case "list": {
          const entries = await getAllCoreMemory(params.tier);
          if (entries.length === 0) {
            return { content: [{ type: "text", text: "No core memory entries found." }], details: null };
          }
          const formatted = entries.map((e, i) => {
            const sid = e.session_id ? ` session:${e.session_id}` : "";
            return `${i + 1}. [T${e.tier}/${e.category}/p${e.priority}${sid}] ${e.id}\n   ${e.text.slice(0, 200)}`;
          }).join("\n\n");
          return {
            content: [{ type: "text", text: `${entries.length} core memory entries:\n\n${formatted}` }],
            details: { count: entries.length },
          };
        }

        case "add": {
          if (!params.text) {
            return { content: [{ type: "text", text: "Error: 'text' is required for add action." }], details: null };
          }
          const tier = params.tier ?? 0;
          const sid = tier === 1 ? (params.session_id ?? ctx.sessionId) : undefined;
          const id = await createCoreMemory(
            params.text,
            params.category ?? "general",
            params.priority ?? 50,
            tier,
            sid,
          );
          if (!id) {
            return {
              content: [{ type: "text", text: "FAILED: Core memory entry was not created. The database may be unavailable." }],
              details: { error: true },
            };
          }
          invalidateCoreMemoryCache();
          return {
            content: [{ type: "text", text: `Created core memory entry: ${id} (tier ${tier}, ${params.category ?? "general"}, p${params.priority ?? 50})` }],
            details: { id },
          };
        }

        case "update": {
          if (!params.id) {
            return { content: [{ type: "text", text: "Error: 'id' is required for update action." }], details: null };
          }
          const fields: Record<string, unknown> = {};
          if (params.text !== undefined) fields.text = params.text;
          if (params.category !== undefined) fields.category = params.category;
          if (params.priority !== undefined) fields.priority = params.priority;
          if (params.tier !== undefined) fields.tier = params.tier;
          const updated = await updateCoreMemory(params.id, fields);
          if (!updated) {
            return {
              content: [{ type: "text", text: `FAILED: Could not update ${params.id}. Record may not exist.` }],
              details: { error: true },
            };
          }
          invalidateCoreMemoryCache();
          return {
            content: [{ type: "text", text: `Updated core memory: ${params.id}` }],
            details: { id: params.id },
          };
        }

        case "deactivate": {
          if (!params.id) {
            return { content: [{ type: "text", text: "Error: 'id' is required for deactivate action." }], details: null };
          }
          await deleteCoreMemory(params.id);
          invalidateCoreMemoryCache();
          return {
            content: [{ type: "text", text: `Deactivated core memory: ${params.id}` }],
            details: { id: params.id },
          };
        }
      }
    } catch (err) {
      return { content: [{ type: "text", text: `Core memory operation failed: ${err}` }], details: null };
    }
  },
});
