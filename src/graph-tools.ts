/**
 * Re-export shim — tools have moved to src/tools/.
 * Kept for backwards compat with existing imports (tests, subagent, etc).
 */
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { createTool as createRecall } from "./tools/recall.js";
import { createTool as createCore } from "./tools/core-memory.js";
import { createTool as createIntrospectFactory } from "./tools/introspect.js";

export function createRecallTool(sessionId: string): AgentTool<any> {
  return createRecall({ sessionId, cwd: "" });
}

export function createCoreMemoryTool(sessionId: string): AgentTool<any> {
  return createCore({ sessionId, cwd: "" });
}

export function createIntrospectTool(sessionId: string): AgentTool<any> {
  return createIntrospectFactory({ sessionId, cwd: "" });
}
