import type { AgentTool } from "@mariozechner/pi-agent-core";
import type { ToolContext } from "./types.js";

import { createTool as recall } from "./recall.js";
import { createTool as coreMemory } from "./core-memory.js";
import { createTool as introspect } from "./introspect.js";

const factories = [recall, coreMemory, introspect];

export function createGraphTools(ctx: ToolContext): AgentTool<any>[] {
  return factories.map(f => f(ctx));
}

export type { ToolContext } from "./types.js";
