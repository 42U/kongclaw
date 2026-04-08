import type { AgentTool } from "@mariozechner/pi-agent-core";

export interface ToolContext {
  sessionId: string;
  cwd: string;
}

export type ToolFactory = (ctx: ToolContext) => AgentTool<any>;
