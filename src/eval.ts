import { Agent } from "@mariozechner/pi-agent-core";
import type { AgentEvent, AgentMessage } from "@mariozechner/pi-agent-core";
import { streamSimple, getModel } from "@mariozechner/pi-ai";
import type { AssistantMessage, UserMessage, Usage } from "@mariozechner/pi-ai";
import { createTools } from "./tools.js";
import { graphTransformContext, getLastContextStats, type ContextStats } from "./graph-context.js";

const SYSTEM_PROMPT = `You are Zeraclaw, a capable coding assistant. Be concise and direct.`;

export interface EvalResult {
  prompt: string;
  full: TurnResult;
  graph: TurnResult;
  tokenSavings: number;
  tokenSavingsPct: number;
  qualityMatch: "same" | "graph-better" | "full-better" | "unclear";
  qualityNotes: string;
}

interface TurnResult {
  responseText: string;
  inputTokens: number;
  outputTokens: number;
  totalCost: number;
  toolCalls: string[];
  errors: string[];
  durationMs: number;
}

function passthroughContext(messages: AgentMessage[]): Promise<AgentMessage[]> {
  return Promise.resolve(messages);
}

async function runSinglePrompt(
  cwd: string,
  prompt: string,
  transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>,
): Promise<TurnResult> {
  const model = getModel("anthropic", "claude-sonnet-4-6"); // eval uses sonnet for cost consistency
  const tools = createTools(cwd);

  const agent = new Agent({
    transformContext,
    streamFn: streamSimple,
  });

  agent.setSystemPrompt(SYSTEM_PROMPT);
  agent.setModel(model);
  agent.setThinkingLevel("medium");
  agent.setTools(tools);

  let responseText = "";
  const toolCalls: string[] = [];
  const errors: string[] = [];
  let usage: Usage | undefined;

  const unsub = agent.subscribe((event: AgentEvent) => {
    if (event.type === "message_end") {
      const msg = event.message as AssistantMessage;
      if (msg.role === "assistant") {
        const texts = msg.content.filter((c) => c.type === "text") as { text: string }[];
        responseText += texts.map((t) => t.text).join("\n");
        usage = msg.usage;
      }
    } else if (event.type === "tool_execution_end") {
      toolCalls.push(event.toolName);
      if (event.isError) errors.push(`${event.toolName}: error`);
    }
  });

  const start = Date.now();
  try {
    await agent.prompt(prompt);
  } catch (err) {
    errors.push(String(err));
  }
  const durationMs = Date.now() - start;
  unsub();

  return {
    responseText,
    inputTokens: usage?.input ?? 0,
    outputTokens: usage?.output ?? 0,
    totalCost: usage?.cost?.total ?? 0,
    toolCalls,
    errors,
    durationMs,
  };
}

function assessQuality(full: TurnResult, graph: TurnResult): { match: EvalResult["qualityMatch"]; notes: string } {
  const notes: string[] = [];

  // Both errored
  if (full.errors.length > 0 && graph.errors.length > 0) {
    return { match: "unclear", notes: "Both had errors" };
  }
  // Only one errored
  if (graph.errors.length > 0 && full.errors.length === 0) {
    return { match: "full-better", notes: `Graph had errors: ${graph.errors.join(", ")}` };
  }
  if (full.errors.length > 0 && graph.errors.length === 0) {
    return { match: "graph-better", notes: `Full had errors: ${full.errors.join(", ")}` };
  }

  // Compare response length (very rough — short isn't always worse)
  const lenRatio = graph.responseText.length / Math.max(full.responseText.length, 1);
  if (lenRatio < 0.3) {
    notes.push(`Graph response much shorter (${graph.responseText.length} vs ${full.responseText.length} chars)`);
  }

  // Compare tool usage
  if (full.toolCalls.length > 0 && graph.toolCalls.length === 0) {
    notes.push("Full used tools but graph didn't");
  } else if (graph.toolCalls.length > 0 && full.toolCalls.length === 0) {
    notes.push("Graph used tools but full didn't");
  } else if (full.toolCalls.length === graph.toolCalls.length) {
    notes.push("Same number of tool calls");
  }

  // If both produced non-empty responses with no errors, call it same
  if (full.responseText.length > 0 && graph.responseText.length > 0 && notes.length === 0) {
    return { match: "same", notes: "Both produced responses, same tool usage pattern" };
  }

  // If only minor differences
  if (notes.length > 0 && lenRatio >= 0.3) {
    return { match: "same", notes: notes.join("; ") };
  }

  return { match: "unclear", notes: notes.join("; ") || "Unable to determine" };
}

export async function runEval(cwd: string, prompt: string): Promise<EvalResult> {
  console.log(`\n   ── A/B Eval ──`);
  console.log(`   Prompt: "${prompt.slice(0, 80)}${prompt.length > 80 ? "..." : ""}"\n`);

  // Run full context (no transform — all messages passed through)
  console.log("   [A] Running with FULL context...");
  const full = await runSinglePrompt(cwd, prompt, passthroughContext);
  console.log(`   [A] Done: ${full.inputTokens}→${full.outputTokens} tokens, ${full.toolCalls.length} tools, ${full.durationMs}ms`);

  // Run graph context
  console.log("   [B] Running with GRAPH context...");
  const graph = await runSinglePrompt(cwd, prompt, graphTransformContext);
  console.log(`   [B] Done: ${graph.inputTokens}→${graph.outputTokens} tokens, ${graph.toolCalls.length} tools, ${graph.durationMs}ms`);

  const tokenSavings = full.inputTokens - graph.inputTokens;
  const tokenSavingsPct = full.inputTokens > 0 ? (tokenSavings / full.inputTokens) * 100 : 0;
  const { match, notes } = assessQuality(full, graph);

  const result: EvalResult = {
    prompt,
    full,
    graph,
    tokenSavings,
    tokenSavingsPct,
    qualityMatch: match,
    qualityNotes: notes,
  };

  console.log(`\n   ── Results ──`);
  console.log(`   Input tokens:  ${full.inputTokens} (full) → ${graph.inputTokens} (graph) | saved ${tokenSavings} (${tokenSavingsPct.toFixed(1)}%)`);
  console.log(`   Output tokens: ${full.outputTokens} (full) → ${graph.outputTokens} (graph)`);
  console.log(`   Cost:          $${full.totalCost.toFixed(4)} (full) → $${graph.totalCost.toFixed(4)} (graph)`);
  console.log(`   Quality:       ${match}`);
  console.log(`   Notes:         ${notes}`);
  console.log("");
  console.log(`   ── Full Response (A) ──`);
  console.log(`   ${full.responseText.slice(0, 500)}`);
  console.log(`\n   ── Graph Response (B) ──`);
  console.log(`   ${graph.responseText.slice(0, 500)}`);
  console.log("");

  return result;
}

export async function runEvalSuite(cwd: string): Promise<void> {
  const prompts = [
    "What files are in the current directory?",
    "Read the package.json and tell me what dependencies this project uses",
    "What is the purpose of the graph-context.ts file?",
    "Write a hello world function in TypeScript",
    "How does the transformContext function work in this project?",
  ];

  console.log("\n   ════════════════════════════════════");
  console.log("   Zeraclaw Quality + Token Eval Suite");
  console.log("   ════════════════════════════════════\n");

  const results: EvalResult[] = [];
  for (const prompt of prompts) {
    const result = await runEval(cwd, prompt);
    results.push(result);
  }

  // Summary
  const totalFullInput = results.reduce((s, r) => s + r.full.inputTokens, 0);
  const totalGraphInput = results.reduce((s, r) => s + r.graph.inputTokens, 0);
  const totalFullCost = results.reduce((s, r) => s + r.full.totalCost, 0);
  const totalGraphCost = results.reduce((s, r) => s + r.graph.totalCost, 0);
  const qualityCounts = { same: 0, "graph-better": 0, "full-better": 0, unclear: 0 };
  for (const r of results) qualityCounts[r.qualityMatch]++;

  console.log("\n   ════════════════════════════");
  console.log("   EVAL SUMMARY");
  console.log("   ════════════════════════════");
  console.log(`   Prompts run:     ${results.length}`);
  console.log(`   Total input:     ${totalFullInput} (full) → ${totalGraphInput} (graph) | ${((1 - totalGraphInput / totalFullInput) * 100).toFixed(1)}% reduction`);
  console.log(`   Total cost:      $${totalFullCost.toFixed(4)} (full) → $${totalGraphCost.toFixed(4)} (graph)`);
  console.log(`   Quality:         ${qualityCounts.same} same | ${qualityCounts["graph-better"]} graph-better | ${qualityCounts["full-better"]} full-better | ${qualityCounts.unclear} unclear`);
  console.log("");
}
