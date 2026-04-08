/**
 * Subagent child process worker.
 *
 * Spawned by the parent via fork(). Receives config over IPC,
 * initializes its own embeddings + SurrealDB connection,
 * creates a ZeraAgent, runs the task, and reports back.
 */
import type { AgentEvent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, AssistantMessageEvent } from "@mariozechner/pi-ai";
import { createZeraAgent } from "./agent.js";
import { initEmbeddings, disposeEmbeddings } from "./embeddings.js";
import { initSurreal, closeSurreal } from "./surreal.js";
import { preflight, postflight, recordToolCall } from "./orchestrator.js";
import type {
  IpcStartMessage,
  IpcCompleteMessage,
  IpcOutputMessage,
  IpcErrorMessage,
  SubagentResult,
} from "./subagent.js";

function send(msg: IpcCompleteMessage | IpcOutputMessage | IpcErrorMessage): void {
  if (process.send) {
    process.send(msg);
  }
}

function buildSystemPrompt(mode: string, task: string): string {
  const modeNote = mode === "incognito"
    ? "You are running in INCOGNITO mode — you have your own isolated memory graph. The parent agent cannot see your work unless they explicitly merge your knowledge."
    : "You are running in FULL mode — you share the parent agent's memory graph and can read/write to it.";

  return `You are a Zeraclaw subagent — an autonomous agent spawned to handle a specific task.

${modeNote}

You have tools for reading, writing, and editing files, running shell commands, searching code, and searching your memory graph via the \`recall\` tool.

Your task: ${task}

Focus on completing the task efficiently. When done, provide a clear summary of what you accomplished and any important findings.`;
}

process.on("message", async (msg: IpcStartMessage) => {
  if (msg.type !== "start") return;

  const startTime = Date.now();
  const { config, surrealConfig, anthropicApiKey, embeddingModelPath } = msg;

  // Set API key for this process
  if (anthropicApiKey.startsWith("sk-ant-oat")) {
    process.env.ANTHROPIC_OAUTH_TOKEN = anthropicApiKey;
  } else {
    process.env.ANTHROPIC_API_KEY = anthropicApiKey;
  }

  try {
    // Initialize infrastructure (own module scope = own singletons)
    await Promise.all([
      initEmbeddings({ modelPath: embeddingModelPath, dimensions: 1024 }).catch(() => {}),
      initSurreal(surrealConfig).catch(() => {}),
    ]);

    const systemPrompt = config.systemPrompt ?? buildSystemPrompt(config.mode, config.task);
    const cwd = config.cwd ?? process.cwd();
    const modelId = config.modelId ?? "claude-opus-4-6";

    const zera = await createZeraAgent(cwd, systemPrompt, modelId);

    // Track metrics
    let turnCount = 0;
    let toolCalls = 0;
    let lastAssistantText = "";

    const unsub = zera.subscribe((event: AgentEvent) => {
      if (event.type === "message_update") {
        const e = event.assistantMessageEvent as AssistantMessageEvent;
        if (e.type === "text_delta") {
          lastAssistantText += e.delta;
          send({ type: "output", text: e.delta });
        }
      } else if (event.type === "message_end") {
        const msg = event.message as AssistantMessage;
        if (msg.role === "assistant") {
          turnCount++;
        }
      } else if (event.type === "tool_execution_start") {
        toolCalls++;
      }
    });

    // Run preflight for adaptive config
    const pf = await preflight(config.task, zera.sessionId);
    zera.configureForTurn(pf.config);

    // Execute the task
    await zera.prompt(config.task);

    // Run postflight
    await postflight(config.task, pf, toolCalls, 0, 0, Date.now() - startTime, zera.sessionId);

    // Cleanup (session summary, causal extraction, skill extraction, reflection)
    await zera.cleanup();

    unsub();

    const result: SubagentResult = {
      sessionId: zera.sessionId,
      mode: config.mode,
      status: "completed",
      summary: lastAssistantText.slice(0, 2000) || "Task completed.",
      incognitoId: config.incognitoId,
      turnCount,
      toolCalls,
      durationMs: Date.now() - startTime,
    };

    send({ type: "complete", result });
  } catch (err: any) {
    send({ type: "error", message: err.message ?? "Unknown error" });
  } finally {
    await closeSurreal();
    await disposeEmbeddings().catch(() => {});
    // Give IPC time to flush before exiting
    setTimeout(() => process.exit(0), 100);
  }
});
