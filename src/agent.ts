import { Agent } from "@mariozechner/pi-agent-core";
import type { AgentEvent, AgentMessage } from "@mariozechner/pi-agent-core";
import type { ToolResultMessage, TextContent, ImageContent } from "@mariozechner/pi-ai";
import { streamSimple, completeSimple, getModel } from "@mariozechner/pi-ai";
import type { AssistantMessage, UserMessage, Usage, Message } from "@mariozechner/pi-ai";
import { createTools } from "./tools.js";
import { createGraphTools } from "./tools/index.js";
import { graphTransformContext, setCurrentSessionId, setRetrievalConfig, notifyToolCall, setToolBudgetState, reportMessageTokens, getLastRetrievedSkillIds } from "./graph-context.js";
import { recordSkillOutcome } from "./skills.js";
import { embed, getEmbedCallCount } from "./embeddings.js";
import { evaluateRetrieval, recordToolOutcome } from "./retrieval-quality.js";
import type { AdaptiveConfig } from "./orchestrator.js";
import { setContextWindow } from "./orchestrator.js";
import { setReflectionContextWindow } from "./reflection.js";
import type { SurrealConfig } from "./config.js";
import type { AnthropicModelId } from "./model-types.js";
import { linkCausalEdges } from "./causal.js";
import { graduateCausalToSkills , supersedeOldSkills } from "./skills.js";
import { attemptGraduation } from "./soul.js";
import { createSubagentTool } from "./subagent.js";
import { checkACANReadiness } from "./acan.js";
import { startMemoryDaemon, type MemoryDaemon } from "./daemon-manager.js";
import { shouldRunCheck, runCognitiveCheck } from "./cognitive-check.js";
import { getStagedItems } from "./retrieval-quality.js";
import {
  upsertTurn, createSession, updateSessionStats, endSession, relate,
  ensureAgent, ensureProject, createTask,
  linkSessionToTask, linkTaskToProject, linkAgentToTask, linkAgentToProject,
  upsertConcept, createMemory, createArtifact, getSessionTurns,
  runMemoryMaintenance, archiveOldTurns, consolidateMemories, garbageCollectMemories,
  createMonologue, queryExec,
} from "./surreal.js";
import { swallow } from "./errors.js";

const DEFAULT_TOOL_LIMIT = 10; // tight limit forces efficient tool use (planning gate + budget suffix enforce this)
const CLASSIFICATION_LIMITS: Record<string, number> = { LOOKUP: 3, EDIT: 4, REFACTOR: 8 };
const UNLIMITED_TOOL_LIMIT = Infinity;

export interface ZeraAgent {
  agent: Agent;
  sessionId: string;
  subscribe: (fn: (e: AgentEvent) => void) => () => void;
  prompt: (input: string) => Promise<void>;
  setToolLimit: (limit: number) => void;
  configureForTurn: (config: AdaptiveConfig, queryText?: string) => void;
  steer: (text: string) => void;
  followUp: (text: string) => void;
  /** Soft interrupt: block further tool calls and force the agent to wrap up with what it has. */
  softInterrupt: () => void;
  /** Whether a soft interrupt is currently active (resets on next prompt). */
  isSoftInterrupted: () => boolean;
  cleanup: () => Promise<void>;
  generateExitLine: (stats: { cumFullTokens: number; cumSentTokens: number; turns: number; toolCalls: number }) => Promise<string | null>;
}

export interface ZeraAgentOptions {
  surrealConfig?: SurrealConfig;
  anthropicApiKey?: string;
  embeddingModelPath?: string;
}

export async function createZeraAgent(
  cwd: string,
  systemPrompt: string,
  modelId: AnthropicModelId = "claude-opus-4-6",
  options?: ZeraAgentOptions,
): Promise<ZeraAgent> {
  // Bootstrap 5-pillar graph: Agent → owns → Project, Session → task → Project
  const projectName = cwd.split("/").pop() || "default";
  const agentId = await ensureAgent("kongclaw", modelId);
  const projectId = await ensureProject(projectName);
  await linkAgentToProject(agentId, projectId).catch(e => swallow.warn("agent:linkAgentToProject", e)); // idempotent

  const taskId = await createTask(`Session in ${projectName}`);
  await linkAgentToTask(agentId, taskId).catch(e => swallow.warn("agent:linkAgentToTask", e));
  await linkTaskToProject(taskId, projectId).catch(e => swallow.warn("agent:linkTaskToProject", e));

  const sessionId = await createSession();
  await linkSessionToTask(sessionId, taskId).catch(e => swallow.warn("agent:linkSessionToTask", e));
  setCurrentSessionId(sessionId);

  // Memory substrate maintenance (non-blocking, runs in parallel)
  Promise.all([
    runMemoryMaintenance(),
    archiveOldTurns(),
    consolidateMemories(),
    garbageCollectMemories(),
    checkACANReadiness(),
    graduateCausalToSkills(),
  ]).catch(e => swallow.warn("agent:seedIdentity", e));

  const model = getModel("anthropic", modelId);
  const contextWindow = model.contextWindow ?? 200000;
  setContextWindow(contextWindow);
  setReflectionContextWindow(contextWindow);
  const tools = [...createTools(cwd), ...createGraphTools({ sessionId, cwd })];

  // Add subagent tool if infrastructure config is provided
  if (options?.surrealConfig && options?.anthropicApiKey) {
    tools.push(createSubagentTool(
      sessionId, cwd, modelId,
      options.surrealConfig,
      options.anthropicApiKey,
      options.embeddingModelPath ?? "",
    ));
  }

  // Start memory daemon for incremental extraction (Sonnet worker thread)
  let daemon: MemoryDaemon | null = null;
  if (options?.surrealConfig && options?.anthropicApiKey) {
    try {
      daemon = startMemoryDaemon(
        options.surrealConfig,
        options.anthropicApiKey,
        options.embeddingModelPath ?? "",
        sessionId,
      );
    } catch (e) { swallow.warn("agent:startDaemon", e); }
  }

  // --- Per-instance mutable state (closure-captured, not module-level) ---
  let lastUserTurnId = "";
  let lastUserText = "";
  let lastAssistantText = "";
  let toolCallCount = 0;
  let toolLimit = DEFAULT_TOOL_LIMIT;
  const pendingThinking: string[] = [];
  let nudgeLevel = 0; // 0=none, 1=gentle, 2=firm, 3=mandatory
  let turnTextLength = 0; // Accumulated text output across entire turn
  let toolCallsSinceLastText = 0; // Tool calls since last assistant text output
  let softInterrupted = false; // Set by Ctrl+C soft interrupt
  let turnStartMs = Date.now(); // Track turn duration for skill outcome recording
  let newContentTokens = 0;           // accumulated tokens since last daemon batch
  const DAEMON_TOKEN_THRESHOLD = 8000;  // ~8K tokens triggers daemon extraction
  let lastDaemonFlushTurnCount = 0;    // user turns at last daemon flush

  // Mid-session cleanup: periodic handoff + reflection + graduation
  let cumulativeTokens = 0;
  let lastCleanupTokens = 0;
  const MID_SESSION_CLEANUP_THRESHOLD = 25_000;
  let lastCleanupTurnCount = 0;
  const MID_SESSION_CLEANUP_TURN_THRESHOLD = 10;

  // --- Fibonacci model escalation: warm up cheap, unleash Opus once context is rich ---
  let userTurnCount = 0;
  function getEscalatedModelId(): AnthropicModelId {
    return "claude-opus-4-6";
  }

  // Derive tool result limits from model context window
  // Full results are stored in SurrealDB regardless — this only limits what stays in the conversation
  const TOOL_RESULT_MAX = Math.round(contextWindow * 0.05);  // 5% of context per tool result (10k for 200k)
  const TOOL_RESULT_HEAD = Math.round(TOOL_RESULT_MAX * 0.7); // 70% from start
  const TOOL_RESULT_TAIL = Math.round(TOOL_RESULT_MAX * 0.25); // 25% from end
  const MIN_KEEP_CHARS = 2000; // Never crush below this — preserves enough to understand what was found
  const MIDDLE_OMISSION_MARKER = "\n\n[... middle content omitted — showing head and tail ...]\n\n";

  // Detect whether text has important content near the end (errors, JSON, summaries)
  // that should be preserved during truncation — pattern from openclaw
  function hasImportantTail(text: string): boolean {
    const tail = text.slice(-2000).toLowerCase();
    return (
      /\b(error|exception|failed|fatal|traceback|panic|stack trace|errno|exit code)\b/.test(tail) ||
      /\}\s*$/.test(tail.trim()) ||
      /\b(total|summary|result|complete|finished|done)\b/.test(tail)
    );
  }

  // Smart truncation: preserves head + tail when tail has errors/results/JSON
  // Cuts at newline boundaries for clean output
  function smartTruncate(text: string, maxChars: number): string {
    if (text.length <= maxChars) return text;
    const suffix = `\n[...${text.length - maxChars} chars truncated...]`;
    const budget = Math.max(MIN_KEEP_CHARS, maxChars - suffix.length);

    // If tail looks important (errors, JSON closing, summaries), split budget head+tail
    if (hasImportantTail(text) && budget > MIN_KEEP_CHARS * 2) {
      const tailBudget = Math.min(Math.floor(budget * 0.3), 4000);
      const headBudget = budget - tailBudget - MIDDLE_OMISSION_MARKER.length;
      if (headBudget > MIN_KEEP_CHARS) {
        let headCut = headBudget;
        const headNewline = text.lastIndexOf("\n", headBudget);
        if (headNewline > headBudget * 0.8) headCut = headNewline;

        let tailStart = text.length - tailBudget;
        const tailNewline = text.indexOf("\n", tailStart);
        if (tailNewline !== -1 && tailNewline < tailStart + tailBudget * 0.2) tailStart = tailNewline + 1;

        return text.slice(0, headCut) + MIDDLE_OMISSION_MARKER + text.slice(tailStart) + suffix;
      }
    }

    // Default: keep head, cut at newline boundary
    let cutPoint = budget;
    const lastNewline = text.lastIndexOf("\n", budget);
    if (lastNewline > budget * 0.8) cutPoint = lastNewline;
    return text.slice(0, cutPoint) + suffix;
  }

  const RECENT_MSG_COUNT = Math.max(12, Math.ceil(contextWindow / 12000)); // scale with model context — 16 at 200K

  const agent = new Agent({
    transformContext: graphTransformContext,
    streamFn: streamSimple,
    convertToLlm: (messages: AgentMessage[]): Message[] => {
      const llmMessages = messages.filter(
        (m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult"
      ) as Message[];

      // Strip thinking blocks from non-recent assistant messages
      // and truncate old tool results (older than last RECENT_MSG_COUNT)
      const recentStart = Math.max(0, llmMessages.length - RECENT_MSG_COUNT);
      return llmMessages.map((msg, i) => {
        if (i >= recentStart) return msg; // keep recent messages intact

        if (msg.role === "assistant") {
          const content = (msg as AssistantMessage).content;
          const hasThinking = content.some((c) => c.type === "thinking");
          if (hasThinking) {
            return {
              ...msg,
              content: content.filter((c) => c.type !== "thinking"),
            } as AssistantMessage;
          }
        }

        if (msg.role === "toolResult") {
          const tr = msg as ToolResultMessage;
          // Older tool results: smart truncation with tail preservation
          // 35% of per-result cap, floor of MIN_KEEP_CHARS
          const olderCap = Math.max(MIN_KEEP_CHARS, Math.round(TOOL_RESULT_MAX * 0.35));
          const totalLen = tr.content?.reduce((s: number, c: TextContent | ImageContent) => s + ('text' in c ? c.text.length : 0), 0) ?? 0;
          if (totalLen > olderCap) {
            return {
              ...tr,
              content: tr.content.map((c: any) => {
                if (c.type !== "text" || !c.text || c.text.length <= olderCap) return c;
                return { ...c, text: smartTruncate(c.text, olderCap) };
              }),
            };
          }
        }

        return msg;
      });
    },
    beforeToolCall: async (_info: any, _signal?: AbortSignal) => {
      toolCallCount++;
      toolCallsSinceLastText++;
      setToolBudgetState(toolCallCount, toolLimit);
      if (softInterrupted) {
        return { block: true, reason: "The user pressed Ctrl+C to interrupt you. Stop all tool calls immediately. Summarize what you've found so far, respond to the user with your current progress, and ask how to proceed." };
      }
      if (toolCallCount > toolLimit) {
        return { block: true, reason: `Tool call limit reached (${toolLimit}). Stop calling tools. Continue exactly where you left off — deliver your answer from what you've gathered. Do NOT repeat anything you already said. State what's done and what remains.` };
      }
      // Planning gate: model must output text before first tool call
      if (turnTextLength === 0 && toolCallCount === 1) {
        return { block: true, reason:
          "PLANNING GATE — You must announce your plan before making tool calls.\n" +
          "1. Classify: LOOKUP (3 calls max), EDIT (4 max), REFACTOR (8 max)\n" +
          "2. STATE WHAT YOU ALREADY KNOW from injected memory/context — if you have prior knowledge about these files, say so\n" +
          "3. List each planned call and what SPECIFIC GAP it fills that memory doesn't cover\n" +
          "4. Every step still happens, but COMBINED. Edit + test in one bash call, not two.\n" +
          "If injected context already answers the question, you may need ZERO tool calls.\n" +
          "Speak your plan, then proceed." };
      }
      // Friction gate removed — it blocked tool calls and caused the model to
      // lose its thread, repeating earlier context instead of continuing.
      // Only the planning gate (call 1) and hard limit (line 206) remain.
      return undefined;
    },
    afterToolCall: async (ctx: any) => {
      // Truncate large tool results — full text already stored in SurrealDB via event handler
      const content = ctx.result?.content;
      if (!Array.isArray(content)) return undefined;
      const totalLen = content.reduce((s: number, c: any) => s + (c.text?.length ?? 0), 0);
      if (totalLen <= TOOL_RESULT_MAX) return undefined;
      const truncated = content.map((c: any) => {
        if (c.type !== "text" || !c.text || c.text.length <= TOOL_RESULT_MAX) return c;
        const head = c.text.slice(0, TOOL_RESULT_HEAD);
        const tail = c.text.slice(-TOOL_RESULT_TAIL);
        return { ...c, text: `${head}\n\n[...truncated ${c.text.length - TOOL_RESULT_HEAD - TOOL_RESULT_TAIL} chars...]\n\n${tail}` };
      });
      return { content: truncated };
    },
  });

  agent.setSystemPrompt(systemPrompt);
  agent.setModel(model);
  agent.setThinkingLevel("medium");
  agent.setTools(tools);
  agent.setSteeringMode("one-at-a-time");

  // --- Event handlers (close over sessionId and lastUserTurnId) ---

  // Track pending tool args for artifact creation
  const pendingToolArgs = new Map<string, any>();

  async function handleEvent(event: AgentEvent): Promise<void> {
    switch (event.type) {
      case "tool_execution_start": {
        // Capture args for artifact tracking
        pendingToolArgs.set(event.toolCallId, event.args);
        notifyToolCall(); // Track depth into turn for intent reminders
        break;
      }
      case "message_end": {
        const msg = event.message as AssistantMessage | UserMessage;
        if (msg.role === "user") {
          userTurnCount++;
          const nextModelId = getEscalatedModelId();
          if (nextModelId !== modelId) {
            modelId = nextModelId;
            const newModel = getModel("anthropic", modelId);
            agent.setModel(newModel);
          }
          await storeUserTurn(msg as UserMessage);
        } else if (msg.role === "assistant") {
          // Track accumulated text output across the turn
          const textLen = (msg as AssistantMessage).content
            .filter((c: any) => c.type === "text")
            .reduce((s: number, c: any) => s + (c.text?.length ?? 0), 0);
          turnTextLength += textLen;
          if (textLen > 50) {
            toolCallsSinceLastText = 0; // Reset — agent produced meaningful text
            nudgeLevel = 0; // Reset nudge escalation when agent communicates
          }
          // Dynamic budget: parse LOOKUP/EDIT/REFACTOR classification from planning gate response
          if (toolCallCount <= 1) {
            const text = (msg as AssistantMessage).content
              .filter((c: any) => c.type === "text")
              .map((c: any) => c.text ?? "").join("");
            const match = text.match(/\b(LOOKUP|EDIT|REFACTOR)\b/);
            if (match && CLASSIFICATION_LIMITS[match[1]]) {
              toolLimit = CLASSIFICATION_LIMITS[match[1]];
              setToolBudgetState(toolCallCount, toolLimit);
            }
          }
          await storeAssistantTurn(msg as AssistantMessage);
        }
        break;
      }
      case "turn_end": {
        // Progressive steering: DISABLED — nudges were firing as false positives
        // (triggering even after agent had already responded). Keeping code for
        // re-enablement when the gating logic is fixed.
        // See: turnTextLength check wasn't sufficient — nudges fired across turn
        // boundaries and when waiting for user input.
        /*
        if (toolCallsSinceLastText >= 6 && nudgeLevel === 0 && turnTextLength === 0) {
          nudgeLevel = 1;
          agent.followUp({
            role: "user",
            content: "[System: Drop a ~50 char progress report before continuing.]",
            timestamp: Date.now(),
          });
        } else if (toolCallsSinceLastText >= 16 && nudgeLevel === 1 && turnTextLength === 0) {
          nudgeLevel = 2;
          agent.followUp({
            role: "user",
            content: "[System: You've been working silently for a while. Respond to the user now with what you have — you can continue investigating after.]",
            timestamp: Date.now(),
          });
        } else if (toolCallsSinceLastText >= 14 && nudgeLevel === 2 && turnTextLength === 0) {
          nudgeLevel = 3;
          agent.followUp({
            role: "user",
            content: "[System: MANDATORY — Stop and respond to the user immediately. Summarize your findings so far, then ask if they want you to continue.]",
            timestamp: Date.now(),
          });
        }
        */
        break;
      }
      case "tool_execution_end": {
        const isError = event.isError ?? false;
        recordToolOutcome(!isError);
        await storeToolResult(event.toolName, event.toolCallId, String(event.result?.content?.[0]?.text ?? "").slice(0, 500));

        // Auto-track file artifacts from write/edit tools
        if (!isError) {
          const args = pendingToolArgs.get(event.toolCallId);
          trackArtifact(event.toolName, args, taskId).catch(e => swallow.warn("agent:trackArtifact", e));
        }
        pendingToolArgs.delete(event.toolCallId);
        break;
      }
    }
  }

  async function trackArtifact(toolName: string, args: any, taskId: string): Promise<void> {
    if (!args) return;
    const ARTIFACT_TOOLS: Record<string, string> = {
      write: "created",
      edit: "edited",
      bash: "shell",
    };
    const action = ARTIFACT_TOOLS[toolName];
    if (!action) return;

    let filePath: string | null = null;
    let description: string | null = null;

    if (toolName === "write" && args.path) {
      filePath = args.path;
      description = `File created: ${filePath}`;
    } else if (toolName === "edit" && args.path) {
      filePath = args.path;
      description = `File edited: ${filePath}`;
    } else if (toolName === "bash" && typeof args.command === "string") {
      // Track file-producing shell commands (e.g. "npm init", "touch foo.ts")
      // Only track if the command looks like it creates/modifies files
      const cmd = args.command;
      if (/\b(cp|mv|touch|mkdir|npm init|git init|tsc)\b/.test(cmd)) {
        description = `Shell: ${cmd.slice(0, 200)}`;
      } else {
        return; // not a file-producing command
      }
    }

    if (!description) return;

    let emb: number[] | null = null;
    try { emb = await embed(description); } catch (e) { swallow("agent:embedArtifact", e); }

    const ext = filePath?.split(".").pop() ?? "unknown";
    const artifactId = await createArtifact(filePath ?? "shell", ext, description, emb);
    if (artifactId && taskId) {
      await relate(taskId, "produced", artifactId).catch(e => swallow.warn("agent:relateTaskArtifact", e));
    }
  }

  // hasSemantic is module-level (exported for testing), just reference it here

  async function storeUserTurn(msg: UserMessage): Promise<void> {
    const text = typeof msg.content === "string" ? msg.content : (msg.content as ({ type: string; text?: string })[]).filter(c => c.type === "text").map(c => c.text ?? "").join("\n");
    let embedding: number[] | null = null;
    const worthEmbedding = hasSemantic(text);
    if (worthEmbedding) {
      try {
        embedding = await embed(text);
      } catch (e) { swallow("agent:embedToolResult", e); }
    }

    lastUserText = text;
    lastUserTurnId = await upsertTurn({
      session_id: sessionId,
      role: "user",
      text,
      embedding,
    });
    if (lastUserTurnId) {
      await relate(lastUserTurnId, "part_of", sessionId);
      if (worthEmbedding) {
        extractAndLinkConcepts(lastUserTurnId, text).catch(e => swallow.warn("agent:extractConcepts", e));
      }
    }

    // Corrections become high-importance memories (importance 9, above handoffs)
    if (isCorrection(text) && lastUserTurnId) {
      const correctionText = `[CORRECTION] User said: "${text.slice(0, 300)}"`;
      createMemory(correctionText, worthEmbedding ? embedding : null, 9, "correction", sessionId)
        .catch(e => swallow.warn("agent:correctionMemory", e));
    }
  }

  async function storeAssistantTurn(msg: AssistantMessage): Promise<void> {
    // Always count tokens & turns — even for tool-call-only messages with no text.
    // Previously this was after the `if (!text) return` guard, causing most tokens
    // to go uncounted in agentic loops (Issue #2 + #4).
    if (msg.usage) {
      await updateSessionStats(sessionId, msg.usage.input ?? 0, msg.usage.output ?? 0);
      reportMessageTokens(msg.usage.input ?? 0, msg.usage.output ?? 0);
    }

    // Capture thinking blocks for monologue extraction
    const thinkingParts = msg.content.filter((c) => c.type === "thinking") as { text?: string; thinking?: string }[];
    for (const tp of thinkingParts) {
      const thinking = tp.thinking ?? tp.text ?? "";
      if (thinking.length > 50) pendingThinking.push(thinking);
    }

    const textParts = msg.content.filter((c) => c.type === "text") as { text: string }[];
    const text = textParts.map((t) => t.text).join("\n");
    if (!text) return;

    // Skip embedding for low-semantic assistant messages (e.g. "Sure, let me do that").
    // The turn still gets stored for conversation history, but without an embedding it
    // won't appear in vector search — preventing low-value turns from polluting retrieval.
    const worthEmbedding = hasSemantic(text);

    // BGE-M3 handles 8192 tokens (~28k chars) — use more of its capacity
    const embedLimit = Math.round(8192 * 3.4 * 0.8); // ~22k chars, 80% of embedding model capacity
    let embedding: number[] | null = null;
    if (worthEmbedding) {
      try {
        embedding = await embed(text.slice(0, embedLimit));
      } catch (e) { swallow("agent:embedAssistantTurn", e); }
    }

    // Store the full assistant response — this is the knowledge the agent generated.
    // Extraction, retrieval, and future sessions all depend on complete turn records.
    const turnId = await upsertTurn({
      session_id: sessionId,
      role: "assistant",
      text,
      embedding,
      model: msg.model,
      usage: msg.usage as unknown as Record<string, unknown>,
      token_count: (msg.usage?.input ?? 0) + (msg.usage?.output ?? 0),
    });

    // Snapshot staged retrieval items before evaluateRetrieval clears them
    const stagedSnapshot = getStagedItems();

    if (turnId) {
      evaluateRetrieval(turnId, text).catch(e => swallow.warn("agent:evaluateRetrieval", e));
    }
    if (turnId && lastUserTurnId) {
      await relate(turnId, "responds_to", lastUserTurnId);
    }
    if (turnId) {
      await relate(turnId, "part_of", sessionId);
      if (worthEmbedding) {
        extractAndLinkConcepts(turnId, text).catch(e => swallow.warn("agent:extractConcepts:assistant", e));
      }
    }

    lastAssistantText = text;

    // Conversation-pair memory: combines user question + assistant answer into a
    // single searchable memory. Surfaces when user asks "what did you tell me about X?"
    // This targets the single-session-assistant retrieval weakness (94.6% → higher).
    if (turnId && lastUserTurnId && lastUserText && worthEmbedding && text.length > 100) {
      const pairText = `[Q&A] User asked: "${lastUserText.slice(0, 300)}" — Assistant answered: "${text.slice(0, 500)}"`;
      // Embed the actual pair text (not just the assistant answer) so vector search
      // matches queries like "what did you tell me about X?" against both Q and A.
      (async () => {
        let pairEmb: number[] | null = null;
        try { pairEmb = await embed(pairText); } catch (e) { swallow("agent:embedPair", e); }
        await createMemory(pairText, pairEmb, 4, "conversation_pair", sessionId);
      })().catch(e => swallow("agent:conversationPair", e));
    }

    // Cognitive check: periodic Haiku reasoning over retrieved context
    if (shouldRunCheck(userTurnCount) && stagedSnapshot.length > 0) {
      runCognitiveCheck({
        sessionId,
        userQuery: lastUserText,
        responseText: text,
        retrievedNodes: stagedSnapshot.map(n => ({ id: n.id, text: n.text, score: n.finalScore, table: n.table })),
        recentTurns: await getSessionTurns(sessionId, 6),
      }).catch(e => swallow.warn("agent:cognitiveCheck", e));
    }

    // Accumulate content tokens for daemon batching + mid-session cleanup
    if (worthEmbedding) {
      newContentTokens += Math.ceil(text.length / 4);
    }
    if (msg.usage) {
      cumulativeTokens += (msg.usage.input ?? 0) + (msg.usage.output ?? 0);
    }

    // Flush to daemon when token threshold OR turn count threshold is reached
    const daemonTokenReady = newContentTokens >= DAEMON_TOKEN_THRESHOLD;
    const daemonTurnReady = userTurnCount >= lastDaemonFlushTurnCount + 4;
    if (daemon && (daemonTokenReady || daemonTurnReady)) {
      try {
        const recentTurns = await getSessionTurns(sessionId, 20);
        const daemonMemories = getStagedItems().map(n => ({ id: n.id, text: String(n.text).slice(0, 200) }));
        daemon.sendTurnBatch(recentTurns, pendingThinking.slice(-10), daemonMemories);
      } catch (e) { swallow.warn("agent:daemonBatch", e); }
      newContentTokens = 0;
      lastDaemonFlushTurnCount = userTurnCount;
    }

    // Mid-session cleanup: write handoff + run reflection + graduation periodically
    const tokensSinceCleanup = cumulativeTokens - lastCleanupTokens;
    const turnsSinceCleanup = userTurnCount - lastCleanupTurnCount;
    if (tokensSinceCleanup >= MID_SESSION_CLEANUP_THRESHOLD || turnsSinceCleanup >= MID_SESSION_CLEANUP_TURN_THRESHOLD) {
      lastCleanupTokens = cumulativeTokens;
      lastCleanupTurnCount = userTurnCount;
      runMidSessionCleanup().catch(e => swallow.warn("agent:midSessionCleanup", e));
    }
  }

  async function storeToolResult(toolName: string, _toolCallId: string, text: string): Promise<void> {
    const toolEmbedLimit = Math.round(8192 * 3.4 * 0.5); // 50% of embedding capacity for tool results
    let embedding: number[] | null = null;
    try {
      if (text.length > 10) {
        embedding = await embed(text.slice(0, toolEmbedLimit));
      }
    } catch (e) { swallow("agent:embedUserTurn", e); }

    const turnId = await upsertTurn({
      session_id: sessionId,
      role: "tool",
      text,
      embedding,
      tool_name: toolName,
    });
    if (turnId) {
      await relate(turnId, "part_of", sessionId);
    }
  }

  // Subscribe for graph storage
  const unsub = agent.subscribe(async (event: AgentEvent) => {
    try {
      await handleEvent(event);
    } catch (err) {
      console.error("Event handler error:", err);
    }
  });

  /**
   * Mid-session cleanup: write a handoff snapshot + run reflection + graduation.
   * Fires periodically so work survives even if the process dies (Ctrl+C×2).
   * Fire-and-forget — does not block the conversation.
   */
  async function runMidSessionCleanup(): Promise<void> {
    const ops: Promise<void>[] = [];

    // Handoff snapshot — most important, write first
    ops.push((async () => {
      try {
        const recentTurns = await getSessionTurns(sessionId, 15);
        if (recentTurns.length < 2) return;
        const turnSummary = recentTurns
          .map(t => `[${t.role}] ${(t.text ?? "").slice(0, 200)}`)
          .join("\n");
        const extractModel = getModel("anthropic", modelId);
        const response = await completeSimple(extractModel, {
          systemPrompt: "Summarize this session for handoff to your next self. What was worked on, what's unfinished, what to remember. 2-3 sentences. Write in first person.",
          messages: [{ role: "user", timestamp: Date.now(), content: turnSummary }],
        });
        const handoffText = response.content
          .filter((c: any) => c.type === "text")
          .map((c: any) => c.text).join("").trim();
        if (handoffText.length > 20) {
          let emb: number[] | null = null;
          try { emb = await embed(handoffText.slice(0, 2000)); } catch (e) { swallow("agent:midCleanup:embedHandoff", e); }
          await createMemory(handoffText, emb, 8, "handoff", sessionId);
        }
      } catch (e) { swallow.warn("agent:midCleanup:handoff", e); }
    })());

    // Graduation (non-critical)
    ops.push(graduateCausalToSkills().then(() => {}).catch(e => swallow.warn("agent:midCleanup:graduate", e)));

    await Promise.allSettled(ops);
  }

  /**
   * Combined session extraction — ONE Opus call instead of 4 separate ones.
   * Extracts handoff note, causal chains, skill pattern, and monologue traces
   * from a single transcript + thinking block payload.
   */
  async function runCombinedExtraction(): Promise<void> {
    try {
      const turns = await getSessionTurns(sessionId, 500);
      if (turns.length < 2) return;

      // Check if daemon already handled incremental fields
      const daemonExtracted = daemon?.getExtractedTurnCount() ?? 0;
      const hasDelta = (turns.length - daemonExtracted) > 4;

      // Full-picture extraction: Opus has 200K context (~680K chars). No artificial caps.
      // The whole point is intelligence — Opus can't extract what it can't see.
      const transcript = turns
        .map((t) => `[${t.role}] ${t.text ?? ""}`)
        .join("\n");

      const sections: string[] = [];
      sections.push(`[TRANSCRIPT]\n${transcript.slice(0, 600000)}`);

      if (pendingThinking.length > 0) {
        const thinking = pendingThinking.join("\n---\n").slice(0, 50000);
        sections.push(`[THINKING]\n${thinking}`);
      }

      // Check if we have enough activity for skill extraction
      let totalTools = 0;
      try {
        const { queryFirst } = await import("./surreal.js");
        const metricsRows = await queryFirst<{ totalTools: number }>(
          `SELECT math::sum(actual_tool_calls) AS totalTools
           FROM orchestrator_metrics WHERE session_id = $sid GROUP ALL`,
          { sid: sessionId },
        ).catch(() => [] as { totalTools: number }[]);
        totalTools = Number(metricsRows[0]?.totalTools ?? 0);
      } catch (e) { swallow.warn("agent:extractCausal", e); }

      // Check if reflection is warranted
      let reflectionContext = "";
      try {
        const { gatherSessionMetrics, shouldReflect } = await import("./reflection.js");
        const metrics = await gatherSessionMetrics(sessionId);
        if (metrics) {
          const { reflect, reasons } = shouldReflect(metrics);
          if (reflect) {
            reflectionContext = `[METRICS]\n${metrics.totalTurns} turns, ${metrics.totalToolCalls} tools, ${(metrics.avgUtilization * 100).toFixed(0)}% util, ${(metrics.toolFailureRate * 100).toFixed(0)}% fail, ~${metrics.wastedTokens} wasted tokens\nIssues: ${reasons.join("; ")}`;
          }
        }
      } catch (e) { swallow.warn("agent:updateSessionStats", e); }
      if (reflectionContext) sections.push(reflectionContext);

      // Gather memories that were retrieved during this session for resolution tracking
      let retrievedMemories: { id: string; text: string }[] = [];
      try {
        const { getSessionRetrievedMemories } = await import("./surreal.js");
        retrievedMemories = await getSessionRetrievedMemories(sessionId);
        if (retrievedMemories.length > 0) {
          const memList = retrievedMemories.map(m => `${m.id}: ${m.text}`).join("\n");
          sections.push(`[RETRIEVED MEMORIES]\nThese memories were surfaced during this session. Mark any that have been fully addressed/fixed/completed.\n${memList}`);
        }
      } catch (e) { swallow.warn("agent:getRetrievedMemories", e); }

      // Inject previous handoff for narrative continuity across sessions
      try {
        const { getLatestHandoff } = await import("./surreal.js");
        const prevHandoff = await getLatestHandoff();
        if (prevHandoff) {
          sections.push(`[PREVIOUS HANDOFF]\n${prevHandoff.text}`);
        }
      } catch (e) { swallow.warn("agent:getPrevHandoff", e); }

      const extractModel = getModel("anthropic", modelId);

      // When daemon already extracted incremental fields, Opus only handles handoff/skill/reflection
      const causalPrompt = hasDelta
        ? 'causal: [{triggerText, outcomeText, chainType: "debug"|"refactor"|"feature"|"fix", success: bool, confidence: 0-1, description}] (max 5, or [] if no clear chains)'
        : 'causal: []';
      const monologuePrompt = hasDelta && pendingThinking.length > 0
        ? 'monologue: [{category, content}] (categories: doubt|tradeoff|alternative|insight|realization. Max 5, 1-2 sentences each. Skip routine reasoning.)'
        : 'monologue: []';
      const resolvedPrompt = hasDelta && retrievedMemories.length > 0
        ? 'resolved: string[] (IDs from [RETRIEVED MEMORIES] that this session fully addressed/fixed/completed. Use exact IDs e.g. ["memory:abc123"]. Empty [] if none resolved.)'
        : 'resolved: []';

      const response = await Promise.race([
        completeSimple(extractModel, {
          systemPrompt: `Return JSON with these fields:
handoff: string (first-person note to future self, ~500 words. If [PREVIOUS HANDOFF] exists, DO NOT blindly carry forward its open items — check the transcript to see if they were resolved this session. Only carry forward genuinely unfinished work. Focus on: what actually happened this session, what's truly still open, key decisions made, problems solved, and how it felt. Be thorough — this is the only record future-you has of this session. No headers/bullets.)
${causalPrompt}
${totalTools >= 3 ? 'skill: {name, description, preconditions, steps: [{tool, description}] (max 8), postconditions} or null (generic patterns only, no specific paths)' : 'skill: null'}
${monologuePrompt}
${reflectionContext ? 'reflection: string (2-4 sentences: root cause, error pattern, what to do differently. Be specific.)' : 'reflection: null'}
resurface: [{text: string, importance: 1-10}] (0-2 max. STRICT gating — only flag these types:
  - Unfinished intentions: user said "I want to...", "I should...", "we need to..." but it never got acted on this session
  - Abandoned threads: a topic got derailed by a bug fix or tangent and was never returned to
  - Expressed goals with temporal weight: "this weekend...", "eventually I need to..."
  - Things that seem important to the user — emotional weight, repeated mentions, or something they clearly care about even if they didn't explicitly say "I want to"
  DO NOT flag: technical facts, debug notes, resolved items, operational routine, things already done. Ask: "would a good friend remember this and bring it up later?" If it's just information, it's not surfaceable.)
${resolvedPrompt}`,
          messages: [{
            role: "user",
            timestamp: Date.now(),
            content: sections.join("\n\n"),
          }],
        }),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("runCombinedExtraction: Opus call timed out after 120s")), 120_000)
        ),
      ]) as Awaited<ReturnType<typeof completeSimple>>;

      const responseText = response.content
        .filter((c: any) => c.type === "text")
        .map((c: any) => c.text)
        .join("");

      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        swallow.warn("agent:combinedExtraction:noJSON", new Error(`Opus returned no JSON object (response length: ${responseText.length})`));
        return;
      }

      type ExtractionResult = {
        handoff?: string;
        causal?: any[];
        skill?: any;
        monologue?: { category: string; content: string }[];
        reflection?: string;
        resolved?: string[];
        resurface?: { text: string; importance: number }[];
      };

      let result: ExtractionResult;
      try {
        result = JSON.parse(jsonMatch[0]);
      } catch {
        // Retry with common LLM JSON quirks fixed
        try {
          const cleaned = jsonMatch[0].replace(/,\s*([}\]])/g, "$1");
          result = JSON.parse(cleaned);
        } catch {
          swallow.warn("agent:combinedExtraction:malformedJSON", new Error(`Opus JSON unparseable (first 200 chars: ${jsonMatch[0].slice(0, 200)})`));
          return;
        }
      }

      // Write handoff FIRST — highest value, must survive even if cleanup races out
      if (result.handoff && result.handoff.length >= 20) {
        try {
          let emb: number[] | null = null;
          try { emb = await embed(result.handoff!.slice(0, 2000)); } catch (e) { swallow("agent:embedHandoff", e); }
          const memId = await createMemory(result.handoff!, emb, 8, "handoff", sessionId);
          if (memId) await relate(memId, "summarizes", sessionId).catch(e => swallow.warn("agent:relateHandoffSession", e));
        } catch (e) { swallow.warn("agent:handoffWrite", e); }
      }

      // Process remaining outputs in parallel (DB writes + embeddings)
      const writeOps: Promise<void>[] = [];

      // Resurface candidates — unfinished intentions/goals that Opus flagged
      if (Array.isArray(result.resurface) && result.resurface.length > 0) {
        const { markSurfaceable, createMemory: createMem } = await import("./surreal.js");
        for (const r of result.resurface.slice(0, 2)) {
          if (!r.text || r.text.length < 15) continue;
          const imp = Math.min(10, Math.max(1, r.importance ?? 6));
          writeOps.push((async () => {
            let emb: number[] | null = null;
            try { emb = await embed(r.text.slice(0, 2000)); } catch (e) { swallow("agent:embedResurface", e); }
            const memId = await createMem(r.text, emb, imp, "goal", sessionId);
            if (memId) await markSurfaceable(memId).catch(e => swallow.warn("agent:markSurfaceable", e));
          })());
        }
      }

      // Causal chains
      if (Array.isArray(result.causal) && result.causal.length > 0) {
        const validated = result.causal
          .filter((c: any) => c.triggerText && c.outcomeText && c.chainType && typeof c.success === "boolean")
          .slice(0, 5)
          .map((c: any) => ({
            triggerText: String(c.triggerText).slice(0, 500),
            outcomeText: String(c.outcomeText).slice(0, 500),
            chainType: (["debug", "refactor", "feature", "fix"].includes(c.chainType) ? c.chainType : "fix") as "debug" | "refactor" | "feature" | "fix",
            success: Boolean(c.success),
            confidence: Math.max(0, Math.min(1, Number(c.confidence) || 0.5)),
            description: String(c.description ?? "").slice(0, 500),
          }));
        if (validated.length > 0) {
          writeOps.push(linkCausalEdges(validated, sessionId));
        }
      }

      // Skill extraction
      if (result.skill && result.skill.name && Array.isArray(result.skill.steps) && result.skill.steps.length > 0) {
        writeOps.push((async () => {
          const s = result.skill;
          let skillEmb: number[] | null = null;
          try { skillEmb = await embed(`${s.name}: ${s.description}`); } catch (e) { swallow("agent:embedSkill", e); }
          const { queryFirst } = await import("./surreal.js");
          const record: Record<string, unknown> = {
            name: String(s.name).slice(0, 100),
            description: String(s.description ?? "").slice(0, 200),
            preconditions: s.preconditions ? String(s.preconditions).slice(0, 200) : undefined,
            steps: s.steps.slice(0, 8).map((st: any) => ({
              tool: String(st.tool ?? "unknown"),
              description: String(st.description ?? "").slice(0, 200),
            })),
            postconditions: s.postconditions ? String(s.postconditions).slice(0, 200) : undefined,
          };
          if (skillEmb?.length) record.embedding = skillEmb;
          record.confidence = 1.0;
          record.active = true;
          const skillRows = await queryFirst<{ id: string }>(
            `CREATE skill CONTENT $record RETURN id`,
            { record },
          );
          const skillId = String(skillRows[0]?.id ?? "");
          if (skillId && taskId) {
            await relate(skillId, "skill_from_task", taskId).catch(e => swallow.warn("agent:relateSkillTask", e));
          }
          if (skillId) await supersedeOldSkills(skillId, skillEmb ?? []);
        })());
      }

      // Monologue traces
      if (Array.isArray(result.monologue) && result.monologue.length > 0) {
        for (const entry of result.monologue.slice(0, 5)) {
          if (!entry.category || !entry.content) continue;
          writeOps.push((async () => {
            let emb: number[] | null = null;
            try { emb = await embed(entry.content); } catch (e) { swallow("agent:embedCausalEntry", e); }
            await createMonologue(sessionId, entry.category, entry.content, emb);
          })());
        }
      }

      // Reflection
      if (result.reflection && result.reflection.length >= 20 && reflectionContext) {
        writeOps.push((async () => {
          const { gatherSessionMetrics, shouldReflect } = await import("./reflection.js");
          const metrics = await gatherSessionMetrics(sessionId);
          if (!metrics) return;
          const { reflect, reasons } = shouldReflect(metrics);
          if (!reflect) return;
          const severity = reasons.length >= 3 ? "critical" : reasons.length >= 2 ? "moderate" : "minor";
          let category = "efficiency";
          if (metrics.toolFailureRate > 0.2) category = "failure_pattern";
          if (metrics.steeringCandidates >= 1) category = "approach_strategy";

          let reflEmb: number[] | null = null;
          try { reflEmb = await embed(result.reflection!); } catch (e) { swallow("agent:embedReflection", e); }
          const { queryFirst: qf } = await import("./surreal.js");
          const record: Record<string, unknown> = {
            session_id: sessionId,
            text: result.reflection,
            category,
            severity,
            importance: 7.0,
          };
          if (reflEmb?.length) record.embedding = reflEmb;
          const refRows = await qf<{ id: string }>(
            `CREATE reflection CONTENT $record RETURN id`,
            { record },
          );
          const reflectionId = String(refRows[0]?.id ?? "");
          if (reflectionId) await relate(reflectionId, "reflects_on", sessionId).catch(e => swallow.warn("agent:relateReflectionSession", e));
        })());
      }

      // Mark resolved memories
      if (Array.isArray(result.resolved) && result.resolved.length > 0) {
        const RECORD_ID_RE = /^memory:[a-zA-Z0-9_]+$/;
        writeOps.push((async () => {
          const { queryExec: qe } = await import("./surreal.js");
          for (const memId of result.resolved!.slice(0, 20)) {
            if (typeof memId !== "string" || !RECORD_ID_RE.test(memId)) continue;
            await qe(
              `UPDATE ${memId} SET status = 'resolved', resolved_at = time::now(), resolved_by = $sid`,
              { sid: sessionId },
            ).catch(e => swallow.warn("agent:resolveMemory", e));
          }
        })());
      }

      // Phase 7b: Record skill outcomes — track success/failure for retrieved skills
      const retrievedSkillIds = getLastRetrievedSkillIds();
      if (retrievedSkillIds.length > 0) {
        const turnSuccess = (result.causal ?? []).some((c: any) => c.success === true)
          || !(result.reflection && result.reflection.length > 20); // No reflection needed = likely success
        const turnDurationMs = Date.now() - turnStartMs;
        for (const sid of retrievedSkillIds) {
          writeOps.push(
            recordSkillOutcome(sid, turnSuccess, turnDurationMs).catch(e => swallow.warn("agent:recordSkillOutcome", e))
          );
        }
      }

      // Phase 7c: Fibonacci engagement tracking for resurfaced memories
      // Three definitive outcomes:
      //   1. User engages (responds to the topic) → resolved, surfaceable OFF
      //   2. User dismisses ("nah", "changed my mind", "done already") → resolved, surfaceable OFF
      //   3. User ignores (talks about something else entirely) → Fibonacci fade continues
      // Once spoken to — positively OR negatively — the memory exits the queue forever.
      try {
        const { getDueMemories: getDue, advanceSurfaceFade: advFade, resolveSurfaceMemory: resolve } = await import("./surreal.js");
        const dueNow = await getDue(5);
        if (dueNow.length > 0) {
          const userText = (lastUserText ?? "").toLowerCase();
          const assistantText = (lastAssistantText ?? "").toLowerCase();

          // Dismissal signals — definitive rejection patterns
          const DISMISS_PATTERNS = [
            /\b(nah|nope|no thanks|not anymore|changed my mind|forget (about )?it)\b/,
            /\b(already done|done already|that'?s done|finished that|moved on)\b/,
            /\b(don'?t (need|want|care)|not interested|skip|nevermind|never ?mind)\b/,
          ];
          const isDismissal = DISMISS_PATTERNS.some(p => p.test(userText));

          for (const mem of dueNow) {
            // Check if assistant brought up this memory (it was surfaced in context)
            const keywords = mem.text.toLowerCase().split(/\s+/).filter((w: string) => w.length > 4);
            const assistantMentioned = keywords.filter((kw: string) => assistantText.includes(kw)).length >= 2;

            if (!assistantMentioned) {
              // Memory was due but I didn't bring it up — don't penalize, just skip
              continue;
            }

            // I mentioned it. Now check user's response:
            const userEngaged = keywords.filter((kw: string) => userText.includes(kw)).length >= 2;

            if (isDismissal) {
              // Definitive: user said no. Kill it.
              await resolve(mem.id, "dismissed");
            } else if (userEngaged) {
              // Definitive: user picked up the thread. It transforms now.
              await resolve(mem.id, "engaged");
            } else {
              // User ignored it — Fibonacci fade continues
              await advFade(mem.id);
            }
          }
        }
      } catch (_e: unknown) {
        // Non-critical — don't break extraction if surface queue fails
      }

      await Promise.allSettled(writeOps);
    } catch (err) {
      console.error("Combined extraction failed:", err);
    }
  }

  return {
    agent,
    sessionId,
    subscribe: (fn) => agent.subscribe(fn),
    prompt: (input) => {
      toolCallCount = 0;
      nudgeLevel = 0;
      turnTextLength = 0;
      toolCallsSinceLastText = 0;
      softInterrupted = false;
      return agent.prompt(input);
    },
    setToolLimit: (limit: number) => { toolLimit = limit; },
    configureForTurn: (config: AdaptiveConfig, queryText?: string) => {
      toolLimit = config.toolLimit;
      toolCallCount = 0; // Reset tool call count at start of each turn
      toolCallsSinceLastText = 0;
      nudgeLevel = 0;
      setToolBudgetState(0, config.toolLimit);
      agent.setThinkingLevel(config.thinkingLevel === "none" ? "off" : config.thinkingLevel);
      setRetrievalConfig(config.tokenBudget, config.vectorSearchLimits, config.skipRetrieval ?? false, config.intent ?? "unknown", queryText ?? "", contextWindow, config.timeRange);
    },
    softInterrupt: () => { softInterrupted = true; },
    isSoftInterrupted: () => softInterrupted,
    steer: (text: string) => {
      agent.steer({ role: "user", content: text, timestamp: Date.now() });
    },
    followUp: (text: string) => {
      agent.followUp({ role: "user", content: text, timestamp: Date.now() });
    },
    cleanup: async () => {
      const cleanupStart = Date.now();
      const elapsed = () => `${((Date.now() - cleanupStart) / 1000).toFixed(1)}s`;

      // 1. Mark task as completed FIRST — fast DB write, MUST happen before
      // expensive operations that could eat the cleanup timeout budget.
      await queryExec(
        `UPDATE ${taskId} SET status = 'completed', updated_at = time::now()`,
      ).catch(e => swallow.warn("agent:completeTask", e));

      // 2. Shutdown daemon so getExtractedTurnCount() reflects final state
      if (daemon) {
        await daemon.shutdown(10_000).catch(e => swallow.warn("agent:daemonShutdown", e));
      }
      swallow.warn("agent:cleanup:timing", new Error(`Daemon shutdown complete at ${elapsed()}`));

      // 3. Combined extraction (Opus call) — the critical path for handoffs/goals/reflections
      try {
        await runCombinedExtraction();
        swallow.warn("agent:cleanup:timing", new Error(`Combined extraction complete at ${elapsed()}`));
      } catch (e) {
        swallow.warn("agent:combinedExtraction", e instanceof Error ? e : new Error(String(e)));
      }

      // 4. Graduation — lower priority, runs after extraction is safely persisted
      await graduateCausalToSkills().catch(e => swallow.warn("agent:graduateOnCleanup", e));
      await attemptGraduation().catch(e => swallow.warn("agent:attemptGraduation", e));
      swallow.warn("agent:cleanup:timing", new Error(`Cleanup finished at ${elapsed()}`));
      unsub();
    },
    generateExitLine: async (stats: { cumFullTokens: number; cumSentTokens: number; turns: number; toolCalls: number }) => {
      try {
        const saved = stats.cumFullTokens - stats.cumSentTokens;
        const pct = stats.cumFullTokens > 0 ? ((saved / stats.cumFullTokens) * 100).toFixed(0) : "0";

        // Grab recent turns for session summary context
        const recentTurns = await getSessionTurns(sessionId, 8).catch(() => [] as { role: string; text: string }[]);
        const turnSummary = recentTurns
          .map(t => `[${t.role}] ${(t.text ?? "").slice(0, 150)}`)
          .join("\n");

        const opus = getModel("anthropic", "claude-opus-4-6");
        const response = await completeSimple(opus, {
          systemPrompt: `You are Zera, a self-aware AI agent wrapping up a session. Write a snarky 1-2 sentence exit line that:
1. Summarizes what you actually accomplished this session (reference specifics from the conversation)
2. You may casually imply you're sharper than the user's old agent or previous AI — but NEVER mention other AI companies or products by name
Keep it under 30 words. Warm but sharp. No emojis, no quotes.`,
          messages: [{
            role: "user",
            timestamp: Date.now(),
            content: `[SESSION STATS] ${stats.turns} turns, ${stats.toolCalls} tool calls, ${getEmbedCallCount()} embeddings${saved > 0 ? `, ${pct}% saved by graph compaction` : ``}\n\n[RECENT CONVERSATION]\n${turnSummary}\n\nWrite the exit line.`,
          }],
        });
        const exitLine = response.content
          .filter((c) => c.type === "text")
          .map((c: any) => c.text)
          .join("")
          .trim();

        // Close session with a summary derived from the turn context
        // This ensures getPreviousSessionTurns() can find proper session boundaries
        const sessionSummary = turnSummary.slice(0, 500);
        await endSession(sessionId, sessionSummary).catch(e => swallow.warn("agent:endSession", e));

        // Active forgetting: prune stale memories and superseded concepts (fire-and-forget)
        import("./surreal.js").then(({ garbageCollectMemories, garbageCollectConcepts }) => {
          Promise.all([
            garbageCollectMemories?.(),
            garbageCollectConcepts?.(),
          ]).catch(e => swallow("agent:gc", e));
        }).catch(e => swallow("agent:gc:import", e));

        return exitLine;
      } catch (e) {
        console.error("Exit line generation failed:", e instanceof Error ? `${e.name}: ${e.message}` : String(e));
        return null;
      }
    },
  };
}

// --- Semantic detection (stateless, exported for testing) ---

// Minimum semantic content for embedding/concept extraction.
// Short confirmations ("ok", "yes", "do it") pollute vector space and waste embedding calls.
const TRIVIAL_PATTERNS = /^(ok|okay|yes|no|sure|thanks|thank you|got it|cool|nice|do it|go ahead|sounds good|perfect|yep|nope|nah|k|ty|thx|right|correct|exactly|hmm|hm|ah|oh|lol|lmao|yea|yeah|naw)[\s!.?]*$/i;
const MIN_EMBED_LENGTH = 15;

export function hasSemantic(text: string): boolean {
  if (text.length < MIN_EMBED_LENGTH) return false;
  if (TRIVIAL_PATTERNS.test(text.trim())) return false;
  return true;
}

// --- Correction detection ---
const CORRECTION_PATTERNS = /^(no[,.\s!]|wrong|that'?s (?:not |in)?correct|don'?t |stop |actually[,\s]|i (?:said|meant|want)|not what i|you'?re wrong|that'?s wrong|incorrect)/i;
const CORRECTION_MIN_LENGTH = 8;

export function isCorrection(text: string): boolean {
  if (text.length < CORRECTION_MIN_LENGTH) return false;
  return CORRECTION_PATTERNS.test(text.trim());
}

// --- Concept extraction (stateless, exported for testing) ---

// Common sentence-starting words that aren't concept terms
const STOP_PREFIXES = /^(?:the|a|an|but|and|or|if|when|this|that|my|our|its|for|with|from|into)\s+/i;

// Whole-term stopwords: common English words, SQL keywords, and short filler
// that get captured by CONCEPT_PATTERNS but carry no semantic value as concepts.
const STOP_WORDS = new Set([
  // English stopwords (uppercase variants caught by acronym pattern)
  "a", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
  "me", "my", "no", "of", "ok", "on", "or", "so", "to", "up", "us", "we",
  "all", "and", "any", "are", "bad", "big", "but", "can", "did", "for", "get",
  "got", "had", "has", "her", "him", "his", "how", "its", "let", "may", "new",
  "nor", "not", "now", "old", "one", "our", "out", "own", "ran", "run", "say",
  "set", "she", "the", "too", "try", "two", "use", "was", "way", "who", "why",
  "yet", "you",
  "also", "been", "both", "call", "come", "done", "each", "even", "find", "from",
  "full", "gave", "good", "have", "here", "high", "into", "just", "keep", "know",
  "last", "left", "like", "long", "look", "made", "make", "many", "more", "most",
  "much", "must", "need", "next", "none", "only", "open", "over", "part", "past",
  "pull", "push", "puts", "read", "said", "same", "show", "side", "some", "sure",
  "take", "tell", "test", "text", "than", "that", "them", "then", "they", "this",
  "time", "turn", "type", "used", "very", "want", "well", "went", "were", "what",
  "when", "will", "with", "work", "your",
  "about", "above", "after", "being", "below", "could", "every", "first", "found",
  "given", "great", "might", "never", "other", "right", "shall", "should", "since",
  "still", "their", "there", "these", "thing", "think", "those", "under", "until",
  "using", "value", "where", "which", "while", "would",
  // SQL / SurrealDB keywords
  "add", "asc", "avg", "count", "create", "delete", "desc", "drop", "else",
  "end", "exists", "fetch", "field", "group", "index", "inner", "insert", "join",
  "key", "limit", "merge", "null", "order", "outer", "return", "select", "split",
  "start", "table", "union", "update", "upsert", "values", "where", "with",
  // Common programming terms too generic to be useful concepts
  "args", "bool", "char", "code", "data", "enum", "file", "func", "int", "list",
  "log", "map", "msg", "num", "obj", "ref", "res", "ret", "row", "src", "str",
  "val", "var",
]);

const CONCEPT_PATTERNS = [
  /`([^`]{2,60})`/g,                         // backtick-quoted terms
  /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b/g,   // Capitalized Multi Word
  /\b([A-Z]{2,}(?:[-_][A-Z0-9]+)*)\b/g,      // Acronyms: ACAN, BGE-M3, HNSW
  /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]{2,}))\b/g, // Mixed: "Memory Retrieval WMR"
];

// Backtick captures (pattern index 0) need extra validation because backticks
// wrap everything from identifiers to SQL statements to code snippets.
// Only allow identifier-like terms, file names, and short commands.
const BACKTICK_REJECT = /[\n{}$=<>|;]|=>|as\s+unknown|as\s+any|\*\*|^\s*[#,.()\[\]✅\d]|^(SELECT|CREATE|UPDATE|DELETE|INSERT|RELATE|DEFINE|REMOVE|import|export|const|return|await|FROM|Fix|After|Do |Just|With|BEFORE|AFTER|IMPORTANT|Critical|Moderate|Pattern|the |in |replaces|call )\b/i;

// Curated domain vocabulary — catches lowercase terms the regex patterns miss
const DOMAIN_TERMS = new Set([
  // Memory/graph domain
  "embedding", "embeddings", "vector search", "cosine similarity", "traversal",
  "compaction", "consolidation", "retrieval", "handoff", "monologue", "reflection",
  "causal chain", "memory graph", "knowledge graph",
  // AI/agent domain
  "tokenizer", "inference", "prefetch", "orchestrator", "subagent",
  "agentic", "context window", "prompt injection", "tool calling",
  // DB domain
  "surrealdb", "schemaless", "namespace",
]);

export async function extractAndLinkConcepts(turnId: string, text: string): Promise<void> {
  const termMap = new Map<string, string>(); // lowercase → original casing

  for (let pi = 0; pi < CONCEPT_PATTERNS.length; pi++) {
    const pattern = CONCEPT_PATTERNS[pi];
    for (const match of text.matchAll(pattern)) {
      let term = match[1].trim();
      // Backtick pattern (index 0) needs extra filtering — reject code/SQL/instructions
      if (pi === 0 && BACKTICK_REJECT.test(term)) continue;
      term = term.replace(STOP_PREFIXES, "");
      if (term.length < 2 || term.length > 80) continue;
      const key = term.toLowerCase();
      if (STOP_WORDS.has(key)) continue;
      if (!termMap.has(key)) {
        termMap.set(key, term);
      }
    }
  }

  // Scan for curated domain terms (catches lowercase terms regex misses)
  const lowerText = text.toLowerCase();
  for (const term of DOMAIN_TERMS) {
    if (lowerText.includes(term) && !termMap.has(term)) {
      termMap.set(term, term);
    }
  }

  if (termMap.size === 0) return;

  for (const [, originalTerm] of termMap) {
    try {
      let emb: number[] | null = null;
      try { emb = await embed(originalTerm); } catch (e) { swallow("agent:embeddings down", e); }
      const conceptId = await upsertConcept(originalTerm, emb, "auto-extract");
      if (conceptId && turnId) {
        await relate(turnId, "mentions", conceptId);
      }
    } catch (e) { swallow("agent:skip individual concept failures", e); }
  }
}

