/**
 * TUI — pi-tui based terminal UI for Zeraclaw (Phase 2).
 *
 * Component-based TUI: pinned editor at bottom, scrollable chat log above,
 * animated status bar, styled markdown output, stateful tool display boxes.
 */
import chalk from "chalk";
import type { AgentEvent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage as PiAssistantMessage, AssistantMessageEvent } from "@mariozechner/pi-ai";
import {
  CombinedAutocompleteProvider,
  Container,
  Editor,
  type EditorTheme,
  type MarkdownTheme,
  matchesKey,
  ProcessTerminal,
  Text,
  TUI,
} from "@mariozechner/pi-tui";
import { highlight } from "cli-highlight";
import { createZeraAgent, type ZeraAgent, type ZeraAgentOptions } from "./agent.js";
import { getLastContextStats } from "./graph-context.js";
import { runEval, runEvalSuite } from "./eval.js";
import { getSessionQualityStats } from "./retrieval-quality.js";
import { getReflectionCount } from "./reflection.js";
import { preflight, postflight, recordToolCall, type PreflightResult } from "./orchestrator.js";
import { predictQueries, prefetchContext } from "./prefetch.js";
import { spawnSubagent, mergeFromIncognito, listSubagents } from "./subagent.js";
import {
  hasUserIdentity, findWakeupFile, readWakeupFile,
  deleteWakeupFile, saveUserIdentity, buildWakeupPrompt,
} from "./identity.js";
import { synthesizeWakeup, synthesizeStartupCognition, synthesizeBirthCognition } from "./wakeup.js";
import { deactivateSessionMemories, closeSurreal, markShutdown, createCoreMemory } from "./surreal.js";
import { disposeEmbeddings } from "./embeddings.js";
import { invalidateCoreMemoryCache } from "./graph-context.js";
import { swallow } from "./errors.js";
import { buildEffectivePrompt } from "./prompt.js";
import { setActiveSessionState, clearActiveSessionState } from "./index.js";
import { runDeferredCleanup } from "./deferred-cleanup.js";
import {
  AssistantMessage,
  FooterBar,
  StatusLine,
  SystemMessage,
  ToolExecutionComponent,
  UserMessage,
  formatTokenCount,
} from "./tui-components.js";

// ── Theme ────────────────────────────────────────────────────────────────────

const mdTheme: MarkdownTheme = {
  heading: (t) => chalk.bold.cyan(t),
  link: (t) => chalk.cyan.underline(t),
  linkUrl: (t) => chalk.dim(t),
  code: (t) => chalk.cyan(t),
  codeBlock: (t) => t,
  codeBlockBorder: (t) => chalk.dim(t),
  quote: (t) => chalk.dim(t),
  quoteBorder: (t) => chalk.dim(t),
  hr: (t) => chalk.dim(t),
  listBullet: (t) => chalk.cyan(t),
  bold: (t) => chalk.bold(t),
  italic: (t) => chalk.italic(t),
  strikethrough: (t) => chalk.strikethrough(t),
  underline: (t) => chalk.underline(t),
  highlightCode: (code, lang) => {
    try {
      return highlight(code, { language: lang || "plaintext" }).split("\n");
    } catch {
      return code.split("\n");
    }
  },
  codeBlockIndent: "  ",
};

const edTheme: EditorTheme = {
  borderColor: (t: string) => chalk.cyan(t),
  selectList: {
    selectedPrefix: (t: string) => chalk.cyan(t),
    selectedText: (t: string) => chalk.bgCyan.black(t),
    description: (t: string) => chalk.dim(t),
    scrollInfo: (t: string) => chalk.dim(t),
    noMatch: (t: string) => chalk.dim(t),
  },
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function truncate(s: string, max = 60): string {
  if (!s) return "";
  const clean = s.replace(/\n/g, "↵").trim();
  return clean.length > max ? clean.slice(0, max) + "…" : clean;
}

/** Whether TUI is in graceful shutdown. Exported for index.ts SIGINT coordination. */
export let shuttingDown = false;

// ── Main TUI entry ──────────────────────────────────────────────────────────

export async function startTui(cwd: string, modelId?: string, agentOptions?: ZeraAgentOptions): Promise<void> {
  // ── Build component tree ──
  const terminal = new ProcessTerminal();
  const tui = new TUI(terminal);
  const root = new Container();

  const header = new Text(
    chalk.cyan.bold("⟡ Zeraclaw") + chalk.dim("  loading..."),
    1, 0,
  );
  const chatLog = new Container();
  const statusLine = new StatusLine(tui);
  const footerBar = new FooterBar();
  const editor = new Editor(tui, edTheme);

  root.addChild(header);
  root.addChild(chatLog);
  root.addChild(statusLine);
  root.addChild(editor);


  root.addChild(footerBar);
  tui.addChild(root);
  tui.setFocus(editor);

  // Autocomplete for slash commands
  const slashCommands = [
    { name: "/help", description: "Show available commands" },
    { name: "/stats", description: "Session statistics" },
    { name: "/quit", description: "Exit Zeraclaw" },
    { name: "/clear", description: "Clear chat log" },
    { name: "/eval", description: "Run eval suite" },
    { name: "/compare", description: "Compare retrieval" },
    { name: "/spawn", description: "Spawn subagent" },
    { name: "/merge", description: "Merge incognito data" },
    { name: "/wakeup", description: "Set up identity" },
    { name: "/agents", description: "List subagents" },
    { name: "/unlimited", description: "Remove tool limit" },
    { name: "/export-training", description: "Export training data" },
  ];
  const autocomplete = new CombinedAutocompleteProvider(slashCommands, cwd);
  editor.setAutocompleteProvider(autocomplete);

  // ── TUI helpers ──
  function addSystem(text: string): void {
    if (shuttingDown) return;
    chatLog.addChild(new SystemMessage(text));
    tui.requestRender();
  }

  function pruneMessages(): void {
    if (chatLog.children.length > 150) {
      const excess = chatLog.children.length - 120;
      chatLog.children.splice(0, excess);
      chatLog.invalidate();
      tui.requestRender();
    }
  }

  // ── State ──
  let shutdownResolve: (() => void) | null = null;
  let currentAssistant: AssistantMessage | null = null;
  let currentText = "";
  let wakeupActive = false;
  let toolCallsThisTurn = 0;
  let sessionToolCalls = 0;
  const pendingTools = new Map<string, ToolExecutionComponent>();
  const pendingToolArgs = new Map<string, { name: string; args: any }>();
  let sessionTurns = 0;
  let totalTokensIn = 0;
  let totalTokensOut = 0;
  let totalCacheRead = 0;
  let totalCacheWrite = 0;
  let totalCost = 0;
  let sessionCost = 0;
  let agentActive = false;
  let muted = false;
  // Per-turn stats — updated at end of each response, displayed pinned in footer row 2
  let lastTurnCtxSent = 0;
  let lastTurnCtxFull = 0;
  let lastTurnGraphNodes = 0;
  let lastTurnToolsCount = 0;

  function updateFooter(): void {
    const stats = getLastContextStats();
    const compressionPct = stats && stats.cumFullTokens > 0
      ? (1 - stats.cumSentTokens / stats.cumFullTokens) * 100
      : undefined;
    footerBar.update({
      turns: sessionTurns,
      toolCalls: sessionToolCalls,
      cost: sessionCost,
      tokensIn: totalTokensIn,
      tokensOut: totalTokensOut,
      model: modelId,
      compressionPct,
      lastCtxSent: lastTurnCtxSent,
      lastCtxFull: lastTurnCtxFull,
      graphNodes: lastTurnGraphNodes,
      lastTurnTools: lastTurnToolsCount,
      lastTurnSessionTools: sessionToolCalls,
    });
    if (!shuttingDown) tui.requestRender();
  }

  // ── Start TUI ──
  tui.start();

  // ── Wake up ──
  statusLine.setBusy("Waking up...");
  tui.requestRender();

  const wakeStart = performance.now();
  let startupGreeting = "";
  let startupThoughts: string[] = [];
  let effectivePrompt: string;
  try {
    const [briefing, cognition] = await Promise.all([
      synthesizeWakeup(modelId, undefined, cwd),
      synthesizeStartupCognition(modelId),
    ]);
    effectivePrompt = buildEffectivePrompt(briefing);
    if (cognition) {
      startupGreeting = cognition.greeting;
      startupThoughts = cognition.thoughts;
    } else {
      const birth = await synthesizeBirthCognition(modelId);
      if (birth) {
        startupGreeting = birth.greeting;
        startupThoughts = birth.thoughts;
      }
    }
  } catch (e) {
    swallow("tui:non-critical — proceed without briefing", e);
    effectivePrompt = buildEffectivePrompt();
  }

  const zera = await createZeraAgent(cwd, effectivePrompt, modelId, agentOptions);

  // Register session for sync handoff on abrupt exit
  setActiveSessionState({
    sessionId: zera.sessionId,
    lastUserText: "",
    lastAssistantText: "",
    unextractedTokens: 0,
    cwd,
  });

  // Deferred cleanup: process orphaned sessions from previous crashes (background)
  runDeferredCleanup(zera.sessionId)
    .then(n => { if (n > 0) console.log(`  Deferred cleanup: processed ${n} orphaned session(s)`); })
    .catch(e => swallow.warn("tui:deferredCleanup", e));

  // Pin proactive thoughts
  if (startupThoughts.length > 0) {
    const thoughtsText = startupThoughts.map(t => `- ${t}`).join("\n");
    createCoreMemory(
      `[PROACTIVE THOUGHTS]\n${thoughtsText}`,
      "proactive", 8, 1, zera.sessionId,
    ).then(() => invalidateCoreMemoryCache()).catch(e => swallow("tui:proactive-thoughts", e));
  }

  // Soft ambient hint — follow the thread if user responds, let it fade if they pivot
  if (startupGreeting) {
    createCoreMemory(
      `[SESSION OPEN] Greeting shown to user: "${startupGreeting}"\nIf the user responds to this, continue that thread. If they start a different topic, let this fade — do not force it.`,
      "greeting", 15, 1, zera.sessionId,
    ).then(() => invalidateCoreMemoryCache()).catch(e => swallow("tui:startup-greeting", e));
  }

  statusLine.setIdle();

  const wakeSec = ((performance.now() - wakeStart) / 1000).toFixed(1);
  header.setText(
    chalk.cyan.bold("⟡ Zeraclaw") + chalk.dim(`  ${zera.sessionId}`),
  );
  addSystem(
    chalk.green("Ready") + chalk.dim(` (${wakeSec}s)`) +
    (startupGreeting ? `\n${chalk.italic(startupGreeting)}` : ""),
  );
  if (modelId) {
  }

  // ── Agent event subscriber ──
  zera.subscribe((event: AgentEvent) => {
    if (event.type === "message_update") {
      const e = event.assistantMessageEvent as AssistantMessageEvent;
      if (e.type === "text_delta" && !muted) {
        currentText += e.delta;
        // Lazy creation: handles deltas arriving before agent_start
        if (!currentAssistant) {
          currentAssistant = new AssistantMessage(mdTheme);
          chatLog.addChild(currentAssistant);
        }
        currentAssistant.setText(currentText);
        tui.requestRender();
      }
    } else if (event.type === "message_end") {
      const msg = event.message as PiAssistantMessage;
      if (msg.role === "assistant") {
        const u = msg.usage;
        if (u) {
          totalTokensIn += u.input ?? 0;
          totalTokensOut += u.output ?? 0;
          totalCacheRead += u.cacheRead ?? 0;
          totalCacheWrite += u.cacheWrite ?? 0;
          totalCost += u.cost?.total ?? 0;
        }
        // Show stats after final response (not after tool_use stops)
        if (u && msg.stopReason !== "toolUse" && !muted) {
          // Capture per-turn stats into state vars — footer row 2 displays them pinned
          const ctxStats = getLastContextStats();
          if (ctxStats) {
            lastTurnCtxSent = ctxStats.cumSentTokens;
            lastTurnCtxFull = ctxStats.cumFullTokens;
            lastTurnGraphNodes = ctxStats.graphNodes ?? 0;
          }
          lastTurnToolsCount = toolCallsThisTurn;
        }
        currentText = "";
        currentAssistant = null;
      }
    } else if (event.type === "tool_execution_start") {
      toolCallsThisTurn++;
      sessionToolCalls++;
      recordToolCall(event.toolName);
      pendingToolArgs.set(event.toolCallId, { name: event.toolName, args: event.args });

      // Finalize current assistant text before tool box
      currentText = "";
      currentAssistant = null;

      // Add stateful tool component
      const toolComp = new ToolExecutionComponent(event.toolName, event.args);
      pendingTools.set(event.toolCallId, toolComp);
      chatLog.addChild(toolComp);

      // Update status to show running tool
      statusLine.setBusy(`Running ${event.toolName}...`);
      tui.requestRender();
    } else if (event.type === "tool_execution_end") {
      const toolComp = pendingTools.get(event.toolCallId);
      const args = pendingToolArgs.get(event.toolCallId)?.args;
      pendingTools.delete(event.toolCallId);
      pendingToolArgs.delete(event.toolCallId);

      if (toolComp) {
        toolComp.setResult(event.result, event.isError, args);
      }

      // Revert status
      if (pendingTools.size === 0) {
        statusLine.setBusy("Thinking...");
      }
      tui.requestRender();
    } else if (event.type === "agent_start") {
      toolCallsThisTurn = 0;
      sessionTurns++;
      totalTokensIn = 0;
      totalTokensOut = 0;
      totalCacheRead = 0;
      totalCacheWrite = 0;
      totalCost = 0;
      currentText = "";
      currentAssistant = null;

      statusLine.setBusy("Thinking...");
      tui.requestRender();
    }
    // Track session-wide cost
    if (event.type === "message_end") {
      const cost = (event.message as { usage?: { cost?: { total?: number } } })?.usage?.cost?.total;
      if (cost) sessionCost += cost;
      updateFooter();
    }
  });

  // ── Identity extraction helper ──
  async function extractAndSaveIdentity(agent: ZeraAgent): Promise<void> {
    addSystem(chalk.dim("Saving identity to persistent memory..."));
    muted = true;
    let extractedText = "";
    const unsub = agent.subscribe((event: AgentEvent) => {
      if (event.type === "message_update") {
        const e = event.assistantMessageEvent as AssistantMessageEvent;
        if (e.type === "text_delta") extractedText += e.delta;
      }
    });
    await agent.prompt(`Based on the identity we just established, output ONLY a JSON array of 4-8 identity statements — short, declarative sentences defining who you are, your personality, tone, role, and principles. Each should be standalone and retrievable independently. No explanation, just the JSON array.`);
    unsub();
    muted = false;

    try {
      const jsonMatch = extractedText.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const chunks: string[] = JSON.parse(jsonMatch[0]);
        if (Array.isArray(chunks) && chunks.length > 0) {
          const saved = await saveUserIdentity(chunks);
          addSystem(chalk.green(`Saved ${saved} identity chunks to memory graph.`));
          return;
        }
      }
      addSystem(chalk.yellow("Warning: Could not parse identity extraction. Try /wakeup again."));
    } catch (e) {
      swallow("tui:identity-extraction-failed", e);
      addSystem(chalk.yellow("Warning: Identity extraction failed. Try /wakeup again."));
    }
  }

  // ── WAKEUP.md check ──
  const wakeupPath = findWakeupFile(cwd);
  const needsWakeup = wakeupPath && !(await hasUserIdentity());
  if (needsWakeup && wakeupPath) {
    const wakeupContent = readWakeupFile(wakeupPath);
    const { systemAddition, firstMessage } = buildWakeupPrompt(wakeupContent);
    addSystem(chalk.cyan("First run detected — establishing identity from WAKEUP.md"));
    zera.steer(systemAddition);
    await zera.prompt(firstMessage);
    await extractAndSaveIdentity(zera);
    deleteWakeupFile(wakeupPath);
    addSystem(chalk.dim("WAKEUP.md deleted. Identity established."));
  }

  // ── Keyboard shortcuts ──
  let lastCtrlCAt = 0;
  tui.addInputListener((data) => {
    // Ctrl+C — soft interrupt / exit
    if (matchesKey(data, "ctrl+c")) {
      if (shuttingDown) {
        process.exit(1);
      }
      if (agentActive && !zera.isSoftInterrupted()) {
        zera.softInterrupt();
        addSystem(chalk.yellow("Soft interrupt — wrapping up current work..."));
        return { consume: true };
      }
      const now = Date.now();
      if (now - lastCtrlCAt < 1000) {
        gracefulShutdown();
        return { consume: true };
      }
      lastCtrlCAt = now;
      addSystem(chalk.dim("Press Ctrl+C again to exit"));
      return { consume: true };
    }

    // Ctrl+L — clear chat log
    if (matchesKey(data, "ctrl+l")) {
      chatLog.clear();
      chatLog.invalidate();
      tui.requestRender();
      return { consume: true };
    }

    // Escape — clear status line
    if (matchesKey(data, "escape")) {
      if (!agentActive) {
        statusLine.setIdle();
        tui.requestRender();
        return { consume: true };
      }
    }

    return undefined;
  });

  // ── Async slash command guard ──
  async function runSlashAsync(fn: () => Promise<void>): Promise<void> {
    editor.disableSubmit = true;
    try {
      await fn();
    } catch (e) {
      addSystem(chalk.red(`Error: ${e}`));
    } finally {
      editor.disableSubmit = false;
      if (!shuttingDown) tui.requestRender();
    }
  }

  // ── Slash command handler ──
  async function handleSlashCommand(input: string): Promise<boolean> {
    // Normalize: allow // prefix as alias for /
    if (input.startsWith("//")) input = input.slice(1);

    if (input === "/help") {
      addSystem(
        chalk.bold.cyan("Commands:") + "\n" +
        chalk.cyan("  /help") + chalk.dim("             — this message\n") +
        chalk.cyan("  /stats") + chalk.dim("            — session statistics\n") +
        chalk.cyan("  /clear") + chalk.dim("            — clear chat log\n") +
        chalk.cyan("  /eval") + chalk.dim("             — run eval suite\n") +
        chalk.cyan("  /compare <prompt>") + chalk.dim(" — compare retrieval\n") +
        chalk.cyan("  /spawn <mode> <task>") + chalk.dim(" — spawn subagent\n") +
        chalk.cyan("  /merge <id>") + chalk.dim("       — merge incognito data\n") +
        chalk.cyan("  /wakeup") + chalk.dim("           — set up/reset identity\n") +
        chalk.cyan("  /agents") + chalk.dim("           — list subagents\n") +
        chalk.cyan("  /unlimited") + chalk.dim("        — remove tool limit\n") +
        chalk.cyan("  /export-training") + chalk.dim("  — export training pairs\n") +
        chalk.cyan("  /quit") + chalk.dim("             — exit"),
      );
      return true;
    }
    if (input === "/quit" || input === "/exit") {
      gracefulShutdown();
      return true;
    }
    if (input === "/clear") {
      chatLog.clear();
      chatLog.invalidate();
      tui.requestRender();
      return true;
    }
    if (input === "/unlimited") {
      zera.setToolLimit(Infinity);
      addSystem(chalk.dim("Tool call limit removed for next prompt."));
      return true;
    }
    if (input === "/agents") {
      runSlashAsync(async () => {
        const agents = await listSubagents();
        if (agents.length === 0) {
          addSystem(chalk.dim("No subagents spawned yet."));
        } else {
          const lines = agents.map(a => {
            const idStr = a.incognito_id ? ` [${a.incognito_id}]` : "";
            return `  ${a.mode} | ${a.status} | ${truncate(a.task, 60)}${idStr}`;
          });
          addSystem(chalk.dim(`Spawned subagents (${agents.length}):\n`) + chalk.dim(lines.join("\n")));
        }
      });
      return true;
    }
    if (input === "/stats") {
      runSlashAsync(async () => {
        const stats = getLastContextStats();
        if (!stats) {
          addSystem(chalk.dim("No stats yet — send a message first."));
          return;
        }
        const quality = await getSessionQualityStats(zera.sessionId);
        const reflCount = await getReflectionCount();
        const lines: string[] = [];
        lines.push(chalk.bold.cyan("Session Statistics"));
        lines.push(chalk.dim("─".repeat(40)));
        lines.push(`Graph nodes:      ${stats.graphNodes}`);
        lines.push(`Context tokens:   ${formatTokenCount(stats.cumSentTokens)} / ${formatTokenCount(stats.cumFullTokens)}`);
        lines.push(`Compression:      ${stats.cumFullTokens > 0 ? ((1 - stats.cumSentTokens / stats.cumFullTokens) * 100).toFixed(0) : 0}%`);
        lines.push(`Turns:            ${sessionTurns}`);
        lines.push(`Tool calls:       ${sessionToolCalls}`);
        lines.push(`Session cost:     $${sessionCost.toFixed(3)}`);
        lines.push(`Reflections:      ${reflCount}`);
        if (quality) {
          lines.push(chalk.dim("─".repeat(40)));
          lines.push(`Utilization:      ${quality.avgUtilization?.toFixed(2) ?? "—"}`);
          lines.push(`Retrievals:       ${quality.totalRetrievals ?? 0}`);
        }
        addSystem(lines.join("\n"));
      });
      return true;
    }
    if (input === "/eval") {
      runSlashAsync(async () => {
        addSystem(chalk.dim("Running full eval suite (5 prompts × 2 runs each)..."));
        await runEvalSuite(cwd);
        addSystem(chalk.green("Eval suite complete."));
      });
      return true;
    }
    if (input.startsWith("/compare ")) {
      const prompt = input.slice("/compare ".length).trim();
      if (!prompt) {
        addSystem(chalk.dim("Usage: /compare <your prompt here>"));
      } else {
        runSlashAsync(async () => { await runEval(cwd, prompt); });
      }
      return true;
    }
    if (input.startsWith("/spawn ")) {
      const rest = input.slice("/spawn ".length).trim();
      const spaceIdx = rest.indexOf(" ");
      if (spaceIdx < 0) {
        addSystem(chalk.dim("Usage: /spawn <full|incognito> <task>"));
        return true;
      }
      const mode = rest.slice(0, spaceIdx) as "full" | "incognito";
      if (mode !== "full" && mode !== "incognito") {
        addSystem(chalk.dim("Mode must be 'full' or 'incognito'."));
        return true;
      }
      const task = rest.slice(spaceIdx + 1).trim();
      if (!task) {
        addSystem(chalk.dim("Task description required."));
        return true;
      }
      const surrealConfig = agentOptions?.surrealConfig;
      const apiKey = agentOptions?.anthropicApiKey ?? "";
      const embPath = agentOptions?.embeddingModelPath ?? "";
      if (!surrealConfig) {
        addSystem(chalk.red("Cannot spawn subagent: no SurrealDB config available."));
        return true;
      }
      runSlashAsync(async () => {
        addSystem(chalk.dim(`Spawning ${mode} subagent...`));
        const result = await spawnSubagent(
          { mode, task, parentSessionId: zera.sessionId, cwd },
          surrealConfig, apiKey, embPath,
          (text) => { addSystem(text); },
        );
        addSystem(chalk.green(`Subagent done: ${result.status} | ${result.turnCount} turns, ${result.toolCalls} tools, ${(result.durationMs / 1000).toFixed(1)}s`));
        if (result.incognitoId) addSystem(chalk.dim(`Incognito ID: ${result.incognitoId}`));
      });
      return true;
    }
    if (input.startsWith("/merge ")) {
      const incognitoId = input.slice("/merge ".length).trim();
      if (!incognitoId) {
        addSystem(chalk.dim("Usage: /merge <incognito_id>"));
        return true;
      }
      const surrealConfig = agentOptions?.surrealConfig;
      if (!surrealConfig) {
        addSystem(chalk.red("Cannot merge: no SurrealDB config available."));
        return true;
      }
      runSlashAsync(async () => {
        addSystem(chalk.dim(`Merging from ${incognitoId}...`));
        const result = await mergeFromIncognito(incognitoId, surrealConfig);
        const lines = [`Merged ${result.merged} nodes (${result.skippedDuplicates} duplicates skipped)`];
        for (const [table, count] of Object.entries(result.tables)) {
          lines.push(`  ${table}: ${count}`);
        }
        addSystem(chalk.dim(lines.join("\n")));
      });
      return true;
    }
    if (input.startsWith("/wakeup")) {
      const wp = findWakeupFile(cwd);
      if (wp) {
        runSlashAsync(async () => {
          const content = readWakeupFile(wp);
          const { systemAddition, firstMessage } = buildWakeupPrompt(content);
          addSystem(chalk.dim("Establishing identity from WAKEUP.md..."));
          zera.steer(systemAddition);
          await zera.prompt(firstMessage);
          await extractAndSaveIdentity(zera);
          deleteWakeupFile(wp);
          addSystem(chalk.dim("WAKEUP.md deleted."));
        });
      } else {
        runSlashAsync(async () => {
          addSystem(chalk.dim("Starting conversational identity setup..."));
          wakeupActive = true;
          zera.steer(`
IDENTITY SETUP MODE
The user wants to define or redefine your personality, tone, and behavioral guidelines.
Ask them about: personality/tone, role, communication style, and principles.
Keep it conversational — ask one or two questions at a time.
After each response, confirm what you heard and ask if there's more.
When the user says they're done or confirms the identity feels right (e.g. "yes", "that's good", "perfect", "done", "looks good"), respond with EXACTLY the phrase "[IDENTITY_CONFIRMED]" at the start of your message, followed by a brief confirmation.`);
          await zera.prompt("I want to set up your identity. Ask me what you need to know.");
        });
      }
      return true;
    }
    if (input.startsWith("/export-training")) {
      runSlashAsync(async () => {
        const { exportTrainingData, getTrainingDataCount } = await import("./acan.js");
        const count = await getTrainingDataCount();
        if (count === 0) {
          addSystem(chalk.dim("No training data yet."));
        } else {
          const args = input.slice("/export-training".length).trim();
          const exported = await exportTrainingData(args || undefined);
          addSystem(chalk.dim(`Exported ${exported} training pairs (${count} total available).\nTrain with: python scripts/train_acan.py`));
        }
      });
      return true;
    }
    return false;
  }

  // ── Editor submit ──
  editor.onSubmit = async (input: string) => {
    let text = input.trim();
    if (!text) return;

    // Normalize // prefix early — autocomplete prepends / to command names that already start with /
    if (text.startsWith("//")) text = text.slice(1);

    editor.addToHistory(text);

    // Echo user input
    chatLog.addChild(new UserMessage(text));
    tui.requestRender();
    pruneMessages();

    if (await handleSlashCommand(text)) return;

    // Preflight
    const turnStart = performance.now();
    let pf: PreflightResult;
    try {
      pf = await preflight(text, zera.sessionId);
      zera.configureForTurn(pf.config, text);

      if (!pf.fastPath && pf.intent.category !== "unknown") {
        statusLine.setIntent(pf.intent.category, pf.config.thinkingLevel, pf.config.toolLimit, pf.preflightMs);
        tui.requestRender();
      }
    } catch (e) {
      swallow("tui:preflight", e);
      pf = {
        intent: { category: "unknown", confidence: 0, scores: [] },
        complexity: { level: "moderate", estimatedToolCalls: 15, suggestedThinking: "medium" },
        config: { thinkingLevel: "medium", toolLimit: 15, tokenBudget: 1500, vectorSearchLimits: { turn: 10, identity: 5, concept: 5, memory: 5, artifact: 3 } },
        preflightMs: 0,
        fastPath: true,
      };
    }

    // Predictive prefetch
    if (!pf.fastPath && pf.intent.category !== "unknown") {
      const predicted = predictQueries(text, pf.intent.category);
      if (predicted.length > 0) {
        prefetchContext(predicted, zera.sessionId).catch(e => swallow("tui:prefetchContext", e));
      }
    }

    // Update sync handoff state with latest user text
    const sessionState = { sessionId: zera.sessionId, lastUserText: text, lastAssistantText: "", unextractedTokens: 0, cwd };
    setActiveSessionState(sessionState);

    try {
      editor.disableSubmit = true;
      agentActive = true;
      await zera.prompt(text);
    } catch (err) {
      addSystem(chalk.red(`Error: ${err}`));
    } finally {
      agentActive = false;
      editor.disableSubmit = false;
      statusLine.setIdle();
      if (!shuttingDown) tui.requestRender();
    }

    // Auto-save identity during wakeup flow
    if (wakeupActive && currentText.includes("[IDENTITY_CONFIRMED]")) {
      wakeupActive = false;
      await extractAndSaveIdentity(zera);
    }

    // Postflight metrics
    const turnDuration = performance.now() - turnStart;
    postflight(text, pf, toolCallsThisTurn, totalTokensIn, totalTokensOut, turnDuration, zera.sessionId).catch(e => swallow.warn("tui:postflight", e));
  };

  updateFooter();

  // Keep the function alive until shutdown — without this, startTui() returns
  // immediately after setup and Node exits because the event loop drains.
  await new Promise<void>((resolve) => {
    shutdownResolve = resolve;
  });

  // ── Graceful shutdown ──
  async function gracefulShutdown(): Promise<void> {
    if (shuttingDown) return;
    shuttingDown = true;
    markShutdown();

    editor.disableSubmit = true;
    statusLine.setBusy("Going to sleep...");
    tui.requestRender();

    const stats = getLastContextStats();
    const cumFull = stats?.cumFullTokens ?? 0;
    const cumSent = stats?.cumSentTokens ?? 0;

    deactivateSessionMemories(zera.sessionId).catch(e => swallow.warn("tui:deactivateSessionMemories", e));

    const EXIT_LINE_TIMEOUT = 15_000;
    const CLEANUP_TIMEOUT = 150_000;
    let exitLine: string | null = null;

    const exitLineP = Promise.race([
      zera.generateExitLine({
        cumFullTokens: cumFull, cumSentTokens: cumSent,
        turns: sessionTurns, toolCalls: sessionToolCalls,
      }).then(line => { if (line) exitLine = line; }).catch(e => swallow.error("tui:exitLine", e)),
      new Promise<void>(r => setTimeout(r, EXIT_LINE_TIMEOUT)),
    ]);

    let cleanupTimedOut = false;
    const cleanupP = Promise.race([
      zera.cleanup().catch(e => swallow.warn("tui:cleanup", e)),
      new Promise<void>(r => setTimeout(() => { cleanupTimedOut = true; r(); }, CLEANUP_TIMEOUT)),
    ]).then(() => {
      if (cleanupTimedOut) swallow.warn("tui:cleanup:timeout", new Error(`Cleanup timed out after ${CLEANUP_TIMEOUT / 1000}s — extraction/graduation may be incomplete`));
    });

    await exitLineP;
    statusLine.dispose();

    // Summary
    const delta = cumFull - cumSent;
    const summaryLines: string[] = [];
    summaryLines.push(chalk.dim("─".repeat(40)));
    summaryLines.push(chalk.dim(`Turns: ${sessionTurns} · Tools: ${sessionToolCalls} · Cost: $${sessionCost.toFixed(2)}`));
    if (cumFull > 0) {
      summaryLines.push(chalk.dim(`Context: ${formatTokenCount(cumSent)}/${formatTokenCount(cumFull)} tokens (saved ${formatTokenCount(delta)})`));
    }
    if (exitLine) {
      summaryLines.push("");
      summaryLines.push(chalk.italic(exitLine));
    }
    summaryLines.push(chalk.dim("─".repeat(40)));
    chatLog.addChild(new SystemMessage(summaryLines.join("\n")));
    tui.requestRender();

    await cleanupP;
    clearActiveSessionState(); // Graceful exit — no need for sync handoff file
    await Promise.allSettled([closeSurreal(), disposeEmbeddings()]);

    // Small delay so the user can see the exit message
    await new Promise(r => setTimeout(r, 500));

    try { tui.stop(); } catch { /* ignore EBADF */ }
    if (shutdownResolve) shutdownResolve();
    process.exit(0);
  }
}
