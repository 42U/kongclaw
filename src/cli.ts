import * as readline from "node:readline";
import type { AgentEvent } from "@mariozechner/pi-agent-core";
import type { AssistantMessage, AssistantMessageEvent } from "@mariozechner/pi-ai";
import { completeSimple, getModel } from "@mariozechner/pi-ai";
import { createZeraAgent, type ZeraAgent, type ZeraAgentOptions } from "./agent.js";
import { getLastContextStats } from "./graph-context.js";
import { runEval, runEvalSuite } from "./eval.js";
import { getSessionQualityStats } from "./retrieval-quality.js";
import { getReflectionCount } from "./reflection.js";
import { preflight, postflight, recordToolCall, type PreflightResult } from "./orchestrator.js";
import { predictQueries, prefetchContext } from "./prefetch.js";
import { spawnSubagent, mergeFromIncognito, listSubagents, type SubagentConfig } from "./subagent.js";
import type { SurrealConfig } from "./config.js";
import { logError } from "./logger.js";
import {
  hasUserIdentity, findWakeupFile, readWakeupFile,
  deleteWakeupFile, saveUserIdentity, buildWakeupPrompt,
} from "./identity.js";
import { synthesizeWakeup, synthesizeStartupCognition, synthesizeBirthCognition } from "./wakeup.js";
import { deactivateSessionMemories, closeSurreal, markShutdown, createCoreMemory } from "./surreal.js";
import { disposeEmbeddings } from "./embeddings.js";
import { swallow } from "./errors.js";
import * as render from "./render.js";
import { buildEffectivePrompt } from "./prompt.js";
import { setActiveSessionState, clearActiveSessionState } from "./index.js";
import { runDeferredCleanup } from "./deferred-cleanup.js";

/** Whether CLI is in graceful shutdown (cleanup in progress). Exported for index.ts SIGINT coordination. */
export let shuttingDown = false;

export async function startCli(cwd: string, modelId?: string, agentOptions?: ZeraAgentOptions): Promise<void> {
  render.writeBanner();

  // Phase 10: Constitutive memory — synthesize wake-up briefing before agent creation
  // Animated spinner while waking up (same braille animation as shutdown)
  const wakeFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let wakeFrameIdx = 0;
  const wakeStart = performance.now();
  render.writeSpinner(wakeFrames[0], "Waking up...", "0");
  const wakeSpinner = setInterval(() => {
    const elapsed = ((performance.now() - wakeStart) / 1000).toFixed(0);
    render.writeSpinner(wakeFrames[wakeFrameIdx++ % wakeFrames.length], "Waking up...", elapsed);
  }, 80);

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
      // True first boot — no prior sessions, no memories. Birth moment.
      const birth = await synthesizeBirthCognition(modelId);
      if (birth) {
        startupGreeting = birth.greeting;
        startupThoughts = birth.thoughts;
      }
    }
  } catch (e) {
    swallow("cli:non-critical — proceed without briefing", e);
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
    .then(n => { if (n > 0) render.writeLine(`  Deferred cleanup: processed ${n} orphaned session(s)`); })
    .catch(e => swallow.warn("cli:deferredCleanup", e));

  // Pin proactive thoughts as Tier 1 core_memory (after agent creation so we have sessionId)
  if (startupThoughts.length > 0) {
    const thoughtsText = startupThoughts.map(t => `- ${t}`).join("\n");
    createCoreMemory(
      `[PROACTIVE THOUGHTS]\n${thoughtsText}`,
      "proactive", 8, 1, zera.sessionId,
    ).catch(e => swallow("cli:proactive-thoughts — non-critical", e));
  }

  // Soft ambient hint — follow the thread if user responds, let it fade if they pivot
  if (startupGreeting) {
    createCoreMemory(
      `[SESSION OPEN] Greeting shown to user: "${startupGreeting}"\nIf the user responds to this, continue that thread. If they start a different topic, let this fade — do not force it.`,
      "greeting", 3, 1, zera.sessionId,
    ).catch(e => swallow("cli:startup-greeting", e));
  }

  clearInterval(wakeSpinner);
  const wakeSec = ((performance.now() - wakeStart) / 1000).toFixed(1);
  render.writeReady(wakeSec, startupGreeting, zera.sessionId);
  render.initStatusBar(zera.sessionId);

  // Helper: extract identity chunks from conversation and save to graph
  async function extractAndSaveIdentity(agent: ZeraAgent): Promise<void> {
    render.writeInfo("Saving identity to persistent memory...");
    render.setMuted(true);
    let extractedText = "";
    const unsub = agent.subscribe((event: AgentEvent) => {
      if (event.type === "message_update") {
        const e = event.assistantMessageEvent as AssistantMessageEvent;
        if (e.type === "text_delta") extractedText += e.delta;
      }
    });
    await agent.prompt(`Based on the identity we just established, output ONLY a JSON array of 4-8 identity statements — short, declarative sentences defining who you are, your personality, tone, role, and principles. Each should be standalone and retrievable independently. No explanation, just the JSON array.`);
    unsub();
    render.setMuted(false);

    try {
      const jsonMatch = extractedText.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        const chunks: string[] = JSON.parse(jsonMatch[0]);
        if (Array.isArray(chunks) && chunks.length > 0) {
          const saved = await saveUserIdentity(chunks);
          render.writeSuccess(`Saved ${saved} identity chunks to memory graph.`);
          return;
        }
      }
      render.writeError("Warning: Could not parse identity extraction. Try /wakeup again.");
    } catch (e) {
      swallow("cli:identity-extraction-failed", e);
      render.writeError("Warning: Identity extraction failed. Try /wakeup again.");
    }
  }

  // Subscribe for streaming output
  let currentText = "";
  let wakeupActive = false;
  let toolCallsThisTurn = 0;
  let sessionToolCalls = 0;
  const pendingToolArgs = new Map<string, { name: string; args: any; count: number }>();
  let sessionTurns = 0;
  let totalTokensIn = 0;
  let totalTokensOut = 0;
  let totalCacheRead = 0;
  let totalCacheWrite = 0;
  let totalCost = 0;
  let sessionCost = 0;
  let agentActive = false; // true while zera.prompt() is running
  zera.subscribe((event: AgentEvent) => {
    if (event.type === "message_update") {
      const e = event.assistantMessageEvent as AssistantMessageEvent;
      if (e.type === "text_delta") {
        render.writeText(e.delta);
        currentText += e.delta;
      }
    } else if (event.type === "message_end") {
      const msg = event.message as AssistantMessage;
      if (msg.role === "assistant") {
        if (currentText && !render.isMuted()) {
          render.flushText();
        }
        // Accumulate usage across all API calls in this turn
        const u = msg.usage;
        if (u) {
          totalTokensIn += u.input ?? 0;
          totalTokensOut += u.output ?? 0;
          totalCacheRead += u.cacheRead ?? 0;
          totalCacheWrite += u.cacheWrite ?? 0;
          totalCost += u.cost?.total ?? 0;
        }
        if (u && msg.stopReason !== "toolUse" && !render.isMuted()) {
          const stats = getLastContextStats();
          render.writeStats(
            { tokensIn: totalTokensIn, tokensOut: totalTokensOut, cacheRead: totalCacheRead, cacheWrite: totalCacheWrite, cost: totalCost },
            stats,
            toolCallsThisTurn,
            sessionToolCalls,
          );
        }
        currentText = "";
      }
    } else if (event.type === "tool_execution_start") {
      toolCallsThisTurn++;
      sessionToolCalls++;
      recordToolCall(event.toolName);
      pendingToolArgs.set(event.toolCallId, { name: event.toolName, args: event.args, count: toolCallsThisTurn });
      render.writeToolStart(toolCallsThisTurn, event.toolName, event.args);
    } else if (event.type === "tool_execution_end") {
      const pending = pendingToolArgs.get(event.toolCallId);
      const args = pending?.args;
      const count = pending?.count ?? toolCallsThisTurn;
      pendingToolArgs.delete(event.toolCallId);
      render.writeToolEnd(count, event.toolName, args, event.result, event.isError);
    } else if (event.type === "agent_start") {
      toolCallsThisTurn = 0;
      sessionTurns++;
      totalTokensIn = 0;
      totalTokensOut = 0;
      totalCacheRead = 0;
      totalCacheWrite = 0;
      totalCost = 0;
      render.updateStatusBar({ turns: sessionTurns });
    }
    // Track session-wide cost
    if (event.type === "message_end") {
      const cost = (event.message as { usage?: { cost?: { total?: number } } })?.usage?.cost?.total;
      if (cost) sessionCost += cost;
      render.updateStatusBar({ cost: sessionCost, toolCalls: sessionToolCalls, tokensIn: totalTokensIn, tokensOut: totalTokensOut });
    }
  });

  // ── WAKEUP.md — First-run identity establishment ──
  const wakeupPath = findWakeupFile(cwd);
  const needsWakeup = wakeupPath && !(await hasUserIdentity());
  if (needsWakeup && wakeupPath) {
    const wakeupContent = readWakeupFile(wakeupPath);
    const { systemAddition, firstMessage } = buildWakeupPrompt(wakeupContent);

    render.writeInfo("⚡ First run detected — establishing identity from WAKEUP.md\n");
    zera.steer(systemAddition);
    await zera.prompt(firstMessage);
    await extractAndSaveIdentity(zera);
    deleteWakeupFile(wakeupPath);
    render.writeInfo("WAKEUP.md deleted. Identity established.\n");
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: render.getPromptString(),
  });

  rl.prompt();

  // Three-stage Ctrl+C: soft interrupt → graceful shutdown → force exit
  rl.on("SIGINT", () => {
    if (shuttingDown) {
      // Ctrl+C during shutdown → force exit immediately
      render.writeLine("\n   Force exit.");
      process.exit(1);
    } else if (agentActive && !zera.isSoftInterrupted()) {
      // First Ctrl+C during active prompt → soft interrupt
      zera.softInterrupt();
      render.writeLine("\n   ⚡ Soft interrupt — wrapping up current work...");
    } else {
      // Second Ctrl+C, or Ctrl+C when idle → graceful close
      rl.close();
    }
  });

  rl.on("line", async (line) => {
    const input = line.trim();
    if (!input) {
      rl.prompt();
      return;
    }
    if (input === "/quit" || input === "/exit") {
      rl.close();
      return;
    }
    if (input === "/eval") {
      render.writeInfo("Running full eval suite (5 prompts × 2 runs each)...");
      await runEvalSuite(cwd);
      rl.prompt();
      return;
    }
    if (input.startsWith("/compare ")) {
      const prompt = input.slice("/compare ".length).trim();
      if (!prompt) {
        render.writeLine("\n   Usage: /compare <your prompt here>");
      } else {
        await runEval(cwd, prompt);
      }
      rl.prompt();
      return;
    }
    if (input === "/help") {
      render.writeHelp();
      rl.prompt();
      return;
    }
    if (input === "/unlimited") {
      zera.setToolLimit(Infinity);
      render.writeLine("\n   Tool call limit removed for next prompt.");
      rl.prompt();
      return;
    }
    if (input.startsWith("/spawn ")) {
      const rest = input.slice("/spawn ".length).trim();
      const spaceIdx = rest.indexOf(" ");
      if (spaceIdx < 0) {
        render.writeLine("\n   Usage: /spawn <full|incognito> <task description>");
        rl.prompt();
        return;
      }
      const mode = rest.slice(0, spaceIdx) as "full" | "incognito";
      if (mode !== "full" && mode !== "incognito") {
        render.writeLine("\n   Mode must be 'full' or 'incognito'.");
        rl.prompt();
        return;
      }
      const task = rest.slice(spaceIdx + 1).trim();
      if (!task) {
        render.writeLine("\n   Task description required.");
        rl.prompt();
        return;
      }
      render.writeLine(`\n   Spawning ${mode} subagent...`);
      try {
        const surrealConfig = agentOptions?.surrealConfig;
        const apiKey = agentOptions?.anthropicApiKey ?? "";
        const embPath = agentOptions?.embeddingModelPath ?? "";
        if (!surrealConfig) {
          render.writeLine("   Cannot spawn subagent: no SurrealDB config available.");
          rl.prompt();
          return;
        }
        const result = await spawnSubagent(
          { mode, task, parentSessionId: zera.sessionId, cwd },
          surrealConfig, apiKey, embPath,
          (text) => process.stdout.write(text),
        );
        render.writeLine(`\n\n   Subagent done: ${result.status} | ${result.turnCount} turns, ${result.toolCalls} tools, ${(result.durationMs / 1000).toFixed(1)}s`);
        if (result.incognitoId) {
          render.writeLine(`   Incognito ID: ${result.incognitoId}`);
        }
      } catch (err: any) {
        render.writeLine(`\n   Subagent failed: ${err.message}`);
      }
      rl.prompt();
      return;
    }
    if (input.startsWith("/merge ")) {
      const incognitoId = input.slice("/merge ".length).trim();
      if (!incognitoId) {
        render.writeLine("\n   Usage: /merge <incognito_id>");
        rl.prompt();
        return;
      }
      const surrealConfig = agentOptions?.surrealConfig;
      if (!surrealConfig) {
        render.writeLine("   Cannot merge: no SurrealDB config available.");
        rl.prompt();
        return;
      }
      render.writeLine(`\n   Merging from ${incognitoId}...`);
      try {
        const result = await mergeFromIncognito(incognitoId, surrealConfig);
        render.writeLine(`   Merged ${result.merged} nodes (${result.skippedDuplicates} duplicates skipped)`);
        for (const [table, count] of Object.entries(result.tables)) {
          render.writeLine(`     ${table}: ${count}`);
        }
      } catch (err: any) {
        render.writeLine(`   Merge failed: ${err.message}`);
      }
      rl.prompt();
      return;
    }
    if (input.startsWith("/wakeup")) {
      const wp = findWakeupFile(cwd);
      if (wp) {
        // File-based: read WAKEUP.md, process, save, delete
        const content = readWakeupFile(wp);
        const { systemAddition, firstMessage } = buildWakeupPrompt(content);
        render.writeLine("\n   Establishing identity from WAKEUP.md...\n");
        zera.steer(systemAddition);
        await zera.prompt(firstMessage);
        await extractAndSaveIdentity(zera);
        deleteWakeupFile(wp);
        render.writeLine("   WAKEUP.md deleted.");
      } else {
        // Conversational: agent interviews user, then auto-saves
        render.writeLine("\n   Starting conversational identity setup...\n");
        wakeupActive = true;
        zera.steer(`
IDENTITY SETUP MODE
The user wants to define or redefine your personality, tone, and behavioral guidelines.
Ask them about: personality/tone, role, communication style, and principles.
Keep it conversational — ask one or two questions at a time.
After each response, confirm what you heard and ask if there's more.
When the user says they're done or confirms the identity feels right (e.g. "yes", "that's good", "perfect", "done", "looks good"), respond with EXACTLY the phrase "[IDENTITY_CONFIRMED]" at the start of your message, followed by a brief confirmation.`);
        await zera.prompt("I want to set up your identity. Ask me what you need to know.");
      }
      rl.prompt();
      return;
    }
    if (input === "/agents") {
      const agents = await listSubagents();
      if (agents.length === 0) {
        render.writeLine("\n   No subagents spawned yet.");
      } else {
        render.writeLine(`\n   Spawned subagents (${agents.length}):`);
        for (const a of agents) {
          const idStr = a.incognito_id ? ` [${a.incognito_id}]` : "";
          render.writeLine(`   ${a.mode} | ${a.status} | ${a.task.slice(0, 60)}${idStr}`);
        }
      }
      rl.prompt();
      return;
    }
    if (input.startsWith("/export-training")) {
      const { exportTrainingData, getTrainingDataCount } = await import("./acan.js");
      const count = await getTrainingDataCount();
      if (count === 0) {
        render.writeLine("\n   No training data yet. Training data accumulates automatically from retrieval outcomes.");
      } else {
        const args = input.slice("/export-training".length).trim();
        const exported = await exportTrainingData(args || undefined);
        render.writeLine(`\n   Exported ${exported} training pairs (${count} total available).`);
        render.writeLine(`   Train with: python scripts/train_acan.py`);
      }
      rl.prompt();
      return;
    }
    if (input === "/stats") {
      const stats = getLastContextStats();
      if (!stats) {
        render.writeInfo("No stats yet — send a message first.");
      } else {
        const quality = await getSessionQualityStats(zera.sessionId);
        const reflCount = await getReflectionCount();
        await render.writeDetailedStats(stats, quality, reflCount);
      }
      rl.prompt();
      return;
    }

    // ISMAR-GENT: preflight analysis
    const turnStart = performance.now();
    let pf: PreflightResult;
    try {
      pf = await preflight(input, zera.sessionId);
      zera.configureForTurn(pf.config, input);

      // Show intent info (only when non-default and non-fast-path)
      if (!pf.fastPath && pf.intent.category !== "unknown") {
        render.writePreflight(pf.intent.category, pf.config.thinkingLevel, pf.config.toolLimit, pf.preflightMs);
      }
    } catch (e) {
        swallow("cli:silent", e);
      // Preflight failed — use defaults, don't block the user
      pf = {
        intent: { category: "unknown", confidence: 0, scores: [] },
        complexity: { level: "moderate", estimatedToolCalls: 15, suggestedThinking: "medium" },
        config: { thinkingLevel: "medium", toolLimit: 15, tokenBudget: 1500, vectorSearchLimits: { turn: 10, identity: 5, concept: 5, memory: 5, artifact: 3 } },
        preflightMs: 0,
        fastPath: true,
      };
    }

    // Phase 7d: Predictive prefetch — fire in background during LLM thinking
    if (!pf.fastPath && pf.intent.category !== "unknown") {
      const predicted = predictQueries(input, pf.intent.category);
      if (predicted.length > 0) {
        prefetchContext(predicted, zera.sessionId).catch(e => swallow("cli:prefetchContext", e));
      }
    }

    // Update sync handoff state with latest user text
    setActiveSessionState({ sessionId: zera.sessionId, lastUserText: input, lastAssistantText: "", unextractedTokens: 0, cwd });

    try {
      agentActive = true;
      await zera.prompt(input);
    } catch (err) {
      render.writeError(`Error: ${err}`);
    } finally {
      agentActive = false;
    }

    // Auto-save identity when agent confirms during wakeup flow
    if (wakeupActive && currentText.includes("[IDENTITY_CONFIRMED]")) {
      wakeupActive = false;
      await extractAndSaveIdentity(zera);
    }

    // Async postflight — record metrics
    const turnDuration = performance.now() - turnStart;
    postflight(input, pf, toolCallsThisTurn, totalTokensIn, totalTokensOut, turnDuration, zera.sessionId).catch(e => swallow.warn("cli:postflight", e));

    rl.prompt();
  });

  rl.on("close", async () => {
    shuttingDown = true;
    render.destroyStatusBar();
    markShutdown();

    // Animated spinner while summarizing
    const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let frameIdx = 0;

    // Spinner starts immediately — no waiting on Haiku
    let shutdownMessage = "Going to sleep...";
    const spinner = setInterval(() => {
      render.writeSpinner(frames[frameIdx++ % frames.length], shutdownMessage, "");
    }, 80);
    render.writeSpinner(frames[0], shutdownMessage, "");

    // Stats are available immediately — no DB needed
    const stats = getLastContextStats();
    const cumFull = stats?.cumFullTokens ?? 0;
    const cumSent = stats?.cumSentTokens ?? 0;
    const delta = cumFull - cumSent;

    // Deactivate session-scoped Tier 1 core memories (fire-and-forget)
    deactivateSessionMemories(zera.sessionId).catch(e => swallow.warn("cli:deactivateSessionMemories", e));

    // --- Shutdown: exit line + cleanup run in PARALLEL ---
    // Cleanup is the critical path (extraction, graduation, task completion).
    // Exit line is cosmetic. Both start immediately; we wait for both.
    const EXIT_LINE_TIMEOUT = 15_000;
    const CLEANUP_TIMEOUT = 150_000;
    let exitLine: string | null = null;
    let exitLineTimedOut = false;

    // Fire exit line generation — spinner picks up the message via closure
    const exitLineP = Promise.race([
      zera.generateExitLine({
        cumFullTokens: cumFull,
        cumSentTokens: cumSent,
        turns: sessionTurns,
        toolCalls: sessionToolCalls,
      }).then(
        (line) => { if (line) { exitLine = line; shutdownMessage = line; } },
        (e) => { 
          swallow.error("cli:shutdown:exitLine", e);
          logError("cli:shutdown:exitLine", e);
        },
      ),
      new Promise<void>(r => setTimeout(() => { exitLineTimedOut = true; r(); }, EXIT_LINE_TIMEOUT)),
    ]);

    // Fire cleanup in parallel — don't wait for exit line
    let cleanupTimedOut = false;
    const cleanupP = Promise.race([
      zera.cleanup().catch(e => {
        swallow.warn("cli:shutdown:cleanup", e);
        logError("cli:shutdown:cleanup", e);
      }),
      new Promise<void>(r => setTimeout(() => { cleanupTimedOut = true; r(); }, CLEANUP_TIMEOUT)),
    ]).then(() => {
      if (cleanupTimedOut) swallow.warn("cli:cleanup:timeout", new Error(`Cleanup timed out after ${CLEANUP_TIMEOUT / 1000}s — extraction/graduation may be incomplete`));
    });

    // Wait for exit line first (to show it), then wait for cleanup to finish
    await exitLineP;

    clearInterval(spinner);
    render.writeLine(""); // newline after spinner

    render.writeShutdownSummary(sessionTurns, sessionToolCalls, sessionCost, cumFull, cumSent);
    render.writeExitLine(exitLine, exitLineTimedOut);

    // Wait for cleanup to finish (likely already done since it ran in parallel)
    await cleanupP;
    clearActiveSessionState(); // Graceful exit — no need for sync handoff file

    // Release DB connections and embedding models
    await Promise.allSettled([closeSurreal(), disposeEmbeddings()]);

    process.exit(0);
  });
}
