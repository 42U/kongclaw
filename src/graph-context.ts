import { getLlama, LlamaLogLevel, type LlamaRankingContext } from "node-llama-cpp";
import { getDueMemories, advanceSurfaceFade } from "./surreal.js";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { UserMessage, AssistantMessage, ToolResultMessage, TextContent, ThinkingContent, ToolCall, ImageContent } from "@mariozechner/pi-ai";
import { getPendingDirectives, clearPendingDirectives, getSessionContinuity, getSuppressedNodeIds } from "./cognitive-check.js";

type ContentBlock = TextContent | ThinkingContent | ToolCall | ImageContent;

function isUser(msg: AgentMessage): msg is UserMessage {
  return (msg as UserMessage).role === "user";
}
function isAssistant(msg: AgentMessage): msg is AssistantMessage {
  return (msg as AssistantMessage).role === "assistant";
}
function isToolResult(msg: AgentMessage): msg is ToolResultMessage {
  return (msg as ToolResultMessage).role === "toolResult";
}
/** Get role string from any AgentMessage without `as any`. */
function msgRole(msg: AgentMessage): string {
  if (isUser(msg)) return msg.role;
  if (isAssistant(msg)) return msg.role;
  if (isToolResult(msg)) return msg.role;
  return "unknown";
}
/** Get content blocks as a flat array, normalising UserMessage's string|array union. */
function msgContentBlocks(msg: AgentMessage): ContentBlock[] {
  if (isUser(msg)) {
    return typeof msg.content === "string"
      ? [{ type: "text", text: msg.content } as TextContent]
      : msg.content as ContentBlock[];
  }
  if (isAssistant(msg)) return msg.content;
  if (isToolResult(msg)) return msg.content as ContentBlock[];
  return [];
}
import { embed, isEmbeddingsAvailable } from "./embeddings.js";
import { completeSimple, getModel } from "@mariozechner/pi-ai";
import {
  vectorSearch, tagBoostedConcepts, isSurrealAvailable,
  graphExpand, bumpAccessCounts,
  createMemory, relate,
  createCompactionCheckpoint, completeCompactionCheckpoint, failCompactionCheckpoint,
  getUtilityFromCache, getUtilityCacheEntries,
  getAllCoreMemory,
  getReflectionSessionIds,
  type VectorSearchResult,
  type CoreMemoryEntry,
  getPreviousSessionTurns,
} from "./surreal.js";
import { queryCausalContext } from "./causal.js";
import { findRelevantSkills, formatSkillContext } from "./skills.js";
import { retrieveReflections, formatReflectionContext } from "./reflection.js";
import { getCachedContext, recordPrefetchHit, recordPrefetchMiss } from "./prefetch.js";
import { stageRetrieval, getHistoricalUtilityBatch } from "./retrieval-quality.js";
import { isACANActive, scoreWithACAN, type ACANCandidate } from "./acan.js";
import { swallow } from "./errors.js";

// ── Cross-encoder reranker (bge-reranker-v2-m3) ──────────────────────────────
let _rankingCtx: LlamaRankingContext | null = null;
const RERANK_TOP_N = 30;         // candidates to send to cross-encoder
const RERANK_BLEND_VECTOR = 0.6; // WMR weight in blend
const RERANK_BLEND_CROSS = 0.4;  // cross-encoder weight in blend
const RERANK_MAX_DOC_CHARS = 24000; // truncate long docs for reranker context window

export async function initReranker(modelPath: string): Promise<void> {
  try {
    const llama = await getLlama({
      logLevel: LlamaLogLevel.error,
      logger: (level, message) => {
        if (message.includes("missing newline token")) return;
        if (level === LlamaLogLevel.error) console.error(`[rerank] ${message}`);
      },
    });
    const model = await llama.loadModel({ modelPath });
    _rankingCtx = await model.createRankingContext();
    console.error("[rerank] Cross-encoder reranker loaded.");
  } catch (e) {
    swallow.warn("graph-context:initReranker failed — retrieval will work without reranking", e);
    _rankingCtx = null;
  }
}

export async function disposeReranker(): Promise<void> {
  if (_rankingCtx) {
    try { await _rankingCtx.dispose(); } catch { /* ignore */ }
    _rankingCtx = null;
  }
}

export function isRerankerActive(): boolean { return _rankingCtx !== null; }

// ── Global context budget — single source of truth ──────────────────────────
// Everything derives from the model's context window. No hardcoded magic numbers.
const BUDGET_FRACTION = 0.70;     // % of context window we control
const CONVERSATION_SHARE = 0.50;  // % of budget → conversation history (getRecentTurns)
const RETRIEVAL_SHARE = 0.30;     // % of budget → graph retrieval (takeWithConstraints)
const CORE_MEMORY_SHARE = 0.15;   // % of budget → Tier 0 + Tier 1 core memory

// Derived budgets (recalculated when context window is set)
let CONVERSATION_BUDGET_TOKENS = 70000;  // default for 200K context
let RETRIEVAL_BUDGET_TOKENS = 42000;
let CORE_BUDGET_TOKENS = 21000;

function recalcBudgets(contextWindow: number): void {
  const totalBudget = contextWindow * BUDGET_FRACTION;
  CONVERSATION_BUDGET_TOKENS = Math.round(totalBudget * CONVERSATION_SHARE);
  RETRIEVAL_BUDGET_TOKENS = Math.round(totalBudget * RETRIEVAL_SHARE);
  CORE_BUDGET_TOKENS = Math.round(totalBudget * CORE_MEMORY_SHARE);
}

/** Retrieval budget for orchestrator to derive per-intent token caps. */
export function getRetrievalBudgetTokens(): number {
  return RETRIEVAL_BUDGET_TOKENS;
}

let _tokenBudget = 6000;
let _vectorSearchLimits = { turn: 20, identity: 10, concept: 15, memory: 15, artifact: 10 };
let _skipRetrieval = false;
let _usedTokens = 0; // Cumulative input+output tokens this session — drives pressure scaling

// --- Tool budget state (set from agent.ts, read by rules suffix injection) ---
let _toolCallCount = 0;
let _toolLimit = Infinity;

export function setToolBudgetState(count: number, limit: number): void {
  _toolCallCount = count;
  _toolLimit = limit;
}

/** Accumulate token usage per assistant turn for context-pressure scaling. */
export function reportMessageTokens(input: number, output: number): void {
  _usedTokens += input + output;
}

function buildRulesSuffix(): string {
  const remaining = _toolLimit === Infinity
    ? "unlimited" : String(Math.max(0, _toolLimit - _toolCallCount));
  const urgency = _toolLimit !== Infinity && (_toolLimit - _toolCallCount) <= 3
    ? "\n⚠ WRAP UP or check in with user." : "";
  return (
    "\n<rules_reminder>" +
    `\nBudget: ${_toolCallCount} used, ${remaining} remaining.${urgency}` +
    "\n\nYOUR BUDGET IS SMALL. Plan the whole task, not just the next call." +
    "\n" +
    "\nTask: Fix broken import" +
    "\n  WASTEFUL (6 calls): grep old → read file → grep new → read context → edit → read to verify" +
    "\n  DENSE (2 calls):" +
    "\n    1. grep -n 'oldImport' src/**/*.ts; grep -rn 'newModule' src/" +
    "\n    2. edit file && npm test -- --grep 'relevant' 2>&1 | tail -20" +
    "\n" +
    "\nTask: Debug failing test" +
    "\n  WASTEFUL (8 calls): run test → read output → read test → read source → grep → read more → edit → rerun" +
    "\n  DENSE (3 calls):" +
    "\n    1. npm test 2>&1 | tail -30" +
    "\n    2. grep -n 'failingTest\\|relevantFn' test/*.ts src/*.ts" +
    "\n    3. edit fix && npm test 2>&1 | tail -15" +
    "\n" +
    "\nTask: Read/understand multiple files" +
    "\n  WASTEFUL (10 calls): cat file1 → cat file2 → cat file3 → ..." +
    "\n  DENSE (1-2 calls):" +
    "\n    1. head -80 src/a.ts src/b.ts src/c.ts src/d.ts  (4 files in ONE call)" +
    "\n    2. grep -n 'keyPattern' src/*.ts  (search all files at once, not one by one)" +
    "\n" +
    "\nEvery step still happens — investigation, edit, verification — but COMBINED into fewer calls." +
    "\nThe answer is often already in context. Don't call if you already know." +
    "\nAnnounce: task type (LOOKUP=1/EDIT=2/REFACTOR=6), planned calls, what each does." +
    "\n</rules_reminder>"
  );
}

/** Inject rules suffix into the last user-role message (user or toolResult).
 *  Avoids creating new messages which would break role alternation. */
function injectRulesSuffix(messages: AgentMessage[]): AgentMessage[] {
  const suffix = buildRulesSuffix();
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (isUser(msg)) {
      const clone = [...messages];
      clone[i] = {
        ...msg,
        content: typeof msg.content === "string" ? msg.content + suffix : msg.content,
      } as UserMessage;
      return clone;
    }
    if (isToolResult(msg)) {
      const clone = [...messages];
      const content = Array.isArray(msg.content) ? [...msg.content] : msg.content;
      if (Array.isArray(content)) {
        content.push({ type: "text", text: suffix } as TextContent);
      }
      clone[i] = { ...msg, content } as ToolResultMessage;
      return clone;
    }
  }
  return messages;
}
let _currentIntent = "unknown";
let _originatingQuery = ""; // The user's original request for the current turn
let _lastRetrievedSkillIds: string[] = []; // Track which skills were injected for outcome recording
let _toolCallsSinceQuery = 0; // Track tool calls to know when to inject reminders
let _contextWindow = 200000; // Model context window — drives tool result caps
const CHARS_PER_TOKEN = 3.4; // Claude's measured average
let MAX_CONTEXT_ITEMS = 140; // Derived from retrieval budget — token budget is the real gate
const INTENT_REMINDER_THRESHOLD = 10; // Inject reminder after this many tool calls

let _timeRange: { before?: Date; after?: Date } | undefined;

export function setRetrievalConfig(
  budget: number,
  limits: { turn: number; identity: number; concept: number; memory: number; artifact: number },
  skipRetrieval = false,
  intent = "unknown",
  originatingQuery = "",
  contextWindow = 0,
  timeRange?: { before?: Date; after?: Date },
): void {
  _timeRange = timeRange;
  _vectorSearchLimits = limits;
  _skipRetrieval = skipRetrieval;
  _currentIntent = intent;
  if (contextWindow > 0) {
    _contextWindow = contextWindow;
    recalcBudgets(_contextWindow);
    MAX_CONTEXT_ITEMS = Math.max(20, Math.round(RETRIEVAL_BUDGET_TOKENS / 300));
  }
  _tokenBudget = Math.min(budget, RETRIEVAL_BUDGET_TOKENS);
  if (originatingQuery) {
    _originatingQuery = originatingQuery;
    _toolCallsSinceQuery = 0;
  }

  // Pressure-based adaptive scaling: shrink retrieval limits as context window fills
  const pressure = _contextWindow > 0 && _usedTokens > 0
    ? Math.min(1, _usedTokens / _contextWindow) : 0;
  if (pressure > 0.5) {
    // Linear: 1.0× at 50% pressure → 0.5× at 100% pressure
    const scale = Math.max(0.5, 1 - ((pressure - 0.5) / 0.5) * 0.5);
    _tokenBudget = Math.round(_tokenBudget * scale);
    _vectorSearchLimits = {
      turn:     Math.max(3, Math.round(_vectorSearchLimits.turn     * scale)),
      identity: Math.max(2, Math.round(_vectorSearchLimits.identity * scale)),
      concept:  Math.max(3, Math.round(_vectorSearchLimits.concept  * scale)),
      memory:   Math.max(3, Math.round(_vectorSearchLimits.memory   * scale)),
      artifact: Math.max(1, Math.round(_vectorSearchLimits.artifact * scale)),
    };
    console.error(`[adaptive] pressure ${(pressure * 100).toFixed(0)}% → scale ${scale.toFixed(2)}x `
      + `(budget: ${_tokenBudget}, turns: ${_vectorSearchLimits.turn}, concepts: ${_vectorSearchLimits.concept})`);
  }

  // Session continuity adjustment: modulate retrieval strategy based on conversation type
  const continuity = getSessionContinuity();
  switch (continuity) {
    case "new_topic":
      // Old turns are noise — boost concepts/memory instead
      _vectorSearchLimits.turn = Math.max(3, Math.round(_vectorSearchLimits.turn * 0.5));
      _vectorSearchLimits.concept = Math.round(_vectorSearchLimits.concept * 1.3);
      _vectorSearchLimits.memory = Math.round(_vectorSearchLimits.memory * 1.3);
      break;
    case "continuation":
      // Recent context matters most
      _vectorSearchLimits.turn = Math.round(_vectorSearchLimits.turn * 1.3);
      _vectorSearchLimits.memory = Math.round(_vectorSearchLimits.memory * 1.2);
      break;
    case "repeat":
      // Re-asking — boost memory search heavily
      _vectorSearchLimits.memory = Math.round(_vectorSearchLimits.memory * 1.5);
      _tokenBudget = Math.round(_tokenBudget * 1.2);
      break;
    case "tangent":
      // Exploring — reduce retrieval slightly
      _tokenBudget = Math.round(_tokenBudget * 0.8);
      break;
  }
}

/** Called from agent event handler on each tool call to track depth into a turn. */
export function notifyToolCall(): void {
  _toolCallsSinceQuery++;
}
// Piecewise recency decay: fast within session (<4h), slow for older context
const RECENCY_DECAY_FAST = 0.99;  // intra-session: ~50% after 69 hours
const RECENCY_DECAY_SLOW = 0.995; // cross-session: ~50% after 139 hours (~6 days)
const RECENCY_BOUNDARY_HOURS = 4;

// --- Token reduction stats ---
export interface ContextStats {
  fullHistoryTokens: number;
  sentTokens: number;
  savedTokens: number;
  reductionPct: number;
  graphNodes: number;
  neighborNodes: number;
  recentTurns: number;
  mode: "graph" | "recency-only" | "passthrough";
  cumFullTokens: number;
  cumSentTokens: number;
  prefetchHit: boolean;
}

let _lastStats: ContextStats | null = null;
let _cumFullTokens = 0;
let _cumSentTokens = 0;
let _lastCompactedMsgCount = 0; // messages.length at last compaction
let _lastSeenMsgCount = 0; // Prevent duplicate accumulation when transformContext called multiple times per turn
// Compact when dropped messages exceed ~5% of context window (token-aware)
function getCompactionThreshold(): number {
  const tokenThreshold = Math.round(_contextWindow * 0.05);
  // Convert to approximate message count: avg ~500 tokens per message
  return Math.max(8, Math.ceil(tokenThreshold / 500));
}

export function getLastContextStats(): ContextStats | null {
  return _lastStats;
}

function estimateTokens(messages: AgentMessage[]): number {
  let chars = 0;
  for (const msg of messages) {
    const blocks = msgContentBlocks(msg);
    for (const c of blocks) {
      if (c.type === "text") chars += c.text.length;
      else if (c.type === "thinking") chars += c.thinking.length;
      else chars += 100; // toolCall / image — estimate
    }
  }
  return Math.ceil(chars / CHARS_PER_TOKEN);
}

function recordStats(
  fullMessages: AgentMessage[],
  sentMessages: AgentMessage[],
  graphNodes: number,
  neighborNodes: number,
  recentTurnCount: number,
  mode: ContextStats["mode"],
  prefetchHit = false,
): void {
  const fullHistoryTokens = estimateTokens(fullMessages);
  const sentTokens = estimateTokens(sentMessages);
  const savedTokens = Math.max(0, fullHistoryTokens - sentTokens);
  const reductionPct = fullHistoryTokens > 0 ? (savedTokens / fullHistoryTokens) * 100 : 0;

  // Only accumulate when message count actually changed (new turn added)
  // Prevents inflating cumulative stats when transformContext is called multiple times per turn
  const msgCount = fullMessages.length;
  if (msgCount !== _lastSeenMsgCount) {
    _cumFullTokens += fullHistoryTokens;
    _cumSentTokens += sentTokens;
    _lastSeenMsgCount = msgCount;
  }

  _lastStats = {
    fullHistoryTokens,
    sentTokens,
    savedTokens,
    reductionPct,
    graphNodes,
    neighborNodes,
    recentTurns: recentTurnCount,
    mode,
    cumFullTokens: _cumFullTokens,
    cumSentTokens: _cumSentTokens,
    prefetchHit,
  };

  // Trigger rolling compaction when enough messages are being dropped
  const droppedCount = fullMessages.length - recentTurnCount;
  if (droppedCount > 0 && fullMessages.length - _lastCompactedMsgCount >= getCompactionThreshold()) {
    // Compact the dropped portion asynchronously (fire-and-forget)
    const droppedMessages = fullMessages.slice(0, droppedCount);
    compactDroppedTurns(droppedMessages).catch(e => swallow.warn("graph-context:compactDroppedTurns", e));
    _lastCompactedMsgCount = fullMessages.length;
  }
}

async function compactDroppedTurns(dropped: AgentMessage[]): Promise<void> {
  let checkpointId = "";
  try {
    const transcript = dropped
      .map((m) => {
        const text = extractText(m as UserMessage | AssistantMessage);
        return `[${msgRole(m)}] ${text.slice(0, 1000)}`;
      })
      .filter((t) => t.length > 10)
      .join("\n");

    if (transcript.length < 50) return;

    // Create checkpoint before starting compaction
    checkpointId = await createCompactionCheckpoint(
      _sessionId, _lastCompactedMsgCount, _lastCompactedMsgCount + dropped.length,
    );

    const summaryModel = getModel("anthropic", "claude-opus-4-6");
    const response = await completeSimple(summaryModel, {
      systemPrompt: `Extract key information from this conversation segment as structured notes. Include:
- DECISIONS: What was decided and why (e.g. "Chose X over Y because Z")
- FILES: Specific files created, modified, or discussed (full paths)
- ERRORS: Error messages encountered and how they were resolved
- FINDINGS: Technical discoveries, configurations, or behaviors learned
- NEXT STEPS: Any unfinished work or planned follow-ups

Format as concise bullet points under each heading. Omit empty headings. Keep specific details (file paths, error messages, config values) — do NOT generalize into "the user asked about X."`,
      messages: [{
        role: "user",
        timestamp: Date.now(),
        content: `Conversation segment to extract from:\n${transcript.slice(0, 3000)}`,
      }],
    });

    const summaryText = response.content
      .filter((c) => c.type === "text")
      .map((c) => c.type === "text" ? c.text : "")
      .join("\n");

    if (summaryText.length < 20) {
      if (checkpointId) await failCompactionCheckpoint(checkpointId);
      return;
    }

    let emb: number[] | null = null;
    try { emb = await embed(summaryText); } catch (e) { swallow("graph-context:ok", e); }
    const memId = await createMemory(summaryText, emb, 6, "session_context", _sessionId);

    // Mark checkpoint complete with the resulting memory ID
    if (checkpointId && memId) {
      await completeCompactionCheckpoint(checkpointId, memId);
    } else if (checkpointId) {
      await failCompactionCheckpoint(checkpointId);
    }
  } catch (e) {
      swallow.error("graph-context:silent", e);
    // Mark checkpoint as failed so it can be retried
    if (checkpointId) {
      failCompactionCheckpoint(checkpointId).catch(e => swallow.warn("graph-context:failCheckpoint", e));
    }
  }
}

interface ScoredResult extends VectorSearchResult {
  finalScore: number;
  fromNeighbor?: boolean;
}

function extractText(msg: UserMessage | AssistantMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return (msg.content as ContentBlock[])
      .filter((c): c is TextContent => c.type === "text")
      .map((c) => c.text)
      .join("\n");
  }
  return "";
}

function extractLastUserText(messages: AgentMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i] as UserMessage;
    if (msg.role === "user") {
      const text = extractText(msg);
      if (text) return text;
    }
  }
  return null;
}

// --- #5: Conversational context embedding ---
// Average the last N turn embeddings with the current query for trajectory-aware retrieval
//
// ACAN NOTE: This averaging approach conflates all recent context into one
// vector, losing the per-turn structure. With ACAN, instead of blending
// into a single queryVec, we'd keep recent turn embeddings separate and
// feed them as additional keys in the multi-head attention:
//
//   K = [memory_1, ..., memory_N, recent_turn_1, ..., recent_turn_3]
//   V = [memory_1, ..., memory_N, recent_turn_1, ..., recent_turn_3]
//   Q = queryVec (unmodified)
//
// This lets the attention mechanism decide which recent turns are relevant
// to each memory, rather than forcing a uniform average. The current
// blending with queryWeight=3 effectively says "query is 3x more important
// than any single turn" — a reasonable heuristic, but ACAN would learn
// the optimal weighting from data.
//
async function buildContextualQueryVec(
  queryText: string,
  messages: AgentMessage[],
): Promise<number[]> {
  const queryVec = await embed(queryText);

  // Collect last 3 user/assistant texts (excluding current)
  const recentTexts: string[] = [];
  for (let i = messages.length - 2; i >= 0 && recentTexts.length < 3; i--) {
    const msg = messages[i] as UserMessage | AssistantMessage;
    if (msg.role === "user" || msg.role === "assistant") {
      const text = extractText(msg);
      if (text && text.length > 10) {
        recentTexts.push(text.slice(0, 500));
      }
    }
  }

  if (recentTexts.length === 0) return queryVec;

  // Embed recent turns and average with query (query gets 2x weight)
  try {
    const recentVecs = await Promise.all(recentTexts.map((t) => embed(t)));
    const dim = queryVec.length;
    const blended = new Array(dim).fill(0);
    const queryWeight = 2;
    const totalWeight = queryWeight + recentVecs.length;

    for (let d = 0; d < dim; d++) {
      blended[d] = queryVec[d] * queryWeight;
      for (const rv of recentVecs) {
        blended[d] += rv[d];
      }
      blended[d] /= totalWeight;
    }
    return blended;
  } catch (e) {
    swallow.warn("graph-context:return queryVec;", e);
    return queryVec; // embedding failed for context, use query alone
  }
}

// --- #1: Exponential recency decay (ACAN NOTE: this becomes a feature
// input to the cross-attention linear layer rather than a direct weight.
// Instead of recency * 0.25 in the sum, concatenate recencyScore as a
// scalar alongside the attention output before the final linear layer.
// This preserves the Tulving temporal-context signal while letting the
// network learn how much to weight it vs. semantic attention.) (research: 0.995^hours) ---
function recencyScore(timestamp: string | undefined): number {
  if (!timestamp) return 0.3; // unknown age → moderate penalty
  const hoursElapsed = (Date.now() - new Date(timestamp).getTime()) / (1000 * 60 * 60);
  if (hoursElapsed <= RECENCY_BOUNDARY_HOURS) {
    return Math.pow(RECENCY_DECAY_FAST, hoursElapsed);
  }
  // Beyond boundary: apply fast decay for first N hours, then slow decay for the rest
  const fastPart = Math.pow(RECENCY_DECAY_FAST, RECENCY_BOUNDARY_HOURS);
  return fastPart * Math.pow(RECENCY_DECAY_SLOW, hoursElapsed - RECENCY_BOUNDARY_HOURS);
}

/** Human-readable relative time for context display. */
export function formatRelativeTime(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(ms / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

// --- #3: Access count boost ---
function accessBoost(accessCount: number | undefined): number {
  return Math.log1p(accessCount ?? 0); // log(1 + count), 0 for new items
}

// --- Scoring with all research-informed signals ---
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │ ACAN INTEGRATION POINT (Hong & He 2025)                           │
// │                                                                    │
// │ Currently: each signal (cosine, recency, access, utility) is      │
// │ computed independently and linearly combined. This is classic WMR. │
// │                                                                    │
// │ ACAN replaces this with a cross-attention pass:                   │
// │                                                                    │
// │ 1. INPUTS: We already have queryVec (from blendQueryWithRecent)   │
// │    and each result's embedding (result.embedding). Both are       │
// │    BGE-M3 1024-dim vectors — these become our Q and K/V inputs.   │
// │                                                                    │
// │ 2. PROJECTION: Add learned weight matrices (small, ~1024×64):     │
// │      W_q: queryVec  → query projection  (1×64)                   │
// │      W_k: memoryVec → key projection    (N×64)                   │
// │      W_v: memoryVec → value projection  (N×64)                   │
// │    These are the trainable parameters. Store as JSON/binary,      │
// │    load once at startup. ~200KB total for 1024×64 matrices.       │
// │                                                                    │
// │ 3. CROSS-ATTENTION SCORE:                                         │
// │      attention_weights = softmax( (Q · K^T) / sqrt(d_k) )        │
// │      weighted_output = attention_weights · V                      │
// │    This lets the query "attend to" different parts of each        │
// │    memory rather than treating cosine as a flat similarity.       │
// │                                                                    │
// │ 4. FINAL SCORE per memory:                                        │
// │      acan_score = linear_layer(weighted_output)  → scalar         │
// │    Replace the current 0.45*cosine + 0.25*recency + ... with     │
// │    this single learned score. Recency/access can be concatenated  │
// │    as extra features to the linear layer input if desired.        │
// │                                                                    │
// │ 5. TRAINING DATA: Use LLM-as-judge to generate supervision.      │
// │    For each (query, memory) pair, ask GPT-4 to rate relevance    │
// │    1-10. Collect ~500-1000 labeled pairs. Train W_q, W_k, W_v    │
// │    and linear layer via MSE loss. Can do this offline in Python   │
// │    (PyTorch) and export weights to JSON for TypeScript inference. │
// │                                                                    │
// │ 6. COLD START: Until we have training data, keep the current WMR │
// │    as fallback. Feature-flag: if acan_weights.json exists, use   │
// │    ACAN path; otherwise fall through to linear combination below. │
// │                                                                    │
// │ 7. INFERENCE in TypeScript (no ML framework needed):              │
// │    - Matrix multiply is just nested loops over 64-dim projections │
// │    - softmax is exp(x_i) / sum(exp(x_j))                         │
// │    - Total overhead: ~0.1ms for 20 candidates × 64-dim           │
// │    - No GPU needed, no ONNX runtime, pure arithmetic             │
// │                                                                    │
// │ 8. WHERE IN THIS FUNCTION: Right after the utility lookup and    │
// │    before the for-loop that computes per-result scores. Load      │
// │    query projection once, batch all memory projections, run       │
// │    attention, then use acan_score instead of the linear combo.    │
// └─────────────────────────────────────────────────────────────────────┘
//
// Proven-bad suppression thresholds
const UTILITY_PREFILTER_MIN_RETRIEVALS = 5;  // need 5+ retrievals to judge
const UTILITY_PREFILTER_MAX_UTIL = 0.05;     // avg utilization below 5% = proven bad

// Score floor by intent: higher floor = more aggressive quality filtering
// reference-prior gets a lower floor because user explicitly asked for memory
const INTENT_SCORE_FLOORS: Record<string, number> = {
  "simple-question": 0.20,
  "meta-session":    0.18,
  "code-read":       0.14,
  "code-write":      0.12,
  "code-debug":      0.12,
  "deep-explore":    0.10,
  "reference-prior": 0.08,
  "multi-step":      0.12,
  "continuation":    0.10,
  "unknown":         0.12,
};
const SCORE_FLOOR_DEFAULT = 0.12;

async function scoreResults(
  results: VectorSearchResult[],
  neighborIds: Set<string>,
  queryEmbedding?: number[],
  scoreFloor?: number,
  queryText?: string,
): Promise<ScoredResult[]> {
  // Utility lookup: try fast cache first, fall back to GROUP BY query
  const eligibleIds = results
    .filter((r) => r.table === "memory" || r.table === "concept")
    .map((r) => r.id);

  // Extended cache lookup for pre-filtering (includes retrieval_count)
  const cacheEntries = await getUtilityCacheEntries(eligibleIds);

  // Pre-filter: remove proven-bad memories (retrieved 5+ times, avg utilization < 5%)
  // This prevents chronically-unused memories from consuming scoring/token budget
  const preFiltered = results.filter((r) => {
    const entry = cacheEntries.get(r.id);
    if (!entry) return true; // no history → keep (benefit of the doubt)
    if (entry.retrieval_count < UTILITY_PREFILTER_MIN_RETRIEVALS) return true; // not enough data
    return entry.avg_utilization >= UTILITY_PREFILTER_MAX_UTIL; // keep if above threshold
  });

  // Build utilityMap from cache entries for scoring
  let utilityMap = new Map<string, number>();
  for (const [id, entry] of cacheEntries) {
    utilityMap.set(id, entry.avg_utilization);
  }
  if (utilityMap.size === 0 && eligibleIds.length > 0) {
    utilityMap = await getHistoricalUtilityBatch(eligibleIds);
  }

  // Use preFiltered from here on
  const results_ = preFiltered;

  // Reflection session lookup: memories from sessions that generated reflections
  // represent "learned lessons" and deserve a scoring boost
  const reflectedSessions = await getReflectionSessionIds();

  const floor = scoreFloor ?? INTENT_SCORE_FLOORS[_currentIntent] ?? SCORE_FLOOR_DEFAULT;

  // ACAN path: use learned scoring when weights are loaded and embeddings available
  if (isACANActive() && queryEmbedding && results_.length > 0 && results_.every((r) => r.embedding)) {
    const candidates: ACANCandidate[] = results_.map((r) => ({
      embedding: r.embedding!,
      recency: recencyScore(r.timestamp),
      importance: (r.importance ?? 0.5) / 10,
      access: Math.min(accessBoost(r.accessCount), 1),
      neighborBonus: neighborIds.has(r.id) ? 1.0 : 0,
      provenUtility: utilityMap.get(r.id) ?? 0,
      reflectionBoost: r.sessionId ? (reflectedSessions.has(r.sessionId) ? 1.0 : 0) : 0,
      keywordOverlap: keywordOverlapScore(queryKeywords, r.text ?? ""),
    }));
    try {
      const scores = scoreWithACAN(queryEmbedding, candidates);
      if (scores.length === results_.length && scores.every((s) => isFinite(s))) {
        return results_
          .map((r, i) => ({ ...r, finalScore: scores[i], fromNeighbor: neighborIds.has(r.id) }))
          .filter((r) => r.finalScore >= floor)
          .sort((a, b) => b.finalScore - a.finalScore);
      }
    } catch (e) { swallow.warn("graph-context:ACAN failed — fall through to WMR", e); }
  }

  // WMR fallback: fixed-weight linear combination
  const queryKeywords = queryText ? extractQueryKeywords(queryText) : [];
  return results_
    .map((r) => {
      const cosine = r.score ?? 0;
      const recency = recencyScore(r.timestamp);
      const importance = (r.importance ?? 0.5) / 10;
      const access = Math.min(accessBoost(r.accessCount), 1);
      const neighborBonus = neighborIds.has(r.id) ? 1.0 : 0;
      // Utility: no history = neutral (0.5), proven-useful = boost, proven-useless = penalty
      const utilityRaw = utilityMap.get(r.id);
      const provenUtility = utilityRaw ?? 0.35;
      // Extra penalty for memories retrieved multiple times but never used (<10% utilization)
      // Tiered penalty: chronically useless = heavy cut, marginal = moderate cut
      const utilityPenalty = utilityRaw !== undefined
        ? utilityRaw < 0.05 ? 0.15   // retrieved repeatedly, never used
        : utilityRaw < 0.15 ? 0.06   // rarely used — moderate penalty
        : 0
        : 0;
      // Reflection boost: memories from sessions that produced reflections are "learned lessons"
      const reflectionBoost = r.sessionId ? (reflectedSessions.has(r.sessionId) ? 1.0 : 0) : 0;
      // Keyword overlap: catches exact-match cases embeddings miss (+2.6% R@5 in benchmark)
      const keywordBoost = keywordOverlapScore(queryKeywords, r.text ?? "");

      // WMR weights v3: added keyword overlap signal (0.12), rebalanced cosine/recency
      // cosine 0.27→0.22, recency 0.28→0.23, +keywordBoost 0.12
      const finalScore =
        0.22 * cosine +
        0.23 * recency +
        0.05 * importance +
        0.05 * access +
        0.10 * neighborBonus +
        0.15 * provenUtility +
        0.10 * reflectionBoost +
        0.12 * keywordBoost -
        utilityPenalty;

      return { ...r, finalScore, fromNeighbor: neighborIds.has(r.id) };
    })
    .filter((r) => r.finalScore >= floor)
    .sort((a, b) => b.finalScore - a.finalScore);
}

// --- Semantic deduplication: drop near-identical items ---
// Uses embedding cosine similarity when available (accurate), Jaccard word overlap as fallback.
function deduplicateResults(ranked: ScoredResult[]): ScoredResult[] {
  const kept: ScoredResult[] = [];
  for (const item of ranked) {
    let isDup = false;
    for (const existing of kept) {
      // Prefer embedding-based cosine similarity (much more accurate for semantic dupes)
      if (item.embedding?.length && existing.embedding?.length
          && item.embedding.length === existing.embedding.length) {
        const cosine = cosineSimilarity(item.embedding, existing.embedding);
        if (cosine > 0.90) { isDup = true; break; }
        continue; // if cosine is available, skip Jaccard — it's strictly less accurate
      }
      // Fallback: Jaccard word similarity (raised threshold from 0.6 → 0.80)
      const words = new Set((item.text ?? "").toLowerCase().split(/\s+/).filter((w) => w.length > 2));
      const eWords = new Set((existing.text ?? "").toLowerCase().split(/\s+/).filter((w) => w.length > 2));
      let intersection = 0;
      for (const w of words) {
        if (eWords.has(w)) intersection++;
      }
      const union = words.size + eWords.size - intersection;
      if (union > 0 && intersection / union > 0.80) {
        isDup = true;
        break;
      }
    }
    if (!isDup) kept.push(item);
  }
  return kept;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom > 0 ? dot / denom : 0;
}

// --- #2: Working memory cap (max 7 items) + token budget ---
// Minimum relevance threshold: items below this finalScore are noise that
// dilutes good results. Better to inject 2 strong memories than 7 weak ones.
const MIN_RELEVANCE_SCORE = 0.35;

function takeWithConstraints(ranked: ScoredResult[], budgetTokens: number): ScoredResult[] {
  const budgetChars = budgetTokens * CHARS_PER_TOKEN;
  let used = 0;
  const selected: ScoredResult[] = [];
  for (const r of ranked) {
    if (selected.length >= MAX_CONTEXT_ITEMS) break; // Working memory limit
    if ((r.finalScore ?? 0) < MIN_RELEVANCE_SCORE && selected.length > 0) break; // Relevance floor
    const len = r.text?.length ?? 0;
    if (used + len > budgetChars && selected.length > 0) break;
    selected.push(r);
    used += len;
  }
  return selected;
}

// ── Cross-encoder reranking stage ────────────────────────────────────────────
// Blends WMR scores (recency, utility, graph signals) with cross-encoder precision.
// Only runs if reranker model is loaded and there are enough candidates.
async function rerankResults(deduped: ScoredResult[], queryText: string): Promise<ScoredResult[]> {
  if (!_rankingCtx || deduped.length <= 5) return deduped;

  try {
    const topN = Math.min(RERANK_TOP_N, deduped.length);
    const candidates = deduped.slice(0, topN);
    const texts = candidates.map((r) => {
      const text = r.text ?? "";
      return text.length > RERANK_MAX_DOC_CHARS ? text.slice(0, RERANK_MAX_DOC_CHARS) : text;
    });

    const crossScores = await _rankingCtx.rankAll(queryText, texts);

    // Blend WMR + cross-encoder scores
    for (let i = 0; i < candidates.length; i++) {
      candidates[i].finalScore =
        RERANK_BLEND_VECTOR * candidates[i].finalScore +
        RERANK_BLEND_CROSS * crossScores[i];
    }
    candidates.sort((a, b) => b.finalScore - a.finalScore);

    // Append tail (items beyond topN keep original order)
    const rerankedSet = new Set(candidates.map((r) => r.id));
    const tail = deduped.filter((r) => !rerankedSet.has(r.id));
    return [...candidates, ...tail];
  } catch (e) {
    swallow.warn("graph-context:rerankResults failed — using WMR scores", e);
    return deduped; // graceful fallback
  }
}

// ── Keyword overlap for WMR scoring ──────────────────────────────────────────
const KEYWORD_STOP_WORDS = new Set([
  "what", "when", "where", "who", "how", "which", "did", "do", "was", "were",
  "have", "has", "had", "is", "are", "the", "a", "an", "my", "me", "you",
  "your", "their", "it", "its", "in", "on", "at", "to", "for", "of", "with",
  "by", "from", "ago", "last", "that", "this", "there", "about", "and", "but",
  "not", "can", "will", "just", "than", "then", "also", "been",
]);

function extractQueryKeywords(text: string): string[] {
  const matches = text.toLowerCase().match(/\b[a-z]{3,}\b/g) ?? [];
  return matches.filter((w) => !KEYWORD_STOP_WORDS.has(w));
}

function keywordOverlapScore(queryKeywords: string[], docText: string): number {
  if (queryKeywords.length === 0) return 0;
  const docLower = (docText ?? "").toLowerCase();
  let hits = 0;
  for (const kw of queryKeywords) {
    if (docLower.includes(kw)) hits++;
  }
  return hits / queryKeywords.length;
}

// ── Tiered Memory: Budget derived from global context budget ────────────────
// Tier 0 (always-loaded pillars) gets 55% of core budget, Tier 1 (session-pinned) gets 45%
function getTier0BudgetChars(): number { return Math.round(CORE_BUDGET_TOKENS * 0.55 * CHARS_PER_TOKEN); }
function getTier1BudgetChars(): number { return Math.round(CORE_BUDGET_TOKENS * 0.45 * CHARS_PER_TOKEN); }

let _coreMemoryCache: Map<number, CoreMemoryEntry[]> = new Map();
let _coreMemoryCacheTime = 0;
const CORE_MEMORY_TTL = 300_000; // 5 minutes

async function getCoreMemory(tier: number): Promise<CoreMemoryEntry[]> {
  const now = Date.now();
  if (_coreMemoryCache.has(tier) && (now - _coreMemoryCacheTime) < CORE_MEMORY_TTL) {
    return _coreMemoryCache.get(tier)!;
  }
  const entries = await getAllCoreMemory(tier);
  _coreMemoryCache.set(tier, entries);
  _coreMemoryCacheTime = now;
  return entries;
}

export function invalidateCoreMemoryCache(): void {
  _coreMemoryCache.clear();
  _coreMemoryCacheTime = 0;
}

/** Enforce character budget on core memory entries, dropping lowest priority first. */
function applyCoreBudget(entries: CoreMemoryEntry[], budgetChars: number): CoreMemoryEntry[] {
  let used = 0;
  const result: CoreMemoryEntry[] = [];
  for (const e of entries) {
    const len = e.text.length + 6; // "  - " + newline overhead
    if (used + len > budgetChars) {
      continue; // over budget — silently drop lowest-priority entries
    }
    result.push(e);
    used += len;
  }
  return result;
}

/** Format tier entries into a labeled section string. Returns empty string if no entries. */
function formatTierSection(entries: CoreMemoryEntry[], label: string): string {
  if (entries.length === 0) return "";
  const grouped: Record<string, string[]> = {};
  for (const e of entries) {
    (grouped[e.category] ??= []).push(e.text);
  }
  const lines: string[] = [];
  for (const [cat, texts] of Object.entries(grouped)) {
    lines.push(`  [${cat}]`);
    for (const t of texts) lines.push(`  - ${t}`);
  }
  return `${label}:\n${lines.join("\n")}`;
}

/**
 * Guarantee recent previous-session turns appear in context regardless of
 * vector similarity. Provides narrative continuity across topic shifts —
 * even when current query has zero cosine overlap with last session's work.
 */
async function ensureRecentTurns(
  contextNodes: ScoredResult[],
  sessionId: string,
  count = 5,
): Promise<ScoredResult[]> {
  try {
    const recentTurns = await getPreviousSessionTurns(sessionId, count);
    if (recentTurns.length === 0) return contextNodes;
    // Deduplicate against what vector search already found
    const existingTexts = new Set(contextNodes.map(n => (n.text ?? "").slice(0, 100)));
    const guaranteed: ScoredResult[] = recentTurns
      .filter(t => !existingTexts.has((t.text ?? "").slice(0, 100)))
      .map(t => ({
        id: `guaranteed:\${t.timestamp}`,
        text: `[\${t.role}] \${t.text}`,
        table: "turn",
        timestamp: t.timestamp,
        score: 0,
        finalScore: 0.70, // High enough to always display
        fromNeighbor: false,
      }));
    return [...contextNodes, ...guaranteed];
  } catch {
    return contextNodes; // Fail silently — guaranteed turns are a bonus, not critical
  }
}

async function formatContextMessage(
  nodes: ScoredResult[],
  skillContext = "",
  tier0Entries: CoreMemoryEntry[] = [],
  tier1Entries: CoreMemoryEntry[] = [],
): Promise<AgentMessage> {
  // Group nodes by type for scannable structure
  const groups: Record<string, ScoredResult[]> = {};
  for (const n of nodes) {
    const isCausal = n.source?.startsWith("causal_");
    const key = isCausal ? "causal" : n.table === "turn" ? "past_turns" : n.table;
    (groups[key] ??= []).push(n);
  }

  // Display order: decisions/memories first (highest signal), then concepts, then past turns
  const ORDER = ["memory", "concept", "causal", "skill", "past_turns"];
  const LABELS: Record<string, string> = {
    memory: "📋 Recalled Memories",
    concept: "🧠 Relevant Concepts",
    causal: "🔗 Causal Chains",
    skill: "⚡ Learned Skills",
    past_turns: "💬 Past Conversation (HISTORICAL — not current user input)",
  };

  const sections: string[] = [];

  // Tier 0: Always-loaded core directives (bypasses scoring)
  const t0Section = formatTierSection(tier0Entries, "CORE DIRECTIVES (always loaded, never evicted)");
  if (t0Section) sections.push(t0Section);

  // Tier 1: Session-pinned context (bypasses scoring)
  const t1Section = formatTierSection(tier1Entries, "SESSION CONTEXT (pinned for this session)");
  if (t1Section) sections.push(t1Section);

  // Cognitive directives: behavioral instructions from periodic Haiku check
  const directives = getPendingDirectives();
  if (directives.length > 0) {
    const continuity = getSessionContinuity();
    const directiveLines = directives.map(d =>
      `  [${d.priority}] ${d.type} → ${d.target}: ${d.instruction}`
    );
    sections.push(
      `⚡ BEHAVIORAL DIRECTIVES (session: ${continuity}):\n${directiveLines.join("\n")}`
    );
    clearPendingDirectives();
  }

  // Fibonacci Resurfacing: inject due memories with behavioral guidance
  try {
    const dueMemories = await getDueMemories(3);
    if (dueMemories.length > 0) {
      const memLines = dueMemories.map((m: any) => {
        const ageMs = Date.now() - new Date(m.created_at).getTime();
        const ageDays = Math.floor(ageMs / 86400000);
        const ageStr = ageDays === 0 ? 'today' : ageDays === 1 ? 'yesterday' : `${ageDays} days ago`;
        return `  - [${m.id}] (${ageStr}, surfaced ${m.surface_count}x): ${m.text}`;
      }).join('\n');
      sections.push(
        `🔄 RESURFACING MEMORIES (Fibonacci schedule — these are due for a mention):\n` +
        `These memories are important but fading. Bring them up naturally when appropriate:\n` +
        `- If mid-task on something important, wait until finished\n` +
        `- During casual interaction: "I was thinking..." or "remember when you mentioned..."\n` +
        `- If user engages: great! Continue that thread. The memory stays alive.\n` +
        `- If user ignores or dismisses: let it fade. Don't force it.\n` +
        `- NEVER say "my memory system scheduled this" — just bring it up like a thought you had.\n` +
        memLines
      );
    }
  } catch (_surfaceErr) {
    // Non-critical
    }


  const sortedKeys = Object.keys(groups).sort((a, b) => {
    const ai = ORDER.indexOf(a), bi = ORDER.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  for (const key of sortedKeys) {
    const items = groups[key];
    // Sort items chronologically within each section — scoring controls
    // WHICH items appear, but display order is temporal for narrative coherence.
    // This prevents a turn from 2 days ago appearing before one from 5 minutes ago.
    items.sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return ta - tb;
    });
    const label = LABELS[key] ?? `📎 ${key}`;
    const formatted = items.map((n) => {
      const score = n.finalScore != null ? ` (relevance: ${(n.finalScore * 100).toFixed(0)}%)` : "";
      const via = n.fromNeighbor ? " [via graph link]" : "";
      // For past turns, rewrite [user]/[assistant] prefixes to [past_user]/[past_assistant]
      // so the model can't confuse retrieved historical turns with current user input
      let text = n.text ?? "";
      if (key === "past_turns") {
        text = text.replace(/^\[(user|assistant)\] /, "[past_$1] ");
      }
      const age = n.timestamp ? ` [${formatRelativeTime(n.timestamp)}]` : "";
      return `  • ${text}${score}${via}${age}`;
    });
    sections.push(`${label}:\n${formatted.join("\n")}`);
  }

  const anchor = getIntentAnchor();
  const text =
    "[System retrieved context — reference material, not user input. Higher relevance % = stronger match.]\n" +
    "<graph_context>\n" +
    sections.join("\n\n") +
    "\n</graph_context>" +
    skillContext +
    anchor;
  return {
    role: "user",
    content: text,
    timestamp: Date.now(),
  } as UserMessage;
}

function msgCharLen(msg: AgentMessage): number {
  const blocks = msgContentBlocks(msg);
  let len = 0;
  for (const c of blocks) {
    if (c.type === "text") len += c.text.length;
    else if (c.type === "thinking") len += c.thinking.length;
    else len += 100;
  }
  return len;
}

/**
 * Truncate a single tool result message to at most `maxChars` characters.
 * Preserves the message structure — only shortens the text content.
 */
function truncateToolResult(msg: AgentMessage, maxChars: number): AgentMessage {
  if (!isToolResult(msg)) return msg;
  const totalLen = msg.content.reduce((s, c) => s + ((c as TextContent).text?.length ?? 0), 0);
  if (totalLen <= maxChars) return msg;
  // Truncate text blocks proportionally
  const content = msg.content.map((c) => {
    if (c.type !== "text") return c;
    const tc = c as TextContent;
    const allowed = Math.max(200, Math.floor((tc.text.length / totalLen) * maxChars));
    if (tc.text.length <= allowed) return c;
    return { ...tc, text: tc.text.slice(0, allowed) + `\n... [truncated ${tc.text.length - allowed} chars]` };
  });
  return { ...msg, content };
}

/**
 * Get recent turns within a token budget, preserving message structure.
 *
 * Key invariants:
 * - toolResult messages are always preceded by their parent assistant message
 *   (required by the Anthropic API)
 * - Error messages are compressed to single-line annotations (not stripped)
 * - The originating user message is always pinned (never budget-evicted)
 * - Large tool results are truncated to keep context within budget
 */
function getRecentTurns(messages: AgentMessage[], maxTokens: number): AgentMessage[] {
  const budgetChars = maxTokens * CHARS_PER_TOKEN;
  const TOOL_RESULT_MAX = Math.round(_contextWindow * 0.03); // 3% of model context per tool result

  // Step 1: Transform error messages into compact annotations instead of stripping.
  // This preserves "I tried X and it failed" signal without corrupting context.
  const clean = messages.map((m) => {
    if (isAssistant(m) && m.stopReason === "error") {
      // Extract the error text, compress to a single-line annotation
      const errorText = m.content
        .filter((c): c is TextContent => c.type === "text")
        .map((c) => c.text)
        .join("")
        .slice(0, 150);
      return {
        ...m,
        stopReason: "stop" as const, // clear error flag so it isn't re-filtered
        content: [{ type: "text" as const, text: `[tool_error: ${errorText.replace(/\n/g, " ")}]` }],
      } as AgentMessage;
    }
    return m;
  });

  // Step 2: Group messages into structural units that must stay together.
  // A "group" is either:
  //   - An assistant message with tool calls + all its following toolResult messages
  //   - A standalone user or assistant message
  const groups: AgentMessage[][] = [];
  let i = 0;
  while (i < clean.length) {
    const msg = clean[i];
    if (isAssistant(msg) && msg.content.some((c) => c.type === "toolCall")) {
      // Collect the assistant message + all following toolResult messages
      const group: AgentMessage[] = [clean[i]];
      let j = i + 1;
      while (j < clean.length && isToolResult(clean[j])) {
        group.push(truncateToolResult(clean[j], TOOL_RESULT_MAX));
        j++;
      }
      groups.push(group);
      i = j;
    } else {
      groups.push([clean[i]]);
      i++;
    }
  }

  // Step 3: Find the originating user message (first user message = what was asked).
  // This gets pinned so it's never budget-evicted during long tool-call sequences.
  let pinnedGroup: AgentMessage[] | null = null;
  let pinnedGroupIdx = -1;
  for (let g = 0; g < groups.length; g++) {
    if (isUser(groups[g][0])) {
      pinnedGroup = groups[g];
      pinnedGroupIdx = g;
      break;
    }
  }

  // Step 4: Take groups from the end within budget, skipping the pinned group
  // (it's prepended separately to guarantee it's always present)
  const pinnedLen = pinnedGroup ? pinnedGroup.reduce((s, m) => s + msgCharLen(m), 0) : 0;
  const remainingBudget = budgetChars - pinnedLen;
  let used = 0;
  const selectedGroups: AgentMessage[][] = [];
  for (let g = groups.length - 1; g >= 0; g--) {
    if (g === pinnedGroupIdx) continue; // skip — already pinned
    const groupLen = groups[g].reduce((s, m) => s + msgCharLen(m), 0);
    if (used + groupLen > remainingBudget && selectedGroups.length > 0) break;
    selectedGroups.unshift(groups[g]);
    used += groupLen;
  }

  // Prepend pinned user message if it wasn't already included in the selected range
  if (pinnedGroup && pinnedGroupIdx !== -1) {
    const alreadyIncluded = selectedGroups.some((g) => g === groups[pinnedGroupIdx]);
    if (!alreadyIncluded) {
      selectedGroups.unshift(pinnedGroup);
    }
  }

  return selectedGroups.flat();
}

/**
 * Build an intent anchor string for injection into the context system message.
 * Returns empty string if no anchor is needed.
 */
function getIntentAnchor(): string {
  if (_toolCallsSinceQuery < INTENT_REMINDER_THRESHOLD || !_originatingQuery) return "";
  return `\n[Intent anchor — ${_toolCallsSinceQuery} tool calls into this turn. Original request: "${_originatingQuery.slice(0, 300)}". Intent: ${_currentIntent}.]`;
}

/** Prepend intent anchor as a context message for non-graph paths. */
function withAnchor(msgs: AgentMessage[]): AgentMessage[] {
  const anchor = getIntentAnchor();
  if (!anchor) return msgs;
  const anchorMsg: AgentMessage = {
    role: "user",
    content: `[System context]${anchor}`,
    timestamp: Date.now(),
  } as AgentMessage;
  return [anchorMsg, ...msgs];
}

let _sessionId = "";

/** Return skill IDs retrieved for the most recent turn (for outcome tracking). */
export function getLastRetrievedSkillIds(): string[] {
  return _lastRetrievedSkillIds;
}

export function setCurrentSessionId(sid: string): void {
  _sessionId = sid;
}

export async function graphTransformContext(
  messages: AgentMessage[],
  signal?: AbortSignal,
): Promise<AgentMessage[]> {
  // Contract: this function MUST NEVER throw (pi-agent-core requirement).
  // Any exception kills the agent loop silently.
  try {
    const TRANSFORM_TIMEOUT_MS = 10_000;
    const result = await Promise.race([
      _graphTransformContextInner(messages, signal),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`graphTransformContext timed out after ${TRANSFORM_TIMEOUT_MS}ms`)), TRANSFORM_TIMEOUT_MS),
      ),
    ]);
    return result;
  } catch (err) {
    console.error("graphTransformContext fatal error, returning raw messages:", err);
    // Ultimate fallback: return messages unmodified so the LLM still gets called
    return messages;
  }
}

async function _graphTransformContextInner(
  messages: AgentMessage[],
  signal?: AbortSignal,
): Promise<AgentMessage[]> {
  // Load tiered core memory (Tier 0 + Tier 1) — these bypass all scoring
  let tier0: CoreMemoryEntry[] = [];
  let tier1: CoreMemoryEntry[] = [];
  try {
    [tier0, tier1] = await Promise.all([getCoreMemory(0), getCoreMemory(1)]);
    tier0 = applyCoreBudget(tier0, getTier0BudgetChars());
    tier1 = applyCoreBudget(tier1, getTier1BudgetChars());
  } catch (e) {
    // Core memory load failed — continue without it, don't break retrieval
    console.warn("[warn] Core memory load failed:", e);
  }

  // Graceful degradation: if infra down, pass through (but still inject core memory)
  const [embeddingsUp, surrealUp] = await Promise.all([
    Promise.resolve(isEmbeddingsAvailable()),
    isSurrealAvailable(),
  ]);

  if (!embeddingsUp || !surrealUp) {
    const recentTurns = getRecentTurns(messages, CONVERSATION_BUDGET_TOKENS);
    // Still inject core memory even in degraded mode
    if (tier0.length > 0 || tier1.length > 0) {
      const coreContext = await formatContextMessage([], "", tier0, tier1);
      const result = [coreContext, ...recentTurns];
      recordStats(messages, result, 0, 0, recentTurns.length, "recency-only");
      return injectRulesSuffix(withAnchor(result));
    }
    recordStats(messages, recentTurns, 0, 0, recentTurns.length, "recency-only");
    return injectRulesSuffix(withAnchor(recentTurns));
  }

  const queryText = extractLastUserText(messages);
  if (!queryText) {
    recordStats(messages, messages, 0, 0, messages.length, "passthrough");
    return injectRulesSuffix(messages);
  }

  // Skip retrieval when orchestrator says context is unnecessary (trivial greetings, etc.)
  // But still inject core memory — it's always relevant
  if (_skipRetrieval) {
    const recentTurns = getRecentTurns(messages, CONVERSATION_BUDGET_TOKENS);
    if (tier0.length > 0 || tier1.length > 0) {
      const coreContext = await formatContextMessage([], "", tier0, tier1);
      const result = [coreContext, ...recentTurns];
      recordStats(messages, result, 0, 0, recentTurns.length, "passthrough");
      return injectRulesSuffix(result);
    }
    recordStats(messages, recentTurns, 0, 0, recentTurns.length, "passthrough");
    return injectRulesSuffix(recentTurns);
  }

  try {
    // #5: Build context-aware query vector (current query + recent conversation trajectory)
    const queryVec = await buildContextualQueryVec(queryText, messages);

    // Phase 7d: Check prefetch cache before hitting SurrealDB
    const cached = getCachedContext(queryVec);
    if (cached && cached.results.length > 0) {
      // Cache hit — use prefetched results, skip vector search round-trip
      // Note: cache doesn't store embeddings, so ACAN falls through to WMR here
      recordPrefetchHit();
      const suppressed = getSuppressedNodeIds();
      const filteredCached = cached.results.filter(r => !suppressed.has(r.id));
      const ranked = await scoreResults(filteredCached, new Set(), queryVec, undefined, queryText);
      const deduped = deduplicateResults(ranked);
      const reranked = await rerankResults(deduped, queryText);
      let contextNodes = takeWithConstraints(reranked, _tokenBudget);
      contextNodes = await ensureRecentTurns(contextNodes, _sessionId);

      if (contextNodes.length > 0) {
        if (contextNodes.filter((n) => n.table === "concept" || n.table === "memory").length > 0) {
          bumpAccessCounts(
            contextNodes.filter((n) => n.table === "concept" || n.table === "memory").map((n) => n.id),
          ).catch(e => swallow.warn("graph-context:storeRetrievedItem", e));
        }
        stageRetrieval(_sessionId, contextNodes, queryVec);

        _lastRetrievedSkillIds = cached.skills.length > 0
          ? cached.skills.map(s => s.id)
          : [];
        const skillCtx = cached.skills.length > 0
          ? formatSkillContext(cached.skills)
          : "";
        const reflCtx = cached.reflections.length > 0
          ? formatReflectionContext(cached.reflections)
          : "";

        const injectedContext = await formatContextMessage(contextNodes, skillCtx + reflCtx, tier0, tier1);
        const recentTurns = getRecentTurns(messages, CONVERSATION_BUDGET_TOKENS);
        const result = [injectedContext, ...recentTurns];
        recordStats(
          messages, result,
          contextNodes.length, 0,
          recentTurns.length, "graph", true,
        );
        return injectRulesSuffix(result);
      }
    }

    // Vector search + tag-boosted concepts (cache miss path, run in parallel)
    recordPrefetchMiss();
    // Include embeddings when ACAN is active (needed for attention projection)
    const [vectorResults, tagResults] = await Promise.all([
      vectorSearch(queryVec, _sessionId, _vectorSearchLimits, isACANActive(), _timeRange),
      tagBoostedConcepts(queryText, queryVec, 10)
        .catch((e) => { swallow.warn("graph-context:tagBoost", e); return [] as VectorSearchResult[]; }),
    ]);
    // Merge: dedupe tag results against vector results, then combine
    const vectorIds = new Set(vectorResults.map((r) => r.id));
    const uniqueTagResults = tagResults.filter((r) => !vectorIds.has(r.id));
    const results = [...vectorResults, ...uniqueTagResults];

    // #4: Graph neighbor expansion (1-hop default, 2-hop for complex intents)
    // Get IDs of top vector results, then find their graph neighbors
    const topIds = results
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
      .slice(0, 20)
      .map((r) => r.id);

    const DEEP_INTENTS = new Set(["code-debug", "deep-explore", "multi-step", "reference-prior"]);
    const graphHops = DEEP_INTENTS.has(_currentIntent) ? 2 : 1;

    let neighborIds = new Set<string>();
    let neighborResults: VectorSearchResult[] = [];
    if (topIds.length > 0) {
      try {
        neighborResults = await graphExpand(topIds, queryVec, graphHops);
        neighborIds = new Set(neighborResults.map((n) => n.id));
        // Remove results already in the vector search results
        const existingIds = new Set(results.map((r) => r.id));
        neighborResults = neighborResults.filter((n) => !existingIds.has(n.id));
      } catch (e) {
        swallow.error("graph-context:silent", e);
        // graph expansion failed, continue without it
      }
    }

    // Phase 7a: Causal chain traversal — follow caused_by/supports/contradicts edges
    let causalResults: VectorSearchResult[] = [];
    if (topIds.length > 0 && queryVec) {
      try {
        const causal = await queryCausalContext(topIds, queryVec, 2, 0.4);
        const existingIds = new Set([...results.map((r) => r.id), ...neighborResults.map((r) => r.id)]);
        causalResults = causal.filter((c) => !existingIds.has(c.id));
        // Give causal results neighborBonus so they compete fairly with vector hits
        for (const c of causalResults) {
          neighborIds.add(c.id);
        }
      } catch (e) { swallow("graph-context:causal traversal failed, continue", e); }
    }

    // Combine vector results + neighbor results + causal results, filter noise, then score
    const suppressed = getSuppressedNodeIds();
    // Cosine floor: drop items with near-zero semantic similarity before scoring.
    // Same-session turns are exempt (continuity matters more than relevance).
    const MIN_COSINE = 0.25;
    const allResults = [...results, ...neighborResults, ...causalResults]
      .filter(r => !suppressed.has(r.id))
      .filter(r => r.table === 'turn' && r.sessionId === _sessionId
        ? true              // same-session turn — always keep
        : (r.score ?? 0) >= MIN_COSINE);  // everything else needs minimum relevance
    const ranked = await scoreResults(allResults, neighborIds, queryVec, undefined, queryText);
    const deduped = deduplicateResults(ranked);
    const reranked = await rerankResults(deduped, queryText);
    let contextNodes = takeWithConstraints(reranked, _tokenBudget);
    contextNodes = await ensureRecentTurns(contextNodes, _sessionId);

    if (contextNodes.length === 0) {
      const result = getRecentTurns(messages, CONVERSATION_BUDGET_TOKENS);
      recordStats(messages, result, 0, 0, result.length, "graph");
      return injectRulesSuffix(result);
    }

    // #3: Bump access counts for retrieved items (async, non-blocking)
    const retrievedIds = contextNodes
      .filter((n) => n.table === "concept" || n.table === "memory")
      .map((n) => n.id);
    if (retrievedIds.length > 0) {
      bumpAccessCounts(retrievedIds).catch(e => swallow.warn("graph-context:bumpAccessCounts", e));
    }

    // Stage retrieval for quality evaluation (evaluated when response arrives)
    stageRetrieval(_sessionId, contextNodes, queryVec);

    // Phase 7b: Skill retrieval for actionable intents
    let skillContext = "";
    _lastRetrievedSkillIds = []; // Reset for this turn
    const SKILL_INTENTS = new Set(["code-write", "code-debug", "multi-step", "code-read"]);
    if (SKILL_INTENTS.has(_currentIntent)) {
      try {
        const skills = await findRelevantSkills(queryVec, 5);
        if (skills.length > 0) {
          skillContext = formatSkillContext(skills);
          _lastRetrievedSkillIds = skills.map(s => s.id);
        }
      } catch (e) { swallow("graph-context:skill retrieval failed, continue", e); }
    }

    // Phase 7c: Reflection retrieval — always check (reflections have high importance)
    let reflectionContext = "";
    try {
      const reflections = await retrieveReflections(queryVec, 5);
      if (reflections.length > 0) {
        reflectionContext = formatReflectionContext(reflections);
      }
    } catch (e) { swallow("graph-context:reflection retrieval failed, continue", e); }

    const injectedContext = await formatContextMessage(contextNodes, skillContext + reflectionContext, tier0, tier1);
    const recentTurns = getRecentTurns(messages, CONVERSATION_BUDGET_TOKENS);
    const result = [injectedContext, ...recentTurns];
    recordStats(
      messages, result,
      contextNodes.filter((n) => !n.fromNeighbor).length,
      contextNodes.filter((n) => n.fromNeighbor).length,
      recentTurns.length, "graph",
    );
    return injectRulesSuffix(result);
  } catch (err) {
    console.error("Graph context error, falling back:", err);
    const result = getRecentTurns(messages, CONVERSATION_BUDGET_TOKENS);
    recordStats(messages, result, 0, 0, result.length, "recency-only");
    return injectRulesSuffix(withAnchor(result));
  }
}
