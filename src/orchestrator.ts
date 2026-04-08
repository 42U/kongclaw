/**
 * ISMAR-GENT Orchestration Layer
 *
 * Pre/post processing pipeline between user input and agent.prompt().
 * Classifies intent, adapts agent configuration, records metrics.
 * Target: <25ms for non-trivial prompts, <1ms for simple ones.
 */

import { classifyIntent, estimateComplexity } from "./intent.js";
import type { IntentResult, ComplexityEstimate, ThinkingLevel, IntentCategory } from "./intent.js";
import { queryExec } from "./surreal.js";
import { isSurrealAvailable } from "./surreal.js";
import { getRecentUtilizationAvg } from "./retrieval-quality.js";
import { getRetrievalBudgetTokens } from "./graph-context.js";
import { swallow } from "./errors.js";

// Detects inputs that reference memory/history — prevents skipping retrieval on misclassified simple-questions
const MEMORY_REFERENCE_RE = /\b(we|our|yesterday|earlier|before|last time|prior|remember|recall|previous|discussed|decided|talked about|worked on|you said|you mentioned)\b/i;

// Temporal pattern detection — extracts time ranges for filtered retrieval
const TEMPORAL_PATTERNS: { re: RegExp; resolve: () => { before?: Date; after?: Date } }[] = [
  { re: /\byesterday\b/i, resolve: () => ({ after: new Date(Date.now() - 48 * 3600_000), before: new Date(Date.now() - 24 * 3600_000) }) },
  { re: /\blast week\b/i, resolve: () => ({ after: new Date(Date.now() - 14 * 86400_000), before: new Date(Date.now() - 7 * 86400_000) }) },
  { re: /\b(\d+)\s+days?\s+ago\b/i, resolve: (m?: RegExpMatchArray) => {
    const n = parseInt(m?.[1] ?? "1", 10);
    return { after: new Date(Date.now() - (n + 1) * 86400_000), before: new Date(Date.now() - (n - 1) * 86400_000) };
  }},
  { re: /\btoday\b/i, resolve: () => ({ after: new Date(new Date().setHours(0, 0, 0, 0)) }) },
  { re: /\bthis week\b/i, resolve: () => ({ after: new Date(Date.now() - 7 * 86400_000) }) },
  { re: /\blast month\b/i, resolve: () => ({ after: new Date(Date.now() - 60 * 86400_000), before: new Date(Date.now() - 30 * 86400_000) }) },
];

function extractTimeRange(input: string): { before?: Date; after?: Date } | undefined {
  for (const p of TEMPORAL_PATTERNS) {
    const match = input.match(p.re);
    if (match) return (p.resolve as (m?: RegExpMatchArray) => { before?: Date; after?: Date })(match);
  }
  return undefined;
}

let _contextWindow = 200000; // Updated via setContextWindow from agent setup
export function setContextWindow(cw: number): void { _contextWindow = cw; }

// --- Types ---

export interface AdaptiveConfig {
  thinkingLevel: ThinkingLevel;
  toolLimit: number;
  tokenBudget: number;
  retrievalShare?: number; // % of retrieval budget (0-1) — tokenBudget derived from this at preflight
  skipRetrieval?: boolean; // true = skip graph retrieval entirely (trivial first-turn greetings)
  intent?: string; // classified intent category, passed to graph-context for skill retrieval
  timeRange?: { before?: Date; after?: Date }; // temporal filtering for "what happened last week?" queries
  vectorSearchLimits: {
    turn: number;
    identity: number;
    concept: number;
    memory: number;
    artifact: number;
  };
}

export interface PreflightResult {
  intent: IntentResult;
  complexity: ComplexityEstimate;
  config: AdaptiveConfig;
  preflightMs: number;
  fastPath: boolean;
}

interface TurnEvent {
  type: string;
  toolName?: string;
  isError?: boolean;
  tokens?: { input: number; output: number };
}

// --- Default config — tokenBudget is a fallback; retrievalShare derives it from global budget ---

const DEFAULT_CONFIG: AdaptiveConfig = {
  thinkingLevel: "medium",
  toolLimit: 15,
  tokenBudget: 6000,
  retrievalShare: 0.15,
  vectorSearchLimits: { turn: 25, identity: 10, concept: 20, memory: 20, artifact: 10 },
};

// --- Intent → Config mapping ---
// retrievalShare: % of global retrieval budget (42K at 200K context) allocated to this intent.
// tokenBudget: fallback value used before context window is known. Overridden by retrievalShare.
// Tight tool limits force efficient call planning (LOOKUP=1, EDIT=2, REFACTOR=6).

const INTENT_CONFIG: Record<IntentCategory, AdaptiveConfig> = {
  "simple-question": {
    thinkingLevel: "low",
    toolLimit: 3,
    tokenBudget: 4000,
    retrievalShare: 0.10,
    vectorSearchLimits: { turn: 15, identity: 5, concept: 12, memory: 12, artifact: 3 },
  },
  "code-read": {
    thinkingLevel: "medium",
    toolLimit: 5,
    tokenBudget: 6000,
    retrievalShare: 0.15,
    vectorSearchLimits: { turn: 25, identity: 8, concept: 20, memory: 20, artifact: 10 },
  },
  "code-write": {
    thinkingLevel: "high",
    toolLimit: 8,
    tokenBudget: 8000,
    retrievalShare: 0.20,
    vectorSearchLimits: { turn: 30, identity: 10, concept: 20, memory: 20, artifact: 15 },
  },
  "code-debug": {
    thinkingLevel: "high",
    toolLimit: 10,
    tokenBudget: 8000,
    retrievalShare: 0.20,
    vectorSearchLimits: { turn: 30, identity: 8, concept: 20, memory: 25, artifact: 15 },
  },
  "deep-explore": {
    thinkingLevel: "medium",
    toolLimit: 15,
    tokenBudget: 6000,
    retrievalShare: 0.15,
    vectorSearchLimits: { turn: 25, identity: 8, concept: 15, memory: 15, artifact: 8 },
  },
  "reference-prior": {
    thinkingLevel: "medium",
    toolLimit: 5,
    tokenBudget: 10000,
    retrievalShare: 0.25,
    vectorSearchLimits: { turn: 40, identity: 10, concept: 25, memory: 30, artifact: 10 },
  },
  "meta-session": {
    thinkingLevel: "low",
    toolLimit: 2,
    tokenBudget: 3000,
    retrievalShare: 0.07,
    skipRetrieval: false,
    vectorSearchLimits: { turn: 8, identity: 5, concept: 5, memory: 8, artifact: 0 },
  },
  "multi-step": {
    thinkingLevel: "high",
    toolLimit: 12,
    tokenBudget: 8000,
    retrievalShare: 0.20,
    vectorSearchLimits: { turn: 30, identity: 10, concept: 20, memory: 20, artifact: 15 },
  },
  "continuation": {
    thinkingLevel: "low",
    toolLimit: 8,
    tokenBudget: 4000,
    skipRetrieval: true,
    vectorSearchLimits: { turn: 0, identity: 0, concept: 0, memory: 0, artifact: 0 },
  },
  "unknown": { ...DEFAULT_CONFIG },
};

// --- State ---

let _lastConfig: AdaptiveConfig = { ...DEFAULT_CONFIG };
let _turnIndex = 0;

// --- Steering tracking (Phase 4: logged only, not actioned) ---

interface SteeringCandidate {
  type: "runaway" | "budget_warning" | "scope_drift";
  toolCall: number;
  detail: string;
}

let _currentTurnTools: { name: string; args?: string }[] = [];
let _steeringCandidates: SteeringCandidate[] = [];

// --- Public API ---

export async function preflight(input: string, sessionId: string): Promise<PreflightResult> {
  const start = performance.now();
  _turnIndex++;
  _currentTurnTools = [];
  _steeringCandidates = [];

  // Fast path: only truly trivial inputs on the FIRST turn skip classification.
  // After turn 1, short inputs like "yes plz", "do it", "docs" are continuations, not greetings.
  const isTrivial = _turnIndex <= 1 && input.length < 20 && !input.includes("?");
  if (isTrivial) {
    const elapsed = performance.now() - start;
    const result: PreflightResult = {
      intent: { category: "unknown", confidence: 0, scores: [] },
      complexity: { level: "simple", estimatedToolCalls: 0, suggestedThinking: "low" },
      config: { thinkingLevel: "low", toolLimit: 15, tokenBudget: 300, skipRetrieval: true,
          vectorSearchLimits: { turn: 0, identity: 0, concept: 0, memory: 0, artifact: 0 } },
      preflightMs: elapsed,
      fastPath: true,
    };
    _lastConfig = result.config;
    return result;
  }

  // Non-first-turn short inputs: treat as continuation — skip retrieval, context is in the thread
  if (_turnIndex > 1 && input.length < 20 && !input.includes("?")) {
    const elapsed = performance.now() - start;
    const inheritedLimit = Math.max(_lastConfig.toolLimit, 25); // floor of 25
    const result: PreflightResult = {
      intent: { category: "continuation", confidence: 0.9, scores: [] },
      complexity: { level: "moderate", estimatedToolCalls: 15, suggestedThinking: "medium" },
      config: { ..._lastConfig, toolLimit: inheritedLimit, skipRetrieval: true,
        vectorSearchLimits: { turn: 0, identity: 0, concept: 0, memory: 0, artifact: 0 } },
      preflightMs: elapsed,
      fastPath: true,
    };
    _lastConfig = result.config;
    return result;
  }

  // Full classification path
  const intent = await classifyIntent(input);
  const complexity = estimateComplexity(input, intent);

  // Build config from intent mapping
  // Low-confidence classifications get conservative retrieval to avoid wasting tokens
  const LOW_CONFIDENCE_CONFIG: AdaptiveConfig = {
    thinkingLevel: "low",
    toolLimit: 15,
    tokenBudget: 3000,
    retrievalShare: 0.08,
    vectorSearchLimits: { turn: 12, identity: 5, concept: 8, memory: 12, artifact: 3 },
  };

  let config: AdaptiveConfig;
  if (intent.category === "continuation") {
    // Inherit previous turn's config, but with a reasonable floor
    config = { ..._lastConfig };
    config.toolLimit = Math.max(config.toolLimit, 15);
  } else if (intent.confidence < 0.40) {
    // Low confidence — don't trust classification, use conservative defaults
    config = { ...LOW_CONFIDENCE_CONFIG };
  } else {
    config = { ...(INTENT_CONFIG[intent.category] ?? DEFAULT_CONFIG) };
  }
  config.intent = intent.category;

  // Gate retrieval: skip vector search for simple-question / meta-session
  // unless the input references memory ("we decided", "yesterday", "remember", etc.)
  // Tier 0 core memory is always loaded regardless — this only skips vector search.
  if (
    (intent.category === "simple-question" || intent.category === "meta-session") &&
    intent.confidence >= 0.70 &&
    !MEMORY_REFERENCE_RE.test(input)
  ) {
    config.skipRetrieval = true;
    config.vectorSearchLimits = { turn: 0, identity: 0, concept: 0, memory: 0, artifact: 0 };
  }

  // Derive tokenBudget from global retrieval budget when retrievalShare is set
  if (config.retrievalShare != null && config.retrievalShare > 0) {
    const retrievalBudget = getRetrievalBudgetTokens();
    config.tokenBudget = Math.round(retrievalBudget * config.retrievalShare);
  }

  // Override thinking if complexity demands it
  if (complexity.suggestedThinking === "high" && config.thinkingLevel !== "high") {
    config.thinkingLevel = "high";
  }

  // Override tool limit if complexity estimates more — but cap at 1.5x intent limit (max 20)
  if (complexity.estimatedToolCalls > config.toolLimit) {
    const hardCeiling = 20;
    config.toolLimit = Math.min(complexity.estimatedToolCalls, Math.ceil(config.toolLimit * 1.5), hardCeiling);
  }

  // Adaptive token budget: scale based on rolling retrieval quality
  if (!config.skipRetrieval) {
    const recentUtil = await getRecentUtilizationAvg(sessionId, 10).catch(() => null);
    if (recentUtil !== null) {
      // 0% util → 0.5x budget, 50% util → 1.0x, 100% util → 1.3x
      const scale = Math.max(0.5, Math.min(1.3, 0.5 + recentUtil * 0.8));
      config.tokenBudget = Math.round(config.tokenBudget * scale);
    }
  }

  // Temporal reasoning: extract time ranges from natural language
  const timeRange = extractTimeRange(input);
  if (timeRange) config.timeRange = timeRange;

  _lastConfig = config;
  const elapsed = performance.now() - start;

  return {
    intent,
    complexity,
    config,
    preflightMs: elapsed,
    fastPath: false,
  };
}

/**
 * Record a tool call for steering analysis (Phase 4).
 * Called from agent event handler on tool_execution_start.
 */
export function recordToolCall(name: string, args?: string): void {
  _currentTurnTools.push({ name, args });

  // Runaway detection: 5+ consecutive same tool
  if (_currentTurnTools.length >= 5) {
    const last5 = _currentTurnTools.slice(-5);
    if (last5.every((t) => t.name === last5[0].name)) {
      _steeringCandidates.push({
        type: "runaway",
        toolCall: _currentTurnTools.length,
        detail: `${last5[0].name} called 5+ times consecutively`,
      });
    }
  }

  // Budget warning: within 15% of limit
  const budgetWarnAt = Math.floor(_lastConfig.toolLimit * 0.85);
  if (_lastConfig.toolLimit !== Infinity && _currentTurnTools.length >= budgetWarnAt) {
    _steeringCandidates.push({
      type: "budget_warning",
      toolCall: _currentTurnTools.length,
      detail: `${_currentTurnTools.length}/${_lastConfig.toolLimit} tool calls used`,
    });
  }
}

/**
 * Async postflight: record metrics to SurrealDB.
 * Non-blocking — failures are silently ignored.
 */
export async function postflight(
  input: string,
  result: PreflightResult,
  actualToolCalls: number,
  actualTokensIn: number,
  actualTokensOut: number,
  turnDurationMs: number,
  sessionId: string,
): Promise<void> {
  try {
    if (!(await isSurrealAvailable())) return;
    await queryExec(`CREATE orchestrator_metrics CONTENT $data`, {
      data: {
        session_id: sessionId,
        turn_index: _turnIndex,
        input_length: input.length,
        intent: result.intent.category,
        intent_confidence: result.intent.confidence,
        complexity: result.complexity.level,
        thinking_level: result.config.thinkingLevel,
        tool_limit: result.config.toolLimit === Infinity ? -1 : result.config.toolLimit,
        token_budget: result.config.tokenBudget,
        actual_tool_calls: actualToolCalls,
        actual_tokens_in: actualTokensIn,
        actual_tokens_out: actualTokensOut,
        preflight_ms: result.preflightMs,
        turn_duration_ms: turnDurationMs,
        steering_candidates: _steeringCandidates.length,
        steering_details: _steeringCandidates.length > 0
          ? _steeringCandidates.map((c) => `${c.type}: ${c.detail}`).join("; ")
          : undefined,
        fast_path: result.fastPath,
      },
    });
  } catch (e) {
    swallow("orchestrator:silent", e);
    // non-critical telemetry
  }
}

/**
 * Get the last adaptive config (for CLI display).
 */
export function getLastPreflightConfig(): AdaptiveConfig {
  return _lastConfig;
}

export function getSteeringCandidates(): SteeringCandidate[] {
  return _steeringCandidates;
}
