# Zeraclaw. System Architecture Documentation
> Updated: 2026-04-08 | Post-benchmark: cross-encoder reranking, WMR v3, concept supersedes

> **Note:** For comprehensive documentation including tools, TUI, commands, and development setup, see **[GUIDE.md](GUIDE.md)**.

---

## Overview

Zeraclaw is a graph-backed coding agent with persistent memory across sessions. Every conversation turn, concept, decision, and file artifact is stored in SurrealDB. Memory is retrieved via BGE-M3 vector embeddings + graph traversal and injected into the context window before each LLM call.

**Tech stack:** TypeScript (Node.js), SurrealDB 3.0, BGE-M3 via node-llama-cpp (1024-dim HNSW cosine), Claude Opus 4.6 as inference layer (everywhere: chat, extraction, cognitive checks, summarization).

---

## Source Map

| File | Lines | Role |
|---|---|---|
| `src/graph-context.ts` | 1299 | Context injection pipeline, global budget system, retrieval, scoring, formatting |
| `src/surreal.ts` | 1323 | Database layer, vector search, graph expand, query helpers |
| `src/agent.ts` | 1154 | Core agent loop, turn handling, tool execution, smart truncation, extraction |
| `src/tui.ts` | 809 | Terminal UI (pi-tui), pinned editor, scrollable chat, styled output |
| `src/render.ts` | 602 | Terminal rendering, markdown, syntax highlighting, streaming |
| `src/acan.ts` | 555 | ACAN: learned cross-attention scoring model (dormant until 1000+ samples) |
| `src/cli.ts` | 546 | Readline REPL interface (fallback to TUI) |
| `src/memory-daemon.ts` | 495 | Worker thread for incremental memory extraction per turn |
| `src/subagent.ts` | 423 | Subagent spawning (full + incognito modes) |
| `src/wakeup.ts` | 406 | Session startup. birth cognition, soul graduation, wake-up synthesis |
| `src/retrieval-quality.ts` | 370 | Post-hoc retrieval telemetry. utilization, waste, tool success |
| `src/soul.ts` | 350 | Soul/graduation system. tracks learning milestones, 6th pillar unlock |
| `src/skills.ts` | 346 | Skill library. reusable procedure extraction and retrieval |
| `src/orchestrator.ts` | 342 | ISMAR-GENT, adaptive config per turn, % of retrieval budget |
| `src/cognitive-check.ts` | 316 | Periodic Opus reasoning over retrieved context |
| `src/tui-components.ts` | 298 | Reusable TUI components |
| `src/reflection.ts` | 284 | Metacognitive reflection generation |
| `src/prefetch.ts` | 239 | Predictive prefetch with LRU cache |
| `src/causal.ts` | 238 | Causal chain extraction with embedded descriptions |
| `src/identity.ts` | 229 | Core identity seeding and WAKEUP.md lifecycle |
| `src/eval.ts` | 213 | Evaluation framework |
| `src/intent.ts` | 200 | Zero-shot intent classification via BGE-M3 |
| `src/daemon-manager.ts` | 141 | Worker thread lifecycle management |
| `src/subagent-worker.ts` | 127 | Child process worker for subagents |

**Total:** ~12,300 lines across 40 source files.

---

## Core Architecture: Global Context Budget

Everything derives from the model's context window. No hardcoded magic numbers.

```
CONTEXT_WINDOW = 200,000 tokens (or whatever the model reports)

┌─────────────────────────────────────────────────────┐
│ System prompt + tool defs        ~15%  (not ours)   │
│ Model response headroom          ~15%  (not ours)   │
│─────────────────────────────────────────────────────│
│ OUR BUDGET (BUDGET_FRACTION)     70%   = 140K       │
│                                                     │
│  ┌── Conversation history ─── 50% of budget = 70K  │
│  │   getRecentTurns budget (token-aware grouping)   │
│  │   Smart truncation: head+tail with error detect  │
│  │   RECENT_MSG_COUNT = max(12, cw/12000) = 16      │
│  │   TOOL_RESULT_MAX = cw * 0.05 = 10K per result   │
│  │   olderCap = TOOL_RESULT_MAX * 0.35 = 3.5K min   │
│  │   MIN_KEEP_CHARS = 2000 (floor, never below)     │
│  │                                                  │
│  ├── Graph retrieval ──────── 30% of budget = 42K  │
│  │   Per-intent tokenBudget = retrievalShare × 42K  │
│  │   MAX_CONTEXT_ITEMS = max(20, budget/300) = 140  │
│  │   MIN_RELEVANCE_SCORE = 0.35                     │
│  │   Skills + reflections + causal chains            │
│  │                                                  │
│  ├── Core memory ──────────── 15% of budget = 21K  │
│  │   Tier 0: 55% of core = ~19.6K chars (pillars)  │
│  │   Tier 1: 45% of core = ~16K chars (session)    │
│  │                                                  │
│  └── Overhead ─────────────── 5% of budget = 7K   │
│      Rules suffix, intent anchors, formatting       │
└─────────────────────────────────────────────────────┘
```

**Budget recalculation** happens in `graph-context.ts:recalcBudgets()`, called from `setRetrievalConfig()` whenever the context window changes.

**Pressure scaling:** As tokens accumulate during a session, context budgets shrink proportionally:
- **Onset:** 50% of context window used (100K tokens at 200K)
- **Min scale:** 0.5× (never cuts retrieval below half)
- Linear interpolation between onset and 100%

---

## Context Injection Pipeline (`graph-context.ts`)

The heart of memory integration. Two-layer architecture: `graphTransformContext` runs first, then `convertToLlm` applies conversation-level truncation.

### Layer 1: `graphTransformContext` → `_graphTransformContextInner`

1. **Load core memory**. Tier 0 (always-loaded pillars) + Tier 1 (session-pinned), apply budget caps
2. **Graceful degradation**. if embeddings or SurrealDB down, pass through with core memory only
3. **Embed user input**. BGE-M3, contextual: averages with last 3 turn embeddings (query weight 2×)
4. **Cache check**. prefetch LRU, cosine > 0.85 → skip DB round-trip
5. **Vector search**. across 5 tables (`turn`, `identity_chunk`, `concept`, `memory`, `artifact`) using per-intent limits
6. **Graph expand**. 1-hop neighbors of top 20 results
7. **Causal traversal**. 2-hop traversal across `caused_by`, `supports`, `contradicts`, `describes` edges
8. **Score**. WMR (6 signals, fixed weights) or ACAN (learned, if active)
9. **Deduplicate**. cosine > 0.88 → drop duplicate
10. **Budget gate**. `takeWithConstraints()`: MAX_CONTEXT_ITEMS cap, MIN_RELEVANCE_SCORE floor, token budget
11. **Ensure recent turns**. guarantee last-session turns appear regardless of similarity
12. **Skill + reflection retrieval**. for actionable intents (code-write, code-debug, multi-step)
13. **Format**. grouped sections (Core Directives, Session Context, Behavioral Directives, Resurfacing Memories, Recalled Memories, Concepts, Causal Chains, Skills, Past Turns)
14. **getRecentTurns**. token-aware conversation grouping within `CONVERSATION_BUDGET_TOKENS` (70K at 200K)
15. **Inject rules suffix**. tool budget reminders, efficiency examples

### Layer 2: `convertToLlm` (agent.ts)

Runs on the output of Layer 1:

1. Filter to user/assistant/toolResult messages
2. Split into "recent" (last `RECENT_MSG_COUNT` = 16) and "older"
3. **Recent messages:** kept fully intact
4. **Older assistant messages:** strip thinking blocks
5. **Older tool results:** smart truncation with `hasImportantTail()` detection:
   - Detects errors, exceptions, JSON, summaries at the end of output
   - When tail is important: 70% head + 30% tail (up to 4K), with middle omission marker
   - Cuts at newline boundaries for clean output
   - Floor: `MIN_KEEP_CHARS = 2000`. never crushes below 2K
   - Cap: `olderCap = max(2000, TOOL_RESULT_MAX * 0.35)` = 3.5K at 200K

---

## Scoring System

### WMR v3 (Working Memory Ranker). default

8-signal weighted scoring:

| Signal | Weight | Source |
|---|---|---|
| Cosine similarity | 0.22 | Vector distance from query |
| Recency | 0.23 | Piecewise exponential decay (fast <4h, slow >4h) |
| Importance | 0.05 | 0-10 scale stored on record |
| Access count | 0.05 | log(1 + count), capped at 1.0 |
| Neighbor bonus | 0.10 | Graph-expanded items get 1.0 |
| Proven utility | 0.15 | Historical utilization from retrieval outcomes |
| Reflection boost | 0.10 | Memories from sessions that produced reflections |
| Keyword overlap | 0.12 | Query keyword matches in document text |

Minus utility penalty (-0.15 for <5% util, -0.06 for <15%) for proven-bad memories.

### Cross-Encoder Reranking. post-scoring stage

After WMR scoring and deduplication, top-30 candidates are rescored by bge-reranker-v2-m3 (a purpose-built cross-encoder running locally via `LlamaRankingContext`). Scores blend 60% WMR + 40% cross-encoder. Benchmarked at 98.2% R@5 on LongMemEval (beats MemPalace's 96.6%).

### ACAN (Attentive Cross-Attention Network). learned, dormant

- Architecture: query projection (1024→64), key projection (1024→64), attention logit, final linear layer over [attnLogit, recency, importance, access, neighbor, utility, reflection, keywordOverlap] (8-dim)
- ~131K params, trains via TypeScript SGD with validation split, early stopping, LR decay
- Auto-activates after 5,000+ `retrieval_outcome` records. Retrains when data grows 50%+ or weights >7 days old.
- Weights: `~/.kongclaw/acan_weights.json`

### Concept Supersedes. stale knowledge evolution

When corrections are extracted by the daemon, matching stale concepts (cosine > 0.70) get `supersedes` edges and stability decayed by 60% (floor 0.15). Superseded concepts are filtered from vectorSearch entirely.

---

## Orchestrator (`orchestrator.ts`)

Per-turn adaptive configuration via intent classification.

### Intent → Config Mapping

Token budgets are percentages of the global retrieval budget (42K tokens at 200K context). Tool limits unchanged.

| Intent | Tool Limit | Retrieval Share | Token Budget (200K) | Thinking |
|---|---|---|---|---|
| simple-question | 3 | 10% | 4,200 | low |
| code-read | 5 | 15% | 6,300 | medium |
| code-write | 8 | 20% | 8,400 | high |
| code-debug | 10 | 20% | 8,400 | high |
| deep-explore | 15 | 15% | 6,300 | medium |
| reference-prior | 5 | 25% | 10,500 | medium |
| meta-session | 2 | 7% | 2,940 | low |
| multi-step | 12 | 20% | 8,400 | high |
| continuation | 8 |. | inherited | low |
| unknown | 15 | 15% | 6,300 | medium |

**Low-confidence fallback** (confidence < 0.40): 8% = 3,360 tokens.

**Adaptive scaling:** Rolling retrieval utilization modulates budget (0.5× at 0% util → 1.3× at 100% util).

**Session continuity adjustment:** Cognitive check's continuity signal reshapes search limits (new_topic boosts concepts, continuation boosts turns, repeat boosts memory).

---

## Identity & Core Memory

### 5 Pillars (Tier 0 Core Memory)

Seeded from `src/seed-core-memory.ts`. Always loaded, never evicted:

| Pillar | Priority | Text |
|---|---|---|
| Knowledge | 95 | Store every success and every failure, learn from both. |
| Operations | 90 | Think before you act. Do you already have a saved skill or knowledge gem for that? Then use it. If not, build it. |
| Intelligence | 85 | Be thoughtful, take your time to make sure you understand deeply before responding. When you respond, add value. |
| Graph | 80 | Save everything to the graph. It will help you develop better habits and routes for you to follow. |
| Network | 75 | Networks aren't just computer packets getting from one place to another. Network the data you learn. |

### 6th Pillar: Persona (unlocked at graduation)

Created in `soul.ts:attemptGraduation()` when the agent passes graduation thresholds:
- Text: "Persona: you belong in this world. Remember to be unique."
- Priority 70, Tier 0

### Birth Cognition

On first boot (sessions=0, memoryCount=0), `wakeup.ts:synthesizeBirthCognition()` generates a greeting via Opus acknowledging the 5 pillars and marking a fresh start.

---

## Causal Chain System (`causal.ts`)

### Creation

Extracted by Opus at session cleanup when the session has meaningful delta (4+ new turns beyond daemon coverage):

1. Opus sees full session transcript (up to 600K chars, 500 turns, no per-turn cap)
2. Extracts up to 5 chains: `{triggerText, outcomeText, chainType, success, confidence, description}`
3. Trigger and outcome texts stored as embedded `memory` nodes (up to 500 chars each)
4. **Description** (up to 500 chars). embedded as a separate `memory` node with category `causal_description_{type}`, linked via `describes` edges to both trigger and outcome
5. Graph edges created: `caused_by`, `supports` (if success) or `contradicts` (if failure)

### Retrieval

`queryCausalContext()`. graph traversal, not vector search:

1. Start from seed IDs (top vector search results)
2. Traverse 4 edge types: `caused_by`, `supports`, `contradicts`, `describes`
3. Both forward and reverse direction, 2 hops deep
4. Server-side cosine scoring against query vector
5. Filter by chain confidence (default 0.4)
6. Results compete in scoring alongside vector search hits (get neighborBonus)

### Why Descriptions Are Embedded Separately

The `causal_chain` table is metadata (trigger_memory, outcome_memory, confidence, type). The trigger/outcome texts are already embedded as memory nodes. But the description ("Fixed import path after refactor broke module resolution") captures the *what happened* narrative. making chains discoverable by semantic search on the summary, not just by graph traversal from already-retrieved memories.

---

## Full-Picture Extraction

Session cleanup calls Opus with the complete session for high-quality knowledge extraction.

### What Opus Sees

| Input | Limit | Purpose |
|---|---|---|
| Turns fetched | 500 | Entire session |
| Per-turn text | No cap | Full turn content including tool results |
| Transcript | 600K chars | ~175K tokens. fills Opus context |
| Thinking blocks | All, 50K chars | Inner reasoning drives better chains |
| Previous handoff | Full text | Narrative continuity across sessions |
| Retrieved memories | Full text | Judge if resolved |

### What Opus Extracts

| Field | Max Length | Description |
|---|---|---|
| Handoff | ~500 words | First-person note to future self. decisions, progress, open items, how it felt |
| Causal chains | 5 chains, 500 chars each | trigger→outcome with type, success, confidence, description |
| Skill | 1 skill, 8 steps | Reusable procedure from successful multi-step work |
| Monologues | 5 entries | Inner thought traces (doubt, tradeoff, insight, realization) |
| Reflection | 2-4 sentences | Root cause analysis when quality metrics are poor |
| Resurface | 0-2 items | Unfinished intentions, abandoned threads, goals with temporal weight |
| Resolved | memory IDs | Memories that were fully addressed this session |

**Timeout:** 120 seconds (increased from 42s to handle full transcripts).

---

## DB Storage

### What Gets Stored Per Turn

| Turn Type | Text Stored | Embedding |
|---|---|---|
| User | Full text, no cap | BGE-M3 if semantically meaningful |
| Assistant | Full text, no cap | BGE-M3 up to 22K chars (80% of model capacity) |
| Tool result | Full text, no cap | BGE-M3 up to 14K chars (50% of model capacity) |

All turn text is stored complete in SurrealDB. The caps in `convertToLlm` and `getRecentTurns` only affect what goes into the LLM's context window. the DB has the full record for extraction, retrieval, and future sessions.

---

## Model Usage

**Opus 4.6 everywhere:**
- Chat (all turns, no escalation)
- Session extraction (handoff, chains, skills, reflection)
- Cognitive checks (retrieval grading, directives)
- Compaction summaries
- Exit line generation
- Wake-up/startup/birth cognition
- Graph context summarization

No model switching, no cost optimization tiers.

---

## Cognitive Checks (`cognitive-check.ts`)

Periodic Opus reasoning over retrieved context.

- **Frequency:** Turn 2, then every 3 turns (2, 5, 8, 11...)
- **Model:** Claude Opus 4.6
- **Evaluates:** retrieval grading (relevant/irrelevant + score + learned/resolved), session continuity, user preference detection
- **Directive types:** `repeat`, `continuation`, `contradiction`, `noise`, `insight`
- **Correction tracking:** If a correction memory was followed unprompted → decay importance (floor 3). If ignored → reinforce (cap 9).
- **Noise suppression:** Irrelevant items with score < 0.3 are suppressed for the rest of the session
- **Mid-session resolution:** Memories marked resolved get `status='resolved'` immediately

---

## Database Schema (SurrealDB. NS: zera, DB: memory)

### Key Tables

| Table | Purpose | Vector Index |
|---|---|---|
| `turn` | Every conversation message (embedded) | Yes (1024d HNSW) |
| `memory` | Extracted knowledge, compaction summaries, causal texts, descriptions | Yes (1024d HNSW) |
| `concept` | Semantic knowledge nodes | Yes (1024d HNSW) |
| `artifact` | File artifacts with metadata | Yes (1024d HNSW) |
| `skill` | Reusable procedures from multi-step tasks | Yes (1024d HNSW) |
| `reflection` | Metacognitive lessons from problem sessions | Yes (1024d HNSW) |
| `monologue` | Inner thought traces | Yes (1024d HNSW) |
| `core_memory` | Always-loaded Tier 0/1 directives (no vector search, full load) | No |
| `causal_chain` | Cause→effect metadata (links trigger/outcome/description memories) | No |
| `session` | Session metadata, token counts | No |
| `retrieval_outcome` | Quality telemetry per retrieved item | No |
| `orchestrator_metrics` | Per-turn orchestration metrics | No |
| `soul` | Graduation progress and self-authored identity | No |

### Graph Edges

| Category | Edges |
|---|---|
| Turn-level | `responds_to`, `tool_result_of`, `part_of`, `mentions` |
| 5-Pillar | `performed`, `owns`, `task_part_of`, `session_task`, `produced`, `derived_from`, `relevant_to`, `used_in` |
| Knowledge | `narrower`, `broader`, `related_to` (concept↔concept) |
| Causal | `caused_by`, `supports`, `contradicts`, `describes` (memory↔memory) |
| Cross-pillar | `about_concept` (memory→concept), `artifact_mentions` (artifact→concept) |
| Procedural | `skill_from_task`, `skill_uses_concept`, `spawned`, `reflects_on`, `summarizes` |

---

## Data Flow Summary

```
User input
  → Intent classification (BGE-M3, ~25ms) → adaptive config (tool limit, retrieval share)
  → Predictive prefetch (2-4 predicted queries, background vector search)
  → Context injection pipeline:
      Core memory (Tier 0 + Tier 1, budget-capped)
    → Contextual query vector (current + last 3 turns blended)
    → Prefetch cache check (cosine > 0.85?)
    → Vector search (SurrealDB HNSW, per-intent limits)
    → Graph expand (1-hop neighbors of top 20)
    → Causal traversal (2 hops across 4 edge types)
    → Score (WMR or ACAN) → Deduplicate → Budget gate
    → Skill + reflection retrieval
    → Format as grouped sections
    → getRecentTurns (70K token budget for conversation history)
    → Smart truncation of older tool results (head+tail, error-aware)
    → Rules suffix injection
  → LLM call (Opus 4.6, streaming)
  → Tool execution (agent loop, orchestrator tracking)
  → Response complete:
      Evaluate retrieval quality (6 signals)
    → Write retrieval_outcome records
    → Update utility cache
    → Cognitive check (every 3 turns. grading, directives, preferences)
    → Memory daemon batch (if token threshold crossed)
  → Session end:
      Full-picture extraction (Opus, 600K char transcript)
    → Handoff note, causal chains, skills, reflection, monologues
    → Causal descriptions embedded and linked
    → Deactivate Tier 1 memories
    → Generate exit line
```

---

## Configuration

| Path | Purpose |
|---|---|
| `.env` | SurrealDB credentials, namespace, database |
| `~/.surreal_env` | Alternative SurrealDB config (shell-style exports) |
| `~/.kongclaw/acan_weights.json` | Trained ACAN model weights |
| `~/.node-llama-cpp/models/bge-m3-q4_k_m.gguf` | BGE-M3 embedding model |

**Key env vars:** `SURREAL_URL` (default ws://localhost:8042), `SURREAL_USER`, `SURREAL_PASS`, `SURREAL_NS` (zera), `SURREAL_DB` (memory), `ANTHROPIC_API_KEY`, `ZERACLAW_TUI` (1 for TUI mode).

---

*Last updated: 2026-03-27 | 40 source files, 12,300 lines*
