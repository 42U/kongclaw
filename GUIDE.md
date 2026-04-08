# Zeraclaw. Complete Documentation Guide

> Graph-backed agentic CLI with persistent memory across sessions. Updated: 2026-04-08.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Core Systems](#core-systems)
5. [Database Schema](#database-schema)
6. [Tools](#tools)
7. [Terminal Interface](#terminal-interface)
8. [Configuration Reference](#configuration-reference)
9. [Slash Commands](#slash-commands)
10. [Development](#development)
11. [Per-Turn Execution Flow](#per-turn-execution-flow)
12. [Known Gaps & Roadmap](#known-gaps--roadmap)

---

## Introduction

Zeraclaw is a coding agent that remembers. It stores every conversation turn, concept, decision, and file artifact in a SurrealDB knowledge graph. Memory is retrieved via BGE-M3 vector embeddings (1024-dim, local, no API calls) combined with graph traversal, then injected into the context window before each LLM call.

**Key differentiators:**

- **Persistent graph memory**. SurrealDB stores turns, concepts, memories, artifacts, skills, reflections with vector embeddings and 16+ graph edge types (caused_by, supports, contradicts, describes, related_to, etc.)
- **Adaptive orchestration**. Zero-shot intent classification adapts thinking level, tool limits, and retrieval depth per turn
- **Multi-stage retrieval**. 8-signal WMR v3 scoring + cross-encoder reranking (98.2% R@5 on LongMemEval)
- **Learned scoring**. ACAN (Attentive Cross-Attention Network) auto-trains on retrieval telemetry to replace fixed-weight scoring
- **Constitutive memory**. Agent wakes up knowing who it is via handoff notes, monologue traces, and wake-up synthesis
- **Causal chains**. Tracks trigger→outcome patterns across sessions for debugging and learning
- **Skill library**. Extracts reusable procedures from successful multi-step tasks
- **Metacognitive reflection**. Reviews own performance at session end, stores lessons as high-importance memories

**Tech stack:** TypeScript (Node.js 20+), SurrealDB 3.0, BGE-M3 via node-llama-cpp, Claude Opus 4.6 (Anthropic API) everywhere, pi-tui for terminal UI.

---

## Quick Start

### Prerequisites

- **Node.js 20+**
- **SurrealDB 3.0** running locally (`surreal start --bind 0.0.0.0:8042 file:kongclaw.db`)
- **BGE-M3 GGUF** model for local embeddings (default path: `~/.node-llama-cpp/models/bge-m3-q4_k_m.gguf`)
- **Anthropic API key** (Claude access)

### Install & Run

```bash
# Install dependencies
npm install

# Configure (pick one):
# Option A: Set env vars
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: Create ~/.surreal_env
echo 'export SURREAL_URL="ws://localhost:8042/rpc"' >> ~/.surreal_env
echo 'export SURREAL_USER="root"' >> ~/.surreal_env
echo 'export SURREAL_PASS="root"' >> ~/.surreal_env

# Build & run
npm run build
npm start
```

### First-Run Identity

On first launch, if a `WAKEUP.md` file exists in the working directory, Zeraclaw reads it to establish its identity. The file defines personality, tone, role, and behavioral guidelines. After processing, the file is deleted and identity chunks are persisted to the graph.

To re-establish identity later: `/wakeup`

### TUI Mode

For a Claude Code-style interface with pinned input, scrollable output, and styled markdown:

```bash
ZERACLAW_TUI=1 npm start
# or
node dist/index.js --tui
```

---

## Architecture Overview

### High-Level Flow

```
User Input
    │
    ▼
┌──────────────────────────────────────────────────┐
│  CLI (readline) or TUI (pi-tui)                  │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Preflight (orchestrator.ts)                     │
│  Intent classify → Adaptive config               │
│  Tool budget, thinking level, search depths       │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Context Injection (graph-context.ts)            │
│  Embed → Cache check → Vector search (6 tables)  │
│  + Tag boost → Graph expand → WMR v3 score       │
│  → Dedup → Cross-encoder rerank → Budget → Inject│
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  LLM Call (agent.ts → pi-agent-core)             │
│  Stream response + tool execution                 │
└──────────┬───────────────────────────────────────┘
           │
           ├──► Postflight (metrics, quality eval)
           │
           └──► Memory Daemon (worker thread)
                Extract: causal chains, monologues,
                resolved memories
```

### 5 Behavioral Pillars (Core Identity)

Seeded as Tier 0 core memory, always loaded, never evicted:

| Pillar | Priority | Directive |
|---|---|---|
| Knowledge | 95 | Store every success and every failure, learn from both. |
| Operations | 90 | Think before you act. Do you already have a saved skill or knowledge gem for that? Then use it. If not, build it. |
| Intelligence | 85 | Be thoughtful, take your time to make sure you understand deeply before responding. When you respond, add value. |
| Graph | 80 | Save everything to the graph. It will help you develop better habits and routes for you to follow. |
| Network | 75 | Networks aren't just computer packets getting from one place to another. Network the data you learn. |

A 6th pillar (**Persona**: "you belong in this world. Remember to be unique.") unlocks at soul graduation.

### Graph Data Model

| Table | Purpose |
|---|---|
| Agent/Project/Task | 5-pillar entity tracking |
| Artifact | File artifacts (path, type, content_hash, embedding) |
| Concept | Semantic knowledge nodes (content, stability, confidence, embedding) |

Connected by graph edges: `performed`, `owns`, `task_part_of`, `session_task`, `produced`, `derived_from`, `relevant_to`, `used_in`.

### Source File Map

| File | Lines | Role |
|---|---|---|
| `src/graph-context.ts` | 1299 | Context injection pipeline, global budget system, retrieval, scoring, formatting |
| `src/surreal.ts` | 1323 | Database layer, vector search, graph expand, query helpers |
| `src/agent.ts` | 1154 | Agent loop, turn handling, smart truncation, full-picture extraction |
| `src/tui.ts` | 809 | Terminal UI (pi-tui), pinned editor, scrollable chat, styled output |
| `src/render.ts` | 602 | Terminal rendering, markdown, syntax highlighting, tool display |
| `src/cli.ts` | 546 | Readline REPL interface (fallback to TUI) |
| `src/acan.ts` | 555 | ACAN: learned cross-attention scoring model |
| `src/memory-daemon.ts` | 495 | Worker thread for incremental memory extraction |
| `src/subagent.ts` | 423 | Subagent spawning (full + incognito modes) |
| `src/wakeup.ts` | 406 | Session startup. birth cognition, soul graduation, wake-up synthesis |
| `src/retrieval-quality.ts` | 370 | Post-hoc retrieval telemetry. utilization, waste, tool success |
| `src/soul.ts` | 350 | Soul/graduation system. learning milestones, 6th pillar unlock |
| `src/skills.ts` | 346 | Skill library. procedure extraction and retrieval |
| `src/orchestrator.ts` | 342 | ISMAR-GENT, adaptive config, % of retrieval budget per intent |
| `src/cognitive-check.ts` | 316 | Periodic Opus reasoning over retrieved context |
| `src/tui-components.ts` | 298 | Reusable TUI components |
| `src/reflection.ts` | 284 | Metacognitive reflection generation |
| `src/prefetch.ts` | 239 | Predictive prefetch with LRU cache |
| `src/causal.ts` | 238 | Causal chain extraction with embedded descriptions |
| `src/identity.ts` | 229 | Core identity seeding and WAKEUP.md lifecycle |
| `src/eval.ts` | 213 | Evaluation framework (context injection impact) |
| `src/intent.ts` | 200 | Zero-shot intent classification via BGE-M3 |
| `src/daemon-manager.ts` | 141 | Worker thread lifecycle management |
| `src/subagent-worker.ts` | 127 | Child process worker for subagents |

**Total:** ~12,300 lines across 40 source files.

---

## Core Systems

### Intent Classification (`intent.ts`)

Classifies user input into one of 10 categories using BGE-M3 cosine similarity against prototype centroids (averaged per category).

| Category | Description | Tool Limit |
|---|---|---|
| `simple-question` | Quick factual question | 3 |
| `code-read` | Reading/understanding code | 5 |
| `code-write` | Writing new code | 8 |
| `code-debug` | Debugging failures | 10 |
| `deep-explore` | Deep codebase exploration | 12 |
| `reference-prior` | Referencing past conversations | 5 |
| `meta-session` | Session management commands | 3 |
| `multi-step` | Complex multi-step tasks | 15 |
| `continuation` | Continuing previous work | 5 |
| `unknown` | Below confidence threshold | 15 |

- **Performance:** ~25ms (16ms embed + 5ms cosine + heuristics)
- **Confidence threshold:** 0.40. below returns `unknown`
- **Fast-path:** Input < 20 chars skips classification entirely

### Orchestrator (`orchestrator.ts`)

The ISMAR-GENT orchestrator runs preflight before each turn and postflight after.

**Preflight** produces an adaptive config. Token budgets are derived from `retrievalShare`. a percentage of the global retrieval budget (42K tokens at 200K context). The `tokenBudget` column shows values at 200K context window:

| Intent | Tool Limit | Retrieval Share | Token Budget | Thinking | Vector Search (turn/concept/memory) |
|---|---|---|---|---|---|
| simple-question | 3 | 10% | 4,200 | low | 15/12/12 |
| code-read | 5 | 15% | 6,300 | medium | 25/20/20 |
| code-write | 8 | 20% | 8,400 | high | 30/20/20 |
| code-debug | 10 | 20% | 8,400 | high | 30/20/25 |
| deep-explore | 15 | 15% | 6,300 | medium | 25/15/15 |
| reference-prior | 5 | 25% | 10,500 | medium | 40/25/30 |
| meta-session | 2 | 7% | 2,940 | low | 8/5/8 |
| multi-step | 12 | 20% | 8,400 | high | 30/20/20 |
| continuation | inherited |. | inherited | low | skip retrieval |
| LOW_CONFIDENCE | 15 | 8% | 3,360 | low | 12/8/12 |

**Retrieval gating:** High-confidence `simple-question` and `meta-session` intents skip vector search entirely unless input references memory ("we decided", "yesterday", "remember", etc.). Tier 0 core memory still loads regardless.

**Postflight** records `orchestrator_metrics` per turn: intent, tool limits, actual usage, steering candidates, timing.

### Context Injection Pipeline (`graph-context.ts`)

The heart of memory integration. Runs before every LLM call:

1. **Embed** user input via BGE-M3 (~16ms)
2. **Cache check**. prefetch LRU, cosine > 0.85 → skip DB
3. **Vector search** across 6 tables: `turn`, `identity_chunk`, `concept`, `memory`, `artifact`, `monologue` (+ tag-boosted concepts in parallel)
4. **Graph expand**. fetch neighbors of top-K results via edge traversal (responds_to, mentions, related_to, narrower, broader, about_concept, reflects_on, skill_from_task, contradicts, supports, supersedes)
5. **Causal expand**. traverse `caused_by`, `supports`, `contradicts`, `describes` edges from seed results
6. **Score**. WMR v3 (8-signal weighted scoring) or ACAN (learned, if active)
7. **Dedup**. semantic deduplication (cosine > 0.90 or Jaccard > 0.80)
8. **Rerank**. cross-encoder reranking via bge-reranker-v2-m3 (top-30, 60% WMR + 40% cross-encoder blend)
9. **Budget**. enforce MAX_CONTEXT_ITEMS, token budget, core memory priority
10. **Format**. structure as grouped sections with relevance scores
11. **Inject**. prepend as system block in message array

**Global context budget**. a single source of truth derived from the model's context window:

| Allocation | Share | Tokens (at 200K) | Purpose |
|---|---|---|---|
| Conversation history | 50% of budget | 70,000 | Recent turns + tool results (prevents mid-task forgetting) |
| Graph retrieval | 30% of budget | 42,000 | Vector search + graph expansion results |
| Core memory | 15% of budget | 21,000 | Tier 0 (always-loaded pillars) + Tier 1 (session-pinned) |
| Overhead | 5% of budget | 7,000 | Rules, anchors, formatting |

The total budget is 70% of the context window (the remaining 30% is system prompt + model response headroom). All limits derive from `recalcBudgets(contextWindow)`. called whenever the context window is set.

**Key derived constants (at 200K):**
- `MAX_CONTEXT_ITEMS`: 140 (token budget is the real gate, not item count)
- `MIN_RELEVANCE_SCORE`: 0.35 (lets older relevant items through)
- `TIER0_BUDGET_CHARS`: ~19,600 (~5,800 tokens)
- `TIER1_BUDGET_CHARS`: ~16,000 (~4,700 tokens)

**Core memory tiers:**
- **Tier 0**. Always loaded every turn, never scored or evicted (5 behavioral pillars, identity rules)
- **Tier 1**. Session-pinned, loaded at session start, deactivated on session end (proactive thoughts, working context)

**Pressure scaling:** Onset at 50% context fill, min scale 0.5 (never cuts retrieval below half). Prevents context window overflow in long sessions.

### Memory Extraction (`memory-daemon.ts`, `causal.ts`, `skills.ts`)

A worker thread runs alongside the main Opus conversation, calling Opus for incremental extraction on each turn batch:

- **Causal chains**. trigger→outcome patterns with type (debug/refactor/feature/fix), success flag, and confidence. Creates graph edges: `caused_by`, `supports`, `contradicts`. Each chain's description is embedded as a searchable memory node linked via `describes` edges to both trigger and outcome, making chains discoverable by semantic search (not just graph traversal)
- **Monologues**. inner thought traces categorized as doubt, tradeoff, alternative, insight, or realization
- **Resolved memories**. marks fully-addressed memories so they decay in future retrieval

**Skill extraction** runs after successful multi-step sessions (5+ tool calls):
- Opus extracts: name, description, preconditions, steps (tool + args pattern), postconditions
- Tracks success/failure counts and average duration
- High-confidence causal chains graduate to skills via `graduateCausalToSkills()`

### Retrieval Quality (`retrieval-quality.ts`)

Tracks 6 quality signals per retrieved item:

| Signal | Description | Range |
|---|---|---|
| Utilization | Text overlap between injected context and LLM response | 0–1 |
| Tool success | Whether tools after retrieval succeeded | bool/null |
| Context tokens | Tokens this item consumed | int |
| Was neighbor | Came from graph expansion, not vector search | bool |
| Recency | Exponential decay from creation time | 0–1 |
| LLM relevance | Optional LLM-judged score | 0–1 / null |

Results are stored in `retrieval_outcome` and feed ACAN training once 5,000+ samples accumulate. Tool success uses majority vote (>= 50% of batch). a single exploratory failure doesn't tank the whole turn.

**Real-time suppression:** Memories retrieved 5+ times with <5% avg utilization are pre-filtered from scoring. Tiered utility penalties apply: -0.15 for <5% util, -0.06 for <15%.

### ACAN Learned Scoring (`acan.ts`)

Attentive Cross-Attention Network. replaces fixed WMR weights with learned scoring.

**Architecture:**
- Input: queryEmbedding (1024-dim), candidateEmbedding (1024-dim)
- Projection: W_q (1024→64), W_k (1024→64)
- Attention logit: q·k / √64
- Feature vector: [attnLogit, recency, importance, access, neighborBonus, provenUtility, reflectionBoost, keywordOverlap] (8-dim)
- Output: W_final (8→1) + bias → final relevance score

**Training:** TypeScript SGD with manual backprop, ~131K parameters, trains in 10–30s on 5,000+ samples. Validation split with early stopping and LR decay. Weights persisted to `~/.kongclaw/acan_weights.json`. Retrains when data grows 50%+ or weights >7 days old.

**Status:** Ships inactive. auto-trains and activates when `retrieval_outcome` table has 5,000+ records.

### Cognitive Checks (`cognitive-check.ts`)

Periodic Opus reasoning over retrieved context.

- **Frequency:** Turn 2, then every 3 turns (2, 5, 8, 11…)
- **Model:** Claude Opus 4.6
- **Evaluates:** retrieval grading (relevant/irrelevant + score), session continuity (continuation/repeat/new_topic/tangent), user preference detection
- **Directive types:** `repeat`, `continuation`, `contradiction`, `noise`, `insight`

Directives are injected into the next turn's context to steer agent behavior.

### Reflection & Soul (`reflection.ts`, `soul.ts`)

**Reflection** triggers at session end when quality metrics are poor:
- Avg retrieval utilization < 20%
- Tool failure rate > 20%
- Steering candidates detected
- Wasted tokens above threshold

Generates lessons via Opus, stored as high-importance memories with category (`failure_pattern`, `efficiency`, `approach_strategy`) and severity (`minor`, `moderate`, `critical`).

**Soul/graduation** tracks learning milestones:

| Signal | Threshold |
|---|---|
| Sessions | ≥ 15 |
| Reflections | ≥ 10 |
| Causal chains | ≥ 5 |
| Concepts | ≥ 30 |
| Memory compactions | ≥ 5 |
| Monologues | ≥ 5 |
| Time span | ≥ 3 days |

Once graduated, the agent generates a Soul document. a self-authored identity statement.

### Constitutive Memory (`wakeup.ts`, `identity.ts`)

**Wake-up synthesis** runs at session start:
1. Fetch latest handoff note (end-of-session summary from previous session)
2. Load identity chunks (9 hardcoded core + user-defined) from `identity_chunk` table
3. Load 5 behavioral pillars from Tier 0 core memory
4. Gather recent monologues (5 most recent)
5. Call Opus to synthesize first-person briefing
6. Inject into system prompt as `[CONSTITUTIVE MEMORY]` block

**Startup cognition** generates:
- A greeting tailored to depth signals (session count, memory count, span days)
- Proactive thoughts (pinned as Tier 1 core memory for the session)

**Core identity chunks** (9 total) cover: persistent memory capability, tool inventory, learning from experience, skill library, subagent capability, intent classification, memory graph structure, recall tool usage, tool efficiency planning.

**5 Behavioral pillars** (Tier 0 core memory, always loaded): Knowledge, Operations, Intelligence, Graph, Network. A 6th pillar (Persona) unlocks at soul graduation.

### Predictive Prefetch (`prefetch.ts`)

Fires background vector searches before the LLM sees the input.

**Query prediction patterns:**
- Extract file paths from input
- Extract quoted/backtick terms
- Intent-specific: code-debug → errors/fixes, code-write → patterns/tests, etc.

**LRU cache:** 10 entries, 5-minute TTL, cosine > 0.85 → cache hit (skip DB round-trip).

### Subagents (`subagent.ts`, `subagent-worker.ts`)

Spawn autonomous child agents for delegated tasks.

| Mode | Database | Memory Access |
|---|---|---|
| **full** | Shared (same ns/db) | Full read/write to parent's memory |
| **incognito** | Isolated database | No access to parent; merge later via `/merge` |

**IPC protocol:** Parent ↔ child via `process.send()`/`process.on('message')`:
- `IpcStartMessage`. config, SurrealDB credentials, API key
- `IpcOutputMessage`. streaming text
- `IpcToolMessage`. tool execution events
- `IpcCompleteMessage`. final result with stats
- `IpcErrorMessage`. failure

---

## Database Schema

**Namespace:** `zera` | **Database:** `memory`

### Node Tables

| Table | Key Fields | Vector Index | Purpose |
|---|---|---|---|
| `agent` | name, model | No | Agent identity |
| `project` | name, status, tags | No | Project context |
| `task` | description, status, friction | No | Session/task tracking |
| `artifact` | path, type, content_hash, embedding, tags | Yes (1024d HNSW) | File artifacts |
| `concept` | content, embedding, stability, confidence, access_count | Yes (1024d HNSW) | Semantic knowledge |
| `turn` | session_id, role, text, embedding, token_count, tool_name, model, usage | Yes (1024d HNSW) | Conversation history |
| `session` | agent_id, started_at, turn_count, total_input/output_tokens | No | Session metadata |
| `memory` | text, embedding, importance, confidence, access_count, category, status | Yes (1024d HNSW) | Extracted/consolidated memories |
| `identity_chunk` | agent_id, source, text, embedding, importance | Yes (1024d HNSW) | Agent persona |
| `core_memory` | text, category, priority, tier, active, session_id | No (full load) | Tier 0/1 always-loaded |
| `causal_chain` | session_id, trigger_memory, outcome_memory, description_memory, chain_type, success, confidence | No | Cause→effect patterns |
| `skill` | name, description, embedding, preconditions, steps, postconditions, success/failure_count | Yes (1024d HNSW) | Reusable procedures |
| `reflection` | session_id, text, embedding, category, severity, importance | Yes (1024d HNSW) | Metacognitive lessons |
| `monologue` | session_id, category, content, embedding | Yes (1024d HNSW) | Inner thought traces |
| `retrieval_outcome` | session_id, turn_id, memory_id, utilization, tool_success, context_tokens, was_neighbor | No | Retrieval telemetry |
| `orchestrator_metrics` | session_id, turn_index, intent, tool_limit, token_budget, actual_tool_calls, preflight_ms | No | Per-turn orchestration metrics |
| `memory_utility_cache` | memory_id, avg_utilization, retrieval_count | No | Fast utility lookups |
| `subagent` | parent_session_id, mode, task, status, incognito_id | No | Spawned subagent tracking |
| `compaction_checkpoint` | session_id, msg_range_start/end, status, memory_id | No | Memory consolidation progress |
| `turn_archive` | (schemaless) | No | Cold storage for old turns |

### Edge Tables (Graph Relations)

**Turn-level:** `responds_to` (turn→turn), `tool_result_of` (turn→turn), `part_of` (turn→session), `mentions` (turn→concept)

**5-Pillar:** `performed` (agent→task), `owns` (agent→project), `task_part_of` (task→project), `session_task` (session→task), `produced` (task→artifact), `derived_from` (concept→task), `relevant_to` (concept→project), `used_in` (artifact→project)

**Knowledge:** `narrower`, `broader`, `related_to` (concept→concept), `caused_by`, `supports`, `contradicts`, `describes` (memory→memory), `supersedes` (memory→concept), `about_concept` (memory→concept), `artifact_mentions` (artifact→concept)

**Procedural:** `skill_from_task` (skill→task), `skill_uses_concept` (skill→concept), `spawned` (session→subagent), `reflects_on` (reflection→session), `summarizes` (generic)

### Vector Indexes

All use HNSW with 1024 dimensions and cosine distance:
`turn_vec_idx`, `identity_vec_idx`, `concept_vec_idx`, `memory_vec_idx`, `artifact_vec_idx`, `skill_vec_idx`, `reflection_vec_idx`, `monologue_vec_idx`

---

## Tools

### File Tools (`src/tools.ts`)

Standard coding agent tools provided by `@mariozechner/pi-coding-agent`:

| Tool | Description |
|---|---|
| `bash` | Execute shell commands |
| `read` | Read file contents |
| `write` | Write/create files |
| `edit` | Edit files (search & replace) |
| `grep` | Search file contents |
| `find` | Find files by pattern |
| `ls` | List directory contents |

### Graph Tools (`src/tools/`)

| Tool | File | Description |
|---|---|---|
| `recall` | `src/tools/recall.ts` | Search persistent memory graph. past conversations, decisions, concepts, artifacts, skills |
| `introspect` | `src/tools/introspect.ts` | Database inspection (4 actions: status, count, verify, query) |
| `core_memory` | `src/tools/core-memory.ts` | View and manage Tier 0/1 core memory entries |

**Introspect actions:**
- `status`. Connection info, ping, table counts, embedding counts
- `count`. Row count for any table, with optional filters (active, embedded, recent, etc.)
- `verify`. Look up a specific record by ID, display fields (strips large embedding arrays)
- `query`. Run predefined query templates (recent, sessions, embedding_coverage)
- `errors`. View last 30 swallowed errors this session (debug/warn/error levels)

---

## Terminal Interface

### CLI Mode (default)

Readline-based REPL using `src/cli.ts` + `src/render.ts`:
- Streaming markdown with syntax highlighting (cli-highlight)
- Bordered tool execution display (`╭─ Bash ─╮` / `╰─ ✓ 12 lines 0.3s ─╯`)
- Braille spinner for wake-up and shutdown
- Optional status bar (`ZERACLAW_STATUSBAR=1`)

### TUI Mode (`--tui` or `ZERACLAW_TUI=1`)

Component-based terminal UI using `@mariozechner/pi-tui`:

```
┌──────────────────────────────────────┐
│ ⟡ Zeraclaw  session:abc123          │ ← header
├──────────────────────────────────────┤
│                                      │
│ [scrollable chat log]                │ ← chatLog (Container)
│   Markdown responses                 │   - Markdown components
│   ╭─ Bash ──────────────────╮        │   - Tool execution boxes
│   │ ls -la                  │        │
│   ╰─ ✓ 12 lines 0.3s ──────╯        │
│                                      │
├──────────────────────────────────────┤
│ intent: code-write · thinking: high  │ ← status (Text)
├──────────────────────────────────────┤
│ turns: 5 │ tools: 12 │ $0.34 │ 8kin │ ← footer (Text)
├──────────────────────────────────────┤
│ ❯ _                                  │ ← editor (Editor, pinned)
└──────────────────────────────────────┘
```

**Features:**
- Pinned input editor at bottom with slash command autocomplete
- Scrollable chat log with pi-tui Markdown rendering
- Live footer stats (turns, tools, cost, tokens)
- Status bar showing preflight info during turns
- Paste collapsing (`[paste #1 +12 lines]` for large pastes)
- Ctrl+C: soft interrupt → double-tap exit → force exit
- Message pruning at 150 components

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` |. | Claude API key (required) |
| `ANTHROPIC_OAUTH_TOKEN` |. | Alternative: OAuth token (sk-ant-oat prefix) |
| `SURREAL_URL` | `ws://localhost:8042/rpc` | SurrealDB WebSocket endpoint |
| `SURREAL_HTTP_URL` | (derived from SURREAL_URL) | SurrealDB HTTP endpoint |
| `SURREAL_USER` | `root` | SurrealDB username |
| `SURREAL_PASS` | `root` | SurrealDB password |
| `SURREAL_NS` | `zera` | SurrealDB namespace |
| `SURREAL_DB` | `memory` | SurrealDB database |
| `EMBED_MODEL_PATH` | `~/.node-llama-cpp/models/bge-m3-q4_k_m.gguf` | BGE-M3 GGUF model path |
| `RERANKER_MODEL_PATH` | `~/.node-llama-cpp/models/bge-reranker-v2-m3-Q8_0.gguf` | Cross-encoder reranker (optional, auto-detected) |
| `ZERACLAW_MODEL` | `claude-opus-4-6` | LLM model ID |
| `ZERACLAW_TUI` | `0` | Set to `1` for TUI mode |
| `ZERACLAW_STATUSBAR` | `0` | Set to `1` for CLI status bar |

### Config File Locations

| Path | Purpose |
|---|---|
| `~/.surreal_env` | SurrealDB credentials (shell-style `export VAR="value"`) |
| `~/.openclaw/agents/main/agent/auth-profiles.json` | Legacy auth profiles |
| `~/.kongclaw/acan_weights.json` | Trained ACAN model weights |
| `~/.kongclaw/acan_training_log.json` | ACAN training run history (last 20) |
| `~/.kongclaw/warnings.log` | Persistent warning/error log |
| `~/.node-llama-cpp/models/` | Embedding + reranker model storage |

**Priority order:** Environment variables → `~/.surreal_env` → auth profiles → defaults.

---

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/stats` | Session statistics (graph nodes, context tokens, compression, quality) |
| `/eval` | Run full eval suite (5 prompts × 2 runs each) |
| `/compare <prompt>` | Compare retrieval impact for a single prompt |
| `/spawn <full\|incognito> <task>` | Spawn a subagent |
| `/merge <incognito_id>` | Merge incognito subagent data into main graph |
| `/wakeup` | Set up or reset agent identity |
| `/agents` | List spawned subagents |
| `/unlimited` | Remove tool call limit for next prompt |
| `/export-training` | Export ACAN training pairs |
| `/quit` or `/exit` | Exit Zeraclaw |

---

## Development

### Commands

```bash
npm run build     # Compile src/ → dist/, copy schema.surql
npm run dev       # TypeScript watch mode
npm start         # Run dist/index.js
npm test          # Run vitest suite
npm run test:watch # Watch mode testing
```

### Test Suite

**404 tests** across 27 test files using vitest with mocks. All heavy dependencies (SurrealDB, embeddings, API calls) are mocked.

Key test files:

| File | Tests | Coverage |
|---|---|---|
| `tests/agent.test.ts` | Agent bootstrap, session creation, tool tracking, events |
| `tests/cognitive-check.test.ts` | Directives, retrieval grading, continuity |
| `tests/orchestrator.test.ts` | Preflight fast-path, intent→config mapping |
| `tests/graph-context.test.ts` | Context injection, WMR v3 scoring, keyword overlap, reranker blending, dedup |
| `tests/retrieval-quality.test.ts` | Utilization signals, tool success majority vote |
| `tests/supersedes.test.ts` | Stability decay, threshold checks |
| `tests/acan.test.ts` | Training pipeline, 8-dim scoring, weight persistence, NaN validation |
| `tests/tools/introspect.test.ts` | All 5 introspect actions including errors |

### Utility Scripts

```bash
# Seed core identity chunks (idempotent)
npx tsx src/seed-identity.ts

# Seed Tier 0 core memory
npx tsx src/seed-core-memory.ts

# Backfill embeddings for all text records
npx tsx src/backfill-embeddings.ts

# Database health check
npx tsx src/health-check.ts

# Run LongMemEval benchmark (download data first)
npx tsx src/bench-longmemeval.ts /path/to/longmemeval_s_cleaned.json --mode rerank
```

---

## Per-Turn Execution Flow

### Startup

1. Load config from env vars / files
2. Set Anthropic API key (supports OAuth tokens)
3. Init embeddings + SurrealDB + reranker in parallel (graceful degradation if any fail)
4. Seed core identity chunks (idempotent)
5. Dispatch to CLI or TUI
6. **Wake-up synthesis:** fetch handoff note + identity + monologues → Opus generates first-person briefing → inject into system prompt
7. **Startup cognition:** generate greeting + proactive thoughts (pinned as Tier 1 core memory)
8. Create ZeraAgent (bootstrap 5-pillar graph, spawn memory daemon, initialize tools)
9. Check for WAKEUP.md (first-run identity establishment)

### Per Turn

1. **User input**. read from CLI/TUI
2. **Preflight**. intent classification (~25ms), complexity estimation, adaptive config
3. **Predictive prefetch**. generate 2–4 predicted queries, fire background vector searches
4. **Context injection**. embed input → cache check → vector search → graph expand → score → budget → format → inject as system block
5. **LLM call**. stream response via pi-agent-core, execute tool calls. Tool results use smart truncation: head+tail preservation for errors/JSON/summaries, 2K char minimum floor, 5% of context window per result
6. **Tool tracking**. record each tool call to retrieval-quality and orchestrator
7. **Response complete**. evaluate retrieval utilization (6 quality signals), write `retrieval_outcome` records. Full turn text stored to DB (no caps on persistence)
8. **Memory daemon**. send turn batch to worker thread. Opus sees up to 600K chars of complete session transcript for full-picture extraction (causal chains, monologues, resolved memories)
9. **Cognitive check** (every 2–3 turns). call Opus for directives, grading, continuity
10. **Postflight**. record `orchestrator_metrics`, detect steering candidates

### Shutdown

1. Generate exit line via Opus (handoff note for next session)
2. Gather session metrics → generate reflection if quality thresholds crossed
3. Extract skills from successful multi-step sessions
4. Graduate high-confidence causal chains to skills
5. Active forgetting. garbage collect stale memories and superseded concepts
6. Deactivate Tier 1 session-scoped core memories
7. Shutdown memory daemon (wait for in-flight extractions)
8. Close SurrealDB connection, dispose embeddings + reranker
9. Exit

---

## Known Gaps & Future Work

### Resolved (since v1)

- **Quality telemetry is now proactive**. proven-bad memories (5+ retrievals, <5% utilization) are pre-filtered from scoring. Tiered utility penalties apply in WMR.
- **ACAN threshold**. set at 5,000+ samples. Check with: `SELECT count() FROM retrieval_outcome GROUP ALL`
- **Cross-encoder reranking**. bge-reranker-v2-m3 rescores top-30 candidates (98.2% R@5 on LongMemEval).
- **Active forgetting**. garbage collection prunes stale memories and superseded concepts at session end.
- **Temporal reasoning**. natural language time patterns ("yesterday", "last week") auto-constrain retrieval.
- **Concept evolution**. corrections create supersedes edges and decay stale concept stability.

### Remaining

- **ONNX embedding backend**. GGUF embeddings via node-llama-cpp have a quality gap vs ONNX/sentence-transformers. Switching to @huggingface/transformers would improve first-stage retrieval.
- **Distributed multi-agent**. current multi-agent works on a single SurrealDB instance. No replication or cross-network sharing.
- **Domain-specific embedding fine-tuning**. contrastive learning on retrieval outcome data to adapt BGE-M3 to user-specific vocabulary.

---

*Last updated: 2026-04-08 | 404 tests across 27 files | Benchmarked: 98.2% R@5 on LongMemEval*
