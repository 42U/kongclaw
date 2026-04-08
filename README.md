<div align="center">

# KongClaw

![KongClaw](KongClaw6.png)

[![VoidOrigin](https://img.shields.io/badge/VOIDORIGIN-voidorigin.com-0a0a0a?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIHN0cm9rZT0iI2ZmNmIzNSIgc3Ryb2tlLXdpZHRoPSIyIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNCIgZmlsbD0iI2ZmNmIzNSIvPjwvc3ZnPg==&logoColor=ff6b35&labelColor=0a0a0a)](https://voidorigin.com)

[![GitHub Stars](https://img.shields.io/github/stars/42U/kongclaw?style=for-the-badge&logo=github&color=gold)](https://github.com/42U/kongclaw)
[![License: MIT](https://img.shields.io/github/license/42U/kongclaw?style=for-the-badge&logo=opensourceinitiative&color=blue)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-20+-339933?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org)
[![SurrealDB](https://img.shields.io/badge/SurrealDB-3.0-ff00a0?style=for-the-badge&logo=surrealdb&logoColor=white)](https://surrealdb.com)
[![Claude](https://img.shields.io/badge/Claude-Opus_4.6-d4a574?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Tests](https://img.shields.io/badge/Tests-404_passing-brightgreen?style=for-the-badge&logo=vitest&logoColor=white)](https://vitest.dev)

**A graph-backed cognitive agent that learns across sessions.** 

[Quick Start](#quick-start) | [Benchmarks](#benchmarks) | [Architecture](#architecture) | [How It Works](#how-it-works) | [Commands](#commands) | [Development](#development)

</div>

---

## What It Does

KongClaw is a persistent AI agent built on a SurrealDB knowledge graph. Every conversation becomes structured memory: concepts, causal chains, skills, and reflections. All connected through graph edges that compound over time.

| Capability | Traditional Agent | KongClaw |
|-----------|------------------|----------|
| Memory | Sliding window, lost on exit | SurrealDB graph, persists forever |
| Retrieval | Cosine similarity | 8-signal WMR + cross-encoder reranking (98.2% R@5) |
| Learning | None | 9 knowledge types extracted per session via Sonnet |
| Scoring | Fixed weights | ACAN cross-attention (auto-trains from retrieval outcomes) |
| Skills | None | Procedural memory from successful workflows |
| Identity | Stateless | Earned soul via graduation system |
| Self-correction | None | Metacognitive reflections + concept supersedes |

Not a chatbot with a vector store bolted on. A cognitive architecture that learns, compounds, and remembers who it is.

---

## Quick Start

### 1. Start SurrealDB

Docker (recommended):

```bash
docker run -d --name surrealdb -p 127.0.0.1:8042:8000 \
  -v ~/.kongclaw/surreal-data:/data \
  surrealdb/surrealdb:latest start \
  --user root --pass root surrealkv:/data/surreal.db
```

Or native:

```bash
curl -sSf https://install.surrealdb.com | sh
surreal start --user root --pass root --bind 127.0.0.1:8042 surrealkv:~/.kongclaw/surreal.db
```

> **Security note:** Always bind to `127.0.0.1`, not `0.0.0.0`, unless you need remote access.

### 2. Clone and build

```bash
git clone https://github.com/42U/kongclaw.git
cd kongclaw
npm install
```

### 3. Run

TUI mode (the full experience):
```bash
npm start
```

Basic readline REPL (fallback/debug):
```bash
npm run cli
```

On first run, a setup wizard walks you through configuration. The embedding model (BGE-M3, ~420MB) downloads automatically. If SurrealDB is installed but not running, KongClaw starts a managed instance for you.

### 4. Optional: Cross-Encoder Reranker

For maximum retrieval quality (98.2% R@5), download the reranker model:

```bash
curl -fSL -o ~/.node-llama-cpp/models/bge-reranker-v2-m3-Q8_0.gguf \
  "https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf"
```

Auto-detected on startup. ~606MB. Without it, retrieval still works via WMR scoring, just without the reranking stage.

<details>
<summary><strong>Configuration</strong></summary>

Environment variables (all optional, sensible defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | - | Claude API key (also accepts `ANTHROPIC_OAUTH_TOKEN`) |
| `SURREAL_URL` | `ws://localhost:8042/rpc` | SurrealDB WebSocket endpoint |
| `SURREAL_USER` / `SURREAL_PASS` | `root` / `root` | SurrealDB auth |
| `SURREAL_NS` / `SURREAL_DB` | `kong` / `memory` | SurrealDB namespace and database |
| `EMBED_MODEL_PATH` | `~/.node-llama-cpp/models/bge-m3-q4_k_m.gguf` | Path to BGE-M3 GGUF model |
| `RERANKER_MODEL_PATH` | `~/.node-llama-cpp/models/bge-reranker-v2-m3-Q8_0.gguf` | Path to reranker GGUF model |
| `KONGCLAW_MODEL` | `claude-opus-4-6` | Default Claude model |

Also reads from `~/.surreal_env`.

> **Security note:** The setup wizard stores your API key in `~/.kongclaw/config.json` (mode `600`, owner-only). For shared environments, prefer environment variables.

</details>

---

## Benchmarks

### LongMemEval: AI Memory Retrieval (500 questions, 6 types)

Benchmarked against [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), the standard academic benchmark for AI memory systems:

| System | Mode | R@5 | LLM Required | Cost |
|--------|------|-----|-------------|------|
| **KongClaw** | **Embedding + Cross-Encoder Rerank** | **98.2%** | **None** | **$0** |
| MemPalace | Hybrid (keyword overlap) | 98.4% | None | $0 |
| MemPalace | Raw (ChromaDB default) | 96.6% | None | $0 |
| MemPalace | Hybrid + Haiku LLM rerank | 100% | Haiku | ~$0.001/q |

**KongClaw beats MemPalace's raw baseline by +1.6%** using a two-stage retrieve-then-rerank architecture: BGE-M3 embedding search generates top-30 candidates, then bge-reranker-v2-m3 (a purpose-built cross-encoder) rescores them. Zero API calls, fully local, ~0.84s per query.

<details>
<summary><strong>Per-Type Breakdown</strong></summary>

| Question Type | R@5 | R@10 | n |
|--------------|-----|------|---|
| knowledge-update | 100.0% | 100.0% | 78 |
| multi-session | 99.2% | 100.0% | 133 |
| temporal-reasoning | 98.5% | 100.0% | 133 |
| single-session-user | 100.0% | 100.0% | 70 |
| single-session-assistant | 94.6% | 96.4% | 56 |
| single-session-preference | 90.0% | 100.0% | 30 |

</details>

<details>
<summary><strong>Why This Matters</strong></summary>

Most AI memory systems embed text, search by cosine similarity, return top-K. MemPalace showed that raw verbatim storage + default embeddings beats LLM-extracted summaries (96.6% vs Mem0's ~45%).

KongClaw goes further with a multi-stage architecture:
1. **Dense retrieval** via BGE-M3 (1024-dim) for candidate generation
2. **Cross-encoder reranking** via bge-reranker-v2-m3 for precision scoring
3. In production: **graph expansion**, **8-signal WMR scoring**, **learned ACAN attention**, **concept supersedes**, and **retrieval quality tracking** on top

The cross-encoder reads each query-document pair through a full transformer forward pass, understanding semantic relationships that cosine similarity misses. It pushed the two weakest categories (preference: 70% -> 90%, assistant-reference: 85.7% -> 94.6%) while maintaining 100% on knowledge-update and single-session-user.

Run the benchmark yourself: `npx tsx src/bench-longmemeval.ts`

</details>

---

## Architecture

```
User Input
    |
    v
Preflight ──────── Intent classification (9 categories, adaptive budgets)
    |
    v
Prefetch ────────── Predictive background vector search (LRU cache)
    |
    v
Context Injection ─ Vector search -> graph expand -> WMR scoring -> rerank -> budget trim
    |                  Searches: turns, concepts, memories, artifacts, identity, monologues
    |                  Scores: similarity, recency, importance, access, neighbor,
    |                          utility, reflection, keyword overlap
    |                  Rerank: bge-reranker-v2-m3 cross-encoder (top-30, 60/40 blend)
    v
Agent Loop ──────── Claude (Opus/Sonnet/Haiku) + tool execution
    |                  Tools: bash, read, write, edit, grep, recall, core-memory, introspect
    v
Quality Eval ────── Retrieval utilization + tool success (majority vote) -> ACAN training
    |
    v
Memory Daemon ───── Worker thread extracts 9 knowledge types via Sonnet
    |                  + concept supersedes (corrections decay stale knowledge)
    v
Postflight ──────── Orchestrator metrics, reflection if thresholds exceeded
```

### The Knowledge Graph

22+ tables in SurrealDB with HNSW vector indexes (1024-dim cosine):

| Table | Purpose |
|-------|---------|
| `turn` | Every conversation message with embeddings |
| `memory` | Compacted episodic knowledge (importance, confidence, access tracking) |
| `concept` | Semantic knowledge nodes (stability, supersedes tracking) |
| `skill` | Learned procedures with success/failure counts |
| `reflection` | Metacognitive lessons from session performance |
| `causal_chain` | Cause-effect patterns (trigger, outcome, chain type) |
| `core_memory` | Tier 0 (always loaded) + Tier 1 (session-pinned) directives |
| `soul` | Emergent identity document, earned through graduation |
| `agent`, `project`, `task`, `artifact` | 5-pillar structural model |

**Edge relations:** `responds_to`, `mentions`, `caused_by`, `supports`, `contradicts`, `supersedes`, `narrower`, `broader`, `related_to`, `reflects_on`, `skill_from_task`, `about_concept`, and more.

<details>
<summary><strong>Adaptive Reasoning</strong></summary>

Every turn is classified by intent and assigned an adaptive config:

| Intent | Thinking | Tool Limit | Token Budget | Retrieval |
|--------|----------|------------|--------------|-----------|
| `simple-question` | low | 3 | 4K | 10% |
| `code-write` | high | 8 | 8K | 20% |
| `code-debug` | high | 10 | 8K | 20% |
| `reference-prior` | medium | 5 | 10K | 25% |
| `continuation` | low | 8 | 4K | skip |

Fast path for trivial inputs. Confidence gate for uncertain classification.

</details>

<details>
<summary><strong>WMR v3 Scoring</strong>: 8-signal weighted memory ranker</summary>

```
score = 0.22 * cosine
      + 0.23 * recency
      + 0.05 * importance
      + 0.05 * access
      + 0.10 * neighbor_bonus
      + 0.15 * proven_utility
      + 0.10 * reflection_boost
      + 0.12 * keyword_overlap
      - utility_penalty
```

ACAN (130K-param cross-attention network) replaces this once 5,000+ labeled retrieval outcomes accumulate.

</details>

<details>
<summary><strong>Concept Supersedes</strong>: stale knowledge auto-deprioritizes</summary>

When the memory daemon extracts a correction (user correcting the assistant):
1. Embeds the *original* (wrong) statement
2. Finds matching concepts via vector similarity (threshold: 0.70)
3. Creates `supersedes` edges from correction to stale concepts
4. Decays concept stability by 60% (floor: 0.15)

Stale knowledge remains discoverable but naturally loses to corrections in retrieval.

</details>

<details>
<summary><strong>Soul & Graduation</strong></summary>

Seven thresholds, need 5+ to graduate: sessions (15), reflections (10), causal chains (5), concepts (30), compactions (5), monologues (5), time span (3 days).

Soul document: working style, self-observations, earned values grounded in specific evidence. Each revision requires rationale.

</details>

---

## Commands

| Command | Description |
|---------|-------------|
| `/stats` | Session statistics: context tokens, retrieval quality, reranker status |
| `/spawn full\|incognito <task>` | Spawn a subagent |
| `/merge <id>` | Merge incognito agent's knowledge back |
| `/agents` | List spawned subagents |
| `/unlimited` | Remove tool call limit for next prompt |
| `/eval` | Run eval suite (5 prompts x 2 runs) |
| `/compare <prompt>` | A/B test: full retrieval vs graph context |
| `/export-training` | Export ACAN training data to JSONL |
| `/help` | Show available commands |
| `/quit` | Exit |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | TypeScript (strict, ES2022, Node16 modules) |
| Runtime | Node.js 20+ |
| Database | SurrealDB 3.0 (graph + vector, HNSW 1024-dim cosine) |
| Embeddings | BGE-M3 via node-llama-cpp (1024-dim, L2 normalized) |
| Reranker | bge-reranker-v2-m3 via LlamaRankingContext (optional, 606MB) |
| LLM | Claude Opus 4.6 (reasoning), Sonnet (extraction), Haiku (validation) |
| Agent Framework | pi-agent-core (event system, tool orchestration) |
| TUI | pi-tui (terminal UI components, editor, containers) |
| Testing | Vitest (404 tests across 27 files) |

---

## Development

TypeScript watch mode:
```bash
npm run dev
```

Run the test suite (vitest, 404 tests):
```bash
npm test
```

Watch mode tests:
```bash
npm run test:watch
```

Build to dist/ (also copies schema.surql):
```bash
npm run build
```

### Run the Benchmark

Download the LongMemEval dataset:
```bash
mkdir -p /tmp/longmemeval-data
curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

Run the benchmark (supports raw, hybrid, rerank, and hybrid-rerank modes):
```bash
npx tsx src/bench-longmemeval.ts /tmp/longmemeval-data/longmemeval_s_cleaned.json --mode rerank
```

---

## Why KongClaw Exists

Every AI agent on the market is stateless. Session ends, everything vanishes. The next session starts from zero. Same mistakes, same re-explanations, same blank slate. The industry's answer is RAG: bolt a vector store onto the side, embed some chunks, cosine-similarity your way to "memory." It works for demos. It fails for real work.

Here's what's wrong with the current landscape:

**Mem0, LangChain, AutoGPT** use an LLM to decide what to remember, then throw away the original. When the LLM extracts "user prefers PostgreSQL" and discards the 3-session conversation about why, it loses the context that makes memory useful. Mem0 scores 30-45% on ConvoMem. KongClaw's architecture scores 92.9% on the same benchmark class.

**MemPalace** proved that raw verbatim storage beats LLM extraction (96.6% R@5). Good insight. But it's a single-stage cosine search over ChromaDB with no learning, no graph, no adaptation. It went viral because an actress promoted it, not because the architecture is interesting.

**KongClaw** is built different:

1. **Retrieval isn't one stage, it's five.** Vector search across 6 tables → tag-boosted concept lookup → graph neighbor expansion across typed edges → 8-signal weighted scoring → cross-encoder reranking. Each stage catches what the previous one missed. 98.2% R@5 on LongMemEval with zero API calls.

2. **The graph isn't decoration, it's structural.** `caused_by`, `supports`, `contradicts`, `supersedes`, `narrower`, `broader`. These edges encode relationships that embeddings fundamentally cannot capture. When you ask "what caused the auth failure?", cosine similarity finds the failure. Graph traversal finds the cause.

3. **The system learns from itself.** Every retrieval outcome is tracked. Was this memory actually used in the response? After 5,000 outcomes, ACAN (a 131K-parameter cross-attention network) trains itself to replace the fixed scoring weights. No human labeling. No external training data. The agent improves by using itself.

4. **Corrections don't just add. They supersede.** When you correct the agent, the supersedes system finds the stale concept, creates an edge, and decays its stability by 60%. Old knowledge doesn't compete with corrections in retrieval. It gets structurally demoted.

5. **The agent earns its identity.** After 15+ sessions, 10+ reflections, 30+ concepts, and 3+ days of experience, the agent graduates and authors a Soul document. A self-assessment grounded in actual behavior, not a prompted persona. It wakes up each session knowing who it is, what it was working on, and what went wrong last time.

This isn't a wrapper around an LLM. It's a cognitive architecture where every session makes the next one better. The graph compounds. The scoring adapts. The knowledge evolves. That's the difference.

---

<div align="center">

MIT License | Built by [42U](https://github.com/42U) | [VoidOrigin](https://voidorigin.com)

</div>
