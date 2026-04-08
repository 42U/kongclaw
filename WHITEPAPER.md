# Graph-Backed Persistent Memory for Autonomous LLM Agents

**A Memory Architecture Beyond RAG**

---

## Abstract

Large language model agents are fundamentally stateless. Each session begins with no memory of previous interactions, forcing users to re-establish context, re-explain preferences, and watch agents repeat mistakes they have already learned from. The dominant approach to adding memory — retrieval-augmented generation (RAG) with flat vector stores — treats all memories as interchangeable text chunks, ignoring the rich structure of relationships, causality, and temporal context that makes human memory useful.

We present a graph-backed memory architecture that addresses these limitations through five interlocking mechanisms: (1) a knowledge graph with typed edges that captures relationships between memories, not just their content; (2) adaptive retrieval orchestration that matches retrieval depth to task complexity; (3) a self-training scoring model that learns from its own retrieval telemetry; (4) a metacognitive loop that extracts causal patterns, skills, and reflections from agent performance; and (5) constitutive memory that lets agents wake up knowing who they are. Together, these mechanisms create agents with genuine persistent memory — agents that learn from experience, improve over time, and maintain identity continuity across sessions.

---

## 1. Introduction

### 1.1 The Problem with Stateless Agents

Modern LLM agents built on frameworks like LangChain, AutoGPT, or Claude's tool-use API share a fundamental limitation: they are stateless. When a session ends, everything the agent learned — the user's preferences, the codebase it explored, the bugs it debugged, the decisions it made — vanishes. The next session starts from zero.

This creates several compounding problems:

**Repetitive context establishment.** Users must re-explain their project, preferences, and current state at the start of every session. A developer who spent three hours with an agent debugging a complex authentication system must re-describe the entire architecture when they return the next day.

**Lost learning.** When an agent discovers that a particular debugging approach works well for a specific codebase, that knowledge dies with the session. The next time a similar bug appears, the agent will try the same failed approaches before stumbling on the same solution.

**No skill accumulation.** Human developers build procedural expertise — they develop reliable workflows for common tasks. Stateless agents cannot accumulate such expertise. Every task is approached as if for the first time.

**Identity discontinuity.** Users develop working relationships with agents — establishing communication preferences, technical context, and collaborative patterns. Statelessness destroys this relationship after every session.

### 1.2 The Problem with Simple RAG

The standard solution is retrieval-augmented generation: store conversation history and retrieved documents in a vector database, embed the user's query, find the most similar stored vectors, and inject them into the context window. While RAG is better than nothing, it has fundamental limitations that become severe as memory grows.

**Flat memory treats all items equally.** A vector store has no concept of relationships between memories. It cannot represent "the fix for bug X caused regression Y" or "approach A contradicts approach B." These relationships are crucial for intelligent retrieval — knowing that two memories are causally linked is often more important than knowing they are semantically similar.

**No adaptation to task complexity.** A RAG system performs the same retrieval regardless of whether the user says "hello" or "debug why the auth middleware is rejecting valid tokens from the refresh endpoint we deployed yesterday." The first needs zero retrieval; the second needs deep, multi-table search across conversation history, code artifacts, and past debugging sessions. Yet most RAG systems have a single retrieval pipeline with fixed parameters.

**No feedback loop.** RAG systems cannot learn which retrievals were useful and which were noise. After injecting 20 memory chunks into the context window, there is no mechanism to evaluate whether those chunks contributed to the response. Over time, this means the system cannot improve — it retrieves with the same accuracy whether it has been running for a day or a year.

**Context window pollution.** Perhaps the most insidious problem: injecting irrelevant memories doesn't just waste tokens — it actively degrades response quality. LLMs attend to all context, and irrelevant context competes with relevant context for attention. A system that retrieves 15 chunks but only 3 are useful is performing worse than a system that retrieves only those 3.

### 1.3 Our Approach

We propose a memory architecture built on five pillars that address each of these limitations:

1. **Knowledge graph with typed edges** — Store memories as nodes connected by semantically typed relationships (caused_by, supports, contradicts, narrower, broader, mentions, etc.). This enables retrieval that follows causal chains and conceptual hierarchies, not just embedding similarity.

2. **Adaptive retrieval orchestration** — Classify user intent in real-time (~25ms) and adapt retrieval depth, thinking level, tool budgets, and vector search limits per table. Simple questions get minimal retrieval; complex debugging tasks get deep multi-table search.

3. **Self-training scoring model** — Replace hand-tuned retrieval weights with a lightweight cross-attention network that trains on the system's own retrieval telemetry. The model starts with fixed weights and progressively learns which memories are genuinely useful for which types of queries.

4. **Metacognitive loop** — Extract causal patterns, reusable skills, and reflective lessons from agent performance. The agent doesn't just use memory — it generates new memory from its own experience, creating a feedback loop of continuous improvement.

5. **Constitutive memory** — Synthesize a first-person memory briefing at session start from handoff notes, identity chunks, and recent internal monologues. The agent wakes up knowing who it is, what it was working on, and what it was thinking about.

The remainder of this paper describes each mechanism in detail, with emphasis on the design decisions, architectural trade-offs, and observable properties that make the system work in practice.

---

## 2. Memory Architecture

### 2.1 The Knowledge Graph

At the foundation of our architecture is a knowledge graph stored in SurrealDB — a multi-model database that supports both document storage and graph relations with HNSW vector indexes.

**Node types** represent different categories of knowledge:

| Node Type | Purpose | Example |
|---|---|---|
| Turn | Conversation history | "User asked to fix the auth bug at 3:14 PM" |
| Concept | Semantic knowledge | "OAuth2 refresh token rotation" |
| Memory | Extracted/consolidated summaries | "The auth middleware requires Bearer tokens with RS256 signing" |
| Artifact | File artifacts | "src/auth/middleware.ts — authentication guard" |
| Skill | Reusable procedures | "Debug auth failures: check token expiry → verify signing key → inspect middleware chain" |
| Reflection | Performance lessons | "Tool calls were wasted on irrelevant files — should have used grep before reading" |
| Monologue | Internal thought traces | "The user seems frustrated — I should be more concise" |
| Identity Chunk | Agent persona | "I am a graph-backed coding agent with persistent memory" |
| Core Memory | Always-loaded context | Tier 0: identity rules; Tier 1: session working context |

**Edge types** capture structured relationships between nodes:

| Edge Type | Direction | Semantics |
|---|---|---|
| `caused_by` | memory → memory | Outcome was caused by trigger |
| `supports` | memory → memory | Evidence supports conclusion |
| `contradicts` | memory → memory | Evidence contradicts conclusion |
| `narrower` | concept → concept | Concept is a specialization |
| `broader` | concept → concept | Concept is a generalization |
| `related_to` | concept → concept | General semantic relation |
| `mentions` | turn → concept | Conversation referenced concept |
| `about_concept` | memory → concept | Memory pertains to concept |
| `responds_to` | turn → turn | Turn is a response to previous turn |
| `produced` | task → artifact | Task produced this artifact |
| `reflects_on` | reflection → session | Reflection evaluates session |
| `skill_from_task` | skill → task | Skill was extracted from task |

**Why graphs over flat stores.** The critical insight is that relationships encode structure that embeddings alone cannot capture. Consider:

- "We tried approach A (mocking the database) and it failed because mocked tests passed but the production migration broke." This is a causal chain: approach → failure → lesson. In a flat store, these are three separate chunks with no connection. In a graph, they form a `caused_by` chain that can be traversed when a similar debugging situation arises.

- "The auth middleware supports both JWT and OAuth2, but OAuth2 contradicts the existing session-based auth." `supports` and `contradicts` edges let the system retrieve conflicting information together, enabling the agent to reason about trade-offs rather than receiving isolated facts.

Embedding similarity finds memories that look similar. Graph traversal finds memories that are related — a fundamentally different and often more useful property.

### 2.2 Tiered Memory Hierarchy

Not all memory should be subject to relevance scoring. We introduce a three-tier hierarchy:

**Tier 0: Constitutional Memory.** Always loaded every turn, never scored or evicted. This includes identity statements, behavioral rules, and operational constraints. These are loaded via a full `SELECT` query — not vector search — because they are always relevant by definition.

Example Tier 0 entries:
- "I have a SurrealDB graph database that stores memories across all past sessions"
- "I can extract causal patterns and learn reusable skills from successful tasks"
- "I should use my recall tool to check capabilities I'm unsure about"

**Tier 1: Session-Pinned Memory.** Loaded at session start, stays active until session ends. This includes working context and proactive thoughts generated during wake-up synthesis. Deactivated on graceful shutdown.

Example Tier 1 entries:
- "Currently investigating the flaky integration test in auth module"
- "User prefers concise responses without trailing summaries"

**Tier 2: Vector-Searched Memory.** Scored per turn via embedding similarity + graph traversal. Subject to budget enforcement and relevance decay. This is the vast majority of stored knowledge — conversation turns, concepts, consolidated memories, artifacts, skills, reflections.

**Why tiers matter.** The most common failure mode of RAG systems is retrieving irrelevant content alongside relevant content, then hoping the LLM sorts it out. By explicitly separating always-relevant (Tier 0), session-relevant (Tier 1), and query-relevant (Tier 2) memory, we guarantee that core context is present without consuming scoring budget, and that session-specific working context doesn't compete with historical memories for retrieval slots.

### 2.3 Hybrid Retrieval: Vector Search + Graph Traversal

Our retrieval pipeline combines two fundamentally different approaches:

**Vector search** provides semantic entry points. We embed the user's query using BGE-M3 (1024-dim, local inference, ~16ms) and search HNSW cosine indexes across multiple tables. This finds memories that are semantically similar to the current query.

**Graph expansion** finds structurally related memories. Starting from the top-K vector search results, we traverse typed edges 1-2 hops to find neighbors. This surfaces memories that are related but not necessarily similar in embedding space.

The combination is powerful because each compensates for the other's weakness:
- Vector search misses causally related memories when they use different vocabulary ("the auth middleware" vs "the token validation bug we fixed")
- Graph traversal misses semantically relevant memories that aren't directly connected (a concept about OAuth2 that was discussed in a different session)

Together, they produce a candidate set that captures both semantic similarity and structural relationships.

**Contextual query construction.** Before vector search, we blend the current query embedding with the embeddings of the last 3 conversation turns, weighted 2:1 in favor of the query. This creates a trajectory-aware query vector that considers conversational context, improving retrieval coherence across multi-turn interactions. The query gets 2x weight to prevent context from drowning out the current question.

### 2.4 Memory Formation Pipeline

Memory isn't just retrieved — it's formed. Our system creates new memories through multiple pathways:

**Synchronous (during agent loop):**
- Every conversation turn is stored with full metadata (session ID, role, token count, model, usage stats)
- Concept extraction identifies semantic entities mentioned in conversation
- Graph edges connect turns to sessions, concepts, and previous turns

**Asynchronous (worker thread):**
- A memory daemon running in a separate worker thread receives turn batches
- For each batch, it calls a mid-tier model (Sonnet) to extract:
  - **Causal chains** — trigger→outcome patterns with type (debug/refactor/feature/fix), success flag, and confidence score
  - **Monologues** — internal thought traces categorized as doubt, tradeoff, alternative, insight, or realization
  - **Resolved memories** — IDs of memories that have been fully addressed
- Extracted data is written to the graph asynchronously, never blocking conversation

**Post-session:**
- **Skill extraction** — If the session involved 5+ tool calls, Opus extracts a reusable procedure (name, preconditions, steps, postconditions)
- **Reflection generation** — If quality metrics are poor, Opus generates lessons learned
- **Causal chain graduation** — High-confidence causal chains are promoted to skills
- **Handoff note** — An exit summary is generated for the next session's wake-up synthesis

**Consolidation:**
- Old conversation turns are archived to cold storage
- High-frequency memories are compacted (summarized and merged)
- Low-utility memories decay via access count attenuation

This pipeline ensures that every interaction enriches the knowledge graph without blocking the conversation loop. The worker thread architecture means memory formation has zero latency impact on the user experience.

---

## 3. Adaptive Retrieval Orchestration

### 3.1 The Intent Classification Problem

Consider two user inputs to a coding agent:
1. "What's a linked list?"
2. "Fix the race condition in the connection pool where concurrent requests are getting stale database handles from the pool we refactored last week"

Input 1 is a simple factual question that requires zero retrieval. The LLM already knows what a linked list is. Any memory injection is pure waste.

Input 2 is a complex debugging task that benefits from deep retrieval: past conversation turns about the connection pool refactoring, the artifact record for the pool implementation file, any causal chains from previous pool-related debugging, and relevant reflections about concurrency issues.

Yet most RAG systems treat these identically — same retrieval depth, same number of results, same token budget for injected context.

### 3.2 Zero-Shot Intent Classification

We solve this with a lightweight intent classifier that runs before retrieval. The classifier uses the same BGE-M3 embedding model already loaded for retrieval, requiring no additional model loading.

**Method:** Embed the user's input and compute cosine similarity against pre-computed prototype centroids. Each intent category has 2-4 prototype sentences whose embeddings are averaged into a single centroid. Classification is the argmax over cosine similarities.

**Performance:** ~25ms total (16ms embedding + 5ms cosine computation + 4ms heuristics). This adds negligible latency to the turn pipeline.

**Categories and their retrieval profiles:**

| Category | Description | Tool Budget | Token Budget | Thinking |
|---|---|---|---|---|
| `simple-question` | Quick factual question | 3 | 2,000 | Low |
| `code-read` | Understanding existing code | 5 | 4,000 | Medium |
| `code-write` | Writing new code | 8 | 5,000 | High |
| `code-debug` | Debugging failures | 10 | 5,500 | High |
| `deep-explore` | Full codebase analysis | 12 | 6,000 | High |
| `reference-prior` | Recalling past conversations | 5 | 4,000 | Medium |
| `meta-session` | Session management | 3 | 2,000 | Low |
| `multi-step` | Complex multi-step tasks | 15 | 6,500 | High |
| `continuation` | Continuing previous work | 5 | 3,500 | Medium |

**Confidence threshold:** Below 0.40 cosine similarity, the system returns `unknown` and uses conservative defaults (tool limit 15, medium thinking). This prevents misclassification from causing under-retrieval on ambiguous inputs.

**Fast-path bypass:** Inputs shorter than 20 characters skip classification entirely and use defaults. This avoids wasting an embedding call on trivial inputs like "yes", "ok", or "thanks".

### 3.3 Per-Table Vector Search Limits

The adaptive config doesn't just set a global retrieval budget — it sets per-table vector search limits:

| Category | turn | identity | concept | memory | artifact |
|---|---|---|---|---|---|
| simple-question | 10 | 5 | 8 | 8 | 3 |
| code-read | 20 | 8 | 15 | 15 | 10 |
| code-write | 25 | 10 | 15 | 15 | 15 |
| code-debug | 30 | 12 | 20 | 20 | 15 |
| multi-step | 35 | 15 | 25 | 25 | 20 |

This granularity matters because different table types have different value densities. For a debugging task, past conversation turns and consolidated memories are high-value (they contain prior debugging context), while identity chunks are low-value (the agent already knows who it is via Tier 0). The per-table limits reflect this: `code-debug` gets 30 turn searches but only 12 identity searches.

### 3.4 Context Pressure Scaling

Long sessions accumulate tokens. A 30-turn debugging session might have consumed 200K input tokens. If retrieval budgets remain constant, the context window eventually fills with a mix of conversation history and injected memories, leaving insufficient room for new context.

Our solution: context budgets shrink proportionally as tokens accumulate. The system tracks cumulative input/output tokens across the session and applies a pressure multiplier that reduces retrieval budgets as the session progresses. Early turns get full retrieval depth; later turns get progressively tighter budgets, ensuring the agent remains responsive without overflowing the context window.

### 3.5 Predictive Prefetch

While the LLM processes the user's input through preflight classification, we fire speculative vector searches in the background.

**Query prediction** generates 2-4 follow-up queries from input patterns:
- Extract file paths (e.g., `src/auth/middleware.ts` → search for artifact memories about that file)
- Extract quoted/backtick terms (e.g., "the `refreshToken` function" → search for concept/memory nodes)
- Intent-specific patterns (code-debug → search for error messages and fix patterns; code-write → search for implementation patterns and test examples)

**LRU cache** stores the results: 10 entries, 5-minute TTL, cosine >0.85 hit threshold. When the main retrieval pipeline runs, it checks the cache first. A cache hit means the vector search result is already available — no database round-trip needed.

In coding-heavy sessions, prefetch reduces database round-trips by approximately 30%, with the reduction concentrated in the most latency-sensitive path (the first retrieval call after user input).

---

## 4. Learned Scoring

### 4.1 The Problem with Fixed Weights

The initial scoring function is a Weighted Mean Reciprocal (WMR) — a linear combination of independently computed signals. WMR has evolved through three versions:

```
WMR v3 (current):
finalScore = 0.22 * cosineSimilarity
           + 0.23 * recencyScore
           + 0.05 * importanceWeight
           + 0.05 * accessBoost
           + 0.10 * neighborBonus
           + 0.15 * provenUtility
           + 0.10 * reflectionBoost
           + 0.12 * keywordOverlap
           - utilityPenalty
```

The evolution from v1 (6 signals) to v3 (8 signals) reflects lessons from benchmarking: keyword overlap adds +2.6% R@5 on LongMemEval by catching exact-match cases that embeddings miss, and reflection boost rewards memories from sessions that produced metacognitive lessons. Utility penalties (-0.15 for <5% utilization, -0.06 for <15%) actively demote proven-bad memories.

These weights work well as a starting point. But they have fundamental limitations:

- **Static.** The weights cannot adapt to different users, projects, or workflows. A user who primarily asks factual questions benefits from higher importance weighting. A user doing exploratory debugging benefits from higher recency weighting. Fixed weights serve both adequately but neither optimally.

- **Linear.** The signals are combined linearly, but their interactions may be nonlinear. A memory that is both highly recent and from a graph neighbor may be more valuable than the linear sum suggests — the recency confirms the neighbor connection is fresh and relevant.

- **No cross-attention.** Cosine similarity between the query and a memory is computed as a flat dot product. But different parts of the query may be relevant to different parts of the memory. A cross-attention mechanism can learn to focus on the most relevant dimensions.

### 4.2 ACAN: Attentive Cross-Attention Network

We introduce a lightweight learned scoring model that progressively replaces WMR:

**Architecture:**

```
queryEmbedding (1024-dim)                  candidateEmbedding (1024-dim)
        │                                            │
   ┌────┴────┐                                 ┌────┴────┐
   │  W_q    │  (1024 → 64)                    │  W_k    │  (1024 → 64)
   └────┬────┘                                 └────┬────┘
        │                                            │
        q (64-dim)                              k (64-dim)
        │                                            │
        └───────────────┬────────────────────────────┘
                        │
               attnLogit = q · k / √64
                        │
        ┌───────────────┼───────────────────────┐
        │               │                       │
   [attnLogit, recency, importance, access, neighbor, utility, reflection, keyword]
        │               (8-dim feature vector)
        │
   ┌────┴────┐
   │ W_final │  (8 → 1) + bias
   └────┬────┘
        │
   finalScore (scalar)
```

**Key design decisions:**

- **No softmax.** Unlike standard attention, we compute a raw logit (`q · k / √64`) without softmax normalization. This makes the score candidate-count invariant — adding or removing candidates from the batch doesn't change existing scores. This is important because the number of retrieval candidates varies per turn.

- **No value projection.** Standard attention has Q, K, and V projections. We drop V because we don't need a weighted output vector — we need a scalar score. The attention logit itself becomes a feature alongside handcrafted signals.

- **Hybrid features.** The 8-dim feature vector combines the learned attention logit with handcrafted signals (recency, importance, access count, neighbor bonus, proven utility, reflection boost, keyword overlap). This lets the model leverage both learned cross-attention and well-understood heuristic signals. Keyword overlap, in particular, catches exact-match cases that embedding similarity alone misses.

- **~130K parameters.** W_q (1024×64) + W_k (1024×64) + W_final (8) + bias = 131,081 parameters. This is small enough to train in seconds and store in a few hundred KB of JSON.

### 4.3 Self-Training from Telemetry

ACAN doesn't require external supervision — it trains on the system's own retrieval telemetry.

**Quality signals per retrieved item:**

| Signal | Description | Source |
|---|---|---|
| Utilization | Text overlap between injected context and LLM response | Post-response trigram/unigram analysis |
| Tool success | Whether tools executed after retrieval succeeded | Tool execution outcome tracking |
| Context tokens | Tokens consumed by this item | Token counting during context injection |
| Was neighbor | Came from graph expansion, not vector search | Retrieval pipeline flag |
| Recency | Exponential time decay from creation | Timestamp-based computation |
| LLM relevance | Optional model-judged relevance score | Cognitive check evaluation |

These signals are stored in a `retrieval_outcome` table after every turn. Each record captures one (query, memory, outcome) triple.

**Training trigger:** When `retrieval_outcome` accumulates 5,000+ records, the system auto-trains ACAN using TypeScript SGD with manual backprop in a worker thread. Training takes 10-30 seconds on the accumulated dataset. Retraining triggers when sample count grows by 50% or weights are older than 7 days.

**The feedback loop:**

```
Retrieve memories → Inject into context → Agent responds → Measure utility
        ↑                                                          │
        │                                                          ▼
   Score with ACAN ← ← ← ← ← ← ← ← ← ← ← ← ← ← Train on outcomes
```

This creates a virtuous cycle: better scoring produces better retrieval, which produces more informative training data, which produces better scoring. The system improves with use without any human supervision or external labeling.

### 4.4 Graceful Transition

The transition from fixed to learned scoring is designed to be invisible:

1. **Day 1:** System starts with WMR (fixed weights). Works immediately, no training data needed.
2. **Days 1-N:** Retrieval telemetry accumulates passively during normal use. Zero user effort.
3. **Day N (1000+ samples):** ACAN auto-trains at session start. Takes 10-30 seconds.
4. **Day N+:** ACAN scoring replaces WMR. If ACAN produces an error, system falls back to WMR silently.
5. **Ongoing:** Weights are persisted to `~/.kongclaw/acan_weights.json` and loaded at startup. Retrained periodically as new telemetry accumulates.

The user never needs to know the transition happened. They may notice improved retrieval relevance — fewer irrelevant chunks in context, more useful memories surfacing — but the mechanism is entirely transparent.

---

## 5. Cross-Encoder Reranking

### 5.1 Beyond Embedding Similarity

Vector search with cosine similarity is a bi-encoder approach: query and document are encoded independently, and their interaction is limited to a dot product in embedding space. This is fast (O(1) per comparison with HNSW) but loses fine-grained semantic relationships.

Cross-encoder reranking addresses this by jointly encoding query-document pairs through a full transformer forward pass. The model sees both texts simultaneously and can attend across them, understanding relationships that cosine similarity misses — paraphrasing, implicit references, negation, and context-dependent meaning.

### 5.2 Two-Stage Retrieve-Then-Rerank

Our pipeline uses a two-stage architecture:

**Stage 1: Candidate generation.** BGE-M3 embedding search + graph expansion + tag-boosted concept retrieval produce a broad candidate set. WMR v3 scoring ranks and deduplicates these candidates.

**Stage 2: Cross-encoder reranking.** The top 30 candidates are rescored using bge-reranker-v2-m3, a purpose-built cross-encoder model that runs locally via node-llama-cpp's `LlamaRankingContext`. Scores are blended 60/40 (WMR:cross-encoder) to preserve the WMR's recency, utility, and graph signals that the cross-encoder cannot see.

```
vectorSearch (6 tables)  ──┐
tagBoostedConcepts ────────┤──→ combine → WMR v3 score → dedup → rerank top-30 → budget trim
graphExpand (1-2 hops) ────┤                                         │
causalContext ─────────────┘                                    bge-reranker-v2-m3
                                                              (60% WMR + 40% cross-encoder)
```

### 5.3 Benchmark Results

Evaluated on LongMemEval (500 questions, 6 types) — the standard academic benchmark for AI memory retrieval:

| Configuration | R@5 | Cost |
|---|---|---|
| BGE-M3 raw cosine | 92.0% | $0 |
| + keyword overlap (hybrid) | 94.6% | $0 |
| + cross-encoder rerank (top-30) | **98.2%** | $0 |
| MemPalace raw (MiniLM via ChromaDB) | 96.6% | $0 |
| MemPalace + Haiku LLM rerank | 100% | ~$0.001/q |

The two-stage pipeline beats MemPalace's raw baseline by +1.6% R@5 with zero API calls and fully local inference (~0.84s per query).

### 5.4 Design Decisions

**Top-30 is the sweet spot.** With ~53 candidate sessions per query in LongMemEval, top-30 (57% of corpus) provides the optimal signal-to-noise ratio for the cross-encoder. Top-20 drops to 97.6% (misses some correct answers). Top-50 drops to 91.2% (reranker drowns in noise).

**Blending preserves production signals.** Pure cross-encoder replacement (100% reranker score) achieves the same 98.2% on the benchmark but would lose WMR's recency, proven utility, and graph neighbor signals in production. The 60/40 blend ensures that a memory's track record and temporal context still influence ranking.

**Graceful degradation.** If the reranker model is not present, the pipeline works without it — WMR scoring alone is the fallback. The reranker is auto-detected at startup from `~/.node-llama-cpp/models/bge-reranker-v2-m3-Q8_0.gguf`.

---

## 6. The Metacognitive Loop

### 6.1 Cognitive Checks

Most agent systems treat retrieval as a one-shot pipeline: retrieve → inject → respond. There is no feedback during the turn about whether the retrieved context is useful, contradictory, or redundant.

We introduce periodic cognitive checks: a lightweight model (Haiku, ~300ms, ~$0.001 per check) evaluates the retrieved context against the conversation state.

**Frequency:** Turn 2, then every 3 turns (2, 5, 8, 11...). This balances insight against cost.

**Evaluation targets:**
- **Retrieval grading** — Each injected memory is rated relevant/irrelevant with a 0-1 score and a reason
- **Session continuity** — Is this turn a continuation, repetition, new topic, or tangent?
- **User preference detection** — Has the user expressed preferences about communication style, depth, or approach?

**Directive generation.** The cognitive check produces typed directives that are injected into the next turn's context:

| Directive Type | Meaning | Example |
|---|---|---|
| `repeat` | User is repeating a question — previous answer was inadequate | "Provide more detail about the auth flow" |
| `continuation` | User is continuing a thread — maintain context | "Keep the connection pool context loaded" |
| `contradiction` | Retrieved memories contradict each other | "Session 12 says use JWT; Session 15 says use sessions" |
| `noise` | Retrieved memory is irrelevant to current task | "The Python memory about list comprehensions is not relevant to this TypeScript task" |
| `insight` | An important pattern or connection was noticed | "User has asked about auth 3 times — may be a recurring pain point" |

These directives steer agent behavior without requiring explicit rules. The metacognitive layer observes and advises; the agent loop incorporates the advice naturally.

### 6.2 Causal Chain Extraction

Human experts remember not just what happened, but what caused what. "I changed the database connection timeout, and that fixed the intermittent failures" is a causal chain — trigger (timeout change) → outcome (failures fixed). This causal structure is the basis of debugging expertise.

Our memory daemon extracts causal chains from every conversation turn batch:

**Structure:**
```
{
  trigger: "Changed connection pool timeout from 5s to 30s",
  outcome: "Intermittent database connection failures stopped",
  type: "fix",
  success: true,
  confidence: 0.85
}
```

**Graph edges created:**
- `caused_by` — outcome was caused by trigger
- `supports` — when outcome confirms trigger (success path)
- `contradicts` — when outcome contradicts trigger (failure path)

**Why this matters for retrieval.** When a similar database connection issue arises in a future session, vector search may find the outcome memory ("intermittent failures stopped"). Graph traversal then follows the `caused_by` edge to find the trigger memory ("changed timeout from 5s to 30s") — providing the actual solution, not just a description of the problem being solved.

This is something flat vector stores fundamentally cannot do. The solution and the problem may have completely different embedding representations. Only the structural relationship connects them.

### 6.3 Skill Extraction

Skills are the highest-level form of memory — reusable procedures extracted from successful task completion.

**Extraction trigger:** A session that involved 5+ tool calls (indicating a non-trivial multi-step task) and completed successfully.

**Structure:**
```
{
  name: "Debug auth middleware failures",
  preconditions: "User reports authentication failures; auth middleware exists in codebase",
  steps: [
    { tool: "grep", description: "Search for error messages in middleware", argsPattern: "grep -n 'auth|token|401' src/auth/**" },
    { tool: "read", description: "Read the middleware implementation", argsPattern: "read src/auth/middleware.ts" },
    { tool: "bash", description: "Run auth-related tests", argsPattern: "npm test -- --grep auth" },
    { tool: "edit", description: "Fix the identified issue", argsPattern: "edit src/auth/middleware.ts" },
    { tool: "bash", description: "Verify fix with tests", argsPattern: "npm test -- --grep auth" }
  ],
  postconditions: "Auth tests pass; error messages resolved",
  success_count: 3,
  failure_count: 0,
  avg_duration_ms: 45000
}
```

**Skill graduation.** High-confidence causal chains (confidence > 0.8, success = true) are automatically promoted to skills. This means the agent builds its skill library from experience, not from explicit programming.

**Skill retrieval.** Skills have their own vector embedding and HNSW index. When a new task arrives, relevant skills surface alongside other memories. The agent sees "I've done this before — here's how" rather than discovering the approach from scratch.

**Success/failure tracking.** Each time a skill is applied and the outcome is observable, the success/failure counts are updated. Skills with high failure rates decay in retrieval priority. Skills with high success rates grow more prominent. This creates a natural selection pressure: effective procedures survive; ineffective ones fade.

### 6.4 Metacognitive Reflection

At session end, the system evaluates its own performance and generates lessons.

**Trigger conditions:**
- Average retrieval utilization < 20% (most retrieved context was unused)
- Tool failure rate > 20% (tools were frequently used incorrectly)
- Steering candidates detected (the orchestrator identified conflicting tool calls)
- Wasted tokens exceed threshold (significant context window space consumed by irrelevant content)

**Reflection structure:**
```
{
  text: "Spent 3 tool calls reading files that weren't relevant to the auth issue. Should have used grep to narrow down before reading.",
  category: "efficiency",
  severity: "moderate",
  importance: 7.0
}
```

**Categories:** `failure_pattern` (what went wrong), `efficiency` (what could be more efficient), `approach_strategy` (what approach would have been better).

**How reflections improve future performance.** Reflections are stored as high-importance memories with their own vector embeddings. When a similar situation arises in a future session, the reflection surfaces in retrieval: "Last time I debugged auth, I wasted tool calls reading irrelevant files. Use grep first." The agent doesn't change its weights or behavior rules — but it receives better context, which produces better behavior.

This is **self-improvement without fine-tuning**. The model's parameters never change. What changes is the information it receives, curated by its own metacognitive evaluation.

### 6.5 Concept Supersedes

When the memory daemon extracts a correction (user correcting the assistant), the supersedes system finds concepts that match the *original* (wrong) knowledge and actively deprioritizes them:

1. Embed the original (incorrect) text
2. Find concepts with cosine similarity > 0.70 to the wrong statement
3. Create `supersedes` edges: `correction_memory → supersedes → stale_concept`
4. Decay concept stability by 60% (multiplicative factor 0.4, floor 0.15)
5. Mark concept with `superseded_at` timestamp

Superseded concepts are filtered from vectorSearch and tag-boosted retrieval entirely (`WHERE superseded_at IS NONE`). This ensures corrections don't just compete with stale knowledge — they replace it structurally in the graph.

**Why this matters.** Without supersedes, a high-cosine stale concept can outrank a lower-cosine correction indefinitely. With supersedes, the correction wins not by scoring higher but by removing its competitor from the candidate pool. The graph records *why* the concept was deprecated (the supersedes edge), preserving the historical chain.

### 6.6 The Soul: Emergent Identity

We introduce a graduation system that tracks the accumulation of experiential data across sessions. Graduation requires crossing multiple thresholds simultaneously:

| Signal | Threshold | Why |
|---|---|---|
| Sessions | ≥ 15 | Sufficient interaction history |
| Reflections | ≥ 10 | Self-awareness capacity demonstrated |
| Causal chains | ≥ 5 | Causal reasoning engaged |
| Concepts | ≥ 30 | Rich semantic knowledge base |
| Memory compactions | ≥ 5 | Memory consolidation active |
| Monologues | ≥ 5 | Internal thought processes engaged |
| Time span | ≥ 3 days | Temporal depth of experience |

Once all thresholds are met, the system generates a "Soul document" — a self-authored identity statement synthesized from the agent's accumulated experience. This is not a programmed personality; it is a personality derived from patterns in the agent's own behavior, preferences, and reflections.

The philosophical implication is worth stating directly: an agent that has debugged hundreds of bugs, reflected on its own failures, extracted reusable skills, and developed a consistent communication style has a form of experiential identity that is meaningfully different from a prompted persona. Whether this constitutes genuine selfhood is a question we leave to philosophers. What we observe is that graduated agents exhibit more consistent behavior, more accurate self-descriptions, and higher user satisfaction than non-graduated agents with static personas.

---

## 7. Constitutive Memory

### 7.1 The Wake-Up Problem

Even with persistent memory, the first turn of a new session is uniquely difficult. The agent must answer two questions before it can be useful:

1. **Who am I?** — What is my identity, personality, communication style, and set of capabilities?
2. **Where was I?** — What was I working on, what was the state of the project, what open threads exist?

RAG systems handle question 1 poorly (identity chunks compete with other memories for retrieval slots) and question 2 not at all (there's no mechanism to summarize the state at session start).

### 7.2 Wake-Up Synthesis

At session start, before the user speaks, the system gathers four inputs:

1. **Handoff note** — The exit summary from the previous session, written by the agent itself
2. **Identity chunks** — Core identity statements (hardcoded + user-defined)
3. **Recent monologues** — The 5 most recent internal thought traces
4. **Depth signals** — Session count, memory count, monologue count, span days

These are fed to Opus with a synthesis prompt: "Write a first-person briefing of what you remember, what you were working on, and what you were thinking about."

The resulting briefing is injected into the system prompt as a `[CONSTITUTIVE MEMORY]` block:

```
[CONSTITUTIVE MEMORY — This is what you remember from before this session started]

I've been working with the user on their authentication system for the past three days.
Yesterday we refactored the connection pool to handle concurrent requests better —
the timeout was the root cause, not the pool size as we initially thought. I noticed
the user prefers concise responses and gets frustrated when I repeat context they
already know. There are still intermittent failures in the refresh token rotation
that we haven't investigated yet.
```

The agent's first response already reflects this context. The user experiences continuity — the agent picks up where it left off — rather than the blank-slate amnesia of stateless systems.

### 7.3 Handoff Notes

The wake-up synthesis is only as good as the handoff note from the previous session. At shutdown, the system generates an exit summary that captures:

- What was accomplished
- What open threads remain
- Key decisions made and their rationale
- Any observations about user preferences or communication patterns

This creates a chain of continuity: session N generates a handoff note → session N+1's wake-up synthesis reads it → session N+1 generates its own handoff note → session N+2 benefits from both.

Over time, the handoff chain becomes a compressed narrative of the agent's experience — not raw conversation logs, but curated summaries of what matters for the next session.

### 7.4 Startup Cognition

Beyond the briefing, we generate **proactive thoughts** at session start. Based on depth signals (how much experience the agent has), the system generates thoughts about:

- Patterns observed across recent sessions
- Open questions or investigations
- Hypotheses about the user's current priorities

These are pinned as Tier 1 core memory for the session — they persist throughout but are deactivated at shutdown. The effect is that the agent arrives with initiative: "I notice we've been working on auth for three sessions. The refresh token rotation is still broken — should we investigate that?"

The agent thinks before the user speaks.

---

## 8. Subagent Architecture

### 8.1 Full Mode: Shared Memory

A full-mode subagent shares the parent's database (same namespace and database). It has full read/write access to the entire knowledge graph. Memories created by the subagent are immediately available to the parent and all other full-mode subagents.

**Use case:** Delegated tasks where the results should contribute to shared knowledge. "Research the best approach for token rotation and report back."

### 8.2 Incognito Mode: Isolated Memory

An incognito subagent operates on a completely isolated database. It cannot read the parent's memories, and its own memories are stored separately. After completion, the user can selectively merge results back via an explicit merge command.

**Use case:** Exploratory or risky tasks where failure shouldn't pollute the main graph. "Try rewriting the auth module using a different approach — I want to see if it works, but don't contaminate the memory graph if it fails."

### 8.3 Memory Isolation as Architecture

The full/incognito distinction is not just a privacy feature — it's an architectural decision about knowledge contamination.

When an agent explores a hypothesis that turns out to be wrong, the exploration creates memories: "I tried approach X and it seemed promising." If those memories are in the main graph, they will surface in future retrieval, potentially leading the agent down the same wrong path again. Worse, the "seemed promising" framing may override the actual failure outcome in retrieval scoring.

Incognito mode solves this by isolating the experiment. If it succeeds, the user merges the results. If it fails, the memories are discarded — the main graph never learns the wrong lesson.

This is analogous to a scientist keeping a separate lab notebook for speculative experiments, only transferring confirmed results to the official record.

---

## 9. Evaluation and Telemetry

### 9.1 What We Measure

Our telemetry system captures three levels of measurement:

**Per-item metrics (retrieval_outcome):**
- Utilization: Did the agent actually reference this memory in its response?
- Tool success: Did tools executed after retrieval succeed?
- Context tokens: How many tokens did this item consume?
- Was neighbor: Did this come from graph expansion or vector search?

**Per-turn metrics (orchestrator_metrics):**
- Intent classification (category, confidence)
- Budget allocation (tool limit, token budget, thinking level)
- Actual usage (tool calls made, tokens consumed)
- Timing (preflight ms, total turn duration)
- Steering candidates detected

**Per-session aggregates:**
- Average retrieval utilization across all turns
- Total wasted tokens (injected but unused context)
- Tool success rate
- Compression ratio (how much context injection saved vs full history replay)

### 9.2 Utilization: The Key Metric

We define utilization as the degree to which injected context appears in the agent's response. This is measured through text overlap analysis:

1. Extract key terms from the injected memory (removing stop words and common code tokens)
2. Check for trigram and unigram matches between the memory and the response
3. Weight by term rarity (rare terms matching is more significant than common terms)
4. Produce a 0-1 utilization score

A score of 0.0 means the memory was completely ignored. A score of 1.0 means the response directly referenced the memory's content.

**Why utilization matters:** It directly measures the value of retrieval. A system with 90% average utilization is injecting almost exclusively useful context. A system with 10% utilization is wasting 90% of its retrieval on noise — noise that actively degrades response quality by competing for attention.

### 9.3 Comparative Evaluation

The system includes built-in evaluation commands:

- `/eval` — Run a predefined suite of prompts with and without graph context, measuring token savings and response quality
- `/compare <prompt>` — Compare a single prompt's performance with full vs minimal context

These enable empirical measurement of the memory system's impact. In practice, graph context reduces token usage by 30-60% on continuation tasks (where prior conversation is relevant) while maintaining or improving response quality.

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

**Partial retrieval gating.** The intent classifier suppresses retrieval for `continuation` and `meta-session` intents, and a fast-path bypass skips classification for trivial inputs (<20 chars). However, "simple-question" classification still triggers reduced vector search rather than eliminating it entirely.

**Cold start for ACAN.** The learned scoring model requires 5,000+ retrieval outcome samples before training. Until that threshold is reached, the system operates on WMR v3 fixed weights. For a new deployment, this means potentially hundreds of sessions before the scoring model activates. The cross-encoder reranker provides immediate quality improvement while ACAN accumulates data.

**Single-node architecture.** The knowledge graph resides on a single SurrealDB instance. There is no replication, no distributed retrieval, and no mechanism for multiple agents to share a graph across network boundaries.

**GGUF embedding quality gap.** BGE-M3 via node-llama-cpp GGUF produces lower-quality embeddings than the same model via sentence-transformers/ONNX (92% vs ~96.6% R@5 on LongMemEval). The cross-encoder reranker compensates (pushing to 98.2%), but the first-stage recall ceiling is limited by the GGUF embedding pipeline. Switching to ONNX-based embeddings would improve the first-stage candidate quality.

**No active forgetting.** The graph only grows. Proven-bad memory suppression filters chronically unused memories from scoring, but doesn't delete them. Over months of heavy use, vectorSearch across tens of thousands of nodes will accumulate latency.

### 10.2 Implemented Since v1

Several items from the original future work have been implemented:

- **Real-time quality suppression** — Proven-bad memory pre-filtering (5+ retrievals with <5% utilization → filtered from candidate pool) and tiered utility penalties in WMR scoring (-0.15/-0.06).
- **Retrieval gating** — `skipRetrieval` for continuation, meta-session, and high-confidence simple-question intents (with memory reference detection override).
- **Cross-encoder reranking** — Two-stage retrieve-then-rerank pipeline achieving 98.2% R@5 on LongMemEval.
- **Concept evolution** — Supersedes system for correction-driven knowledge decay.
- **Keyword overlap scoring** — Catches exact-match cases embeddings miss.
- **Embedding adaptation** — ACAN self-trains on retrieval outcome telemetry (5,000+ samples), replacing fixed WMR weights with a learned cross-attention scorer.
- **Active forgetting** — Two-tier garbage collection: stale never-accessed memories (>14 days, low importance) and proven-useless memories (5+ retrievals, <2% utilization). Cascade edge cleanup prevents orphaned graph relations. Superseded concepts with floor stability are also pruned.
- **Temporal reasoning** — Time-filtered vector search via `timeRange` parameter. Natural language temporal patterns ("yesterday", "last week", "3 days ago") are auto-detected in preflight and constrain retrieval to the relevant time window.
- **Multi-agent attribution** — `agent_id` field on memory, concept, and skill tables for tracking which agent created each record. Incognito subagents auto-merge on successful completion (importance >= 3.0, cosine dedup 0.9, max 50 nodes).

### 10.3 Future Directions

**ONNX embedding backend.** Replace GGUF/node-llama-cpp with @huggingface/transformers (ONNX Runtime) for embedding inference. This would match sentence-transformers quality (~96.6% first-stage R@5) and combined with the cross-encoder reranker could push toward 99%+ R@5.

**Distributed multi-agent memory.** The current architecture supports multi-agent sharing on a single SurrealDB instance. True distributed operation would require replication across network boundaries, conflict resolution for simultaneous writes, and agent-scoped retrieval filtering.

**Domain-specific embedding fine-tuning.** Fine-tune BGE-M3 on the retrieval outcome data using contrastive learning. Positive outcomes (high utilization) provide positive training signal; negative outcomes provide negative signal. This would make the embedding space itself adapt to what the user cares about.

---

## 11. Conclusion

The dominant paradigm for LLM agent memory — flat vector stores with fixed retrieval — is fundamentally insufficient for agents that need to learn, adapt, and maintain identity across sessions. We have presented an alternative architecture built on five interlocking mechanisms:

1. **A knowledge graph** that captures not just content but relationships — causality, support, contradiction, hierarchy — enabling retrieval that follows the structure of knowledge rather than just its surface similarity.

2. **Adaptive orchestration** that matches retrieval depth to task complexity, preventing both under-retrieval (missing relevant context for complex tasks) and over-retrieval (injecting noise for simple tasks).

3. **A self-training scoring model** that starts with hand-tuned weights and progressively learns from its own retrieval telemetry, creating a virtuous cycle of improving retrieval quality without human supervision.

4. **A metacognitive loop** that extracts causal patterns, reusable skills, and performance reflections from the agent's own behavior, enabling self-improvement without fine-tuning.

5. **Constitutive memory** that synthesizes a first-person briefing at session start, providing identity continuity and initiative that stateless agents cannot achieve.

Together, these mechanisms shift the paradigm from "agents with tools" to "agents with memory" — systems that accumulate expertise, learn from mistakes, build procedural knowledge, and maintain coherent identity across arbitrarily many sessions.

The key insight underlying the entire architecture is that memory is not a feature — it is infrastructure. Just as a database is not optional for a web application, persistent memory is not optional for an agent that needs to be useful across more than one session. The question is not whether agents need memory, but what kind of memory serves them best.

We believe the answer is structured, adaptive, self-improving, and metacognitive. The architecture presented here is one realization of that answer. We hope it serves as a foundation for others.

---

## Appendix A: System Configuration

### Key Thresholds

| Parameter | Value | Component |
|---|---|---|
| Intent confidence threshold | 0.40 | `intent.ts` |
| Fast-path input length | < 20 chars | `orchestrator.ts` |
| ACAN training threshold | 5000+ samples | `acan.ts` |
| ACAN projection dimension | 64 | `acan.ts` |
| ACAN feature dimensions | 8 | `acan.ts` |
| ACAN total parameters | ~131K | `acan.ts` |
| Reranker top-N | 30 candidates | `graph-context.ts` |
| Reranker blend | 60% WMR / 40% cross-encoder | `graph-context.ts` |
| Supersede threshold | cosine > 0.70 | `supersedes.ts` |
| Stability decay factor | 0.4 (floor 0.15) | `supersedes.ts` |
| Embedding LRU cache | 512 entries | `embeddings.ts` |
| Cognitive check frequency | Turn 2, then every 3 | `cognitive-check.ts` |
| Reflection utilization trigger | < 20% | `reflection.ts` |
| Reflection tool failure trigger | > 20% | `reflection.ts` |
| Prefetch cache size | 10 entries | `prefetch.ts` |
| Prefetch cache TTL | 5 minutes | `prefetch.ts` |
| Prefetch hit threshold | cosine > 0.85 | `prefetch.ts` |
| Recency decay (fast) | 0.995^hours | `graph-context.ts` |
| Skill extraction threshold | 5+ tool calls | `skills.ts` |
| Causal chain graduation confidence | > 0.8 | `skills.ts` |
| Soul graduation: sessions | ≥ 15 | `soul.ts` |
| Soul graduation: reflections | ≥ 10 | `soul.ts` |
| Soul graduation: causal chains | ≥ 5 | `soul.ts` |
| Soul graduation: concepts | ≥ 30 | `soul.ts` |
| Soul graduation: time span | ≥ 3 days | `soul.ts` |
| Message pruning (TUI) | 150 components | `tui.ts` |

### Environment Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API access |
| `SURREAL_URL` | `ws://localhost:8042/rpc` | SurrealDB endpoint |
| `SURREAL_NS` / `SURREAL_DB` | `zera` / `memory` | Database namespace |
| `EMBED_MODEL_PATH` | `~/.node-llama-cpp/models/bge-m3-q4_k_m.gguf` | Embedding model |
| `RERANKER_MODEL_PATH` | `~/.node-llama-cpp/models/bge-reranker-v2-m3-Q8_0.gguf` | Cross-encoder reranker (optional) |
| `ZERACLAW_MODEL` | `claude-opus-4-6` | LLM model |

---

## Appendix B: Knowledge Graph Schema

### Node Tables with Vector Indexes

| Table | Vector Index | Dimensions | Distance |
|---|---|---|---|
| `turn` | `turn_vec_idx` | 1024 | Cosine |
| `identity_chunk` | `identity_vec_idx` | 1024 | Cosine |
| `concept` | `concept_vec_idx` | 1024 | Cosine |
| `memory` | `memory_vec_idx` | 1024 | Cosine |
| `artifact` | `artifact_vec_idx` | 1024 | Cosine |
| `skill` | `skill_vec_idx` | 1024 | Cosine |
| `reflection` | `reflection_vec_idx` | 1024 | Cosine |
| `monologue` | `monologue_vec_idx` | 1024 | Cosine |

### Edge Types

**Turn-level:** `responds_to`, `tool_result_of`, `part_of`, `mentions`

**5-Pillar:** `performed`, `owns`, `task_part_of`, `session_task`, `produced`, `derived_from`, `relevant_to`, `used_in`

**Knowledge:** `narrower`, `broader`, `related_to`, `caused_by`, `supports`, `contradicts`, `supersedes`, `about_concept`, `artifact_mentions`

**Procedural:** `skill_from_task`, `skill_uses_concept`, `spawned`, `reflects_on`, `summarizes`

---

*Implementation: [github.com/42U/kongclaw](https://github.com/42U/kongclaw) — TypeScript, SurrealDB, BGE-M3, Claude*
