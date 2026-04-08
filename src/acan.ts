/**
 * ACAN — Attentive Cross-Attention Network for learned memory scoring.
 *
 * Replaces the fixed 6-signal WMR weights in scoreResults() with a learned
 * cross-attention model. Ships dormant — auto-trains and activates when
 * enough retrieval outcome data accumulates (5000+ labeled pairs).
 *
 * Architecture (simplified from PROPOSE.md — W_v dropped):
 *   q = W_q · queryEmbedding     (1024 → 64)
 *   k = W_k · candidateEmbedding (1024 → 64)
 *   attnLogit = q · k / √64      (raw, no softmax — candidate-count invariant)
 *   features = [attnLogit, recency, importance, access, neighbor, utility]
 *   finalScore = W_final · features + bias
 *
 * Training: pure TypeScript SGD with manual backprop. ~130K params, trains
 * in ~10-30s on 5000+ samples. Runs automatically at session start.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { existsSync, mkdirSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { getDb, isSurrealAvailable, queryFirst, assertRecordId } from "./surreal.js";
import { swallow } from "./errors.js";

// ── Types ──

export interface ACANWeights {
  W_q: number[][];   // [1024][64] query projection
  W_k: number[][];   // [1024][64] key projection
  W_final: number[]; // [8] final linear layer
  bias: number;
  version: number;
  trainedAt?: number;         // epoch ms when last trained
  trainedOnSamples?: number;  // sample count used for training
}

export interface ACANCandidate {
  embedding: number[];
  recency: number;
  importance: number;
  access: number;
  neighborBonus: number;
  provenUtility: number;
  reflectionBoost?: number;
  keywordOverlap?: number;
}

interface TrainingSample {
  query_embedding: number[];
  memory_embedding: number[];
  retrieval_score: number;
  was_neighbor: boolean;
  utilization: number;
  importance: number;     // 0-1 normalized
  access_count: number;   // 0-1 normalized
  recency: number;        // 0-1 exponential decay
}

// ── Module state ──

let _weights: ACANWeights | null = null;
let _active = false;

const ATTN_DIM = 64;
const EMBED_DIM = 1024;
const FEATURE_COUNT = 8;
const WEIGHTS_FILENAME = "acan_weights.json";

function getZeraDir(): string {
  const dir = join(homedir(), ".kongclaw");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  return dir;
}

// ── Weight loading / saving ──

export function loadWeights(path: string): ACANWeights | null {
  try {
    if (!existsSync(path)) return null;
    const raw = JSON.parse(readFileSync(path, "utf-8"));

    if (raw.version !== 1) return null;
    if (!Array.isArray(raw.W_q) || raw.W_q.length !== EMBED_DIM) return null;
    if (!Array.isArray(raw.W_k) || raw.W_k.length !== EMBED_DIM) return null;
    if (!Array.isArray(raw.W_final) || raw.W_final.length !== FEATURE_COUNT) return null;
    if (typeof raw.bias !== "number") return null;

    // Validate inner dimensions
    if (raw.W_q[0].length !== ATTN_DIM || raw.W_k[0].length !== ATTN_DIM) return null;

    // Validate numeric integrity — NaN/Infinity from corrupted training propagates silently
    if (!Number.isFinite(raw.bias)) return null;
    if (!raw.W_final.every((v: number) => Number.isFinite(v))) return null;
    // Spot-check W_q/W_k at start, middle, end rows
    for (const matrix of [raw.W_q, raw.W_k]) {
      for (const idx of [0, Math.floor(EMBED_DIM / 2), EMBED_DIM - 1]) {
        if (!matrix[idx].every((v: number) => Number.isFinite(v))) return null;
      }
    }

    return raw as ACANWeights;
  } catch (e) {
    swallow("acan:return null;", e);
    return null;
  }
}

function saveWeights(weights: ACANWeights, path: string): void {
  const dir = join(path, "..");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(path, JSON.stringify(weights), "utf-8");
}

export function initACAN(weightsDir?: string): boolean {
  const dir = weightsDir ?? getZeraDir();
  const path = join(dir, WEIGHTS_FILENAME);
  _weights = loadWeights(path);
  _active = _weights !== null;
  return _active;
}

export function isACANActive(): boolean {
  return _active;
}

// Expose for testing
export function _getWeights(): ACANWeights | null {
  return _weights;
}

// ── Linear algebra primitives ──

export function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function projectVec(vec: number[], matrix: number[][]): number[] {
  const out = new Array(matrix[0].length).fill(0);
  for (let i = 0; i < vec.length; i++) {
    if (vec[i] === 0) continue; // skip zero elements
    const row = matrix[i];
    for (let j = 0; j < out.length; j++) {
      out[j] += vec[i] * row[j];
    }
  }
  return out;
}

export function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// ── ACAN inference ──

export function scoreWithACAN(queryEmbedding: number[], candidates: ACANCandidate[]): number[] {
  if (!_weights || candidates.length === 0) return [];

  // Project query into attention space
  const q = projectVec(queryEmbedding, _weights.W_q); // [64]

  // Compute raw attention logits for each candidate (candidate-count invariant)
  const scale = Math.sqrt(ATTN_DIM);
  const scores: number[] = [];
  for (const cand of candidates) {
    const k = projectVec(cand.embedding, _weights.W_k); // [64]
    const attnLogit = dot(q, k) / scale;

    const features = [
      attnLogit,
      cand.recency,
      cand.importance,
      cand.access,
      cand.neighborBonus,
      cand.provenUtility,
      cand.reflectionBoost ?? 0,
      cand.keywordOverlap ?? 0,
    ];
    scores.push(dot(features, _weights.W_final) + _weights.bias);
  }

  return scores;
}

// ── In-process training (pure TypeScript SGD with manual backprop) ──

function initRandomWeights(): ACANWeights {
  // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
  const xavierQK = Math.sqrt(2 / (EMBED_DIM + ATTN_DIM));
  const xavierFinal = Math.sqrt(2 / (FEATURE_COUNT + 1));

  const W_q: number[][] = [];
  const W_k: number[][] = [];
  for (let i = 0; i < EMBED_DIM; i++) {
    W_q.push(Array.from({ length: ATTN_DIM }, () => (Math.random() * 2 - 1) * xavierQK));
    W_k.push(Array.from({ length: ATTN_DIM }, () => (Math.random() * 2 - 1) * xavierQK));
  }
  const W_final = Array.from({ length: FEATURE_COUNT }, () => (Math.random() * 2 - 1) * xavierFinal);
  // Initialize W_final[0] (attention) higher since it should be the primary signal
  W_final[0] = 0.3;

  return { W_q, W_k, W_final, bias: 0.0, version: 1 };
}

/**
 * Train ACAN weights from retrieval outcome data.
 *
 * Backprop through:
 *   q = W_q @ query                (1024 → 64)
 *   k = W_k @ memory               (1024 → 64)
 *   attn = dot(q, k) / sqrt(64)    (scalar)
 *   features = [attn, r, i, a, n, u] (6)
 *   score = dot(features, W_final) + bias
 *   loss = (score - label)^2
 *
 * Gradients:
 *   dL/dscore = 2 * (score - label)
 *   dL/dW_final[j] = dL/dscore * features[j]
 *   dL/dbias = dL/dscore
 *   dL/dattn = dL/dscore * W_final[0]
 *   dL/dq[j] = dL/dattn * k[j] / sqrt(64)
 *   dL/dk[j] = dL/dattn * q[j] / sqrt(64)
 *   dL/dW_q[i][j] = dL/dq[j] * query[i]
 *   dL/dW_k[i][j] = dL/dk[j] * memory[i]
 */
export interface TrainingConfig {
  epochs: number;
  lr: number;
  earlyStopPatience: number;  // stop if val loss doesn't improve for this many epochs
  lrDecayPatience: number;    // halve lr if val loss plateaus for this many epochs
  lrFloor: number;            // minimum learning rate
  valSplit: number;           // fraction held out for validation (0.2 = 20%)
}

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 80,
  lr: 0.001,
  earlyStopPatience: 8,
  lrDecayPatience: 4,
  lrFloor: 0.00005,
  valSplit: 0.2,
};

export interface TrainingResult {
  weights: ACANWeights;
  trainLoss: number;
  valLoss: number;
  actualEpochs: number;
  finalLr: number;
  config: TrainingConfig;
}

/** Shuffle array in-place using Fisher-Yates */
function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Compute MSE loss over a sample subset without updating weights */
function evalLoss(
  samples: TrainingSample[],
  auxFeatures: number[][],
  indices: number[],
  w: ACANWeights,
  scale: number,
): number {
  let total = 0;
  for (const si of indices) {
    const s = samples[si];
    const q = projectVec(s.query_embedding, w.W_q);
    const k = projectVec(s.memory_embedding, w.W_k);
    const attn = dot(q, k) / scale;
    const features = [attn, ...auxFeatures[si]];
    const score = dot(features, w.W_final) + w.bias;
    const err = score - s.utilization;
    total += err * err;
  }
  return total / indices.length;
}

export function trainWeights(
  samples: TrainingSample[],
  config: Partial<TrainingConfig> = {},
  warmStart?: ACANWeights,
): TrainingResult {
  const cfg = { ...DEFAULT_TRAINING_CONFIG, ...config };
  const scale = Math.sqrt(ATTN_DIM);

  // Shuffle and split into train/validation sets
  const indices = shuffle(Array.from({ length: samples.length }, (_, i) => i));
  const valSize = Math.max(1, Math.floor(samples.length * cfg.valSplit));
  const valIndices = indices.slice(0, valSize);
  const trainIndices = indices.slice(valSize);
  const nTrain = trainIndices.length;

  // Warm-start from previous weights or initialize fresh
  const w: ACANWeights = warmStart
    ? JSON.parse(JSON.stringify(warmStart)) // deep clone to avoid mutating caller's copy
    : initRandomWeights();

  // Pre-extract auxiliary features (constant across epochs)
  const auxFeatures: number[][] = samples.map((s) => [
    s.recency,
    s.importance,
    s.access_count,
    s.was_neighbor ? 1.0 : 0.0,
    0.0,  // provenUtility — excluded to avoid data leak
    0.0,  // reflectionBoost — not yet in training samples
    0.0,  // keywordOverlap — not yet in training samples
  ]);

  let lr = cfg.lr;
  let bestValLoss = Infinity;
  let epochsSinceImprovement = 0;
  let epochsSinceLrDecay = 0;
  let lastTrainLoss = Infinity;
  let actualEpochs = 0;

  for (let epoch = 0; epoch < cfg.epochs; epoch++) {
    actualEpochs = epoch + 1;

    // Shuffle training indices each epoch to avoid ordering bias
    shuffle(trainIndices);

    let totalLoss = 0;
    for (const si of trainIndices) {
      const s = samples[si];
      const label = s.utilization;

      // Forward pass
      const q = projectVec(s.query_embedding, w.W_q);
      const k = projectVec(s.memory_embedding, w.W_k);
      const attn = dot(q, k) / scale;

      const features = [attn, ...auxFeatures[si]];
      const score = dot(features, w.W_final) + w.bias;
      const err = score - label;
      totalLoss += err * err;

      // Backward pass
      const dScore = (2 / nTrain) * err;

      for (let j = 0; j < FEATURE_COUNT; j++) {
        w.W_final[j] -= lr * dScore * features[j];
      }
      w.bias -= lr * dScore;

      const dAttn = dScore * w.W_final[0];
      const dQ = new Array(ATTN_DIM);
      const dK = new Array(ATTN_DIM);
      for (let j = 0; j < ATTN_DIM; j++) {
        dQ[j] = dAttn * k[j] / scale;
        dK[j] = dAttn * q[j] / scale;
      }

      for (let i = 0; i < EMBED_DIM; i++) {
        if (s.query_embedding[i] === 0) continue;
        const qi = s.query_embedding[i];
        const row = w.W_q[i];
        for (let j = 0; j < ATTN_DIM; j++) {
          row[j] -= lr * dQ[j] * qi;
        }
      }

      for (let i = 0; i < EMBED_DIM; i++) {
        if (s.memory_embedding[i] === 0) continue;
        const mi = s.memory_embedding[i];
        const row = w.W_k[i];
        for (let j = 0; j < ATTN_DIM; j++) {
          row[j] -= lr * dK[j] * mi;
        }
      }
    }

    lastTrainLoss = totalLoss / nTrain;

    // Evaluate on validation set
    const valLoss = evalLoss(samples, auxFeatures, valIndices, w, scale);

    if (valLoss < bestValLoss) {
      bestValLoss = valLoss;
      epochsSinceImprovement = 0;
      epochsSinceLrDecay = 0;
    } else {
      epochsSinceImprovement++;
      epochsSinceLrDecay++;
    }

    // Learning rate decay: halve lr when validation loss plateaus
    if (epochsSinceLrDecay >= cfg.lrDecayPatience && lr > cfg.lrFloor) {
      lr = Math.max(lr * 0.5, cfg.lrFloor);
      epochsSinceLrDecay = 0;
    }

    // Early stopping: no improvement for too many epochs
    if (epochsSinceImprovement >= cfg.earlyStopPatience) {
      break;
    }
  }

  w.trainedAt = Date.now();
  w.trainedOnSamples = samples.length;
  return { weights: w, trainLoss: lastTrainLoss, valLoss: bestValLoss, actualEpochs, finalLr: lr, config: cfg };
}

// ── Training data fetching ──

export async function getTrainingDataCount(): Promise<number> {
  if (!(await isSurrealAvailable())) return 0;
  try {
    const flat = await queryFirst<{ count: number }>(
      `SELECT count() AS count FROM retrieval_outcome WHERE query_embedding != NONE GROUP ALL`,
    );
    return flat[0]?.count ?? 0;
  } catch (e) {
    swallow("acan:return 0;", e);
    return 0;
  }
}

async function fetchTrainingData(): Promise<TrainingSample[]> {
  if (!(await isSurrealAvailable())) return [];

  const db = getDb();

  // Fetch all retrieval outcomes with query embeddings
  // Prefer LLM-judged relevance (from cognitive check) over heuristic utilization when available
  const outcomes = await queryFirst<any>(
    `SELECT query_embedding, memory_id, memory_table,
            IF llm_relevance != NONE THEN llm_relevance ELSE utilization END AS utilization,
            retrieval_score, was_neighbor,
            importance, access_count, recency, created_at
     FROM retrieval_outcome
     WHERE query_embedding != NONE
     ORDER BY created_at ASC`,
  );
  if (outcomes.length === 0) return [];

  // Fetch embeddings for each unique memory ID
  const uniqueMemIds = [...new Set(outcomes.map((r) => String(r.memory_id)))];
  const embeddingMap = new Map<string, number[]>();
  for (const mid of uniqueMemIds) {
    try {
      assertRecordId(mid);
      const flat = await queryFirst<{ id: string; embedding: number[] }>(
        `SELECT id, embedding FROM ${mid} WHERE embedding != NONE`,
      );
      if (flat[0]?.embedding) {
        embeddingMap.set(mid, flat[0].embedding);
      }
    } catch (e) { swallow("acan:skip", e); }
  }

  // Build training samples
  const samples: TrainingSample[] = [];
  for (const row of outcomes) {
    const memEmb = embeddingMap.get(String(row.memory_id));
    if (!memEmb || !row.query_embedding) continue;
    samples.push({
      query_embedding: row.query_embedding,
      memory_embedding: memEmb,
      retrieval_score: row.retrieval_score ?? 0,
      was_neighbor: row.was_neighbor ?? false,
      utilization: row.utilization ?? 0,
      importance: row.importance ?? 0.5,
      access_count: row.access_count ?? 0,
      recency: row.recency ?? 0.5,
    });
  }

  return samples;
}

// ── JSONL export (optional, for external training or inspection) ──

export async function exportTrainingData(outputPath?: string): Promise<number> {
  const samples = await fetchTrainingData();
  if (samples.length === 0) return 0;

  const outFile = outputPath ?? join(getZeraDir(), "acan_training.jsonl");
  const lines = samples.map((s) => JSON.stringify(s));
  writeFileSync(outFile, lines.join("\n") + "\n", "utf-8");
  return samples.length;
}

// ── Startup: auto-train in background and activate mid-session ──

const TRAINING_THRESHOLD = 5000;

/**
 * Run training in a worker thread so it doesn't block the event loop.
 * The worker receives serialized training data + config, runs SGD with
 * validation split, early stopping, and lr decay, then posts back results.
 */
function trainInBackground(
  samples: TrainingSample[],
  weightsPath: string,
  warmStart?: ACANWeights,
  config?: Partial<TrainingConfig>,
): void {
  const cfg = { ...DEFAULT_TRAINING_CONFIG, ...config };

  // Inline worker as a data: URL for ESM compatibility
  const workerCode = `
    import { parentPort, workerData } from "node:worker_threads";
    const { samples, cfg, warmStart, EMBED_DIM, ATTN_DIM, FEATURE_COUNT } = workerData;

    function dot(a, b) {
      let sum = 0;
      for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
      return sum;
    }
    function projectVec(vec, matrix) {
      const out = new Array(matrix[0].length).fill(0);
      for (let i = 0; i < vec.length; i++) {
        if (vec[i] === 0) continue;
        const row = matrix[i];
        for (let j = 0; j < out.length; j++) out[j] += vec[i] * row[j];
      }
      return out;
    }
    function shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    }

    const n = samples.length;
    const auxFeatures = samples.map(s => [
      s.recency, s.importance, s.access_count, s.was_neighbor ? 1.0 : 0.0, 0.0, 0.0, 0.0,
    ]);

    // Shuffle and split train/val
    const indices = shuffle(Array.from({ length: n }, (_, i) => i));
    const valSize = Math.max(1, Math.floor(n * cfg.valSplit));
    const valIdx = indices.slice(0, valSize);
    const trainIdx = indices.slice(valSize);
    const nTrain = trainIdx.length;

    // Initialize or warm-start weights
    let W_q, W_k, W_final, bias;
    if (warmStart) {
      W_q = JSON.parse(JSON.stringify(warmStart.W_q));
      W_k = JSON.parse(JSON.stringify(warmStart.W_k));
      W_final = [...warmStart.W_final];
      bias = warmStart.bias;
    } else {
      const xavierQK = Math.sqrt(2 / (EMBED_DIM + ATTN_DIM));
      const xavierFinal = Math.sqrt(2 / (FEATURE_COUNT + 1));
      W_q = []; W_k = [];
      for (let i = 0; i < EMBED_DIM; i++) {
        W_q.push(Array.from({ length: ATTN_DIM }, () => (Math.random() * 2 - 1) * xavierQK));
        W_k.push(Array.from({ length: ATTN_DIM }, () => (Math.random() * 2 - 1) * xavierQK));
      }
      W_final = Array.from({ length: FEATURE_COUNT }, () => (Math.random() * 2 - 1) * xavierFinal);
      W_final[0] = 0.3;
      bias = 0.0;
    }

    const scale = Math.sqrt(ATTN_DIM);

    function evalLoss(idxList) {
      let total = 0;
      for (const si of idxList) {
        const s = samples[si];
        const q = projectVec(s.query_embedding, W_q);
        const k = projectVec(s.memory_embedding, W_k);
        const attn = dot(q, k) / scale;
        const features = [attn, ...auxFeatures[si]];
        const score = dot(features, W_final) + bias;
        const err = score - s.utilization;
        total += err * err;
      }
      return total / idxList.length;
    }

    let lr = cfg.lr;
    let bestValLoss = Infinity;
    let epochsSinceImprovement = 0;
    let epochsSinceLrDecay = 0;
    let lastTrainLoss = Infinity;
    let actualEpochs = 0;

    for (let epoch = 0; epoch < cfg.epochs; epoch++) {
      actualEpochs = epoch + 1;
      shuffle(trainIdx);

      let totalLoss = 0;
      for (const si of trainIdx) {
        const s = samples[si];
        const q = projectVec(s.query_embedding, W_q);
        const k = projectVec(s.memory_embedding, W_k);
        const attn = dot(q, k) / scale;
        const features = [attn, ...auxFeatures[si]];
        const score = dot(features, W_final) + bias;
        const err = score - s.utilization;
        totalLoss += err * err;
        const dScore = (2 / nTrain) * err;
        for (let j = 0; j < FEATURE_COUNT; j++) W_final[j] -= lr * dScore * features[j];
        bias -= lr * dScore;
        const dAttn = dScore * W_final[0];
        const dQ = new Array(ATTN_DIM);
        const dK = new Array(ATTN_DIM);
        for (let j = 0; j < ATTN_DIM; j++) {
          dQ[j] = dAttn * k[j] / scale;
          dK[j] = dAttn * q[j] / scale;
        }
        for (let i = 0; i < EMBED_DIM; i++) {
          if (s.query_embedding[i] !== 0) {
            const qi = s.query_embedding[i];
            const row = W_q[i];
            for (let j = 0; j < ATTN_DIM; j++) row[j] -= lr * dQ[j] * qi;
          }
          if (s.memory_embedding[i] !== 0) {
            const mi = s.memory_embedding[i];
            const row = W_k[i];
            for (let j = 0; j < ATTN_DIM; j++) row[j] -= lr * dK[j] * mi;
          }
        }
      }
      lastTrainLoss = totalLoss / nTrain;
      const valLoss = evalLoss(valIdx);

      if (valLoss < bestValLoss) {
        bestValLoss = valLoss;
        epochsSinceImprovement = 0;
        epochsSinceLrDecay = 0;
      } else {
        epochsSinceImprovement++;
        epochsSinceLrDecay++;
      }

      if (epochsSinceLrDecay >= cfg.lrDecayPatience && lr > cfg.lrFloor) {
        lr = Math.max(lr * 0.5, cfg.lrFloor);
        epochsSinceLrDecay = 0;
      }

      if (epochsSinceImprovement >= cfg.earlyStopPatience) break;
    }

    parentPort.postMessage({
      weights: { W_q, W_k, W_final, bias, version: 1, trainedAt: Date.now(), trainedOnSamples: n },
      trainLoss: lastTrainLoss,
      valLoss: bestValLoss,
      actualEpochs,
      finalLr: lr,
      config: cfg,
    });
  `;

  const start = Date.now();

  const worker = new Worker(new URL(`data:text/javascript,${encodeURIComponent(workerCode)}`), {
    workerData: {
      samples,
      cfg,
      warmStart: warmStart ?? null,
      EMBED_DIM,
      ATTN_DIM,
      FEATURE_COUNT,
    },
  });

  worker.on("message", (msg: { weights: ACANWeights } & Omit<TrainingResult, "weights">) => {
    try {
      saveWeights(msg.weights, weightsPath);
      _weights = msg.weights;
      _active = true;
      // Persist training log for self-optimization
      saveTrainingLog({
        trainedAt: msg.weights.trainedAt ?? Date.now(),
        samples: samples.length,
        trainLoss: msg.trainLoss,
        valLoss: msg.valLoss,
        actualEpochs: msg.actualEpochs,
        finalLr: msg.finalLr,
        config: msg.config,
        durationMs: Date.now() - start,
      });
    } catch (_err) {
      // Silent — save failure is non-fatal, old weights still work
    }
  });

  worker.on("error", (_err) => {
    // Silent — training failure is non-fatal
  });
}

// ── Training log: persisted to disk for self-optimization ──

interface TrainingLogEntry {
  trainedAt: number;
  samples: number;
  trainLoss: number;
  valLoss: number;
  actualEpochs: number;
  finalLr: number;
  config: TrainingConfig;
  durationMs: number;
}

const TRAINING_LOG_FILENAME = "acan_training_log.json";
const MAX_LOG_ENTRIES = 20; // keep last 20 training runs

function loadTrainingLog(): TrainingLogEntry[] {
  try {
    const path = join(getZeraDir(), TRAINING_LOG_FILENAME);
    if (!existsSync(path)) return [];
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch { return []; }
}

function saveTrainingLog(entry: TrainingLogEntry): void {
  try {
    const log = loadTrainingLog();
    log.push(entry);
    // Keep only the most recent entries
    const trimmed = log.slice(-MAX_LOG_ENTRIES);
    const path = join(getZeraDir(), TRAINING_LOG_FILENAME);
    writeFileSync(path, JSON.stringify(trimmed, null, 2), "utf-8");
  } catch { /* non-critical */ }
}

/**
 * Auto-tune training config based on historical training runs.
 * Looks at the last few runs to adjust lr and epochs.
 */
function autoTuneConfig(): Partial<TrainingConfig> {
  const log = loadTrainingLog();
  if (log.length < 3) return {}; // not enough history to tune

  const recent = log.slice(-5);
  const adjustments: Partial<TrainingConfig> = {};

  // If recent runs consistently early-stopped well below max epochs,
  // the max epochs setting is fine (early stopping handles it)

  // Learning rate adjustment based on final lr trend:
  // If the last 3 runs all decayed lr significantly, start with a lower lr
  const avgFinalLr = recent.reduce((s, e) => s + e.finalLr, 0) / recent.length;
  const avgStartLr = recent.reduce((s, e) => s + e.config.lr, 0) / recent.length;
  if (avgFinalLr < avgStartLr * 0.3) {
    // LR consistently decays a lot — start lower next time
    adjustments.lr = Math.max(avgFinalLr * 2, DEFAULT_TRAINING_CONFIG.lrFloor * 2);
  }

  // If val loss is trending up across runs (overfitting more), increase patience
  if (recent.length >= 3) {
    const lastThree = recent.slice(-3);
    const valTrend = lastThree[2].valLoss - lastThree[0].valLoss;
    if (valTrend > 0.01) {
      // Val loss getting worse — tighter early stopping
      adjustments.earlyStopPatience = Math.max(4, DEFAULT_TRAINING_CONFIG.earlyStopPatience - 2);
    }
  }

  return adjustments;
}

/** Staleness threshold: retrain if sample count has grown by this factor */
const STALENESS_GROWTH_FACTOR = 0.5; // 50% more samples → retrain
/** Staleness threshold: retrain if weights are older than this (ms) */
const STALENESS_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

export async function checkACANReadiness(): Promise<void> {
  const weightsPath = join(getZeraDir(), WEIGHTS_FILENAME);

  // Try loading existing weights (instant)
  const hasWeights = initACAN();

  // Check training data count (fast DB query)
  const count = await getTrainingDataCount();

  if (hasWeights && _weights) {
    // Check if retrain is needed due to staleness
    const trainedOn = _weights.trainedOnSamples ?? 0;
    const trainedAt = _weights.trainedAt ?? 0;
    const growthRatio = trainedOn > 0 ? (count - trainedOn) / trainedOn : Infinity;
    const ageMs = Date.now() - trainedAt;

    const isStale = growthRatio >= STALENESS_GROWTH_FACTOR || ageMs >= STALENESS_MAX_AGE_MS;

    if (!isStale) {
      return;
    }

    // Weights exist but are stale — retrain with warm-start + auto-tuned config
  } else if (count < TRAINING_THRESHOLD) {
    // No weights and not enough data yet
    return;
  }

  // Fetch training data and retrain
  try {
    const samples = await fetchTrainingData();
    if (samples.length < TRAINING_THRESHOLD) {
      return;
    }

    // Auto-tune config from training history
    const tuned = autoTuneConfig();
    // Warm-start from existing weights on retrain (cold start on first train)
    trainInBackground(samples, weightsPath, hasWeights ? _weights ?? undefined : undefined, tuned);
  } catch (_err) {
    // Silent — training is best-effort
  }
}

