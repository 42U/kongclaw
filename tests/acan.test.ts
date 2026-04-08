/**
 * Tests for ACAN (Attentive Cross-Attention Network) inference.
 *
 * Pure math tests — no SurrealDB or embeddings required.
 */
import { describe, it, expect, beforeAll } from "vitest";

let dot: typeof import("../src/acan.js")["dot"];
let projectVec: typeof import("../src/acan.js")["projectVec"];
let softmax: typeof import("../src/acan.js")["softmax"];
let scoreWithACAN: typeof import("../src/acan.js")["scoreWithACAN"];
let loadWeights: typeof import("../src/acan.js")["loadWeights"];
let isACANActive: typeof import("../src/acan.js")["isACANActive"];
let initACAN: typeof import("../src/acan.js")["initACAN"];
let getTrainingDataCount: typeof import("../src/acan.js")["getTrainingDataCount"];

beforeAll(async () => {
  const mod = await import("../src/acan.js");
  dot = mod.dot;
  projectVec = mod.projectVec;
  softmax = mod.softmax;
  scoreWithACAN = mod.scoreWithACAN;
  loadWeights = mod.loadWeights;
  isACANActive = mod.isACANActive;
  initACAN = mod.initACAN;
  getTrainingDataCount = mod.getTrainingDataCount;
});

describe("ACAN", () => {
  describe("dot product", () => {
    it("computes correct dot product", () => {
      expect(dot([1, 2, 3], [4, 5, 6])).toBe(32); // 4+10+18
    });

    it("returns 0 for orthogonal vectors", () => {
      expect(dot([1, 0], [0, 1])).toBe(0);
    });

    it("handles empty vectors", () => {
      expect(dot([], [])).toBe(0);
    });
  });

  describe("projectVec", () => {
    it("computes matrix-vector product", () => {
      // 2-dim input, 2x2 matrix → 2-dim output
      // vec = [1, 2], matrix = [[1, 0], [0, 1]] (identity) → [1, 2]
      const result = projectVec([1, 2], [[1, 0], [0, 1]]);
      expect(result).toEqual([1, 2]);
    });

    it("computes non-trivial projection", () => {
      // vec = [1, 2], matrix = [[1, 2], [3, 4]] → [1*1+2*3, 1*2+2*4] = [7, 10]
      const result = projectVec([1, 2], [[1, 2], [3, 4]]);
      expect(result[0]).toBe(7);
      expect(result[1]).toBe(10);
    });

    it("skips zero elements", () => {
      const result = projectVec([0, 1], [[999, 999], [3, 4]]);
      expect(result).toEqual([3, 4]);
    });
  });

  describe("softmax", () => {
    it("sums to 1.0", () => {
      const result = softmax([1, 2, 3]);
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0);
    });

    it("highest input gets highest probability", () => {
      const result = softmax([1, 5, 2]);
      expect(result[1]).toBeGreaterThan(result[0]);
      expect(result[1]).toBeGreaterThan(result[2]);
    });

    it("handles equal inputs", () => {
      const result = softmax([1, 1, 1]);
      expect(result[0]).toBeCloseTo(1 / 3);
      expect(result[1]).toBeCloseTo(1 / 3);
      expect(result[2]).toBeCloseTo(1 / 3);
    });

    it("is numerically stable with large values", () => {
      const result = softmax([1000, 1001, 1002]);
      expect(result.every((v) => isFinite(v))).toBe(true);
      expect(result.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0);
    });

    it("is numerically stable with very negative values", () => {
      const result = softmax([-1000, -999, -998]);
      expect(result.every((v) => isFinite(v))).toBe(true);
      expect(result.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0);
    });

    it("returns empty for empty input", () => {
      expect(softmax([])).toEqual([]);
    });
  });

  describe("loadWeights", () => {
    it("returns null for non-existent file", () => {
      expect(loadWeights("/tmp/nonexistent_acan_weights_xyz.json")).toBeNull();
    });

    it("returns null for invalid JSON", () => {
      const { writeFileSync, unlinkSync } = require("fs");
      const path = "/tmp/acan_test_invalid.json";
      writeFileSync(path, "not json", "utf-8");
      expect(loadWeights(path)).toBeNull();
      unlinkSync(path);
    });

    it("returns null for wrong dimensions", () => {
      const { writeFileSync, unlinkSync } = require("fs");
      const path = "/tmp/acan_test_wrong_dims.json";
      writeFileSync(path, JSON.stringify({
        version: 1,
        W_q: [[1, 2]], // wrong: should be 1024 rows
        W_k: [[1, 2]],
        W_final: [1, 2, 3, 4, 5, 6, 7, 8],
        bias: 0,
      }), "utf-8");
      expect(loadWeights(path)).toBeNull();
      unlinkSync(path);
    });

    it("loads valid weights", () => {
      const { writeFileSync, unlinkSync } = require("fs");
      const path = "/tmp/acan_test_valid.json";
      const W_q = Array.from({ length: 1024 }, () => Array.from({ length: 64 }, () => 0.01));
      const W_k = Array.from({ length: 1024 }, () => Array.from({ length: 64 }, () => 0.01));
      writeFileSync(path, JSON.stringify({
        version: 1,
        W_q,
        W_k,
        W_final: [0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.05, 0.1],
        bias: 0.01,
      }), "utf-8");
      const weights = loadWeights(path);
      expect(weights).not.toBeNull();
      expect(weights!.W_q.length).toBe(1024);
      expect(weights!.W_q[0].length).toBe(64);
      expect(weights!.W_final.length).toBe(8);
      unlinkSync(path);
    });
  });

  describe("isACANActive", () => {
    it("returns false by default (no weights loaded)", () => {
      expect(isACANActive()).toBe(false);
    });

    it("returns false when init fails", () => {
      initACAN("/tmp/nonexistent_dir_xyz");
      expect(isACANActive()).toBe(false);
    });
  });

  describe("scoreWithACAN", () => {
    it("returns empty for empty candidates", () => {
      expect(scoreWithACAN([], [])).toEqual([]);
    });

    it("produces scores when weights are loaded", () => {
      const { writeFileSync, unlinkSync } = require("fs");
      const path = "/tmp/acan_test_score";
      const { mkdirSync, existsSync } = require("fs");
      if (!existsSync(path)) mkdirSync(path, { recursive: true });

      // Create simple weights
      const W_q = Array.from({ length: 1024 }, (_, i) =>
        Array.from({ length: 64 }, (_, j) => (i === j && j < 64) ? 1.0 : 0.0),
      );
      const W_k = Array.from({ length: 1024 }, (_, i) =>
        Array.from({ length: 64 }, (_, j) => (i === j && j < 64) ? 1.0 : 0.0),
      );

      writeFileSync(`${path}/acan_weights.json`, JSON.stringify({
        version: 1,
        W_q,
        W_k,
        W_final: [1.0, 0.5, 0.3, 0.1, 0.5, 0.2, 0.1, 0.1], // attn weighted highest
        bias: 0.0,
      }), "utf-8");

      const loaded = initACAN(path);
      expect(loaded).toBe(true);

      // Create two candidates with different embeddings
      const queryEmb = new Array(1024).fill(0);
      queryEmb[0] = 1.0; // unit vector in dim 0

      const cand1Emb = new Array(1024).fill(0);
      cand1Emb[0] = 1.0; // aligned with query
      const cand2Emb = new Array(1024).fill(0);
      cand2Emb[1] = 1.0; // orthogonal to query

      const scores = scoreWithACAN(queryEmb, [
        { embedding: cand1Emb, recency: 0.5, importance: 0.5, access: 0.5, neighborBonus: 0, provenUtility: 0.5 },
        { embedding: cand2Emb, recency: 0.5, importance: 0.5, access: 0.5, neighborBonus: 0, provenUtility: 0.5 },
      ]);

      expect(scores.length).toBe(2);
      expect(scores.every((s) => isFinite(s))).toBe(true);
      // Candidate 1 (aligned) should score higher than candidate 2 (orthogonal)
      expect(scores[0]).toBeGreaterThan(scores[1]);

      // Cleanup
      unlinkSync(`${path}/acan_weights.json`);
      // Reset ACAN state
      initACAN("/tmp/nonexistent_xyz");
    });
  });

  describe("getTrainingDataCount", () => {
    it("returns a number >= 0", async () => {
      const count = await getTrainingDataCount();
      expect(typeof count).toBe("number");
      expect(count).toBeGreaterThanOrEqual(0);
    });
  });
});
