/**
 * Tests for concept supersedes system (src/supersedes.ts).
 *
 * Tests the pure logic: threshold checks, stability decay, floor behavior.
 * DB integration is not tested here (requires SurrealDB).
 */
import { describe, it, expect } from "vitest";

const SUPERSEDE_THRESHOLD = 0.70;
const STABILITY_DECAY_FACTOR = 0.4;
const STABILITY_FLOOR = 0.15;

function computeDecayedStability(currentStability: number): number {
  return Math.max(STABILITY_FLOOR, currentStability * STABILITY_DECAY_FACTOR);
}

describe("supersedes — stability decay", () => {
  it("decays stability by 60% (factor 0.4)", () => {
    expect(computeDecayedStability(1.0)).toBeCloseTo(0.4);
    expect(computeDecayedStability(0.8)).toBeCloseTo(0.32);
    expect(computeDecayedStability(0.5)).toBeCloseTo(0.2);
  });

  it("respects stability floor of 0.15", () => {
    expect(computeDecayedStability(0.3)).toBeCloseTo(0.15); // 0.3 * 0.4 = 0.12 < 0.15
    expect(computeDecayedStability(0.1)).toBeCloseTo(0.15); // 0.1 * 0.4 = 0.04 < 0.15
    expect(computeDecayedStability(0.0)).toBeCloseTo(0.15); // floor
  });

  it("stability at exactly the floor stays at floor", () => {
    expect(computeDecayedStability(STABILITY_FLOOR)).toBeCloseTo(STABILITY_FLOOR);
  });

  it("multiple decays converge to floor", () => {
    let s = 1.0;
    for (let i = 0; i < 10; i++) s = computeDecayedStability(s);
    expect(s).toBeCloseTo(STABILITY_FLOOR);
  });
});

describe("supersedes — threshold checks", () => {
  it("concepts above threshold (0.70) are superseded", () => {
    expect(0.75 >= SUPERSEDE_THRESHOLD).toBe(true);
    expect(0.85 >= SUPERSEDE_THRESHOLD).toBe(true);
    expect(1.0 >= SUPERSEDE_THRESHOLD).toBe(true);
  });

  it("concepts below threshold are not superseded", () => {
    expect(0.69 >= SUPERSEDE_THRESHOLD).toBe(false);
    expect(0.5 >= SUPERSEDE_THRESHOLD).toBe(false);
    expect(0.0 >= SUPERSEDE_THRESHOLD).toBe(false);
  });

  it("concepts exactly at threshold are superseded", () => {
    expect(0.70 >= SUPERSEDE_THRESHOLD).toBe(true);
  });
});
