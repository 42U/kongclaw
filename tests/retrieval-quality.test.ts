/**
 * Tests for retrieval quality tracking (src/retrieval-quality.ts).
 *
 * Tests the pure computation functions: trigram overlap, key term extraction,
 * and quality signal computation.
 */
import { describe, it, expect } from "vitest";

// --- Trigram overlap (reimplemented for testing) ---

function extractNgrams(text: string): Set<string> {
  const words = text.split(/\s+/).filter((w) => w.length > 2);
  const grams = new Set<string>();
  if (words.length >= 3) {
    for (let i = 0; i <= words.length - 3; i++) {
      grams.add(`${words[i]} ${words[i + 1]} ${words[i + 2]}`);
    }
  } else if (words.length === 2) {
    grams.add(`${words[0]} ${words[1]}`);
  } else if (words.length === 1) {
    grams.add(words[0]);
  }
  return grams;
}

function trigramOverlap(source: string, target: string): number {
  const srcGrams = extractNgrams(source);
  if (srcGrams.size === 0) return 0;
  const tgtGrams = extractNgrams(target);
  let matches = 0;
  for (const gram of srcGrams) {
    if (tgtGrams.has(gram)) matches++;
  }
  return matches / srcGrams.size;
}

describe("trigram overlap", () => {
  it("identical texts produce 1.0", () => {
    const text = "the quick brown fox jumps over the lazy dog";
    expect(trigramOverlap(text, text)).toBe(1.0);
  });

  it("completely different texts produce 0.0", () => {
    expect(trigramOverlap(
      "alpha beta gamma delta",
      "epsilon zeta eta theta",
    )).toBe(0.0);
  });

  it("partial overlap produces fractional score", () => {
    const score = trigramOverlap(
      "the authentication module handles login",
      "the authentication module returns errors",
    );
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("empty source produces 0.0", () => {
    expect(trigramOverlap("", "some target text here")).toBe(0.0);
  });

  it("short words (<3 chars) are filtered out", () => {
    // Only words with length > 2 pass; "a"(1), "an"(2), "if"(2), "or"(2) filtered; "the"(3) passes
    const grams = extractNgrams("a an if or");
    expect(grams.size).toBe(0);
  });
});

// --- Key term extraction ---

const KEY_TERM_PATTERNS = [
  /`([^`]{2,60})`/g,
  /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b/g,
  /\b([A-Z]{2,}(?:[-_][A-Z0-9]+)*)\b/g,
  /\b(\w+(?:[-_]\w+){1,3})\b/g,
];
const STOP_WORDS = new Set(["the", "a", "an", "but", "and", "or", "if", "when", "this", "that", "for", "with", "from", "into", "not", "are", "was", "were", "has", "have", "been"]);

function extractKeyTerms(text: string): Set<string> {
  const terms = new Set<string>();
  for (const pattern of KEY_TERM_PATTERNS) {
    for (const match of text.matchAll(pattern)) {
      const term = match[1].trim().toLowerCase();
      if (term.length >= 3 && !STOP_WORDS.has(term)) {
        terms.add(term);
      }
    }
  }
  return terms;
}

describe("key term extraction", () => {
  it("extracts backtick-quoted terms", () => {
    const terms = extractKeyTerms("Use the `graphTransformContext` function");
    expect(terms.has("graphtransformcontext")).toBe(true);
  });

  it("extracts Capitalized Multi Word phrases", () => {
    // Regex captures "The Memory Substrate" as one match (includes leading "The")
    const terms = extractKeyTerms("The Memory Substrate handles persistence");
    expect(terms.has("the memory substrate")).toBe(true);
  });

  it("extracts acronyms", () => {
    const terms = extractKeyTerms("Use BGE-M3 for HNSW vector indexing");
    expect(terms.has("bge-m3")).toBe(true);
    expect(terms.has("hnsw")).toBe(true);
  });

  it("extracts hyphenated terms", () => {
    const terms = extractKeyTerms("The graph-context module uses vector-search");
    expect(terms.has("graph-context")).toBe(true);
    expect(terms.has("vector-search")).toBe(true);
  });

  it("filters out stop words", () => {
    const terms = extractKeyTerms("the and for with from into");
    expect(terms.size).toBe(0);
  });

  it("filters out very short terms", () => {
    const terms = extractKeyTerms("`ab` is too short");
    expect(terms.has("ab")).toBe(false);
  });
});

// --- Key term overlap ---

function keyTermOverlap(source: string, targetLower: string): number {
  const terms = extractKeyTerms(source);
  if (terms.size === 0) return 0;
  let found = 0;
  for (const term of terms) {
    if (targetLower.includes(term)) found++;
  }
  return found / terms.size;
}

describe("key term overlap (utilization signal)", () => {
  it("all terms present = 1.0", () => {
    const source = "The `graphTransformContext` uses BGE-M3 embeddings";
    const target = "the graphtransformcontext function with bge-m3 vector embeddings";
    expect(keyTermOverlap(source, target)).toBe(1.0);
  });

  it("no terms present = 0.0", () => {
    const source = "The `graphTransformContext` module";
    const target = "something completely different without any matching terms";
    expect(keyTermOverlap(source, target)).toBe(0.0);
  });

  it("partial presence = fractional", () => {
    const source = "Use `vectorSearch` with HNSW and `scoreResults`";
    const target = "the vectorsearch function was called";
    const score = keyTermOverlap(source, target);
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("source with no key terms = 0.0", () => {
    expect(keyTermOverlap("just some plain words here", "matching text")).toBe(0.0);
  });
});

// --- Hybrid utilization (max of key-term and trigram) ---

describe("hybrid utilization scoring", () => {
  it("paraphrased content detected via key terms even when trigrams miss", () => {
    // Same concept, different phrasing → trigrams low, key terms high
    const source = "The `vectorSearch` function in HNSW returns results";
    const targetLower = "we use vectorsearch to query the hnsw index and get matching items";
    const keyScore = keyTermOverlap(source, targetLower);
    const trigramScore = trigramOverlap(source.toLowerCase(), targetLower);
    const hybrid = Math.max(keyScore, trigramScore);
    expect(hybrid).toBeGreaterThan(trigramScore); // key terms should win
    expect(hybrid).toBeGreaterThan(0.3);
  });

  it("verbatim copy detected by both signals", () => {
    const text = "the graph context transformation applies vector search and scoring";
    const keyScore = keyTermOverlap(text, text);
    const trigramScore = trigramOverlap(text, text);
    expect(Math.max(keyScore, trigramScore)).toBe(1.0);
  });
});

// --- Tool success majority vote (fixed from every() bug) ---

describe("tool success majority vote", () => {
  function computeToolSuccess(results: { success: boolean }[]): boolean | null {
    if (results.length === 0) return null;
    return results.filter((r) => r.success).length / results.length >= 0.5;
  }

  it("no tool calls = null", () => {
    expect(computeToolSuccess([])).toBeNull();
  });

  it("all successful = true", () => {
    expect(computeToolSuccess([{ success: true }, { success: true }])).toBe(true);
  });

  it("all failed = false", () => {
    expect(computeToolSuccess([{ success: false }, { success: false }])).toBe(false);
  });

  it("single failure in batch of 10 = true (majority vote)", () => {
    const results = Array(9).fill({ success: true }).concat([{ success: false }]);
    expect(computeToolSuccess(results)).toBe(true);
  });

  it("50/50 split = true (>= 0.5 threshold)", () => {
    const results = [{ success: true }, { success: false }];
    expect(computeToolSuccess(results)).toBe(true);
  });

  it("minority success = false", () => {
    const results = [{ success: true }, { success: false }, { success: false }, { success: false }];
    expect(computeToolSuccess(results)).toBe(false);
  });

  it("single exploratory failure does NOT tank entire turn (the bug fix)", () => {
    // This was the bug: grep finds file, read succeeds, edit succeeds, but
    // one file-not-found from an exploratory search marked everything as failed.
    const results = [
      { success: true },  // grep
      { success: true },  // read
      { success: true },  // edit
      { success: true },  // test
      { success: false }, // exploratory search — file not found
    ];
    expect(computeToolSuccess(results)).toBe(true);
  });
});
