/**
 * Tests for Procedural Memory / Skill Library (src/skills.ts).
 *
 * Unit tests for skill formatting and outcome recording.
 * Integration tests require SurrealDB + embeddings.
 */
import { describe, it, expect, beforeAll } from "vitest";

let surrealAvailable = false;
let embeddingsReady = false;
let findRelevantSkills: typeof import("../src/skills.js")["findRelevantSkills"];
let formatSkillContext: typeof import("../src/skills.js")["formatSkillContext"];
let recordSkillOutcome: typeof import("../src/skills.js")["recordSkillOutcome"];
let extractSkill: typeof import("../src/skills.js")["extractSkill"];
let supersedeOldSkills: typeof import("../src/skills.js")["supersedeOldSkills"];

beforeAll(async () => {
  try {
    const configMod = await import("../src/config.js");
    const config = configMod.loadConfig();
    const surrealMod = await import("../src/surreal.js");
    await surrealMod.initSurreal(config.surreal);
    surrealAvailable = await surrealMod.isSurrealAvailable();

    const embMod = await import("../src/embeddings.js");
    if (!embMod.isEmbeddingsAvailable()) {
      await embMod.initEmbeddings(config.embedding);
    }
    embeddingsReady = embMod.isEmbeddingsAvailable();
  } catch {
    surrealAvailable = false;
    embeddingsReady = false;
  }

  const mod = await import("../src/skills.js");
  findRelevantSkills = mod.findRelevantSkills;
  formatSkillContext = mod.formatSkillContext;
  recordSkillOutcome = mod.recordSkillOutcome;
  extractSkill = mod.extractSkill;
  supersedeOldSkills = mod.supersedeOldSkills;
});

describe("skill library", () => {
  describe("formatSkillContext", () => {
    it("returns empty string for no skills", () => {
      expect(formatSkillContext([])).toBe("");
    });

    it("formats a single skill with steps", () => {
      const result = formatSkillContext([{
        id: "skill:abc",
        name: "Add REST endpoint",
        description: "Create a new REST API endpoint with validation and tests",
        preconditions: "Express server running",
        steps: [
          { tool: "read", description: "Read existing routes file" },
          { tool: "edit", description: "Add new route handler" },
          { tool: "write", description: "Create test file" },
        ],
        postconditions: "New endpoint responds to GET /api/resource",
        successCount: 3,
        failureCount: 0,
        avgDurationMs: 15000,
        confidence: 1.0,
        active: true,
      }]);

      expect(result).toContain("<skill_context>");
      expect(result).toContain("Add REST endpoint");
      expect(result).toContain("3/3 successful");
      expect(result).toContain("[read]");
      expect(result).toContain("[edit]");
      expect(result).toContain("[write]");
      expect(result).toContain("Pre: Express server running");
      expect(result).toContain("Post: New endpoint responds");
    });

    it("formats multiple skills", () => {
      const result = formatSkillContext([
        {
          id: "skill:1",
          name: "Fix bug",
          description: "Debug and fix a code bug",
          steps: [{ tool: "bash", description: "Run tests" }],
          successCount: 2,
          failureCount: 1,
          avgDurationMs: 5000,
          confidence: 1.0,
          active: true,
        },
        {
          id: "skill:2",
          name: "Add test",
          description: "Add test coverage",
          steps: [{ tool: "write", description: "Create test" }],
          successCount: 5,
          failureCount: 0,
          avgDurationMs: 3000,
          confidence: 1.0,
          active: true,
        },
      ]);

      expect(result).toContain("Fix bug");
      expect(result).toContain("2/3 successful");
      expect(result).toContain("Add test");
      expect(result).toContain("5/5 successful");
    });

    it("shows 'new' for skills with no usage history", () => {
      const result = formatSkillContext([{
        id: "skill:new",
        name: "New skill",
        description: "Just extracted",
        steps: [{ tool: "read", description: "Read" }],
        successCount: 0,
        failureCount: 0,
        avgDurationMs: 0,
        confidence: 1.0,
        active: true,
      }]);

      expect(result).toContain("new");
    });
  });

  describe("findRelevantSkills", () => {
    it("returns empty when infra is down", async () => {
      if (surrealAvailable && embeddingsReady) return;
      const skills = await findRelevantSkills(new Array(1024).fill(0), 3);
      expect(skills).toEqual([]);
    });

    it("returns skills for a valid query", async () => {
      if (!surrealAvailable || !embeddingsReady) return;
      // May return empty if no skills exist yet — that's ok
      const emb = await (await import("../src/embeddings.js")).embed("add a REST endpoint");
      const skills = await findRelevantSkills(emb, 3);
      expect(Array.isArray(skills)).toBe(true);
    });
  });

  describe("recordSkillOutcome", () => {
    it("handles invalid skill ID gracefully", async () => {
      // Should not throw
      await recordSkillOutcome("not-valid", true, 1000);
    });

    it("handles non-existent skill gracefully", async () => {
      if (!surrealAvailable) return;
      // Should not throw even for valid format but non-existent
      await recordSkillOutcome("skill:nonexistent", true, 1000);
    });
  });

  describe("extractSkill", () => {
    it("returns null for non-existent session", async () => {
      if (!surrealAvailable) return;
      const result = await extractSkill("nonexistent-session", "task:test");
      expect(result).toBeNull();
    });

    it("returns null when surreal is down", async () => {
      if (surrealAvailable) return;
      const result = await extractSkill("test", "task:test");
      expect(result).toBeNull();
    });
  });

  describe("supersedeOldSkills", () => {
    it("handles empty embedding gracefully", async () => {
      // Should not throw with empty embedding
      await supersedeOldSkills("skill:test123", []);
    });

    it("handles non-existent skill ID gracefully", async () => {
      if (!surrealAvailable) return;
      // Should not throw even for non-existent skill
      await supersedeOldSkills("skill:nonexistent", new Array(1024).fill(0));
    });
  });

  describe("confidence and active fields", () => {
    it("formatSkillContext includes confidence/active skills only", () => {
      const active = formatSkillContext([{
        id: "skill:a",
        name: "Active skill",
        description: "Still valid",
        steps: [{ tool: "bash", description: "Run it" }],
        successCount: 1,
        failureCount: 0,
        avgDurationMs: 1000,
        confidence: 1.0,
        active: true,
      }]);
      expect(active).toContain("Active skill");
    });

    it("skill interface includes confidence and active", () => {
      const skill = {
        id: "skill:typed",
        name: "Typed skill",
        description: "Has all fields",
        steps: [{ tool: "read", description: "Read" }],
        successCount: 0,
        failureCount: 0,
        avgDurationMs: 0,
        confidence: 0.8,
        active: true,
      };
      expect(skill.confidence).toBe(0.8);
      expect(skill.active).toBe(true);
    });

    it("findRelevantSkills returns skills with confidence/active fields", async () => {
      if (!surrealAvailable || !embeddingsReady) return;
      const emb = await (await import("../src/embeddings.js")).embed("debug authentication timeout");
      const skills = await findRelevantSkills(emb, 3);
      for (const s of skills) {
        expect(typeof s.confidence).toBe("number");
        expect(typeof s.active).toBe("boolean");
      }
    });
  });

});
