/**
 * Terminal rendering module — all visual output flows through here.
 *
 * Replaces raw ANSI codes and process.stdout.write in cli.ts with
 * structured, chalk-styled rendering functions.
 */
import chalk from "chalk";
import { highlight } from "cli-highlight";
import { swallow } from "./errors.js";

// ── Helpers ──────────────────────────────────────────────────────────────────

function truncate(s: string, max = 60): string {
  if (!s) return "";
  const clean = s.replace(/\n/g, "↵").trim();
  return clean.length > max ? clean.slice(0, max) + "…" : clean;
}

function cols(): number {
  return process.stdout.columns || 80;
}

// ── Streaming markdown renderer ──────────────────────────────────────────────

// Inline markdown patterns
const INLINE_CODE = /`([^`]+)`/g;
const BOLD = /\*\*(.+?)\*\*/g;
const ITALIC = /(?<!\*)\*([^*]+)\*(?!\*)/g;
const ITALIC_US = /(?<!_)_([^_]+)_(?!_)/g;

function renderInline(line: string): string {
  return line
    .replace(INLINE_CODE, (_, code) => chalk.cyan(code))
    .replace(BOLD, (_, text) => chalk.bold(text))
    .replace(ITALIC, (_, text) => chalk.italic(text))
    .replace(ITALIC_US, (_, text) => chalk.italic(text));
}

function renderMarkdownLine(line: string): string {
  // Headers
  const headerMatch = line.match(/^(#{1,6})\s+(.+)/);
  if (headerMatch) {
    const level = headerMatch[1].length;
    const text = renderInline(headerMatch[2]);
    if (level === 1) return chalk.bold.cyan(text);
    if (level === 2) return chalk.bold(text);
    return chalk.bold.dim(text);
  }

  // Horizontal rules
  if (/^[-_]{3,}\s*$/.test(line)) {
    return chalk.dim("─".repeat(Math.min(cols(), 60)));
  }

  // Blockquotes
  if (line.startsWith("> ")) {
    return chalk.dim("│ ") + chalk.dim(renderInline(line.slice(2)));
  }

  // Unordered list items
  const ulMatch = line.match(/^(\s*)[-*]\s+(.*)/);
  if (ulMatch) {
    return ulMatch[1] + chalk.dim("•") + " " + renderInline(ulMatch[2]);
  }

  // Ordered list items
  const olMatch = line.match(/^(\s*)(\d+)\.\s+(.*)/);
  if (olMatch) {
    return olMatch[1] + chalk.dim(olMatch[2] + ".") + " " + renderInline(olMatch[3]);
  }

  return renderInline(line);
}

/** Check if a line has any markdown syntax worth rendering. */
function hasMarkdown(line: string): boolean {
  return /[`*_#>]|^[-*]\s|^\d+\.\s|^[-_]{3,}/.test(line);
}

class StreamMarkdownRenderer {
  private lineBuffer = "";
  private rawLineLen = 0; // chars written raw for current line
  inCodeBlock = false;
  codeBlockLang = "";
  codeBlockLines: string[] = [];

  /** Feed a streaming text chunk. */
  feed(chunk: string): void {
    for (const ch of chunk) {
      if (ch === "\n") {
        this.completeLine();
      } else {
        this.lineBuffer += ch;
        if (!this.inCodeBlock) {
          // Stream raw char so user sees text appearing in real-time
          process.stdout.write(ch);
          this.rawLineLen++;
        }
      }
    }
  }

  /** Called when a full line is available (newline received). */
  private completeLine(): void {
    const line = this.lineBuffer;
    this.lineBuffer = "";

    // Code fence detection
    const fenceMatch = line.match(/^```(\w*)$/);
    if (fenceMatch) {
      if (!this.inCodeBlock) {
        // Opening fence — erase any raw chars we wrote for the ``` line
        this.eraseRawLine();
        this.inCodeBlock = true;
        this.codeBlockLang = fenceMatch[1] || "";
        this.codeBlockLines = [];
        this.rawLineLen = 0;
        return;
      } else {
        // Closing fence — render the buffered code block
        this.renderCodeBlock();
        this.inCodeBlock = false;
        this.rawLineLen = 0;
        return;
      }
    }

    if (this.inCodeBlock) {
      this.codeBlockLines.push(line);
      this.rawLineLen = 0;
      return;
    }

    // Regular line — if it has markdown, rewrite it
    if (hasMarkdown(line) && this.rawLineLen > 0) {
      this.eraseRawLine();
      process.stdout.write(renderMarkdownLine(line) + "\n");
    } else if (this.rawLineLen > 0) {
      // Plain text, already streamed raw — just add newline
      process.stdout.write("\n");
    } else {
      // Empty line
      process.stdout.write("\n");
    }
    this.rawLineLen = 0;
  }

  /** Erase the raw characters written for the current line. */
  private eraseRawLine(): void {
    if (this.rawLineLen > 0) {
      process.stdout.write("\r\x1b[2K");
    }
  }

  /** Render a buffered code block with syntax highlighting. */
  private renderCodeBlock(): void {
    const lang = this.codeBlockLang;
    const w = Math.min(cols() - 4, 76);
    const label = lang ? ` ${lang} ` : "";
    const topRule = "─".repeat(Math.max(w - label.length - 2, 4));

    // Syntax highlight the code block
    const raw = this.codeBlockLines.join("\n");
    let highlighted: string;
    try {
      highlighted = highlight(raw, { language: lang || undefined, ignoreIllegals: true });
    } catch {
      highlighted = raw; // fallback to plain text
    }
    const highlightedLines = highlighted.split("\n");

    process.stdout.write(chalk.dim(`  ╭─${label}${topRule}╮`) + "\n");
    for (const codeLine of highlightedLines) {
      process.stdout.write(chalk.dim("  │ ") + codeLine + "\n");
    }
    process.stdout.write(chalk.dim(`  ╰${"─".repeat(w)}╯`) + "\n");
  }

  /** Flush remaining buffer at end of message. */
  flush(): void {
    if (this.lineBuffer) {
      // Incomplete line — render what we have
      if (hasMarkdown(this.lineBuffer) && this.rawLineLen > 0) {
        this.eraseRawLine();
        process.stdout.write(renderMarkdownLine(this.lineBuffer));
      }
      this.lineBuffer = "";
      this.rawLineLen = 0;
    }
    // If we're still in a code block at end of message, dump it
    if (this.inCodeBlock) {
      this.renderCodeBlock();
      this.inCodeBlock = false;
    }
    process.stdout.write("\n");
  }

  reset(): void {
    this.lineBuffer = "";
    this.rawLineLen = 0;
    this.inCodeBlock = false;
    this.codeBlockLang = "";
    this.codeBlockLines = [];
  }
}

const md = new StreamMarkdownRenderer();

// ── Streaming text output ────────────────────────────────────────────────────

let muted = false;

export function setMuted(value: boolean): void {
  muted = value;
}

export function isMuted(): boolean {
  return muted;
}

/** Write a streaming text chunk from the agent (text_delta events). */
export function writeText(chunk: string): void {
  if (!muted) md.feed(chunk);
}

/** Flush any buffered text and write a trailing newline after a complete message. */
export function flushText(): void {
  if (!muted) md.flush();
  md.reset();
}

// ── Tool display ─────────────────────────────────────────────────────────────

export function formatToolSummary(name: string, args: any): string {
  try {
    switch (name) {
      case "Bash":
        return chalk.dim(truncate(args?.command ?? "", 80));
      case "Read":
        return chalk.dim(`${args?.path ?? ""}${args?.offset ? ` @${args.offset}` : ""}${args?.limit ? ` (${args.limit} lines)` : ""}`);
      case "Write":
        return chalk.dim(args?.path ?? "");
      case "Edit":
        return chalk.dim(args?.path ?? "");
      case "Grep":
        return chalk.dim(`"${truncate(args?.pattern ?? "", 30)}"${args?.glob ? ` in ${args.glob}` : ""}${args?.path ? ` @ ${truncate(args.path, 30)}` : ""}`);
      case "find":
        return chalk.dim(`${args?.pattern ?? ""}${args?.path ? ` in ${truncate(args.path, 40)}` : ""}`);
      case "ls":
        return chalk.dim(args?.path ?? ".");
      case "recall":
        return chalk.dim(`"${truncate(args?.query ?? "", 50)}"${args?.scope ? ` [${args.scope}]` : ""}`);
      case "subagent":
        return chalk.dim(`${args?.mode ?? ""}: ${truncate(args?.task ?? "", 50)}`);
      default:
        return "";
    }
  } catch (e) { swallow("render:tool-summary", e); return ""; }
}

export function formatToolResult(name: string, args: any, result: any, isError: boolean): string {
  if (isError) {
    const errText = result?.content?.[0]?.text ?? "error";
    return chalk.red(truncate(String(errText), 60));
  }
  try {
    const text = String(result?.content?.[0]?.text ?? "");
    switch (name) {
      case "Bash": {
        const lines = text.split("\n").filter(Boolean).length;
        const exitMatch = text.match(/exit code:\s*(\d+)/i);
        const exitCode = exitMatch ? exitMatch[1] : "0";
        return chalk.dim(`${lines} lines${exitCode !== "0" ? ` (exit ${exitCode})` : ""}`);
      }
      case "Read": {
        const lines = text.split("\n").length;
        return chalk.dim(`${lines} lines`);
      }
      case "Write":
        return chalk.dim("written");
      case "Edit":
        return text.includes("Successfully") ? chalk.dim("ok") : chalk.dim(truncate(text, 40));
      case "Grep": {
        const matches = text.split("\n").filter(Boolean).length;
        if (text.includes("No matches")) return chalk.dim("no matches");
        return chalk.dim(`${matches} matches`);
      }
      case "find": {
        const files = text.split("\n").filter(Boolean).length;
        return chalk.dim(`${files} files`);
      }
      case "ls": {
        const entries = text.split("\n").filter(Boolean).length;
        return chalk.dim(`${entries} entries`);
      }
      case "recall": {
        if (text.includes("No relevant")) return chalk.dim("nothing found");
        const sections = (text.match(/\[(memory|concept|artifact|turn|skill)\]/gi) ?? []).length;
        return chalk.dim(`${sections} results`);
      }
      default:
        return chalk.dim("done");
    }
  } catch (e) { swallow("render:tool-result", e); return chalk.dim("done"); }
}

const toolStartTimes = new Map<number, number>();

export function writeToolStart(count: number, name: string, args: any): void {
  toolStartTimes.set(count, Date.now());
  const summary = formatToolSummary(name, args);
  const w = Math.min(cols() - 4, 76);
  const header = ` ${name} `;
  const rule = "─".repeat(Math.max(w - header.length - 2, 4));
  process.stdout.write(`\n  ${chalk.dim("╭─")}${chalk.cyan(header)}${chalk.dim(rule + "╮")}\n`);
  if (summary) {
    process.stdout.write(`  ${chalk.dim("│")} ${summary}\n`);
  }
}

export function writeToolEnd(count: number, name: string, args: any, result: any, isError: boolean): void {
  const summary = formatToolResult(name, args, result, isError);
  const startTime = toolStartTimes.get(count);
  toolStartTimes.delete(count);
  const elapsed = startTime ? ((Date.now() - startTime) / 1000).toFixed(1) : "?";

  const icon = isError ? chalk.red("✗") : chalk.green("✓");
  const durationStr = chalk.dim(`${elapsed}s`);
  const w = Math.min(cols() - 4, 76);
  const footer = ` ${icon} ${summary} ${durationStr} `;
  // Strip ANSI for length calc
  const footerLen = footer.replace(/\x1b\[[0-9;]*m/g, "").length;
  const rule = "─".repeat(Math.max(w - footerLen - 1, 4));
  process.stdout.write(`  ${chalk.dim("╰─")}${footer}${chalk.dim(rule + "╯")}\n`);
}

// ── Stats display ────────────────────────────────────────────────────────────

export interface TurnStats {
  tokensIn: number;
  tokensOut: number;
  cacheRead: number;
  cacheWrite: number;
  cost: number;
}

export interface ContextStats {
  mode: string;
  fullHistoryTokens: number;
  sentTokens: number;
  reductionPct: number;
  graphNodes: number;
  neighborNodes: number;
  recentTurns: number;
  cumFullTokens: number;
  cumSentTokens: number;
  prefetchHit?: boolean;
}

export function writeStats(turn: TurnStats, context: ContextStats | null, toolCallsThisTurn: number, sessionToolCalls: number): void {
  console.log("");
  console.log(
    `   [tokens: ${turn.tokensIn}in ${turn.tokensOut}out | cache: ${turn.cacheRead}r/${turn.cacheWrite}w | $${turn.cost.toFixed(4)}]`,
  );
  if (context) {
    const contextDelta = context.fullHistoryTokens - context.sentTokens;
    let contextLabel: string;
    if (contextDelta > 0) {
      contextLabel = `${context.reductionPct.toFixed(0)}% saved`;
    } else if (contextDelta < 0) {
      const injectedPct = context.fullHistoryTokens > 0
        ? ((-contextDelta / context.fullHistoryTokens) * 100).toFixed(0)
        : "0";
      contextLabel = `+${injectedPct}% context added`;
    } else {
      contextLabel = "even";
    }

    const cumDelta = context.cumFullTokens - context.cumSentTokens;
    let sessionLabel: string;
    if (cumDelta >= 0) {
      const cumPct = context.cumFullTokens > 0 ? ((cumDelta / context.cumFullTokens) * 100).toFixed(1) : "0";
      sessionLabel = `${cumPct}% saved`;
    } else {
      const cumPct = context.cumFullTokens > 0 ? ((-cumDelta / context.cumFullTokens) * 100).toFixed(1) : "0";
      sessionLabel = `+${cumPct}% context`;
    }

    console.log(
      `   [context: ${context.mode} | ${context.fullHistoryTokens}→${context.sentTokens} est. tokens (${contextLabel}) | ${context.graphNodes} vector + ${context.neighborNodes} neighbor + ${context.recentTurns} recent]`,
    );
    console.log(
      `   [session: ~${context.cumFullTokens}→~${context.cumSentTokens} tokens (${sessionLabel}) | tools: ${toolCallsThisTurn} this turn / ${sessionToolCalls} session]`,
    );
  }
}

// ── Startup / Shutdown ───────────────────────────────────────────────────────

export function writeBanner(): void {
  console.log(`${chalk.bold.cyan("⟡")} ${chalk.bold("Zeraclaw")} ${chalk.dim("— Graph-backed AI CLI")}`);
  console.log(`   Type your message. ${chalk.dim("Ctrl+C to soft-interrupt, twice to exit.")}\n`);
}

export function writeReady(elapsed: string, greeting?: string, sessionId?: string): void {
  process.stdout.write(`\r   ${chalk.green("⚡")} Ready (${elapsed}s)` + " ".repeat(20) + "\n");
  if (greeting) {
    console.log(`   💀 ${greeting}`);
  }
  if (sessionId) {
    console.log(`   ${chalk.dim(`Session: ${sessionId}`)}\n`);
  }
}

export function writeSpinner(frame: string, message: string, elapsed: string): void {
  process.stdout.write(`\r   ${frame} ${message} ${elapsed}s`);
}

export function writeShutdownSummary(turns: number, toolCalls: number, cost: number, cumFull: number, cumSent: number): void {
  console.log("");
  console.log(`   Session: ${turns} turns | ${toolCalls} tool calls | $${cost.toFixed(4)}`);
  if (cumFull > 0) {
    const delta = cumFull - cumSent;
    if (delta >= 0) {
      const pct = ((delta / cumFull) * 100).toFixed(1);
      console.log(`   Tokens:  ~${cumFull} would-be -> ~${cumSent} actual (${pct}% saved by graph)`);
    } else {
      console.log(`   Tokens:  ~${cumFull} base + ${-delta} graph context -> ~${cumSent} actual`);
    }
  }
  console.log("");
}

export function writeExitLine(line: string | null, timedOut: boolean): void {
  if (!line) {
    if (timedOut) {
      swallow.warn("render:exitLine", new Error("exit line timed out"));
    }
    console.log("   Until next time.");
  }
  console.log("");
}

// ── Preflight / Intent ───────────────────────────────────────────────────────

export function writePreflight(category: string, thinkingLevel: string, toolLimit: number | string, preflightMs: number): void {
  const limitStr = toolLimit === Infinity ? "∞" : String(toolLimit);
  console.log(`   ${chalk.dim(`[${category} | thinking: ${thinkingLevel} | tools: ${limitStr} | ${preflightMs.toFixed(0)}ms]`)}`);
}

// ── General output ───────────────────────────────────────────────────────────

export function writeLine(text: string): void {
  console.log(text);
}

export function writeInfo(text: string): void {
  console.log(`   ${text}`);
}

export function writeError(text: string): void {
  console.log(`   ${chalk.red(text)}`);
}

export function writeSuccess(text: string): void {
  console.log(`   ${chalk.green(text)}`);
}

export function writeDim(text: string): void {
  console.log(`   ${chalk.dim(text)}`);
}

// ── Prompt ───────────────────────────────────────────────────────────────────

export function getPromptString(): string {
  return `${chalk.dim("─".repeat(Math.min(cols(), 60)))}\n${chalk.bold.cyan("❯")} `;
}

// ── Detailed stats (/stats command) ──────────────────────────────────────────

export interface QualityStats {
  totalRetrievals: number;
  avgUtilization: number;
  wastedTokens: number;
  toolSuccessRate: number | null;
}

export async function writeDetailedStats(
  context: ContextStats,
  quality: QualityStats | null,
  reflectionCount: number,
): Promise<void> {
  const cumDelta = context.cumFullTokens - context.cumSentTokens;
  const turnDelta = context.fullHistoryTokens - context.sentTokens;
  const turnLabel = turnDelta >= 0
    ? `${context.reductionPct.toFixed(1)}% saved`
    : `+${context.fullHistoryTokens > 0 ? ((-turnDelta / context.fullHistoryTokens) * 100).toFixed(1) : "0"}% context added`;
  const sessionLabel = cumDelta >= 0
    ? `${context.cumFullTokens > 0 ? ((cumDelta / context.cumFullTokens) * 100).toFixed(1) : "0"}% saved`
    : `+${context.cumFullTokens > 0 ? ((-cumDelta / context.cumFullTokens) * 100).toFixed(1) : "0"}% context`;

  console.log(`\n   ${chalk.bold("── Token Stats ──")}`);
  console.log(`   Last turn:    ${context.fullHistoryTokens} full → ${context.sentTokens} sent (${turnLabel})`);
  console.log(`   Session:      ${context.cumFullTokens} full → ${context.cumSentTokens} sent (${sessionLabel})`);
  console.log(`   Mode:         ${context.mode}`);
  console.log(`   Graph nodes:  ${context.graphNodes} vector + ${context.neighborNodes} neighbor | Recent turns: ${context.recentTurns}`);

  if (quality) {
    const utilPct = (quality.avgUtilization * 100).toFixed(0);
    const toolStr = quality.toolSuccessRate != null ? `${(quality.toolSuccessRate * 100).toFixed(0)}%` : "n/a";
    console.log(`   ${chalk.bold("── Retrieval Quality ──")}`);
    console.log(`   Retrievals:   ${quality.totalRetrievals} | Avg utilization: ${utilPct}%`);
    console.log(`   Wasted tokens: ~${quality.wastedTokens} | Tool success: ${toolStr}`);
  }

  console.log(`   ${chalk.bold("── Cognitive ──")}`);
  if (reflectionCount > 0) {
    console.log(`   Reflections:  ${reflectionCount} stored`);
  }

  // Engine status — show what's active
  const engineStatus: string[] = [];
  try {
    const { isRerankerActive } = await import("./graph-context.js");
    const { isACANActive } = await import("./acan.js");
    const { getCacheSize, drainCacheHitCount } = await import("./embeddings.js");
    engineStatus.push(`Reranker: ${isRerankerActive() ? chalk.green("active") : chalk.yellow("off")}`);
    engineStatus.push(`ACAN: ${isACANActive() ? chalk.green("active") : chalk.yellow("dormant")}`);
    engineStatus.push(`Embed cache: ${getCacheSize()} entries`);
  } catch { /* silent */ }
  if (engineStatus.length > 0) {
    console.log(`   Engine:       ${engineStatus.join(" | ")}`);
  }
}

// ── Help ─────────────────────────────────────────────────────────────────────

// ── Status bar ───────────────────────────────────────────────────────────────

const statusBarEnabled = process.env.ZERACLAW_STATUSBAR === "1" && process.stdout.isTTY;

interface StatusBarState {
  sessionId: string;
  turns: number;
  toolCalls: number;
  cost: number;
  tokensIn: number;
  tokensOut: number;
}

let statusState: StatusBarState | null = null;
let statusBarInitialized = false;

export function initStatusBar(sessionId: string): void {
  if (!statusBarEnabled) return;
  statusState = { sessionId, turns: 0, toolCalls: 0, cost: 0, tokensIn: 0, tokensOut: 0 };
  const rows = process.stdout.rows || 24;
  // Set scroll region to exclude bottom row
  process.stdout.write(`\x1b[1;${rows - 1}r`);
  process.stdout.write(`\x1b[${rows - 1};1H`); // move cursor to bottom of scroll region
  statusBarInitialized = true;
  renderStatusBar();

  process.stdout.on("resize", () => {
    if (!statusBarEnabled || !statusBarInitialized) return;
    const r = process.stdout.rows || 24;
    process.stdout.write(`\x1b[1;${r - 1}r`);
    renderStatusBar();
  });
}

export function updateStatusBar(update: Partial<StatusBarState>): void {
  if (!statusBarEnabled || !statusState) return;
  Object.assign(statusState, update);
  renderStatusBar();
}

function renderStatusBar(): void {
  if (!statusState || !statusBarInitialized) return;
  const rows = process.stdout.rows || 24;
  const c = cols();
  const sid = statusState.sessionId.length > 20 ? statusState.sessionId.slice(0, 20) + "…" : statusState.sessionId;
  const bar = ` ${sid} │ turns: ${statusState.turns} │ tools: ${statusState.toolCalls} │ $${statusState.cost.toFixed(3)} │ ${statusState.tokensIn}in/${statusState.tokensOut}out `;
  const padded = bar.length < c ? bar + " ".repeat(c - bar.length) : bar.slice(0, c);

  // Save cursor, move to bottom, write bar, restore cursor
  process.stdout.write("\x1b7"); // save
  process.stdout.write(`\x1b[${rows};1H`); // move to status row
  process.stdout.write(`\x1b[7m${padded}\x1b[0m`); // reverse video
  process.stdout.write("\x1b8"); // restore
}

export function destroyStatusBar(): void {
  if (!statusBarEnabled || !statusBarInitialized) return;
  // Reset scroll region to full terminal
  process.stdout.write(`\x1b[r`);
  // Clear the status bar row
  const rows = process.stdout.rows || 24;
  process.stdout.write(`\x1b[${rows};1H\x1b[2K`);
  statusBarInitialized = false;
}

// ── Help ─────────────────────────────────────────────────────────────────────

export function writeHelp(): void {
  console.log(`\n   ${chalk.bold("Commands:")}`);
  console.log(`   ${chalk.cyan("/compare")} ${chalk.dim("<prompt>")}  — Run prompt with full vs graph context, show A/B comparison`);
  console.log(`   ${chalk.cyan("/eval")}              — Run 5-prompt eval suite, show quality + token summary`);
  console.log(`   ${chalk.cyan("/stats")}             — Show context, retrieval, and tool stats for this session`);
  console.log(`   ${chalk.cyan("/unlimited")}         — Remove tool call limit for next prompt`);
  console.log(`   ${chalk.cyan("/spawn")} ${chalk.dim("<mode> <task>")} — Spawn subagent (mode: full|incognito)`);
  console.log(`   ${chalk.cyan("/merge")} ${chalk.dim("<id>")}        — Merge incognito agent's knowledge into parent graph`);
  console.log(`   ${chalk.cyan("/agents")}            — List spawned subagents`);
  console.log(`   ${chalk.cyan("/wakeup")}            — Set identity (from WAKEUP.md or conversationally)`);
  console.log(`   ${chalk.cyan("/export-training")}   — Export ACAN training data to JSONL`);
  console.log(`   ${chalk.cyan("/quit")}              — Exit`);
}
