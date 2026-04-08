/**
 * TUI component wrappers — reusable building blocks for the Zeraclaw TUI.
 *
 * Follows pi-tui patterns: extend Container, expose high-level APIs,
 * store references to inner components for targeted updates.
 */
import chalk from "chalk";
import {
  Box,
  Container,
  Loader,
  Markdown,
  type MarkdownTheme,
  Spacer,
  Text,
  type TUI,
} from "@mariozechner/pi-tui";
import { formatToolSummary, formatToolResult } from "./render.js";

// ── Helpers ──────────────────────────────────────────────────────────────────

function cols(): number {
  return process.stdout.columns || 80;
}

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, "");
}

export function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "k";
  return String(n);
}

// ── AlignedContainer ──────────────────────────────────────────────────────────

/** Wraps children at a % of viewport width, optionally right-aligned (left-padded). */
class AlignedContainer extends Container {
  private pct: number;
  private rightAlign: boolean;

  constructor(pct: number, rightAlign = false) {
    super();
    this.pct = pct;
    this.rightAlign = rightAlign;
  }

  render(width: number): string[] {
    const innerW = Math.max(20, Math.floor(width * this.pct));
    const lines = super.render(innerW);
    if (this.rightAlign) {
      const pad = " ".repeat(Math.max(0, width - innerW));
      return lines.map((l) => pad + l);
    }
    return lines;
  }
}

// ── UserMessage ──────────────────────────────────────────────────────────────

/** Styled echo of the user's input — right-aligned header for visual separation. */
export class UserMessage extends Container {
  constructor(text: string) {
    super();
    const w = cols() - 2; // subtract margin to prevent line wrap
    const label = " You ";
    const ruleL = Math.max(0, w - stripAnsi(label).length - 6);
    const ruleR = 6;
    // Header leans right (many dashes left, label at right) to match right-aligned body
    const header = chalk.dim("─".repeat(ruleL)) + chalk.bold.cyan(label) + chalk.dim("─".repeat(ruleR));
    this.addChild(new Text(header, 1, 0));
    const bodyWrap = new AlignedContainer(0.8, true); // right-aligned
    bodyWrap.addChild(new Text(chalk.white(text), 0, 1));
    this.addChild(bodyWrap);
  }
}

// ── AssistantMessage ─────────────────────────────────────────────────────────

/** Wraps a Markdown component for streaming LLM output — left-aligned body, left-leaning header. */
export class AssistantMessage extends Container {
  private body: Markdown;

  constructor(mdTheme: MarkdownTheme) {
    super();
    const w = cols() - 2; // subtract margin to prevent line wrap
    const label = " ⟡ Zeraclaw ";
    const ruleL = 6;
    const ruleR = Math.max(0, w - stripAnsi(label).length - ruleL);
    // Header leans left (label at left, many dashes right) to match left-aligned body
    this.addChild(new Text(chalk.dim("─".repeat(ruleL)) + chalk.magenta(label) + chalk.dim("─".repeat(ruleR)), 1, 0));
    const bodyWrap = new AlignedContainer(0.8); // left-aligned
    this.body = new Markdown("", 0, 0, mdTheme);
    bodyWrap.addChild(this.body);
    this.addChild(bodyWrap);
  }

  setText(text: string): void {
    this.body.setText(text);
  }
}

// ── SystemMessage ────────────────────────────────────────────────────────────

/** Plain text system message (info, warnings, help output, etc.). */
export class SystemMessage extends Container {
  private body: Text;

  constructor(text: string) {
    super();
    this.body = new Text(text, 1, 0);
    this.addChild(this.body);
  }

  setText(text: string): void {
    this.body.setText(text);
  }
}

// ── ToolExecutionComponent ───────────────────────────────────────────────────

/**
 * Stateful tool display: header on start, footer on end.
 * Tracked by toolCallId so start/end events pair correctly.
 */
export class ToolExecutionComponent extends Container {
  private headerText: Text;
  private footerText: Text | null = null;
  readonly toolName: string;
  private startTime: number;

  constructor(toolName: string, args: any) {
    super();
    this.toolName = toolName;
    this.startTime = Date.now();

    // Build header box
    const w = Math.min(cols() - 4, 76);
    const label = ` ${toolName} `;
    const labelLen = stripAnsi(label).length;
    const rule = "─".repeat(Math.max(w - labelLen - 2, 4));
    let header = chalk.dim("╭─") + chalk.cyan(label) + chalk.dim(rule + "╮");

    const summary = formatToolSummary(toolName, args);
    if (summary) header += "\n" + chalk.dim("│") + " " + summary;

    this.headerText = new Text(header, 1, 0);
    this.addChild(this.headerText);
  }

  /** Call when the tool execution completes. */
  setResult(result: any, isError: boolean, args: any): void {
    const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
    const summary = formatToolResult(this.toolName, args, result, isError);
    const icon = isError ? chalk.red("✗") : chalk.green("✓");
    const durationStr = chalk.dim(`${elapsed}s`);

    const w = Math.min(cols() - 4, 76);
    const footerInner = ` ${icon} ${summary} ${durationStr} `;
    const footerLen = stripAnsi(footerInner).length;
    const rule = "─".repeat(Math.max(w - footerLen - 1, 4));
    const footerLine = chalk.dim("╰─") + footerInner + chalk.dim(rule + "╯");

    this.footerText = new Text(footerLine, 0, 0);
    this.addChild(this.footerText);
  }
}

// ── StatusLine ───────────────────────────────────────────────────────────────

/**
 * Animated status bar that switches between Loader (busy) and Text (idle).
 */
export class StatusLine extends Container {
  private tui: TUI;
  private loader: Loader | null = null;
  private idleText: Text;
  private _busy = false;

  constructor(tui: TUI) {
    super();
    this.tui = tui;
    this.idleText = new Text("", 1, 0);
    this.addChild(this.idleText);
  }

  get busy(): boolean {
    return this._busy;
  }

  /** Switch to animated spinner with message. */
  setBusy(message: string): void {
    if (this._busy && this.loader) {
      // Already busy — just update message
      this.loader.setMessage(message);
      return;
    }
    this._busy = true;
    this.clear();
    // (Spacer removed — caused 1-row height jump vs idle state)
    this.loader = new Loader(this.tui, (t) => chalk.cyan(t), (t) => chalk.dim(t), message);
    this.addChild(this.loader);
    this.loader.start();
  }

  /** Switch to static text (or empty). */
  setIdle(text = ""): void {
    if (this.loader) {
      this.loader.stop();
      this.loader = null;
    }
    this._busy = false;
    this.clear();
    // Use a single Text row (space if empty) to match Loader's 1-row height.
    // Spacer(1) was removed — it caused a persistent blank line below footer content.
    this.idleText = new Text(text || " ", 0, 0);
    this.addChild(this.idleText);
  }

  /** Show intent classification result. */
  setIntent(category: string, thinkingLevel: string, toolLimit: number, ms: number): void {
    this.setIdle(
      chalk.dim(`intent: ${category} · thinking: ${thinkingLevel} · budget: ${toolLimit} tools · ${ms}ms`),
    );
  }

  /** Cleanup — stop any running loader. */
  dispose(): void {
    if (this.loader) {
      this.loader.stop();
      this.loader = null;
    }
  }
}

// ── FooterBar ────────────────────────────────────────────────────────────────

export interface FooterStats {
  turns: number;
  toolCalls: number;
  cost: number;
  tokensIn: number;
  tokensOut: number;
  model?: string;
  compressionPct?: number;
  // Per-turn stats (populated after each response)
  lastCtxSent?: number;
  lastCtxFull?: number;
  lastTurnTools?: number;
  lastTurnSessionTools?: number;
  graphNodes?: number;
}

/** Bottom bar with session stats. */
export class FooterBar extends Container {
  private body: Text;

  constructor() {
    super();
    this.body = new Text("", 0, 0); // no top margin — was causing blank line above footer stats
    this.addChild(this.body);
  }

  update(stats: FooterStats): void {
    const div = chalk.dim(" │ ");

    // Row 1 — session totals (always visible)
    const row1: string[] = [];
    if (stats.model) row1.push(chalk.dim(stats.model));
    row1.push(chalk.dim(`turns: ${stats.turns}`));
    row1.push(chalk.dim(`tools: ${stats.toolCalls}`));
    row1.push(chalk.dim(`$${stats.cost.toFixed(2)}`));
    row1.push(chalk.dim(`${formatTokenCount(stats.tokensIn)}↑ ${formatTokenCount(stats.tokensOut)}↓`));
    if (stats.compressionPct != null) row1.push(chalk.dim(`saved: ${stats.compressionPct.toFixed(0)}%`));
    const sessionLine = chalk.dim("── session  ") + row1.join(div);

    // Row 2 — last-turn breakdown (shown after first response)
    let turnLine = "";
    if (stats.lastCtxSent != null && stats.lastCtxSent > 0) {
      const row2: string[] = [];
      if (stats.lastCtxFull) {
        row2.push(chalk.dim(`ctx: ${formatTokenCount(stats.lastCtxSent)}/${formatTokenCount(stats.lastCtxFull)}`));
      }
      if (stats.graphNodes != null && stats.graphNodes > 0) {
        row2.push(chalk.dim(`graph: ${stats.graphNodes}`));
      }
      if (stats.lastTurnTools != null) {
        row2.push(chalk.dim(`tools: ${stats.lastTurnTools}/${stats.lastTurnSessionTools ?? stats.toolCalls}`));
      }
      turnLine = chalk.dim("─── last turn  ") + row2.join(div);
    }

    // Always render 2 lines — stable height prevents blank line below footer when turn data appears.
    // Same pattern as StatusLine.setIdle() which uses Text(" ") to match Loader height.
    this.body.setText((turnLine || " ") + "\n" + sessionLine);
  }
}
