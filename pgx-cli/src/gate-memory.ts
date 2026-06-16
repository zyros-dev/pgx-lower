import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

export type GateFailureState = {
  gate: "review";
  head: string;
  stepName: string;
  focusedCommand: string[];
  runId?: string;
  transcript?: string;
};

export type GateDecision = {
  blocked: boolean;
  message: string;
  stale?: boolean;
  override?: boolean;
};

export type GateFailureInput = {
  gate: "review";
  head: string;
  stepName: string;
  stepCommand: string[];
  root?: string;
  statePath?: string;
  runId?: string;
  transcript?: string;
};

export function gateFailureStatePath(root: string): string {
  return join(root, ".pgx-cli", "gate-state", "review.json");
}

export function recordGateFailure(input: GateFailureInput): GateFailureState {
  const state = {
    gate: input.gate,
    head: input.head,
    stepName: input.stepName,
    focusedCommand: [...input.stepCommand],
    ...(input.runId ? { runId: input.runId } : {}),
    ...(input.transcript ? { transcript: input.transcript } : {})
  } satisfies GateFailureState;

  if (input.root || input.statePath) {
    writeGateFailureState(state, {
      root: input.root,
      statePath: input.statePath
    });
  }

  return state;
}

export function readGateFailureState(input: { root?: string; statePath?: string }): GateFailureState | undefined {
  const path = resolveStatePath(input);
  if (!existsSync(path)) return undefined;
  const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<GateFailureState>;
  if (
    parsed.gate !== "review" ||
    typeof parsed.head !== "string" ||
    typeof parsed.stepName !== "string" ||
    !Array.isArray(parsed.focusedCommand) ||
    !parsed.focusedCommand.every((part) => typeof part === "string")
  ) {
    return undefined;
  }
  return {
    gate: "review",
    head: parsed.head,
    stepName: parsed.stepName,
    focusedCommand: parsed.focusedCommand,
    ...(typeof parsed.runId === "string" ? { runId: parsed.runId } : {}),
    ...(typeof parsed.transcript === "string" ? { transcript: parsed.transcript } : {})
  };
}

export function writeGateFailureState(
  state: GateFailureState,
  input: { root?: string; statePath?: string }
): void {
  const path = resolveStatePath(input);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(state, null, 2)}\n`);
}

export function clearGateFailureState(input: { root?: string; statePath?: string }): void {
  rmSync(resolveStatePath(input), { force: true });
}

export function satisfyGateFailureState(input: {
  root?: string;
  statePath?: string;
  currentHead: string;
  argv: string[];
}): GateFailureState | undefined {
  const previousFailure = readGateFailureState(input);
  if (!previousFailure) return undefined;
  if (previousFailure.head !== input.currentHead) return undefined;
  if (!focusedCommandMatches({ previousFailure, argv: input.argv })) return undefined;
  clearGateFailureState(input);
  return previousFailure;
}

export function shouldBlockReviewGate(input: {
  strict: boolean;
  currentHead: string;
  previousFailure?: GateFailureState;
  argv: string[];
}): GateDecision {
  const previous = input.previousFailure;
  if (!previous) return { blocked: false, message: "" };
  if (previous.head !== input.currentHead) {
    return {
      blocked: false,
      stale: true,
      message: `gate memory: previous review failure was for ${previous.head}; current HEAD is ${input.currentHead}, clearing stale review gate memory`
    };
  }
  if (hasFullGateOverride(input.argv)) {
    return {
      blocked: false,
      override: true,
      message: `gate memory: full review gate override accepted for ${previous.stepName}; expected focused reproducer: ${formatCommand(previous.focusedCommand)}`
    };
  }

  const message = [
    "gate memory: run focused reproducer first before rerunning the full review gate",
    `failed step: ${previous.stepName}`,
    `focused command: ${formatCommand(previous.focusedCommand)}`,
    ...(previous.runId ? [`previous run: ${previous.runId}`] : []),
    ...(previous.transcript ? [`transcript: ${previous.transcript}`] : []),
    "override: pgx-cli dev gate review --rerun-full"
  ].join("\n");
  return { blocked: input.strict, message };
}

export function focusedCommandMatches(input: { previousFailure: GateFailureState; argv: string[] }): boolean {
  return commandsEqual(normalizeCommand(input.previousFailure.focusedCommand), normalizeCommand(input.argv));
}

function resolveStatePath(input: { root?: string; statePath?: string }): string {
  if (input.statePath) return input.statePath;
  if (!input.root) throw new Error("gate memory requires root or statePath");
  return gateFailureStatePath(input.root);
}

function hasFullGateOverride(argv: string[]): boolean {
  return argv.includes("--rerun-full") || argv.includes("--force-full-gate");
}

function normalizeCommand(command: string[]): string[] {
  if (command[0] !== "pgx-cli") return command;
  if (command[1] === "dev") return command.slice(2);
  return command.slice(1);
}

function commandsEqual(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((part, index) => part === right[index]);
}

function formatCommand(command: string[]): string {
  return command.join(" ");
}
