import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationRunner } from "./managed-operations.js";
import type { OperationOutput } from "./operations.js";
import type { ResolvedOutputConfig, ResolvedSyncConfig } from "./project-config.js";

export type LogsConfig = {
  localProjectPath: string;
  output: ResolvedOutputConfig;
  mutagenSession?: string;
  sshHost?: string;
  remoteProjectPath?: string;
  dockerContainer?: string;
  sync?: ResolvedSyncConfig;
  runningOnRemote?: boolean;
};

const defaultLines = 50;
const pgxErrorsPath = "/tmp/pgx_errors.log";

export async function runLogsCommand(
  args: string[],
  runner: ManagedOperationRunner,
  output: OperationOutput,
  config: LogsConfig
): Promise<number> {
  const [command, idOrFlag, maybeValue] = args;
  if (command === "show" && idOrFlag) {
    return showRun(idOrFlag, args.slice(2), output, config);
  }
  if (command === "latest") {
    const latest = latestRunId(config);
    if (!latest) {
      output.stderr += "logs latest: no run artifacts found\n";
      return 1;
    }
    return showRun(latest, args.slice(1), output, config);
  }

  if ((command === "errors" || command === "docker" || command === "file") && isRemoteLogsConfig(config)) {
    const parsed = parseRemoteLogArgs(args);
    if (!parsed) {
      output.stderr += "Usage: logs <show <run-id>|latest|errors|docker|file <path>> [--head N|--tail N|--lines N|--full]\n";
      return 1;
    }
    const shellCommand = parsed.kind === "errors"
      ? `docker exec ${quoteShell(config.dockerContainer)} bash -lc ${quoteShell(`tail -n ${parsed.lines} ${pgxErrorsPath} 2>/dev/null || true`)}`
      : parsed.kind === "docker"
        ? `docker logs --tail ${parsed.lines} ${quoteShell(config.dockerContainer)}`
        : `tail -n ${parsed.lines} ${quoteShell(parsed.path)}`;
    const result = await runManagedRemoteShell({
      runner,
      output,
      config,
      commandName: `logs-${command}`,
      shellCommand,
      requireMutagenProof: false,
      mutagenProofSkipReason: "remote-only logs command"
    });
    return result.workflowExitCode;
  }

  output.stderr += "Usage: logs <show <run-id>|latest|errors|docker|file <path>> [--head N|--tail N|--lines N|--full]\n";
  return 1;
}

function showRun(runId: string, args: string[], output: OperationOutput, config: LogsConfig): number {
  const runDir = join(config.localProjectPath, config.output.transcript_dir, runId);
  const combinedPath = join(runDir, "combined.log");
  if (!existsSync(combinedPath)) {
    output.stderr += `logs show: missing combined transcript for ${runId}\n`;
    return 1;
  }
  const summaryPath = join(runDir, "summary.json");
  const summary = existsSync(summaryPath) ? readFileSync(summaryPath, "utf8").trim() : "";
  const full = args.includes("--full");
  const head = parseHead(args);
  const tail = parseTail(args);
  const text = readFileSync(combinedPath, "utf8");
  const excerpt = full
    ? text
    : head
      ? headLines(text, head)
      : tailLines(text, tail ?? config.output.failure_tail_lines);
  output.stdout += `run id: ${runId}\n`;
  if (summary) output.stdout += `summary: ${summaryPath}\n`;
  output.stdout += `transcript: ${combinedPath}\n`;
  output.stdout += excerpt.endsWith("\n") ? excerpt : `${excerpt}\n`;
  return 0;
}

function latestRunId(config: LogsConfig): string | undefined {
  const runsDir = join(config.localProjectPath, config.output.transcript_dir);
  if (!existsSync(runsDir)) return undefined;
  return readdirSync(runsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((left, right) => {
      const leftTime = statSync(join(runsDir, left)).mtimeMs;
      const rightTime = statSync(join(runsDir, right)).mtimeMs;
      return leftTime === rightTime ? left.localeCompare(right) : leftTime - rightTime;
    })
    .at(-1);
}

type RemoteLogArgs =
  | { kind: "errors"; lines: number }
  | { kind: "docker"; lines: number }
  | { kind: "file"; path: string; lines: number };

function parseRemoteLogArgs(args: string[]): RemoteLogArgs | undefined {
  const [command, ...rest] = args;
  if (command === "errors") {
    const lines = parseLineCount(rest);
    return lines === undefined ? undefined : { kind: "errors", lines };
  }
  if (command === "docker") {
    const lines = parseLineCount(rest);
    return lines === undefined ? undefined : { kind: "docker", lines };
  }
  if (command === "file") {
    const [path, ...options] = rest;
    if (!path) return undefined;
    const lines = parseLineCount(options);
    return lines === undefined ? undefined : { kind: "file", path, lines };
  }
  return undefined;
}

function parseLineCount(args: string[]): number | undefined {
  if (args.length === 0) {
    return defaultLines;
  }
  if (args.length !== 2 || !["--tail", "--lines", "-n"].includes(args[0])) {
    return undefined;
  }
  const value = Number(args[1]);
  return Number.isInteger(value) && value > 0 ? value : undefined;
}

function parseTail(args: string[]): number | undefined {
  const index = args.indexOf("--tail");
  if (index === -1) return undefined;
  const value = Number(args[index + 1]);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

function parseHead(args: string[]): number | undefined {
  const index = args.indexOf("--head");
  if (index === -1) return undefined;
  const value = Number(args[index + 1]);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

function tailLines(text: string, count: number): string {
  const lines = text.endsWith("\n") ? text.slice(0, -1).split("\n") : text.split("\n");
  return `${lines.slice(Math.max(0, lines.length - count)).join("\n")}\n`;
}

function headLines(text: string, count: number): string {
  const lines = text.endsWith("\n") ? text.slice(0, -1).split("\n") : text.split("\n");
  return `${lines.slice(0, count).join("\n")}\n`;
}

function isRemoteLogsConfig(config: LogsConfig): config is LogsConfig & {
  mutagenSession: string;
  sshHost: string;
  remoteProjectPath: string;
  dockerContainer: string;
  sync: ResolvedSyncConfig;
} {
  return !!config.mutagenSession && !!config.sshHost && !!config.remoteProjectPath && !!config.dockerContainer && !!config.sync;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
