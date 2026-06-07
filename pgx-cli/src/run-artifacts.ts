import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { randomBytes } from "node:crypto";

export type RunArtifactPaths = {
  runId: string;
  runDir: string;
  commandPath: string;
  stdoutPath: string;
  stderrPath: string;
  combinedPath: string;
  syncPreflightPath: string;
  summaryPath: string;
  artifactsPath: string;
};

export function sanitizeRunName(value: string): string {
  const sanitized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-+/g, "-");
  return sanitized || "run";
}

export function createRunArtifactPaths(input: {
  root: string;
  commandName: string;
  transcriptDir?: string;
  now?: Date;
  randomSuffix?: string;
}): RunArtifactPaths {
  const timestamp = (input.now ?? new Date()).toISOString().replace(/[:.]/g, "-");
  const suffix = input.randomSuffix ?? randomBytes(3).toString("hex");
  const runId = `${timestamp}-${sanitizeRunName(input.commandName)}-${suffix}`;
  const runDir = join(input.root, input.transcriptDir ?? ".pgx-cli/runs", runId);
  mkdirSync(runDir, { recursive: true });
  return {
    runId,
    runDir,
    commandPath: join(runDir, "command.txt"),
    stdoutPath: join(runDir, "stdout.log"),
    stderrPath: join(runDir, "stderr.log"),
    combinedPath: join(runDir, "combined.log"),
    syncPreflightPath: join(runDir, "sync-preflight.log"),
    summaryPath: join(runDir, "summary.json"),
    artifactsPath: join(runDir, "artifacts.txt")
  };
}

export function writeCommand(paths: RunArtifactPaths, argv: string[]): void {
  writeFileSync(paths.commandPath, `${argv.join(" ")}\n`);
}

export function writeRunSummary(paths: RunArtifactPaths, summary: Record<string, unknown>): void {
  writeFileSync(paths.summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
}

export function appendArtifactPath(paths: RunArtifactPaths, path: string): void {
  appendFileSync(paths.artifactsPath, `${path}\n`);
}
