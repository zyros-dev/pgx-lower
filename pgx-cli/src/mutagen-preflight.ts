import { appendFileSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { Writable } from "node:stream";
import type { StreamingCommandRunner, StreamingRunResult } from "./commands.js";
import type { RunArtifactPaths } from "./run-artifacts.js";

export type MutagenHealth =
  | { healthy: true; reason: string }
  | { healthy: false; reason: string };

export type MutagenPreflightResult =
  | { ok: true; message: string; transcriptPath?: string; proofSkippedReason?: string }
  | { ok: false; message: string; transcriptPath?: string; next: string[] };

export function evaluateMutagenListJson(json: string, sessionName: string): MutagenHealth {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return { healthy: false, reason: `mutagen session ${sessionName} malformed JSON` };
  }
  if (!Array.isArray(parsed)) {
    return { healthy: false, reason: `mutagen session ${sessionName} malformed JSON` };
  }
  const session = parsed.find((candidate) => stringField(candidate, "name", "Name") === sessionName);
  if (!isObject(session)) {
    return { healthy: false, reason: `mutagen session ${sessionName} missing` };
  }
  if (boolField(session, "paused", "Paused") === true) {
    return { healthy: false, reason: `mutagen session ${sessionName} paused` };
  }
  const alpha = objectField(session, "alpha", "Alpha");
  if (alpha && boolField(alpha, "connected", "Connected") !== true) {
    return { healthy: false, reason: `mutagen session ${sessionName} alpha disconnected` };
  }
  const beta = objectField(session, "beta", "Beta");
  if (beta && boolField(beta, "connected", "Connected") !== true) {
    return { healthy: false, reason: `mutagen session ${sessionName} beta disconnected` };
  }
  if (hasNonEmptyField(session, /conflict/i)) {
    return { healthy: false, reason: `mutagen session ${sessionName} conflict` };
  }
  if (hasNonEmptyField(session, /problem/i)) {
    return { healthy: false, reason: `mutagen session ${sessionName} problem` };
  }
  const status = stringField(session, "status", "Status");
  if (status && !new Set(["watching", "scanning", "reconciling"]).has(status)) {
    return { healthy: false, reason: `mutagen session ${sessionName} unsafe status: ${status}` };
  }
  return {
    healthy: true,
    reason: status
      ? `mutagen session ${sessionName} is healthy (${status})`
      : `mutagen session ${sessionName} is healthy`
  };
}

export async function runMutagenPreflight(input: {
  runner: StreamingCommandRunner;
  artifact?: RunArtifactPaths;
  sessionName: string;
  localProjectPath: string;
  remoteProjectPath: string;
  sshHost: string;
  runId: string;
  flushTimeoutSeconds: number;
  runningOnRemote?: boolean;
  requireProof: boolean;
  proofSkipReason?: string;
  proofPath?: string;
  probeContents?: string;
}): Promise<MutagenPreflightResult> {
  const transcriptPath = input.artifact?.syncPreflightPath;
  if (transcriptPath) {
    mkdirSync(dirname(transcriptPath), { recursive: true });
    writeFileSync(transcriptPath, "");
  }

  if (input.runningOnRemote) {
    writeTranscript(transcriptPath, "mutagen: skipped (already on thor)\n");
    return { ok: true, message: "mutagen: skipped (already on thor)", transcriptPath };
  }

  const timeoutMs = input.flushTimeoutSeconds * 1000;
  const firstHealth = await healthCheck(input.runner, input.sessionName, timeoutMs, transcriptPath);
  if (!firstHealth.ok && !firstHealth.message.includes("unsafe status")) {
    return blocked(firstHealth.message, transcriptPath);
  }

  const flush = await runText(input.runner, "mutagen", ["sync", "flush", input.sessionName], timeoutMs, transcriptPath);
  if (flush.timedOut || flush.result.childExitCode !== 0) {
    return blocked(`mutagen session ${input.sessionName} flush failed`, transcriptPath);
  }

  const secondHealth = await healthCheck(input.runner, input.sessionName, timeoutMs, transcriptPath);
  if (!secondHealth.ok) {
    return blocked(secondHealth.message, transcriptPath);
  }

  if (!input.requireProof) {
    const reason = input.proofSkipReason ?? "not required for this workflow";
    writeTranscript(transcriptPath, `sync proof: skipped (${reason})\n`);
    return {
      ok: true,
      message: `${secondHealth.message}; sync proof skipped: ${reason}`,
      transcriptPath,
      proofSkippedReason: reason
    };
  }

  const proofRelativePath = (input.proofPath ?? ".pgx-cli/sync-probes").replace(/^\/+/, "");
  const proofDir = join(input.localProjectPath, proofRelativePath);
  mkdirSync(proofDir, { recursive: true });
  const probePath = join(proofDir, `${input.runId}.txt`);
  const probeContents = input.probeContents ?? `probe-${input.runId}\n`;
  writeFileSync(probePath, probeContents);

  try {
    const proofFlush = await runText(input.runner, "mutagen", ["sync", "flush", input.sessionName], timeoutMs, transcriptPath);
    if (proofFlush.timedOut || proofFlush.result.childExitCode !== 0) {
      return blocked(`mutagen session ${input.sessionName} proof flush failed`, transcriptPath);
    }
    const remoteProbe = `${input.remoteProjectPath}/${proofRelativePath}/${input.runId}.txt`;
    const probe = await runText(
      input.runner,
      "ssh",
      [input.sshHost, "bash", "-lc", quoteShell(`cat ${quoteShell(remoteProbe)}`)],
      timeoutMs,
      transcriptPath
    );
    if (probe.timedOut || probe.result.childExitCode !== 0) {
      return blocked(`mutagen session ${input.sessionName} sync proof failed`, transcriptPath);
    }
    if (probe.stdout !== probeContents) {
      return blocked(`mutagen session ${input.sessionName} sync proof mismatch`, transcriptPath);
    }
    return { ok: true, message: `mutagen session ${input.sessionName} sync proof ok`, transcriptPath };
  } finally {
    rmSync(probePath, { force: true });
  }
}

async function healthCheck(
  runner: StreamingCommandRunner,
  sessionName: string,
  timeoutMs: number,
  transcriptPath?: string
): Promise<{ ok: true; message: string } | { ok: false; message: string }> {
  const list = await runText(runner, "mutagen", ["sync", "list", sessionName, "--template", "{{json .}}"], timeoutMs, transcriptPath);
  if (list.timedOut || list.result.childExitCode !== 0) {
    return { ok: false, message: `mutagen session ${sessionName} status check failed` };
  }
  const health = evaluateMutagenListJson(list.stdout, sessionName);
  return health.healthy ? { ok: true, message: health.reason } : { ok: false, message: health.reason };
}

async function runText(
  runner: StreamingCommandRunner,
  command: string,
  args: string[],
  timeoutMs: number,
  transcriptPath?: string
): Promise<{ result: StreamingRunResult; stdout: string; stderr: string; timedOut: boolean }> {
  const stdout = new CaptureWritable();
  const stderr = new CaptureWritable();
  writeTranscript(transcriptPath, `$ ${[command, ...args].join(" ")}\n`);
  const result = await runner.runStreaming(command, args, { stdout, stderr, timeoutMs });
  writeTranscript(
    transcriptPath,
    `exit ${result.childExitCode}${result.timedOut ? " (timeout)" : ""}\n${stdout.text}${stderr.text}\n`
  );
  return { result, stdout: stdout.text, stderr: stderr.text, timedOut: result.timedOut };
}

function blocked(message: string, transcriptPath?: string): MutagenPreflightResult {
  return {
    ok: false,
    message,
    transcriptPath,
    next: ["pgx-cli sync status", "pgx-cli sync doctor"]
  };
}

function writeTranscript(path: string | undefined, text: string): void {
  if (!path) return;
  appendFileSync(path, text);
}

class CaptureWritable extends Writable {
  text = "";

  override _write(chunk: Buffer, _encoding: BufferEncoding, callback: (error?: Error | null) => void): void {
    this.text += chunk.toString();
    callback();
  }
}

function stringField(value: unknown, ...keys: string[]): string | undefined {
  if (!isObject(value)) return undefined;
  for (const key of keys) {
    const field = value[key];
    if (typeof field === "string") return field;
  }
  return undefined;
}

function boolField(value: unknown, ...keys: string[]): boolean | undefined {
  if (!isObject(value)) return undefined;
  for (const key of keys) {
    const field = value[key];
    if (typeof field === "boolean") return field;
  }
  return undefined;
}

function objectField(value: unknown, ...keys: string[]): Record<string, unknown> | undefined {
  if (!isObject(value)) return undefined;
  for (const key of keys) {
    const field = value[key];
    if (isObject(field)) return field;
  }
  return undefined;
}

function hasNonEmptyField(value: unknown, pattern: RegExp): boolean {
  if (!isObject(value)) return false;
  for (const [key, field] of Object.entries(value)) {
    if (pattern.test(key) && hasValue(field)) return true;
    if (hasNonEmptyField(field, pattern)) return true;
  }
  return false;
}

function hasValue(value: unknown): boolean {
  if (Array.isArray(value)) return value.length > 0;
  if (isObject(value)) return Object.keys(value).length > 0;
  if (typeof value === "string") return value.length > 0;
  if (typeof value === "number") return value > 0;
  if (typeof value === "boolean") return value;
  return false;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
