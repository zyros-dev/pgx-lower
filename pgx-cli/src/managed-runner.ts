import { createWriteStream } from "node:fs";
import { finished } from "node:stream/promises";
import { Writable } from "node:stream";
import type { StreamingCommandRunner } from "./commands.js";
import { streamSampleToText } from "./output.js";
import type { StreamSample } from "./output.js";
import {
  createRunArtifactPaths,
  writeCommand,
  writeRunSummary
} from "./run-artifacts.js";
import type { RunArtifactPaths } from "./run-artifacts.js";
import { summarizeFailure } from "./failure-summary.js";
import type { FailureSummary } from "./failure-summary.js";

export type OutputBudget = {
  maxLinesPerStream: number;
  maxLinesTotal: number;
  successTailLines: number;
  failureTailLines: number;
};

export type PreviewSelection = {
  mode: "head" | "tail";
  lines: number;
};

export type ManagedRunResult = {
  childExitCode: number;
  workflowExitCode: number;
  stdoutPreview: string;
  stderrPreview: string;
  combinedPreview: string;
  truncated: boolean;
  timedOut: boolean;
  artifact: RunArtifactPaths;
  failureSummary?: FailureSummary;
};

export type ManagedRunPostprocessorResult = {
  workflowExitCode: number;
  postprocessedFailure?: string;
  summary?: Record<string, unknown>;
};

export type ManagedRunPostprocessor = (input: {
  commandName: string;
  childExitCode: number;
  stdoutSample: StreamSample;
  stderrSample: StreamSample;
  artifact: RunArtifactPaths;
}) => Promise<ManagedRunPostprocessorResult> | ManagedRunPostprocessorResult;

export function renderManagedPreview(
  text: string,
  maxLines: number
): { text: string; truncated: boolean } {
  if (maxLines <= 0) {
    throw new Error("maxLines must be positive");
  }
  if (!text) {
    return { text: "", truncated: false };
  }
  const trailingNewline = text.endsWith("\n");
  const lines = (trailingNewline ? text.slice(0, -1) : text).split("\n");
  if (lines.length <= maxLines) {
    return { text, truncated: false };
  }
  if (maxLines === 1) {
    return {
      text: `[... omitted ${lines.length} lines; full transcript in run artifact ...]\n`,
      truncated: true
    };
  }
  const headCount = Math.max(1, Math.floor((maxLines - 1) / 2));
  const tailCount = Math.max(1, maxLines - 1 - headCount);
  const omitted = Math.max(0, lines.length - headCount - tailCount);
  const rendered = [
    ...lines.slice(0, headCount),
    `[... omitted ${omitted} lines; full transcript in run artifact ...]`,
    ...lines.slice(lines.length - tailCount)
  ].join("\n");
  return { text: `${rendered}\n`, truncated: true };
}

export function applyTotalOutputBudget(
  parts: string[],
  maxLinesTotal: number
): { text: string; truncated: boolean } {
  if (maxLinesTotal <= 0) {
    throw new Error("maxLinesTotal must be positive");
  }
  return renderManagedPreview(parts.filter(Boolean).join(""), maxLinesTotal);
}

export class ManagedCommandRunner {
  constructor(private readonly runner: StreamingCommandRunner) {}

  async runManaged(input: {
    command: string;
    args: string[];
    root: string;
    commandName: string;
    artifact?: RunArtifactPaths;
    budget: OutputBudget;
    timeoutMs?: number;
    preview?: PreviewSelection;
    postprocess?: ManagedRunPostprocessor;
    summary?: Record<string, unknown>;
  }): Promise<ManagedRunResult> {
    const artifact = input.artifact ?? createRunArtifactPaths({ root: input.root, commandName: input.commandName });
    writeCommand(artifact, [input.command, ...input.args]);

    const stdoutLog = createWriteStream(artifact.stdoutPath, { flags: "a" });
    const stderrLog = createWriteStream(artifact.stderrPath, { flags: "a" });
    const combinedLog = createWriteStream(artifact.combinedPath, { flags: "a" });
    const stdout = teeStream(stdoutLog, combinedLog);
    const stderr = teeStream(stderrLog, combinedLog);
    const startedAt = new Date();

    const childResult = await this.runner.runStreaming(input.command, input.args, {
      stdout,
      stderr,
      timeoutMs: input.timeoutMs
    });
    stdout.end();
    stderr.end();
    stdoutLog.end();
    stderrLog.end();
    combinedLog.end();
    await Promise.all([finished(stdoutLog), finished(stderrLog), finished(combinedLog)]);

    const postprocessed = input.postprocess
      ? await input.postprocess({
          commandName: input.commandName,
          childExitCode: childResult.childExitCode,
          stdoutSample: childResult.stdoutSample,
          stderrSample: childResult.stderrSample,
          artifact
        })
      : undefined;
    const workflowExitCode = postprocessed?.workflowExitCode ?? childResult.childExitCode;
    const failureSummary = summarizeFailure({
      commandName: input.commandName,
      stdoutSample: childResult.stdoutSample,
      stderrSample: childResult.stderrSample,
      childExitCode: childResult.childExitCode,
      workflowExitCode,
      postprocessedFailure: postprocessed?.postprocessedFailure,
      transcriptPaths: {
        stdoutPath: artifact.stdoutPath,
        stderrPath: artifact.stderrPath,
        combinedPath: artifact.combinedPath
      }
    });

    const stdoutPreview = renderSelectedPreview(streamSampleToText(childResult.stdoutSample), input.budget.maxLinesPerStream, input.preview);
    const stderrPreview = renderSelectedPreview(streamSampleToText(childResult.stderrSample), input.budget.maxLinesPerStream, input.preview);
    const failureText = failureSummary
      ? `failure summary:\n${failureSummary.lines.map((line) => `- ${line}`).join("\n")}\n`
      : "";
    const combinedPreview = applyTotalOutputBudget(
      [
        failureText,
        stdoutPreview.text ? `stdout preview:\n${stdoutPreview.text}` : "",
        stderrPreview.text ? `stderr preview:\n${stderrPreview.text}` : ""
      ],
      input.budget.maxLinesTotal
    );
    const finishedAt = new Date();
    const truncated =
      childResult.stdoutSample.truncated ||
      childResult.stderrSample.truncated ||
      stdoutPreview.truncated ||
      stderrPreview.truncated ||
      combinedPreview.truncated;

    writeRunSummary(artifact, {
      runId: artifact.runId,
      commandName: input.commandName,
      command: [input.command, ...input.args],
      cwd: process.cwd(),
      startedAt: startedAt.toISOString(),
      finishedAt: finishedAt.toISOString(),
      durationMs: finishedAt.getTime() - startedAt.getTime(),
      childExitCode: childResult.childExitCode,
      workflowExitCode,
      timedOut: childResult.timedOut,
      truncated,
      transcripts: {
        stdoutPath: artifact.stdoutPath,
        stderrPath: artifact.stderrPath,
        combinedPath: artifact.combinedPath
      },
      artifacts: {
        registryPath: artifact.artifactsPath,
        commandPath: artifact.commandPath,
        runDir: artifact.runDir
      },
      failureSummary,
      postprocessedFailure: postprocessed?.postprocessedFailure,
      postprocessedSummary: postprocessed?.summary,
      ...(input.summary ?? {})
    });

    return {
      childExitCode: childResult.childExitCode,
      workflowExitCode,
      stdoutPreview: stdoutPreview.text,
      stderrPreview: stderrPreview.text,
      combinedPreview: combinedPreview.text,
      truncated,
      timedOut: childResult.timedOut,
      artifact,
      failureSummary
    };
  }
}

function renderSelectedPreview(
  text: string,
  maxLines: number,
  preview?: PreviewSelection
): { text: string; truncated: boolean } {
  if (!preview) {
    return renderManagedPreview(text, maxLines);
  }
  return preview.mode === "head" ? renderHeadPreview(text, preview.lines) : renderTailPreview(text, preview.lines);
}

function renderHeadPreview(text: string, count: number): { text: string; truncated: boolean } {
  return renderEdgePreview(text, count, "head");
}

function renderTailPreview(text: string, count: number): { text: string; truncated: boolean } {
  return renderEdgePreview(text, count, "tail");
}

function renderEdgePreview(text: string, count: number, mode: "head" | "tail"): { text: string; truncated: boolean } {
  if (count <= 0) {
    throw new Error("preview line count must be positive");
  }
  if (!text) {
    return { text: "", truncated: false };
  }
  const trailingNewline = text.endsWith("\n");
  const lines = (trailingNewline ? text.slice(0, -1) : text).split("\n");
  if (lines.length <= count) {
    return { text, truncated: false };
  }
  const omitted = lines.length - count;
  const selected = mode === "head" ? lines.slice(0, count) : lines.slice(lines.length - count);
  const marker = `[... omitted ${omitted} lines; full transcript in run artifact ...]`;
  const rendered = mode === "head" ? [...selected, marker] : [marker, ...selected];
  return { text: `${rendered.join("\n")}\n`, truncated: true };
}

function teeStream(primary: NodeJS.WritableStream, combined: NodeJS.WritableStream): Writable {
  return new Writable({
    write(chunk: Buffer, _encoding, callback) {
      primary.write(chunk);
      combined.write(chunk);
      callback();
    }
  });
}
