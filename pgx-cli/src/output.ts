import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export type StreamSample = {
  head: string;
  tail: string;
  truncated: boolean;
  omittedBytes?: number;
  omittedLines?: number;
};

export type BufferedOutput = {
  stdout: string;
  stderr: string;
};

type RenderOptions = {
  maxLines?: number;
  transcriptPath?: string;
  truncate?: boolean;
};

const defaultMaxLines = 50;

export function streamSampleToText(sample: StreamSample | string): string {
  if (typeof sample === "string") {
    return sample;
  }
  if (!sample.truncated) {
    return sample.head + sample.tail;
  }
  const omitted = sample.omittedLines !== undefined
    ? `lines: ${sample.omittedLines}`
    : `bytes: ${sample.omittedBytes ?? 0}`;
  return `${sample.head}[... omitted ${omitted}; full transcript in run artifact ...]\n${sample.tail}`;
}

export function renderBufferedOutput(
  output: BufferedOutput,
  options: RenderOptions = {}
): BufferedOutput {
  if (options.truncate === false) {
    return output;
  }
  const maxLines = options.maxLines ?? defaultMaxLines;
  const stdout = limitStream(output.stdout, maxLines);
  const stderr = limitStream(output.stderr, maxLines);
  if (!stdout.truncated && !stderr.truncated) {
    return output;
  }

  const transcriptPath = options.transcriptPath ?? createTranscriptPath();
  writeFileSync(
    transcriptPath,
    `# pgx-cli transcript\n\n## stdout\n${output.stdout}\n## stderr\n${output.stderr}`
  );

  return {
    stdout: stdout.text,
    stderr:
      stderr.text +
      `pgx-cli: output truncated to ${maxLines} lines per stream; full transcript: ${transcriptPath}\n`
  };
}

function limitStream(text: string, maxLines: number): { text: string; truncated: boolean } {
  const lines = splitLines(text);
  if (lines.length <= maxLines) {
    return { text, truncated: false };
  }

  const headCount = Math.floor((maxLines - 1) / 2);
  const tailCount = maxLines - 1 - headCount;
  const omitted = lines.length - headCount - tailCount;
  const limited = [
    ...lines.slice(0, headCount),
    `[... omitted ${omitted} lines ...]`,
    ...lines.slice(lines.length - tailCount)
  ];
  return { text: `${limited.join("\n")}\n`, truncated: true };
}

function splitLines(text: string): string[] {
  if (text.length === 0) {
    return [];
  }

  const lines = text.split(/\r?\n/u);
  if (lines[lines.length - 1] === "") {
    lines.pop();
  }
  return lines;
}

function createTranscriptPath(): string {
  const dir = join(tmpdir(), "pgx-cli-transcripts");
  mkdirSync(dir, { recursive: true });
  return join(dir, `pgx-cli-${new Date().toISOString().replace(/[:.]/g, "-")}-${process.pid}.log`);
}
