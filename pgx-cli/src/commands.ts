import { spawn } from "node:child_process";
import type { StreamSample } from "./output.js";

export type RunResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
};

export type CommandRunner = {
  run(command: string, args: string[]): Promise<RunResult>;
};

export type StreamingRunResult = {
  childExitCode: number;
  stdoutSample: StreamSample;
  stderrSample: StreamSample;
  timedOut: boolean;
};

export type StreamingRunOptions = {
  stdout?: NodeJS.WritableStream;
  stderr?: NodeJS.WritableStream;
  timeoutMs?: number;
};

export type StreamingCommandRunner = CommandRunner & {
  runStreaming(
    command: string,
    args: string[],
    options: StreamingRunOptions
  ): Promise<StreamingRunResult>;
};

export class NodeCommandRunner implements StreamingCommandRunner {
  async run(command: string, args: string[]): Promise<RunResult> {
    return new Promise((resolve) => {
      const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk) => {
        const text = chunk.toString();
        stdout += text;
      });
      child.stderr.on("data", (chunk) => {
        const text = chunk.toString();
        stderr += text;
      });
      child.on("error", (error) => {
        resolve({ exitCode: 1, stdout, stderr: stderr + `${error.message}\n` });
      });
      child.on("close", (code) => {
        resolve({ exitCode: code ?? 1, stdout, stderr });
      });
    });
  }

  async runStreaming(
    command: string,
    args: string[],
    options: StreamingRunOptions
  ): Promise<StreamingRunResult> {
    return new Promise((resolve) => {
      const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
      const stdout = new StreamSampler();
      const stderr = new StreamSampler();
      let timedOut = false;
      let settled = false;
      const timeout = options.timeoutMs
        ? setTimeout(() => {
            timedOut = true;
            child.kill("SIGTERM");
          }, options.timeoutMs)
        : undefined;

      child.stdout.on("data", (chunk: Buffer) => {
        stdout.push(chunk);
        options.stdout?.write(chunk);
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderr.push(chunk);
        options.stderr?.write(chunk);
      });
      child.on("error", (error) => {
        if (settled) return;
        settled = true;
        if (timeout) clearTimeout(timeout);
        const chunk = Buffer.from(`${error.message}\n`);
        stderr.push(chunk);
        options.stderr?.write(chunk);
        resolve({
          childExitCode: 1,
          stdoutSample: stdout.sample(),
          stderrSample: stderr.sample(),
          timedOut
        });
      });
      child.on("close", (code) => {
        if (settled) return;
        settled = true;
        if (timeout) clearTimeout(timeout);
        resolve({
          childExitCode: timedOut ? (code ?? 124) || 124 : code ?? 1,
          stdoutSample: stdout.sample(),
          stderrSample: stderr.sample(),
          timedOut
        });
      });
    });
  }
}

class StreamSampler {
  private readonly maxHeadBytes = 64 * 1024;
  private readonly maxTailBytes = 64 * 1024;
  private head = "";
  private tail = "";
  private totalBytes = 0;
  private totalLines = 0;

  push(chunk: Buffer): void {
    const text = chunk.toString();
    this.totalBytes += chunk.length;
    this.totalLines += (text.match(/\n/g) ?? []).length;
    if (this.head.length < this.maxHeadBytes) {
      const remaining = this.maxHeadBytes - this.head.length;
      this.head += text.slice(0, remaining);
      const leftover = text.slice(remaining);
      if (leftover) this.pushTail(leftover);
      return;
    }
    this.pushTail(text);
  }

  sample(): StreamSample {
    const capturedBytes = Buffer.byteLength(this.head) + Buffer.byteLength(this.tail);
    const truncated = this.totalBytes > capturedBytes;
    return {
      head: this.head,
      tail: this.tail,
      truncated,
      omittedBytes: truncated ? this.totalBytes - capturedBytes : 0,
      omittedLines: truncated ? Math.max(0, this.totalLines - countLines(this.head) - countLines(this.tail)) : 0
    };
  }

  private pushTail(text: string): void {
    this.tail = `${this.tail}${text}`;
    while (Buffer.byteLength(this.tail) > this.maxTailBytes) {
      const nextNewline = this.tail.indexOf("\n");
      if (nextNewline === -1) {
        this.tail = this.tail.slice(Math.max(0, this.tail.length - this.maxTailBytes));
        return;
      }
      this.tail = this.tail.slice(nextNewline + 1);
    }
  }
}

function countLines(text: string): number {
  return (text.match(/\n/g) ?? []).length;
}
