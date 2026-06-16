import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Writable } from "node:stream";
import { describe, expect, test } from "vitest";
import { NodeCommandRunner } from "../src/commands.js";

class RecordingWritable extends Writable {
  chunks: string[] = [];

  override _write(chunk: Buffer, _encoding: BufferEncoding, callback: (error?: Error | null) => void): void {
    this.chunks.push(chunk.toString());
    callback();
  }
}

describe("command runner", () => {
  test("keeps the buffered API for cheap commands", async () => {
    const runner = new NodeCommandRunner();

    const result = await runner.run(process.execPath, ["-e", "process.stdout.write('hello\\n')"]);

    expect(result).toEqual({ exitCode: 0, stdout: "hello\n", stderr: "" });
  });

  test("decodes buffered stdout after split UTF-8 chunks are complete", async () => {
    const runner = new NodeCommandRunner();

    const result = await runner.run(process.execPath, [
      "-e",
      "process.stdout.write(Buffer.from([0xc3])); setTimeout(() => process.stdout.write(Buffer.from([0xa9])), 10);"
    ]);

    expect(result).toEqual({ exitCode: 0, stdout: "é", stderr: "" });
  });

  test("decodes streaming samples after split UTF-8 chunks are complete", async () => {
    const runner = new NodeCommandRunner();

    const result = await runner.runStreaming(process.execPath, [
      "-e",
      "process.stdout.write(Buffer.from([0xc3])); setTimeout(() => process.stdout.write(Buffer.from([0xa9])), 10);"
    ], {});

    expect(result.childExitCode).toBe(0);
    expect(result.stdoutSample.head).toBe("é");
  });

  test("streams stdout and stderr before process close", async () => {
    const runner = new NodeCommandRunner();
    const stdout = new RecordingWritable();
    const stderr = new RecordingWritable();

    const result = await runner.runStreaming(
      process.execPath,
      ["-e", "process.stdout.write('out\\n'); process.stderr.write('err\\n')"],
      { stdout, stderr }
    );

    expect(result.childExitCode).toBe(0);
    expect(stdout.chunks.join("")).toBe("out\n");
    expect(stderr.chunks.join("")).toBe("err\n");
    expect(result.stdoutSample.head).toContain("out");
    expect(result.stderrSample.head).toContain("err");
  });

  test("kills a command after the configured timeout", async () => {
    const runner = new NodeCommandRunner();
    const root = mkdtempSync(join(tmpdir(), "pgx-timeout-"));
    try {
      const result = await runner.runStreaming(process.execPath, ["-e", "setTimeout(() => {}, 10000)"], {
        timeoutMs: 50
      });

      expect(result.timedOut).toBe(true);
      expect(result.childExitCode).not.toBe(0);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
