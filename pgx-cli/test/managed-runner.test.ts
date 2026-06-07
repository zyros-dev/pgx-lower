import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { ManagedCommandRunner, applyTotalOutputBudget, renderManagedPreview } from "../src/managed-runner.js";
import type { StreamSample } from "../src/output.js";

class FakeStreamingRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[]; options: StreamingRunOptions }> = [];
  result: StreamingRunResult = {
    childExitCode: 1,
    stdoutSample: sample("out-1\nout-2\n", "out-199\nout-200\n", true),
    stderrSample: sample("err-1\nerr-2\n", "err-199\nerr-200\n", true),
    timedOut: false
  };

  async run(command: string, args: string[]) {
    const result = await this.runStreaming(command, args, {});
    return { exitCode: result.childExitCode, stdout: result.stdoutSample.head, stderr: result.stderrSample.head };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args, options });
    options.stdout?.write("out-1\nout-2\nout-3\n");
    options.stderr?.write("err-1\nerr-2\nerr-3\n");
    return this.result;
  }
}

function sample(head: string, tail: string, truncated = false): StreamSample {
  return { head, tail, truncated, omittedLines: truncated ? 195 : 0 };
}

describe("managed preview rendering", () => {
  test("keeps short text unchanged", () => {
    expect(renderManagedPreview("a\nb\n", 3)).toEqual({ text: "a\nb\n", truncated: false });
  });

  test("keeps head and tail with omitted marker", () => {
    const rendered = renderManagedPreview(["l1", "l2", "l3", "l4", "l5"].join("\n"), 3);

    expect(rendered.truncated).toBe(true);
    expect(rendered.text).toContain("l1");
    expect(rendered.text).toContain("[... omitted 3 lines; full transcript in run artifact ...]");
    expect(rendered.text).toContain("l5");
  });

  test("applies a total budget across output parts", () => {
    const rendered = applyTotalOutputBudget(
      ["start\n", "failure summary:\n- compile-error: a.cpp:1: error\n", "line1\nline2\nline3\n", "transcript: /tmp/x\nexit: 1\n"],
      6
    );

    expect(rendered.truncated).toBe(true);
    expect(rendered.text).toContain("start");
    expect(rendered.text).toContain("failure summary:");
    expect(rendered.text).toContain("transcript: /tmp/x");
    expect(rendered.text).toContain("exit: 1");
  });
});

describe("managed command runner", () => {
  test("writes transcripts and summary while returning bounded previews", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-runner-"));
    try {
      const fake = new FakeStreamingRunner();
      const runner = new ManagedCommandRunner(fake);

      const result = await runner.runManaged({
        command: "pgx-cli",
        args: ["dev", "test", "unit", "pg_nullability"],
        root,
        commandName: "dev-test-unit-pg_nullability",
        budget: {
          maxLinesPerStream: 2,
          maxLinesTotal: 20,
          successTailLines: 2,
          failureTailLines: 2
        }
      });

      expect(result.childExitCode).toBe(1);
      expect(result.workflowExitCode).toBe(1);
      expect(fake.calls).toHaveLength(1);
      expect(readFileSync(result.artifact.stdoutPath, "utf8")).toContain("out-1");
      expect(readFileSync(result.artifact.stderrPath, "utf8")).toContain("err-1");
      expect(readFileSync(result.artifact.combinedPath, "utf8")).toContain("out-1");
      expect(readFileSync(result.artifact.commandPath, "utf8")).toBe("pgx-cli dev test unit pg_nullability\n");
      const summary = JSON.parse(readFileSync(result.artifact.summaryPath, "utf8"));
      expect(summary.childExitCode).toBe(1);
      expect(summary.workflowExitCode).toBe(1);
      expect(summary.transcripts.combinedPath).toBe(result.artifact.combinedPath);
      expect(summary.artifacts.registryPath).toBe(result.artifact.artifactsPath);
      expect(summary.artifacts.commandPath).toBe(result.artifact.commandPath);
      expect(result.stdoutPreview).toContain("omitted");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("postprocessor can convert child success into workflow failure", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-post-"));
    try {
      const fake = new FakeStreamingRunner();
      fake.result = {
        childExitCode: 0,
        stdoutSample: sample("ok\n", "", false),
        stderrSample: sample("", "", false),
        timedOut: false
      };
      const runner = new ManagedCommandRunner(fake);

      const result = await runner.runManaged({
        command: "pgx-cli",
        args: ["dev", "test", "tpch"],
        root,
        commandName: "dev-test-tpch",
        budget: {
          maxLinesPerStream: 10,
          maxLinesTotal: 30,
          successTailLines: 2,
          failureTailLines: 2
        },
        postprocess: () => ({ workflowExitCode: 1, postprocessedFailure: "REGRESSIONS: tpch_q01" })
      });

      expect(result.childExitCode).toBe(0);
      expect(result.workflowExitCode).toBe(1);
      const summary = JSON.parse(readFileSync(result.artifact.summaryPath, "utf8"));
      expect(summary.postprocessedFailure).toBe("REGRESSIONS: tpch_q01");
      expect(summary.failureSummary.lines.join("\n")).toContain("REGRESSIONS: tpch_q01");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
