import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { createRunArtifactPaths } from "../src/run-artifacts.js";
import { evaluateMutagenListJson, runMutagenPreflight } from "../src/mutagen-preflight.js";

class FakeStreamingRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[]; options: StreamingRunOptions }> = [];
  results: Array<{ exitCode: number; stdout: string; stderr?: string; timedOut?: boolean }> = [];

  async run(command: string, args: string[]) {
    const result = await this.runStreaming(command, args, {});
    return { exitCode: result.childExitCode, stdout: result.stdoutSample.head, stderr: result.stderrSample.head };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args, options });
    const result = this.results.shift() ?? { exitCode: 0, stdout: healthyJson() };
    options.stdout?.write(result.stdout);
    options.stderr?.write(result.stderr ?? "");
    return {
      childExitCode: result.exitCode,
      stdoutSample: { head: result.stdout, tail: "", truncated: false },
      stderrSample: { head: result.stderr ?? "", tail: "", truncated: false },
      timedOut: result.timedOut ?? false
    };
  }
}

function healthyJson(): string {
  return JSON.stringify([
    {
      name: "pgx-lower",
      paused: false,
      status: "watching",
      alpha: { connected: true },
      beta: { connected: true }
    }
  ]);
}

describe("mutagen health parser", () => {
  test("accepts a watching connected session", () => {
    expect(evaluateMutagenListJson(healthyJson(), "pgx-lower")).toEqual({
      healthy: true,
      reason: "mutagen session pgx-lower is healthy (watching)"
    });
  });

  test.each(["scanning", "reconciling"])("accepts connected %s status for proof-backed preflight", (status) => {
    expect(
      evaluateMutagenListJson(
        JSON.stringify([{ name: "pgx-lower", paused: false, status, alpha: { connected: true }, beta: { connected: true } }]),
        "pgx-lower"
      )
    ).toEqual({
      healthy: true,
      reason: `mutagen session pgx-lower is healthy (${status})`
    });
  });

  test.each([
    ["[]", "missing"],
    [JSON.stringify([{ name: "pgx-lower", paused: true, alpha: { connected: true }, beta: { connected: true } }]), "paused"],
    [JSON.stringify([{ name: "pgx-lower", paused: false, alpha: { connected: false }, beta: { connected: true } }]), "alpha disconnected"],
    [JSON.stringify([{ name: "pgx-lower", paused: false, alpha: { connected: true }, beta: { connected: false } }]), "beta disconnected"],
	    [JSON.stringify([{ name: "pgx-lower", paused: false, alpha: { connected: true }, beta: { connected: true }, conflicts: ["x"] }]), "conflict"],
	    [
	      JSON.stringify([
	        {
	          name: "pgx-lower",
	          paused: false,
	          alpha: { connected: true },
	          beta: { connected: true, transitionProblems: [{ path: "src/a.cpp", error: "stale" }] }
	        }
	      ]),
	      "problem"
	    ],
	    [JSON.stringify([{ name: "pgx-lower", paused: false, status: "halted", alpha: { connected: true }, beta: { connected: true } }]), "unsafe status"],
    [JSON.stringify([{ name: "pgx-lower", paused: false, status: "staging", alpha: { connected: true }, beta: { connected: true } }]), "unsafe status"],
    ["not-json", "malformed"]
  ])("rejects unhealthy state containing %s", (json, reason) => {
    expect(evaluateMutagenListJson(json, "pgx-lower")).toMatchObject({
      healthy: false,
      reason: expect.stringContaining(reason)
    });
  });
});

describe("mutagen preflight", () => {
  test("flushes, rechecks health, proves the remote sentinel, and writes a transcript", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-mutagen-preflight-"));
    try {
      const artifact = createRunArtifactPaths({ root, commandName: "dev-test", randomSuffix: "abcdef" });
      const runner = new FakeStreamingRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: "probe-abcdef\n" }
      ];

      const result = await runMutagenPreflight({
        runner,
        artifact,
        sessionName: "pgx-lower",
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        sshHost: "comfy",
        runId: "abcdef",
        flushTimeoutSeconds: 45,
        requireProof: true,
        probeContents: "probe-abcdef\n"
      });

      expect(result.ok).toBe(true);
      expect(runner.calls.map((call) => [call.command, ...call.args].join(" "))).toEqual([
        "mutagen sync list pgx-lower --template {{json .}}",
        "mutagen sync flush pgx-lower",
        "mutagen sync list pgx-lower --template {{json .}}",
        "mutagen sync flush pgx-lower",
        "ssh comfy bash -lc 'cat /home/zel/repos/pgx-lower/.pgx-cli/sync-probes/abcdef.txt'"
      ]);
      expect(runner.calls.every((call) => call.options.timeoutMs === 45000)).toBe(true);
      expect(readFileSync(artifact.syncPreflightPath, "utf8")).toContain("mutagen sync list");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("blocks and does not prove when health is bad", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-mutagen-bad-"));
    try {
      const artifact = createRunArtifactPaths({ root, commandName: "dev-test" });
      const runner = new FakeStreamingRunner();
      runner.results = [{ exitCode: 0, stdout: "[]" }];

      const result = await runMutagenPreflight({
        runner,
        artifact,
        sessionName: "pgx-lower",
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        sshHost: "comfy",
        runId: "blocked",
        flushTimeoutSeconds: 45,
        requireProof: true
      });

      expect(result).toMatchObject({ ok: false, next: ["pgx-cli sync status", "pgx-cli sync doctor"] });
      expect(runner.calls).toHaveLength(1);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("blocks when the flush fails", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-mutagen-flush-fail-"));
    try {
      const artifact = createRunArtifactPaths({ root, commandName: "dev-test" });
      const runner = new FakeStreamingRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 1, stdout: "", stderr: "flush failed\n" }
      ];

      const result = await runMutagenPreflight({
        runner,
        artifact,
        sessionName: "pgx-lower",
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        sshHost: "comfy",
        runId: "flush-fail",
        flushTimeoutSeconds: 45,
        requireProof: true
      });

      expect(result).toMatchObject({ ok: false, message: "mutagen session pgx-lower flush failed" });
      expect(runner.calls.map((call) => [call.command, ...call.args].join(" "))).toEqual([
        "mutagen sync list pgx-lower --template {{json .}}",
        "mutagen sync flush pgx-lower"
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("blocks when the sync proof contents do not match", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-mutagen-proof-mismatch-"));
    try {
      const artifact = createRunArtifactPaths({ root, commandName: "dev-test", randomSuffix: "mismatch" });
      const runner = new FakeStreamingRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: "wrong-probe\n" }
      ];

      const result = await runMutagenPreflight({
        runner,
        artifact,
        sessionName: "pgx-lower",
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        sshHost: "comfy",
        runId: "mismatch",
        flushTimeoutSeconds: 45,
        requireProof: true,
        probeContents: "probe-mismatch\n"
      });

      expect(result).toMatchObject({ ok: false, message: "mutagen session pgx-lower sync proof mismatch" });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("uses the configured sync proof path", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-mutagen-proof-path-"));
    try {
      const artifact = createRunArtifactPaths({ root, commandName: "dev-test", randomSuffix: "custom" });
      const runner = new FakeStreamingRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: "probe-custom\n" }
      ];

      const result = await runMutagenPreflight({
        runner,
        artifact,
        sessionName: "pgx-lower",
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        sshHost: "comfy",
        runId: "custom",
        flushTimeoutSeconds: 45,
        requireProof: true,
        proofPath: ".custom-probes"
      });

      expect(result.ok).toBe(true);
      expect(runner.calls.map((call) => [call.command, ...call.args].join(" "))).toContain(
        "ssh comfy bash -lc 'cat /home/zel/repos/pgx-lower/.custom-probes/custom.txt'"
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("flushes a transient unsafe status before requiring final health", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-mutagen-reconciling-"));
    try {
      const artifact = createRunArtifactPaths({ root, commandName: "dev-test" });
      const runner = new FakeStreamingRunner();
      runner.results = [
        {
          exitCode: 0,
          stdout: JSON.stringify([
            { name: "pgx-lower", paused: false, status: "reconciling", alpha: { connected: true }, beta: { connected: true } }
          ])
        },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() }
      ];

      const result = await runMutagenPreflight({
        runner,
        artifact,
        sessionName: "pgx-lower",
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        sshHost: "comfy",
        runId: "reconciling",
        flushTimeoutSeconds: 45,
        requireProof: false
      });

      expect(result.ok).toBe(true);
      expect(runner.calls.map((call) => [call.command, ...call.args].join(" "))).toEqual([
        "mutagen sync list pgx-lower --template {{json .}}",
        "mutagen sync flush pgx-lower",
        "mutagen sync list pgx-lower --template {{json .}}"
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
