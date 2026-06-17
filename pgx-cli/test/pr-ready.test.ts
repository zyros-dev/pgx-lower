import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runPrReadyCommand } from "../src/pr-ready.js";

const hardeningClaimIds = [
  "policy-rules",
  "policy-wrapper-class",
  "evidence-contract",
  "strict-preflight",
  "pgx-cli-tests",
  "pgx-cli-build",
  "lint-diff",
  "gate-batch",
  "gate-review",
  "dev-qa-loop",
  "workflow-skills"
];

type FakeRunnerState = {
  localStatus?: string;
  branch?: string;
  head?: string;
  pushed?: boolean;
  remoteHead?: string;
  prJson?: Record<string, unknown>;
  pr?: {
    url?: string;
    mergeStateStatus?: string;
    headRefName?: string;
    baseRefName?: string;
  } | null;
};

class FakeRunner implements StreamingCommandRunner {
  readonly calls: Array<{ command: string; args: string[] }> = [];

  constructor(private readonly state: FakeRunnerState = {}) {}

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    if (command === "git" && rendered.includes("status --short")) {
      return ok(this.state.localStatus ?? "");
    }
    if (command === "git" && rendered.includes("branch --show-current")) {
      return ok(`${this.state.branch ?? "feature-a"}\n`);
    }
    if (command === "git" && rendered.includes("rev-parse HEAD")) {
      return ok(`${this.state.head ?? "abc123"}\n`);
    }
    if (command === "git" && rendered.includes("ls-remote --heads origin")) {
      return this.state.pushed === false
        ? ok("")
        : ok(`${this.state.remoteHead ?? this.state.head ?? "abc123"}\trefs/heads/${this.state.branch ?? "feature-a"}\n`);
    }
    if (command === "gh" && rendered.includes("pr view")) {
      if (this.state.pr === null) {
        return { exitCode: 1, stdout: "", stderr: "no pull requests found\n" };
      }
      if (this.state.prJson) {
        return ok(`${JSON.stringify(this.state.prJson)}\n`);
      }
      return ok(`${JSON.stringify({
        url: this.state.pr?.url ?? "https://github.example/pr/1",
        mergeStateStatus: this.state.pr?.mergeStateStatus ?? "CLEAN",
        headRefName: this.state.pr?.headRefName ?? this.state.branch ?? "feature-a",
        baseRefName: this.state.pr?.baseRefName ?? "main"
      })}\n`);
    }
    return ok("");
  }

  async runStreaming(_command: string, _args: string[], _options: StreamingRunOptions): Promise<StreamingRunResult> {
    throw new Error("runStreaming should not be used by pr ready");
  }
}

function ok(stdout: string): RunResult {
  return { exitCode: 0, stdout, stderr: "" };
}

function config(root: string) {
  return {
    localProjectPath: root,
    output: {
      transcript_dir: ".pgx-cli/runs"
    }
  };
}

function writeEvidence(root: string): string {
  const path = join(root, ".pgx-cli", "evidence", "current.json");
  mkdirSync(join(root, ".pgx-cli", "evidence"), { recursive: true });
  writeFileSync(path, `${JSON.stringify({
    contract: "agent-hardening",
    requiredClaims: hardeningClaimIds,
    claims: hardeningClaimIds.map((id) => ({
      id,
      claim: id,
      kind: id === "gate-review" ? "gate" : "behavioral-test",
      greenEvidence: `${id} passed`
    }))
  }, null, 2)}\n`);
  return path;
}

function writeReviewSummary(
  root: string,
  runId = "review-1",
  options: {
    workflowExitCode?: number;
    head?: string | null;
    finishedAt?: string;
    commandName?: string;
    command?: string[];
    includePreflightStep?: boolean;
  } = {}
): string {
  const runDir = join(root, ".pgx-cli", "runs", runId);
  mkdirSync(runDir, { recursive: true });
  const summary: Record<string, unknown> = {
    runId,
    commandName: options.commandName ?? "test-compare-postgres-tpch-correctness",
    command: options.command ?? ["pgx-cli", "test", "compare-postgres", "--workload", "tpch-correctness"],
    workflowExitCode: options.workflowExitCode ?? 0,
    finishedAt: options.finishedAt ?? "2099-01-01T00:00:00.000Z"
  };
  if (options.includePreflightStep !== false) {
    summary.steps = [
      {
        name: "strict preflight",
        command: ["pgx-cli", "dev", "preflight", "--strict"],
        exitCode: 0
      }
    ];
  }
  if (options.head !== null) {
    summary.gitHead = options.head ?? "abc123";
  }
  writeFileSync(join(runDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  return runId;
}

function writeFinalReviewSummary(
  root: string,
  runId = "review-1",
  options: { workflowExitCode?: number; head?: string | null; finishedAt?: string } = {}
): string {
  return writeReviewSummary(root, runId, {
    ...options,
    commandName: "dev-gate-review",
    command: ["pgx-cli", "dev", "gate", "review"]
  });
}

async function withRoot<T>(callback: (root: string) => Promise<T>): Promise<T> {
  const root = mkdtempSync(join(tmpdir(), "pgx-pr-ready-"));
  try {
    return await callback(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

describe("pr ready", () => {
  test("fails when evidence matrix is missing", async () => {
    await withRoot(async (root) => {
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail evidence");
      expect(output.stderr).toContain("evidence file missing");
    });
  });

  test("fails when evidence matrix does not declare the required evidence contract", async () => {
    await withRoot(async (root) => {
      const evidencePath = writeEvidence(root);
      writeFileSync(evidencePath, `${JSON.stringify({
        claims: [
          {
            id: "some-green-claim",
            claim: "arbitrary green claim",
            kind: "behavioral-test",
            greenEvidence: "looks good"
          }
        ]
      }, null, 2)}\n`);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail evidence");
      expect(output.stdout).toContain("pgx-cli agent evidence check --contract agent-hardening --file .pgx-cli/evidence/current.json");
      expect(output.stderr).toContain("evidence requiredClaims missing");
    });
  });

  test("fails when the evidence matrix is missing a required hardening contract row", async () => {
    await withRoot(async (root) => {
      const evidencePath = writeEvidence(root);
      writeFileSync(evidencePath, `${JSON.stringify({
        contract: "agent-hardening",
        requiredClaims: hardeningClaimIds.filter((id) => id !== "pgx-cli-tests"),
        claims: hardeningClaimIds
          .filter((id) => id !== "pgx-cli-tests")
          .map((id) => ({
            id,
            claim: id,
            kind: "behavioral-test",
            greenEvidence: `${id} passed`
          }))
      }, null, 2)}\n`);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail evidence");
      expect(output.stdout).toContain("pgx-cli agent evidence check --contract agent-hardening --file .pgx-cli/evidence/current.json");
      expect(output.stderr).toContain("missing required evidence: pgx-cli-tests");
    });
  });

  test("fails when arbitrary self-declared evidence chooses its own readiness universe", async () => {
    await withRoot(async (root) => {
      const evidencePath = writeEvidence(root);
      writeFileSync(evidencePath, `${JSON.stringify({
        requiredClaims: ["only-one"],
        claims: [
          {
            id: "only-one",
            claim: "arbitrary green claim",
            kind: "behavioral-test",
            greenEvidence: "looks good"
          }
        ]
      }, null, 2)}\n`);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail evidence");
      expect(output.stderr).toContain("missing required evidence: pgx-cli-tests");
    });
  });

  test("accepts an explicitly deferred required hardening row", async () => {
    await withRoot(async (root) => {
      const evidencePath = writeEvidence(root);
      writeFileSync(evidencePath, `${JSON.stringify({
        contract: "agent-hardening",
        requiredClaims: hardeningClaimIds,
        claims: hardeningClaimIds.map((id) => id === "workflow-skills"
          ? {
            id,
            claim: id,
            kind: "deferral",
            deferredReason: "workflow skills unavailable in this local smoke"
          }
          : {
            id,
            claim: id,
            kind: "behavioral-test",
            greenEvidence: `${id} passed`
          })
      }, null, 2)}\n`);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(0);
      expect(output.stdout).toContain("ok evidence");
    });
  });

  test("fails when local status is dirty", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner({ localStatus: " M src/file.cpp\n" }),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail local worktree");
      expect(output.stderr).toContain("local worktree dirty");
    });
  });

  test("fails when no pull request exists unless allow-no-pr is set", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root);

      const output = { stdout: "", stderr: "" };
      expect(await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1"],
        new FakeRunner({ pr: null }),
        output,
        config(root)
      )).toBe(1);
      expect(output.stderr).toContain("pull request missing");

      const allowedOutput = { stdout: "", stderr: "" };
      const allowedRunner = new FakeRunner({ pr: null });
      expect(await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        allowedRunner,
        allowedOutput,
        config(root)
      )).toBe(0);
      expect(allowedOutput.stdout).toContain("ok pull request: skipped");
      expect(allowedRunner.calls.some((call) => call.command === "gh")).toBe(false);
    });
  });

  test("fails when the requested review gate run is missing or records another head", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);

      const missingOutput = { stdout: "", stderr: "" };
      expect(await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "missing", "--allow-no-pr"],
        new FakeRunner(),
        missingOutput,
        config(root)
      )).toBe(1);
      expect(missingOutput.stderr).toContain("review gate summary missing");

      writeFinalReviewSummary(root, "review-2", { head: "def456" });
      const mismatchOutput = { stdout: "", stderr: "" };
      expect(await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-2", "--allow-no-pr"],
        new FakeRunner({ head: "abc123" }),
        mismatchOutput,
        config(root)
      )).toBe(1);
      expect(mismatchOutput.stderr).toContain("review gate HEAD mismatch");
    });
  });

  test("fails when the specified review run is only a compare-postgres child step", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeReviewSummary(root, "compare-child", {
        commandName: "test-compare-postgres-tpch-correctness"
      });
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "compare-child", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail review gate");
      expect(output.stderr).toContain("review gate summary command mismatch");
    });
  });

  test("fails when the review gate summary does not record HEAD", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root, "review-no-head", { head: null });
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-no-head", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail review gate");
      expect(output.stderr).toContain("review gate HEAD missing");
      expect(output.stdout).toContain("pgx-cli dev gate review");
    });
  });

  test("fails when the review gate summary did not include strict preflight", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root, "review-no-preflight", { includePreflightStep: false });
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-no-preflight", "--allow-no-pr"],
        new FakeRunner(),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail review gate");
      expect(output.stderr).toContain("review gate strict preflight missing");
    });
  });

  test("fails when the pull request is not mergeable", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1"],
        new FakeRunner({ pr: { mergeStateStatus: "DIRTY" } }),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail pull request");
      expect(output.stderr).toContain("pull request not mergeable");
    });
  });

  test("fails when pull request JSON is missing required refs", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1"],
        new FakeRunner({
          prJson: {
            url: "https://github.example/pr/1",
            mergeStateStatus: "CLEAN"
          }
        }),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail pull request");
      expect(output.stderr).toContain("pull request headRefName missing");
    });
  });

  test("fails when the pushed branch SHA is stale", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner({ head: "abc123", remoteHead: "def456" }),
        output,
        config(root)
      );

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("fail branch pushed");
      expect(output.stderr).toContain("origin/feature-a stale");
    });
  });

  test("passes the branch push check when origin branch SHA matches local HEAD", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root);
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json", "--review-run", "review-1", "--allow-no-pr"],
        new FakeRunner({ head: "abc123", remoteHead: "abc123" }),
        output,
        config(root)
      );

      expect(exitCode).toBe(0);
      expect(output.stdout).toContain("ok branch pushed");
      expect(output.stdout).toContain("origin/feature-a at abc123");
    });
  });

  test("passes when all readiness checks pass", async () => {
    await withRoot(async (root) => {
      writeEvidence(root);
      writeFinalReviewSummary(root, "old-review", { finishedAt: "2098-01-01T00:00:00.000Z" });
      writeFinalReviewSummary(root, "latest-review", { finishedAt: "2099-01-01T00:00:00.000Z" });
      writeReviewSummary(root, "newer-focused-check", {
        commandName: "dev-check-diff",
        finishedAt: "2100-01-01T00:00:00.000Z"
      });
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };
      const exitCode = await runPrReadyCommand(
        ["ready", "--evidence", ".pgx-cli/evidence/current.json"],
        runner,
        output,
        config(root)
      );

      expect(exitCode).toBe(0);
      expect(output.stderr).toBe("");
      expect(output.stdout).toContain("ok local worktree");
      expect(output.stdout).toContain("ok branch");
      expect(output.stdout).toContain("ok HEAD");
      expect(output.stdout).toContain("ok evidence");
      expect(output.stdout).toContain("ok review gate");
      expect(output.stdout).toContain("ok branch pushed");
      expect(output.stdout).toContain("ok pull request");
      expect(output.stdout).toContain("latest-review");
      expect(runner.calls.map((call) => [call.command, ...call.args].join(" "))).toEqual([
        `git -C ${root} status --short`,
        `git -C ${root} branch --show-current`,
        `git -C ${root} rev-parse HEAD`,
        `git -C ${root} ls-remote --heads origin feature-a`,
        "gh pr view --head feature-a --json url,mergeStateStatus,headRefName,baseRefName"
      ]);
    });
  });
});
