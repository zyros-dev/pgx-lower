import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { auditToolSurface, runRepoCommand } from "../src/repo-audit.js";

class FakeRunner implements CommandRunner {
  async run(): Promise<RunResult> {
    return { exitCode: 0, stdout: "", stderr: "" };
  }
}

function writeExecutable(path: string): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, "#!/usr/bin/env bash\n");
  chmodSync(path, 0o755);
}

describe("repo tooling audit", () => {
  test("accepts the intended non-workflow script surface", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-"));
    writeExecutable(join(root, ".githooks/pre-commit"));
    writeExecutable(join(root, ".githooks/pre-push"));
    writeExecutable(join(root, "docker/dev/entrypoint.sh"));
    writeExecutable(join(root, "docker/benchmark/docker-entrypoint.sh"));
    writeFileSync(join(root, "docker/dev/Dockerfile"), "FROM scratch\n");
    writeFileSync(join(root, "docker/docker-compose.yml"), "services: {}\n");
    writeFileSync(join(root, "pgx-cli.yaml"), "project: {}\n");
    mkdirSync(join(root, "tests"), { recursive: true });
    writeFileSync(join(root, "tests/workloads.yaml"), "workloads: []\n");
    writeExecutable(join(root, "pgx-cli/clion-wrappers/container-clang"));
    writeExecutable(join(root, "pgx-cli/dist/index.js"));
    writeExecutable(join(root, "build-docker-lint/generated.sh"));
    writeExecutable(join(root, "benchmark/tpch/run.py"));

    expect(auditToolSurface(root)).toEqual([]);
  });

  test("ignores generated build and benchmark trees", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-"));
    writeExecutable(join(root, "build-docker-lint/tool.py"));
    writeExecutable(join(root, "benchmark/tpch/run.py"));
    writeExecutable(join(root, "cmake-build-debug/tool.py"));

    expect(auditToolSurface(root)).toEqual([]);
  });

  test("reports loose helper scripts", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-"));
    writeExecutable(join(root, "scratch/random.sh"));

    expect(auditToolSurface(root)).toEqual(["scratch/random.sh"]);
  });

  test("repo audit-tools returns non-zero with unexpected files", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-"));
    writeFileSync(join(root, "Makefile"), "all:\n");
    const output = { stdout: "", stderr: "" };

    const exitCode = await runRepoCommand(["audit-tools"], new FakeRunner(), output, { localProjectPath: root });

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("Unexpected loose tool entrypoints");
    expect(output.stderr).toContain("Makefile");
  });
});
