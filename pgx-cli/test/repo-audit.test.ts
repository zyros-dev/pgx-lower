import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { auditToolInventory, auditToolSurface, runRepoCommand } from "../src/repo-audit.js";

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

function writeText(path: string, text: string): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, text);
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

  test("repo audit-tools reports classified inventory", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-inventory-"));
    writeExecutable(join(root, "scripts", "old.sh"));
    writeFileSync(join(root, "justfile"), "compile:\n\tcmake --build build\n");
    const output = { stdout: "", stderr: "" };

    const exitCode = await runRepoCommand(["audit-tools", "--inventory"], new FakeRunner(), output, {
      localProjectPath: root
    });

    expect(exitCode).toBe(1);
    expect(output.stdout).toContain("path");
    expect(output.stdout).toContain("kind");
    expect(output.stdout).toContain("classification");
    expect(output.stdout).toMatch(/justfile\s+justfile\s+unexpected/);
    expect(output.stdout).toMatch(/scripts\/old\.sh\s+script\s+unexpected/);
    expect(output.stderr).toContain("unexpected");
  });

  test("repo audit-tools inventory accepts classified non-workflow surfaces", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-inventory-"));
    writeExecutable(join(root, ".githooks/pre-commit"));
    writeExecutable(join(root, "docker/dev/entrypoint.sh"));
    writeExecutable(join(root, "pgx-cli/clion-wrappers/container-clang"));
    writeExecutable(join(root, "pgx-cli/dist/index.js"));
    writeExecutable(join(root, "pgx-cli/cmake-tools/runtime-header-tool"));
    writeExecutable(join(root, "extension/pgx_lower.so"));
    writeExecutable(join(root, "benchmark/tpch/run.py"));
    mkdirSync(join(root, "benchmark/tpch/__pycache__"), { recursive: true });
    writeFileSync(join(root, "benchmark/tpch/__pycache__/run.cpython-312.pyc"), "");
    writeFileSync(join(root, "CMakeLists.txt"), "cmake_minimum_required(VERSION 3.29)\n");
    mkdirSync(join(root, "cmake"), { recursive: true });
    writeFileSync(join(root, "cmake/FindPostgreSQL.cmake"), "");
    mkdirSync(join(root, ".codex/rules"), { recursive: true });
    writeFileSync(join(root, ".codex/rules/default.rules"), "");
    const output = { stdout: "", stderr: "" };

    const exitCode = await runRepoCommand(["audit-tools", "--inventory"], new FakeRunner(), output, {
      localProjectPath: root
    });

    expect(exitCode).toBe(0);
    expect(output.stdout).toMatch(/\.githooks\/pre-commit\s+hook\s+allowed/);
    expect(output.stdout).toMatch(/docker\/dev\/entrypoint\.sh\s+docker\s+allowed/);
    expect(output.stdout).toMatch(/pgx-cli\/clion-wrappers\/container-clang\s+wrapper\s+internal/);
    expect(output.stdout).toMatch(/pgx-cli\/cmake-tools\/runtime-header-tool\s+generated\s+generated/);
    expect(output.stdout).toMatch(/pgx-cli\/dist\/index\.js\s+generated\s+generated/);
    expect(output.stdout).toMatch(/extension\/pgx_lower\.so\s+generated\s+generated/);
    expect(output.stdout).toMatch(/benchmark\/tpch\/run\.py\s+benchmark\s+migrated/);
    expect(output.stdout).toMatch(/benchmark\/tpch\/__pycache__\/run\.cpython-312\.pyc\s+generated\s+generated/);
    expect(output.stdout).toMatch(/CMakeLists\.txt\s+cmake\s+internal/);
    expect(output.stdout).toMatch(/cmake\/FindPostgreSQL\.cmake\s+cmake\s+internal/);
    expect(output.stdout).toMatch(/\.codex\/rules\/default\.rules\s+codex-rule\s+internal/);
    expect(output.stderr).toBe("");
  });

  test("classifies the benchmark SQL initializer as an internal fixture", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-inventory-"));
    writeText(join(root, "benchmark/tpch_init.sql"), "-- benchmark schema/data fixture\n");

    expect(auditToolInventory(root)).toContainEqual({
      path: "benchmark/tpch_init.sql",
      kind: "benchmark",
      classification: "internal"
    });
  });

  test("repo audit-tools inventory reports stale helper references in docs and specs", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-audit-inventory-"));
    writeText(join(root, "README.md"), "Run scripts/old.sh for setup.\n");
    writeText(join(root, "AGENTS.md"), "Do not restore tools/scripts/profile.sh.\n");
    writeText(join(root, "wiki/specs/pgx-cli/plans/old-plan.md"), "Legacy path: bench_runner.py\n");
    writeText(join(root, "CMakeLists.txt"), "add_custom_target(old COMMAND scripts/build.sh)\n");
    const output = { stdout: "", stderr: "" };

    const exitCode = await runRepoCommand(["audit-tools", "--inventory"], new FakeRunner(), output, {
      localProjectPath: root
    });

    expect(exitCode).toBe(1);
    expect(output.stdout).toMatch(/AGENTS\.md:1:tools\/scripts\/profile\.sh\s+reference\s+unexpected/);
    expect(output.stdout).toMatch(/CMakeLists\.txt:1:scripts\/build\.sh\s+reference\s+unexpected/);
    expect(output.stdout).toMatch(/README\.md:1:scripts\/old\.sh\s+reference\s+unexpected/);
    expect(output.stdout).toMatch(/wiki\/specs\/pgx-cli\/plans\/old-plan\.md:1:bench_runner\.py\s+reference\s+unexpected/);
    expect(output.stderr).toContain("AGENTS.md:1:tools/scripts/profile.sh");
    expect(output.stderr).toContain("README.md:1:scripts/old.sh");
  });
});
