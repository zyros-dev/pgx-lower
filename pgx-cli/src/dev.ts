import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { CommandRunner } from "./commands.js";
import type { RunResult } from "./commands.js";
import { fullLintShellCommand, targetedLintShellCommand } from "./lint.js";
import { evaluatePgRegressBaseline, hasPgRegressBaselineInput } from "./pg-regress-baseline.js";
import type { OperationConfig, OperationOutput } from "./operations.js";
import { writeUnitSqlFiles } from "./unit-sql.js";

export type WorkflowStep = {
  name: string;
  command: string[];
  exitCode: number;
  logPath?: string;
  jobId?: string;
  summary?: string;
};

export type DevConfig = OperationConfig & {
  localProjectPath: string;
  dockerContainer: string;
  buildQueue: string;
  checkQueue: string;
};

export function formatWorkflowSummary(steps: WorkflowStep[]): string {
  const lines = ["Workflow summary:"];
  for (const step of steps) {
    const status = step.exitCode === 0 ? "ok" : "fail";
    const suffix = step.summary
      ? ` - ${step.summary}`
      : step.logPath
        ? ` - log ${step.logPath}`
        : step.jobId
          ? ` - job ${step.jobId}`
          : "";
    lines.push(`- ${status} ${step.name}: ${step.command.join(" ")}${suffix}`);
  }
  lines.push(`Workflow result: ${steps.every((step) => step.exitCode === 0) ? "ok" : "failed"}`);
  return `${lines.join("\n")}\n`;
}

export async function runDevCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig
): Promise<number> {
  const [command, ...rest] = args;

  if (command === "status") {
    output.stdout += "dev status\n";
    const sync = await runner.run("mutagen", ["sync", "list", config.mutagenSession]);
    output.stdout += sync.stdout;
    output.stderr += sync.stderr;
    if (sync.exitCode !== 0) return sync.exitCode;

    for (const shellCommand of [
      `cd ${quoteShell(config.remoteProjectPath)} && git status --short --branch`,
      `TS_SOCKET=/tmp/${config.buildQueue}.sock tsp`,
      `TS_SOCKET=/tmp/${config.checkQueue}.sock tsp`
    ]) {
      const exitCode = await runRemoteShell(runner, output, config, shellCommand);
      if (exitCode !== 0) return exitCode;
    }
    return 0;
  }

  if (command === "logs") {
    const id = rest[0];
    if (id === "latest") {
      return runRemoteShell(runner, output, config, `TS_SOCKET=/tmp/${config.buildQueue}.sock tsp -t`);
    }
    if (id && /^[0-9]+$/.test(id)) {
      return runRemoteShell(runner, output, config, `TS_SOCKET=/tmp/${config.buildQueue}.sock tsp -t ${id}`);
    }
    output.stderr += "Usage: dev logs <latest|job-id>\n";
    return 1;
  }

  if (command === "lint") {
    const [lintCommand, ...lintArgs] = rest;
    if (lintCommand === "diff") return runLintDiff(runner, output, config);
    if (lintCommand === "file" && lintArgs.length === 1) {
      return runLintFiles(runner, output, config, lintArgs);
    }
    if (lintCommand === "files" && lintArgs.length > 0) {
      return runLintFiles(runner, output, config, lintArgs);
    }
    output.stderr += "Usage: dev lint <file <path>|files <paths...>|diff>\n";
    return 1;
  }

  if (command === "test") {
    const [testCommand, testArg] = rest;
    if (testCommand === "unit" && testArg) return runPostgresUnitTests(runner, output, config, testArg);
    if (testCommand === "tpch") return runTpchTests(runner, output, config);
    if (testCommand === "focused") return runPostgresUnitTests(runner, output, config);
    output.stderr += "Usage: dev test <unit <suite>|tpch|focused>\n";
    return 1;
  }

  if (command === "gate") {
    const [gateCommand] = rest;
    if (gateCommand === "batch") {
      return runWorkflow(runner, output, config, [
        {
          name: "check diff",
          command: ["pgx-cli", "dev", "check", "diff"],
          run: () => runCheckDiff(runner, output, config)
        },
        {
          name: "lint diff",
          command: ["pgx-cli", "dev", "lint", "diff"],
          run: () => runLintDiff(runner, output, config)
        },
        {
          name: "utest-pg",
          command: ["pgx-cli", "dev", "test", "focused"],
          run: () => runPostgresUnitTests(runner, output, config)
        }
      ]);
    }
    if (gateCommand === "review") {
      return runWorkflow(runner, output, config, [
        {
          name: "check diff",
          command: ["pgx-cli", "dev", "check", "diff"],
          run: () => runCheckDiff(runner, output, config)
        },
        {
          name: "lint",
          command: ["pgx-cli", "dev", "lint", "all"],
          run: () => runFullLint(runner, output, config)
        },
        {
          name: "compile",
          command: ["pgx-cli", "dev", "build", "compile", "--profile", "debug"],
          logPath: "/tmp/pgx-compile.out",
          run: () => runCompile(runner, output, config)
        },
        {
          name: "utest-pg",
          command: ["pgx-cli", "dev", "test", "focused"],
          run: () => runPostgresUnitTests(runner, output, config)
        },
        {
          name: "test",
          command: ["pgx-cli", "dev", "test", "all"],
          run: () => runFullTests(runner, output, config)
        }
      ]);
    }
    output.stderr += "Usage: dev gate <batch|review> [--no-bench]\n";
    return 1;
  }

  output.stderr += "Usage: dev <status|lint|test|gate|logs>\n";
  return 1;
}

async function runWorkflow(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  steps: Array<{ name: string; command: string[]; logPath?: string; run: () => Promise<number> }>
): Promise<number> {
  const results: WorkflowStep[] = [];
  for (const step of steps) {
    const exitCode = await step.run();
    results.push({
      name: step.name,
      command: step.command,
      exitCode,
      logPath: step.logPath
    });
    if (exitCode !== 0) {
      output.stdout += formatWorkflowSummary(results);
      return exitCode;
    }
  }
  output.stdout += formatWorkflowSummary(results);
  return 0;
}

async function runCheckDiff(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  return runLocalShell(runner, output, checkDiffScript(config));
}

async function runLintDiff(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  return runLocalShell(runner, output, lintDiffScript(config));
}

async function runLintFiles(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  files: string[]
): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  const command = dockerBashCommand(config, targetedLintShellCommand("/workspace", files));
  return runRemoteShell(runner, output, config, command);
}

async function runFullLint(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  return runRemoteShell(
    runner,
    output,
    config,
    queuedDockerCommand(config, config.buildQueue, fullLintShellCommand("/workspace"))
  );
}

async function runPostgresUnitTests(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  suite?: string
): Promise<number> {
  const generated = writeUnitSqlFiles(config.localProjectPath);
  if (generated.length > 0) output.stdout += `${generated.join("\n")}\n`;

  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;

  const testSelector = suite
    ? `test -f /workspace/tests/unit-tests/sql/${quoteShell(`${suite}.sql`)} && `
    : "";
  const testRunner = suite
    ? `su postgres -c "/usr/local/pgsql/bin/dropdb --if-exists regression_unit && /usr/local/pgsql/bin/createdb regression_unit && /usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression_unit -f /workspace/tests/unit-tests/sql/${quoteShell(`${suite}.sql`)}"`
    : `su postgres -c "/usr/local/pgsql/bin/dropdb --if-exists regression_unit && /usr/local/pgsql/bin/createdb regression_unit" && fail=0; for sql in /workspace/tests/unit-tests/sql/*.sql; do echo "--- $(basename "$sql") ---"; su postgres -c "/usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression_unit -f $sql" || { fail=1; echo FAIL: $sql; }; done; echo; if [ $fail -eq 0 ]; then echo UTEST-PG_OK; else echo UTEST-PG_FAILED; exit 1; fi`;
  const command = [
    "export PATH=/usr/local/pgsql/bin:$PATH",
    testSelector + buildAndInstallCommand(),
    "chmod o+x /workspace/.worktrees 2>/dev/null || true",
    "chmod -R o+rX /workspace/tests/unit-tests",
    testRunner
  ].join(" && ");
  return runRemoteShell(runner, output, config, queuedDockerCommand(config, config.buildQueue, command));
}

async function runTpchTests(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  const command = [
    "set -euo pipefail",
    "export PATH=/usr/local/pgsql/bin:$PATH",
    "mkdir -p /workspace/build-artifacts/ptest /workspace/build-artifacts/ptest/extension",
    "cd /workspace/build-artifacts/ptest",
    "([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache /workspace)",
    "cmake --build .",
    "cmake --install .",
    "mkdir -p /tmp/pgx_ir",
    "chmod 777 /tmp/pgx_ir",
    "chmod o+x /workspace/.worktrees 2>/dev/null || true",
    "chmod -R o+rX /workspace",
    "chown -R postgres:postgres /workspace/build-artifacts/ptest",
    "pg_regress_bin=\"$(pg_config --pkglibdir)/pgxs/src/test/regress/pg_regress\"",
    "(su postgres -c \"$pg_regress_bin --bindir=$(pg_config --bindir) --dlpath=$(pg_config --pkglibdir) --inputdir=/workspace/tests/tpch --outputdir=/workspace/build-artifacts/ptest/extension/tpch --load-extension=pgx_lower init_tpch tpch_no_lower tpch\" 2>&1 | tee /tmp/pg_regress_tpch.out; true)"
  ].join(" && ");
  const result = await runRemoteShellResult(runner, output, config, queuedDockerCommand(config, config.buildQueue, command));
  return evaluateRemotePgRegressResult(result, output, config);
}

async function runCompile(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  return runLocalShell(runner, output, compileScript(config));
}

async function runFullTests(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) return flush;
  const command = [
    buildAndInstallCommand(),
    "mkdir -p /tmp/pgx_ir",
    "chmod 777 /tmp/pgx_ir",
    "chmod o+x /workspace/.worktrees 2>/dev/null || true",
    "chmod -R o+rX /workspace",
    "chown -R postgres:postgres /workspace/build-artifacts/ptest",
    "cd /workspace/build-artifacts/ptest",
    "(su postgres -c \"ctest -V\" 2>&1 | tee /tmp/ctest.out; true)"
  ].join(" && ");
  const result = await runRemoteShellResult(runner, output, config, queuedDockerCommand(config, config.buildQueue, command));
  return evaluateRemotePgRegressResult(result, output, config);
}

async function flushMutagen(runner: CommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  if (config.runningOnRemote) {
    output.stdout += "mutagen: skipped (already on thor)\n";
    return 0;
  }
  const result = await runner.run("mutagen", ["sync", "flush", config.mutagenSession]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

async function runLocalShell(runner: CommandRunner, output: OperationOutput, shellCommand: string): Promise<number> {
  const result = await runner.run("bash", ["-lc", shellCommand]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

async function runRemoteShell(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  shellCommand: string
): Promise<number> {
  const result = await runRemoteShellResult(runner, output, config, shellCommand);
  return result.exitCode;
}

async function runRemoteShellResult(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  shellCommand: string
): Promise<RunResult> {
  const result = config.runningOnRemote
    ? await runner.run("bash", ["-lc", `cd ${quoteShell(config.remoteProjectPath)} && ${shellCommand}`])
    : await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(shellCommand)]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result;
}

function evaluateRemotePgRegressResult(result: RunResult, output: OperationOutput, config: DevConfig): number {
  const raw = result.stdout + result.stderr;
  if (result.exitCode !== 0 && !hasPgRegressBaselineInput(raw)) {
    return result.exitCode;
  }
  const evaluated = evaluatePgRegressBaseline(raw, readBaselineText(config));
  output.stdout += evaluated.stdout;
  output.stderr += evaluated.stderr;
  return evaluated.exitCode;
}

function readBaselineText(config: DevConfig): string {
  return readFileSync(join(config.localProjectPath, "tests/pg_regress_baseline.txt"), "utf8");
}

function checkDiffScript(config: DevConfig): string {
  const dockerPipe = dockerPipeCommand(config, "cd /workspace && clang-format-diff-20 -p1 -style=file");
  return [
    "set -eo pipefail",
    "git fetch origin main --quiet",
    "base=$(git merge-base origin/main HEAD)",
    `diff=$(git diff -U0 "$base" -- 'src/*.c' 'src/*.cc' 'src/*.cpp' 'src/*.h' 'src/*.hpp' 'tests/*.c' 'tests/*.cc' 'tests/*.cpp' 'tests/*.h' 'tests/*.hpp' 'extension/*.c' 'extension/*.h' 2>/dev/null || true)`,
    `if [ -z "$diff" ]; then echo "No C/C++ hunks changed vs origin/main - nothing to check."; exit 0; fi`,
    `echo "Checking hunks changed vs origin/main..."`,
    `printf '%s\n' "$diff" | ${dockerPipe} > /tmp/check-diff.out || true`,
    `if [ ! -s /tmp/check-diff.out ]; then echo "check-diff: clean (your hunks match the project style)"; exit 0; fi`,
    "cat /tmp/check-diff.out",
    "echo",
    `echo "check-diff: your hunks need reformatting. Hand-edit the specific lines above."`,
    "exit 1"
  ].join("\n");
}

function lintDiffScript(config: DevConfig): string {
  const dockerPipe = dockerPipeCommand(
    config,
    'cd /workspace && clang-tidy-diff-20 -p1 -path build-docker-lint -clang-tidy-binary clang-tidy-20 -warnings-as-errors="*"'
  );
  return [
    "set -eo pipefail",
    "git fetch origin main --quiet",
    "base=$(git merge-base origin/main HEAD)",
    `diff=$(git diff -U0 "$base" -- 'src/pgx-lower/*.cpp' 'src/pgx-lower/*.h' 2>/dev/null || true)`,
    `if [ -z "$diff" ]; then echo "No src/pgx-lower hunks changed vs origin/main."; exit 0; fi`,
    `printf '%s\n' "$diff" | ${dockerPipe} > /tmp/lint-diff.out 2>&1 || true`,
    `if grep -qE 'warning:|error:' /tmp/lint-diff.out; then cat /tmp/lint-diff.out; echo "lint-diff: your hunks have clang-tidy violations (above)."; exit 1; fi`,
    `echo "lint-diff: clean (your src/pgx-lower hunks pass clang-tidy)."`
  ].join("\n");
}

function dockerPipeCommand(config: DevConfig, command: string): string {
  const dockerCommand = `docker exec -i ${quoteShell(config.dockerContainer)} bash -c ${quoteShell(command)}`;
  return config.runningOnRemote ? dockerCommand : `ssh ${quoteShell(config.sshHost)} ${quoteShell(dockerCommand)}`;
}

function compileScript(config: DevConfig): string {
  const remoteCommand = queuedDockerCommand(config, config.buildQueue, buildAndInstallCommand());
  return [
    "set -o pipefail",
    "log=/tmp/pgx-compile.out",
    "echo \"compile: full log -> ${log}\"",
    `ssh ${quoteShell(config.sshHost)} ${quoteShell(remoteCommand)} >"\${log}" 2>&1`,
    "rc=$?",
    "if [ \"$rc\" -eq 0 ]; then",
    "  ninja_targets=$(grep -cE '^\\[[0-9]+/[0-9]+\\]' \"${log}\" 2>/dev/null || true)",
    "  echo \"BUILD OK - ${ninja_targets} ninja step(s), pgx_lower.so installed\"",
    "else",
    "  errs=$(grep -cE 'error:|FAILED:' \"${log}\" 2>/dev/null || true)",
    "  echo \"BUILD FAILED - ${errs} error line(s), exit $rc. Last 80 lines from ${log}:\"",
    "  tail -n 80 \"${log}\"",
    "  exit \"$rc\"",
    "fi"
  ].join("\n");
}

function buildAndInstallCommand(): string {
  return [
    "mkdir -p /workspace/build-artifacts/ptest",
    "cd /workspace/build-artifacts/ptest",
    "([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache /workspace)",
    "cmake --build .",
    "cmake --install ."
  ].join(" && ");
}

function queuedDockerCommand(config: DevConfig, queue: string, command: string): string {
  return [
    `export TS_SOCKET=/tmp/${quoteShell(queue)}.sock`,
    "tsp -S 1 >/dev/null",
    `id=$(tsp ${dockerBashCommand(config, command)})`,
    `echo "[job $id queued on ${queue}]"`,
    "tsp -c $id"
  ].join(" && ");
}

function dockerBashCommand(config: DevConfig, command: string): string {
  return `docker exec ${quoteShell(config.dockerContainer)} bash -lc ${quoteShell(command)}`;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
