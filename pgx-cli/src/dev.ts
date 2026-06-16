import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { StreamingCommandRunner } from "./commands.js";
import { fullLintShellCommand, targetedLintShellCommand } from "./lint.js";
import { detectCtestFailure, evaluatePgRegressBaseline, hasPgRegressBaselineInput } from "./pg-regress-baseline.js";
import { appendLiveStdout } from "./operations.js";
import type { OperationOutput } from "./operations.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";
import { applyTotalOutputBudget } from "./managed-runner.js";
import { runDevPreflightCommand } from "./dev-preflight.js";

export type WorkflowStep = {
  name: string;
  command: string[];
  exitCode: number;
  logPath?: string;
  jobId?: string;
  summary?: string;
};

export type DevConfig = ManagedOperationConfig & {
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
  runner: StreamingCommandRunner,
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
      "git status --short --branch",
      `TS_SOCKET=/tmp/${config.buildQueue}.sock tsp`,
      `TS_SOCKET=/tmp/${config.checkQueue}.sock tsp`
    ]) {
      const exitCode = await runRemoteShell({
        runner,
        output,
        config,
        commandName: `dev-status-${resultsSafeName(shellCommand)}`,
        shellCommand,
        requireMutagenProof: false
      });
      if (exitCode !== 0) return exitCode;
    }
    return 0;
  }

  if (command === "logs") {
    const id = rest[0];
    if (id === "latest") {
      return runRemoteShell({
        runner,
        output,
        config,
        commandName: "dev-logs-latest",
        shellCommand: `TS_SOCKET=/tmp/${config.buildQueue}.sock tsp -t`,
        requireMutagenProof: true
      });
    }
    if (id && /^[0-9]+$/.test(id)) {
      return runRemoteShell({
        runner,
        output,
        config,
        commandName: `dev-logs-${id}`,
        shellCommand: `TS_SOCKET=/tmp/${config.buildQueue}.sock tsp -t ${id}`,
        requireMutagenProof: true
      });
    }
    output.stderr += "Usage: dev logs <latest|job-id>\n";
    return 1;
  }

  if (command === "preflight") {
    return runDevPreflightCommand(rest, runner, output, config);
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

  if (command === "check") {
    const [checkCommand] = rest;
    if (checkCommand === "diff") return runCheckDiff(runner, output, config);
    output.stderr += "Usage: dev check diff\n";
    return 1;
  }

  if (command === "format") {
    const [formatCommand] = rest;
    if (formatCommand === "diff") return runFormatDiff(runner, output, config);
    output.stderr += "Usage: dev format diff\n";
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
      return runWorkflow(output, config, "dev-gate-batch", [
        {
          name: "check diff",
          command: ["pgx-cli", "dev", "check", "diff"],
          run: (stepOutput) => runCheckDiff(runner, stepOutput, config)
        },
        {
          name: "lint diff",
          command: ["pgx-cli", "dev", "lint", "diff"],
          run: (stepOutput) => runLintDiff(runner, stepOutput, config)
        },
        {
          name: "utest-pg",
          command: ["pgx-cli", "dev", "test", "focused"],
          run: (stepOutput) => runPostgresUnitTests(runner, stepOutput, config)
        }
      ]);
    }
    if (gateCommand === "review") {
      return runWorkflow(output, config, "dev-gate-review", [
        {
          name: "check diff",
          command: ["pgx-cli", "dev", "check", "diff"],
          run: (stepOutput) => runCheckDiff(runner, stepOutput, config)
        },
        {
          name: "lint",
          command: ["pgx-cli", "dev", "lint", "all"],
          run: (stepOutput) => runFullLint(runner, stepOutput, config)
        },
        {
          name: "compile",
          command: ["pgx-cli", "dev", "build", "compile", "--profile", "debug"],
          logPath: "/tmp/pgx-compile.out",
          run: (stepOutput) => runCompile(runner, stepOutput, config)
        },
        {
          name: "utest-pg",
          command: ["pgx-cli", "dev", "test", "focused"],
          run: (stepOutput) => runPostgresUnitTests(runner, stepOutput, config)
        },
        {
          name: "test",
          command: ["pgx-cli", "dev", "test", "all"],
          run: (stepOutput) => runFullTests(runner, stepOutput, config)
        },
        {
          name: "compare-postgres",
          command: ["pgx-cli", "test", "compare-postgres", "--workload", "tpch-correctness"],
          run: (stepOutput) => runComparePostgresGate(runner, stepOutput, config)
        }
      ]);
    }
    output.stderr += "Usage: dev gate <batch|review> [--no-bench]\n";
    return 1;
  }

  output.stderr += "Usage: dev <status|check|format|lint|test|gate|logs|preflight>\n";
  return 1;
}

async function runWorkflow(
  output: OperationOutput,
  config: DevConfig,
  workflowName: string,
  steps: Array<{ name: string; command: string[]; logPath?: string; run: (stepOutput: OperationOutput) => Promise<number> }>
): Promise<number> {
  const results: WorkflowStep[] = [];
  appendLiveStdout(output, `pgx-cli: starting ${workflowName}\n`);
  const parts: string[] = [];
  for (const step of steps) {
    const stepOutput = { stdout: "", stderr: "" };
    const exitCode = await step.run(stepOutput);
    const stepSummary = workflowStepSummary(stepOutput, step.logPath);
    results.push({
      name: step.name,
      command: step.command,
      exitCode,
      logPath: step.logPath,
      summary: stepSummary
    });
    if (exitCode !== 0) {
      parts.push(renderFailedStepOutput(stepOutput));
      parts.push(formatWorkflowSummary(results));
      output.stdout += applyTotalOutputBudget(parts, config.output.max_lines_total).text;
      return exitCode;
    }
  }
  parts.push(formatWorkflowSummary(results));
  output.stdout += applyTotalOutputBudget(parts, config.output.max_lines_total).text;
  return 0;
}

function workflowStepSummary(output: OperationOutput, logPath?: string): string | undefined {
  const combined = `${output.stdout}\n${output.stderr}`;
  const runId = combined.match(/^run id: (.+)$/m)?.[1];
  const transcript = combined.match(/^transcript: (.+)$/m)?.[1];
  const details = [
    ...(logPath ? [`log ${logPath}`] : []),
    ...(runId ? [`run ${runId}`] : []),
    ...(transcript ? [`transcript: ${transcript}`] : [])
  ];
  return details.length > 0 ? details.join(" - ") : undefined;
}

function renderFailedStepOutput(output: OperationOutput): string {
  const combined = `${output.stderr}${output.stdout}`;
  return combined ? `failed step output:\n${combined}` : "";
}

async function runCheckDiff(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  return runRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-check-diff",
    shellCommand: checkDiffScript({ ...config, runningOnRemote: true }),
    requireMutagenProof: true
  });
}

async function runLintDiff(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  return runRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-lint-diff",
    shellCommand: lintDiffScript({ ...config, runningOnRemote: true }),
    requireMutagenProof: true
  });
}

async function runFormatDiff(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const format = await runRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-format-diff",
    shellCommand: formatDiffScript({ ...config, runningOnRemote: true }),
    requireMutagenProof: true
  });
  if (format !== 0 || config.runningOnRemote) return format;
  const flush = await runner.run("mutagen", ["sync", "flush", config.mutagenSession]);
  output.stdout += flush.stdout;
  output.stderr += flush.stderr;
  return flush.exitCode;
}

async function runLintFiles(
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DevConfig,
  files: string[]
): Promise<number> {
  const command = dockerBashCommand(config, targetedLintShellCommand("/workspace", files));
  return runRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-lint-files",
    shellCommand: command,
    requireMutagenProof: true
  });
}

async function runFullLint(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  return runRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-lint-all",
    shellCommand: queuedDockerCommand(config, config.buildQueue, fullLintShellCommand("/workspace")),
    requireMutagenProof: true,
    artifactPaths: ["/workspace/build-docker-lint"]
  });
}

async function runPostgresUnitTests(
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DevConfig,
  suite?: string
): Promise<number> {
  const testSelector = suite
    ? `test -f /workspace/tests/unit-tests/sql/${quoteShell(`${suite}.sql`)} || { echo "unit-sql: suite ${quoteShell(suite)} not generated under /workspace/tests/unit-tests/sql"; exit 1; }`
    : "true";
  const testRunner = suite
    ? `su postgres -c "/usr/local/pgsql/bin/dropdb --if-exists regression_unit && /usr/local/pgsql/bin/createdb regression_unit && /usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression_unit -f /workspace/tests/unit-tests/sql/${quoteShell(`${suite}.sql`)}"`
    : `su postgres -c "/usr/local/pgsql/bin/dropdb --if-exists regression_unit && /usr/local/pgsql/bin/createdb regression_unit" && fail=0; for sql in /workspace/tests/unit-tests/sql/*.sql; do echo "--- $(basename "$sql") ---"; su postgres -c "/usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression_unit -f $sql" || { fail=1; echo FAIL: $sql; }; done; echo; if [ $fail -eq 0 ]; then echo UTEST-PG_OK; else echo UTEST-PG_FAILED; exit 1; fi`;
  const command = [
    "export PATH=/usr/local/pgsql/bin:$PATH",
    "/workspace/pgx-cli/dist/index.js test unit-sql --root /workspace",
    testSelector,
    buildAndInstallCommand(),
    "chmod o+x /workspace/.worktrees 2>/dev/null || true",
    "chmod -R o+rX /workspace/tests/unit-tests",
    testRunner
  ].join(" && ");
  return runRemoteShell({
    runner,
    output,
    config,
    commandName: `dev-test-unit-${suite ?? "focused"}`,
    shellCommand: `${buildCliCommand()} && ${queuedDockerCommand(config, config.buildQueue, command)}`,
    requireMutagenProof: true,
    artifactPaths: [
      "/workspace/tests/unit-tests/sql",
      "/workspace/build-artifacts/ptest"
    ]
  });
}

async function runTpchTests(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  const command = [
    "set -euo pipefail",
    "export PATH=/usr/local/pgsql/bin:$PATH",
    cleanWorkspaceCmakeArtifactsCommand(),
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
  const result = await runManagedRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-test-tpch",
    shellCommand: `${buildCliCommand()} && ${queuedDockerCommand(config, config.buildQueue, command)}`,
    requireMutagenProof: true,
    postprocess: pgRegressPostprocessor(config),
    artifactPaths: [
      "/workspace/build-artifacts/ptest/extension/tpch",
      "/tmp/pg_regress_tpch.out"
    ]
  });
  return result.workflowExitCode;
}

async function runCompile(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
  return runRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-build-compile-debug",
    shellCommand: `${buildCliCommand()} && ${compileScript(config)}`,
    requireMutagenProof: true,
    metadata: {
      profile: {
        name: "debug",
        buildDir: "/workspace/build-artifacts/ptest"
      }
    },
    artifactPaths: ["/workspace/build-artifacts/ptest"]
  });
}

async function runFullTests(runner: StreamingCommandRunner, output: OperationOutput, config: DevConfig): Promise<number> {
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
  const result = await runManagedRemoteShell({
    runner,
    output,
    config,
    commandName: "dev-test-full",
    shellCommand: `${buildCliCommand()} && ${queuedDockerCommand(config, config.buildQueue, command)}`,
    requireMutagenProof: true,
    postprocess: pgRegressPostprocessor(config),
    metadata: {
      profile: {
        name: "debug",
        buildDir: "/workspace/build-artifacts/ptest"
      }
    },
    artifactPaths: [
      "/workspace/build-artifacts/ptest",
      "/workspace/build-artifacts/ptest/Testing/Temporary/LastTest.log",
      "/tmp/ctest.out",
      "/tmp/pgx_ir"
    ]
  });
  return result.workflowExitCode;
}

async function runComparePostgresGate(
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DevConfig
): Promise<number> {
  const command = [
    buildAndInstallCommand(),
    "chmod -R o+rX /workspace",
    "PGX_COMPARE_POSTGRES_INTERNAL=1 /workspace/pgx-cli/dist/index.js test compare-postgres-internal --workload tpch-correctness --root /workspace --output-dir /workspace/build-artifacts/test-runs/tpch-correctness/compare-postgres --from-managed-runner"
  ].join(" && ");
  const result = await runManagedRemoteShell({
    runner,
    output,
    config,
    commandName: "test-compare-postgres-tpch-correctness",
    shellCommand: `${buildCliCommand()} && ${queuedDockerCommand(config, config.buildQueue, command)}`,
    requireMutagenProof: true,
    metadata: {
      workload: "tpch-correctness"
    },
    artifactPaths: ["/workspace/build-artifacts/test-runs/tpch-correctness/compare-postgres"]
  });
  return result.workflowExitCode;
}

async function runRemoteShell(input: {
  runner: StreamingCommandRunner;
  output: OperationOutput;
  config: DevConfig;
  commandName: string;
  shellCommand: string;
  requireMutagenProof: boolean;
  metadata?: Record<string, unknown>;
  artifactPaths?: string[];
}): Promise<number> {
  const result = await runManagedRemoteShell({
    runner: input.runner,
    output: input.output,
    config: input.config,
    commandName: input.commandName,
    shellCommand: input.shellCommand,
    requireMutagenProof: input.requireMutagenProof,
    metadata: input.metadata,
    artifactPaths: input.artifactPaths
  });
  return result.workflowExitCode;
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

function formatDiffScript(config: DevConfig): string {
  const dockerPipe = dockerPipeCommand(config, "cd /workspace && clang-format-diff-20 -p1 -style=file -i");
  return [
    "set -eo pipefail",
    "git fetch origin main --quiet",
    "base=$(git merge-base origin/main HEAD)",
    `diff=$(git diff -U0 "$base" -- 'src/*.c' 'src/*.cc' 'src/*.cpp' 'src/*.h' 'src/*.hpp' 'tests/*.c' 'tests/*.cc' 'tests/*.cpp' 'tests/*.h' 'tests/*.hpp' 'extension/*.c' 'extension/*.h' 2>/dev/null || true)`,
    `if [ -z "$diff" ]; then echo "format-diff: no C/C++ hunks changed vs origin/main."; exit 0; fi`,
    `printf '%s\n' "$diff" | ${dockerPipe}`,
    `echo "format-diff: applied clang-format to C/C++ hunks changed vs origin/main."`
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
  return queuedDockerCommand(config, config.buildQueue, buildAndInstallCommand());
}

function buildCliCommand(): string {
  return "npm --prefix pgx-cli install && npm --prefix pgx-cli run build";
}

function buildAndInstallCommand(): string {
  return [
    cleanWorkspaceCmakeArtifactsCommand(),
    "mkdir -p /workspace/build-artifacts/ptest",
    "cd /workspace/build-artifacts/ptest",
    "([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache /workspace)",
    "cmake --build .",
    "cmake --install ."
  ].join(" && ");
}

function cleanWorkspaceCmakeArtifactsCommand(): string {
  return [
    "rm -rf /workspace/CMakeFiles /workspace/include/runtime-defs",
    "rm -f /workspace/CMakeCache.txt /workspace/build.ninja /workspace/.ninja_deps /workspace/.ninja_log /workspace/tablegen_compile_commands.yml /workspace/CTestTestfile.cmake /workspace/cmake_install.cmake",
    "find /workspace/src/lingodb/mlir \\( -name CMakeFiles -o -name CTestTestfile.cmake -o -name cmake_install.cmake -o -name '*.inc' -o -name '*.inc.d' -o -name '*.o' -o -name '*.a' \\) -exec rm -rf {} +"
  ].join(" && ");
}

function queuedDockerCommand(config: DevConfig, queue: string, command: string): string {
  return [
    `export TS_SOCKET=/tmp/${quoteShell(queue)}.sock`,
    "tsp -S 1 >/dev/null",
    `id=$(tsp ${dockerBashCommand(config, command)})`,
    `echo "[job $id queued on ${queue}]"`,
    "status=0",
    'tsp -w "$id" || status=$?',
    "cat_status=0",
    'tsp -c "$id" || cat_status=$?',
    'if [ "$cat_status" -ne 0 ] && [ "$status" -eq 0 ]; then exit "$cat_status"; fi',
    'exit "$status"'
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

function pgRegressPostprocessor(config: DevConfig) {
  return ({ artifact, childExitCode }: { artifact: { combinedPath: string }; childExitCode: number }) => {
    const raw = readFileSync(artifact.combinedPath, "utf8");
    const ctestFailure = detectCtestFailure(raw);
    if (ctestFailure) {
      return {
        workflowExitCode: 1,
        postprocessedFailure: ctestFailure,
        summary: {
          ctest: {
            failed: true
          }
        }
      };
    }
    if (childExitCode !== 0 && !hasPgRegressBaselineInput(raw)) {
      return { workflowExitCode: childExitCode };
    }
    const evaluated = evaluatePgRegressBaseline(raw, readBaselineText(config));
    return {
      workflowExitCode: evaluated.exitCode,
      postprocessedFailure: evaluated.exitCode === 0 ? undefined : `${evaluated.stdout}${evaluated.stderr}`.trim(),
      summary: {
        pgRegressBaseline: {
          childExitCode,
          workflowExitCode: evaluated.exitCode
        }
      }
    };
  };
}

function resultsSafeName(value: string): string {
  return value.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "command";
}
