import type { CommandRunner } from "./commands.js";
import type { OperationConfig, OperationOutput } from "./operations.js";

export type WorkflowStep = {
  name: string;
  command: string[];
  exitCode: number;
  logPath?: string;
  jobId?: string;
  summary?: string;
};

export type DevConfig = OperationConfig & {
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
    if (lintCommand === "diff") return runJust(runner, output, config, ["lint-diff"]);
    if (lintCommand === "file" && lintArgs.length === 1) {
      return runJust(runner, output, config, ["lint-files", lintArgs[0]]);
    }
    if (lintCommand === "files" && lintArgs.length > 0) {
      return runJust(runner, output, config, ["lint-files", ...lintArgs]);
    }
    output.stderr += "Usage: dev lint <file <path>|files <paths...>|diff>\n";
    return 1;
  }

  if (command === "test") {
    const [testCommand, testArg] = rest;
    if (testCommand === "unit" && testArg) return runJust(runner, output, config, ["utest-pg-one", testArg]);
    if (testCommand === "tpch") return runJust(runner, output, config, ["test-tpch"]);
    if (testCommand === "focused") return runJust(runner, output, config, ["utest-pg"]);
    output.stderr += "Usage: dev test <unit <suite>|tpch|focused>\n";
    return 1;
  }

  if (command === "gate") {
    const [gateCommand] = rest;
    if (gateCommand === "batch") {
      return runWorkflow(runner, output, config, [
        { name: "check diff", justArgs: ["check-diff"] },
        { name: "lint diff", justArgs: ["lint-diff"] },
        { name: "utest-pg", justArgs: ["utest-pg"] }
      ]);
    }
    if (gateCommand === "review") {
      return runWorkflow(runner, output, config, [
        { name: "check diff", justArgs: ["check-diff"] },
        { name: "lint", justArgs: ["lint"] },
        { name: "compile", justArgs: ["compile"], logPath: "/tmp/pgx-compile.out" },
        { name: "utest-pg", justArgs: ["utest-pg"] },
        { name: "test", justArgs: ["test"] }
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
  steps: Array<{ name: string; justArgs: string[]; logPath?: string }>
): Promise<number> {
  const results: WorkflowStep[] = [];
  for (const step of steps) {
    const exitCode = await runJust(runner, output, config, step.justArgs);
    results.push({
      name: step.name,
      command: ["just", ...step.justArgs],
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

async function flushMutagen(runner: CommandRunner, output: OperationOutput, session: string): Promise<number> {
  const result = await runner.run("mutagen", ["sync", "flush", session]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

async function runJust(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  args: string[]
): Promise<number> {
  const flush = await flushMutagen(runner, output, config.mutagenSession);
  if (flush !== 0) return flush;
  return runRemoteShell(
    runner,
    output,
    config,
    `cd ${quoteShell(config.remoteProjectPath)} && ${["just", ...args].map(quoteShell).join(" ")}`
  );
}

async function runRemoteShell(
  runner: CommandRunner,
  output: OperationOutput,
  config: DevConfig,
  shellCommand: string
): Promise<number> {
  const result = await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(shellCommand)]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
