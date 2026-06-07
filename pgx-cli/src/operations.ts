import type { CommandRunner, StreamingCommandRunner } from "./commands.js";
import { evaluateMutagenListJson } from "./mutagen-preflight.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";

export type OperationOutput = {
  stdout: string;
  stderr: string;
};

export type OperationConfig = {
  mutagenSession: string;
  sshHost: string;
  remoteProjectPath: string;
  localProjectPath?: string;
  runningOnRemote?: boolean;
};

export type SetupConfig = OperationConfig & {
  packageDir: string;
  dockerContainer: string;
};

export async function runSyncCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: Pick<OperationConfig, "mutagenSession">
): Promise<number> {
  const [command] = args;

  if (command === "status") {
    output.stdout += `Mutagen session: ${config.mutagenSession}\n`;
    const result = await runner.run("mutagen", ["sync", "list", config.mutagenSession]);
    output.stdout += result.stdout;
    output.stderr += result.stderr;
    return result.exitCode;
  }

  if (command === "flush") {
    output.stdout += `Flushing Mutagen session: ${config.mutagenSession}\n`;
    const result = await runner.run("mutagen", ["sync", "flush", config.mutagenSession]);
    output.stdout += result.stdout;
    output.stderr += result.stderr;
    return result.exitCode;
  }

  if (command === "doctor") {
    const result = await runner.run("mutagen", ["sync", "list", config.mutagenSession, "--template", "{{json .}}"]);
    const health = result.exitCode === 0
      ? evaluateMutagenListJson(result.stdout, config.mutagenSession)
      : { healthy: false as const, reason: `mutagen session ${config.mutagenSession} status check failed` };
    output.stdout += health.healthy ? "sync doctor: ok\n" : `sync doctor: failed - ${health.reason}\n`;
    if (!health.healthy) {
      output.stdout += "next:\n- pgx-cli sync status\n- pgx-cli sync doctor\n";
    }
    output.stderr += result.stderr;
    return health.healthy ? 0 : 1;
  }

  output.stderr += "Usage: sync <status|flush|doctor>\n";
  return 1;
}

export async function runThorCommand(
  args: string[],
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: ManagedOperationConfig
): Promise<number> {
  const [command, ...rest] = args;

  if (command === "shell") {
    const separatorIndex = rest.indexOf("--");
    if (!rest.includes("--dangerous") || separatorIndex === -1 || separatorIndex === rest.length - 1) {
      output.stderr += "Usage: thor shell --dangerous -- <cmd...>\n";
      return 1;
    }

    const commandArgs = rest.slice(separatorIndex + 1);
    const result = await runManagedRemoteShell({
      runner,
      output,
      config,
      commandName: `thor-shell-${commandArgs[0] ?? "command"}`,
      shellCommand: commandArgs.map(quoteShell).join(" "),
      requireMutagenProof: true
    });
    return result.workflowExitCode;
  }

  output.stderr += "Usage: thor shell --dangerous -- <cmd...>\n";
  return 1;
}

export async function runSetupCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: SetupConfig
): Promise<number> {
  const [command] = args;

  if (command === "install") {
    const existing = await runner.run("sh", ["-lc", "command -v pgx-cli || true"]);
    const existingPath = existing.stdout.trim();
    output.stdout += existingPath ? `Existing pgx-cli: ${existingPath}\n` : "Existing pgx-cli: not found\n";

    for (const npmArgs of [
      ["--prefix", config.packageDir, "install"],
      ["--prefix", config.packageDir, "run", "build"]
    ]) {
      const result = await runner.run("npm", npmArgs);
      output.stdout += result.stdout;
      output.stderr += result.stderr;
      if (result.exitCode !== 0) {
        return result.exitCode;
      }
    }

    const link = await runner.run("sh", ["-lc", `cd ${quoteShell(config.packageDir)} && npm link --force`]);
    output.stdout += link.stdout;
    output.stderr += link.stderr;
    if (link.exitCode !== 0) {
      return link.exitCode;
    }

    output.stdout += "setup install: linked in-repo pgx-cli\n";
    return 0;
  }

  if (command === "doctor") {
    type SetupCheckResult = { stdout: string; stderr: string; exitCode: number };
    const checks: Array<{
      label: string;
      command: string;
      args: string[];
      accept?: (result: SetupCheckResult) => boolean;
    }> = [
      {
        label: "global pgx-cli",
        command: "sh",
        args: ["-lc", "command -v pgx-cli || true"],
        accept: (result) => result.stdout.trim().length > 0
      },
      { label: "mutagen session", command: "mutagen", args: ["sync", "list", config.mutagenSession] },
      { label: "ssh host", command: "ssh", args: [config.sshHost, "true"] },
      { label: "remote checkout", command: "ssh", args: [config.sshHost, "test", "-d", config.remoteProjectPath] },
      { label: "task-spooler", command: "ssh", args: [config.sshHost, "command", "-v", "tsp"] },
      {
        label: "docker container",
        command: "ssh",
        args: [config.sshHost, "docker", "ps", "--format", "{{.Names}}"],
        accept: (result) => result.stdout.split("\n").includes(config.dockerContainer)
      }
    ];

    let failed = false;
    for (const check of checks) {
      const result = await runner.run(check.command, check.args);
      output.stdout += result.stdout;
      output.stderr += result.stderr;
      const ok = result.exitCode === 0 && (check.accept ? check.accept(result) : true);
      output.stdout += `${ok ? "ok" : "fail"} ${check.label}\n`;
      failed ||= !ok;
    }

    output.stdout += failed ? "setup doctor: failed\n" : "setup doctor: ok\n";
    return failed ? 1 : 0;
  }

  output.stderr += "Usage: setup <install|doctor>\n";
  return 1;
}

export async function runQueueCommand(
  args: string[],
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: ManagedOperationConfig
): Promise<number> {
  const [command, id] = args;

  if (command === "status") {
    return runManagedQueue(runner, output, config, "queue-status", 'for q in pgx-build pgx-check; do echo "=== ${q} queue ==="; TS_SOCKET=/tmp/${q}.sock tsp; done');
  }

  if (command === "tail" && isNumericId(id)) {
    return runManagedQueue(runner, output, config, `queue-tail-${id}`, `TS_SOCKET=/tmp/pgx-build.sock tsp -t ${id}`);
  }

  if (command === "cancel" && isNumericId(id)) {
    return runManagedQueue(runner, output, config, `queue-cancel-${id}`, `TS_SOCKET=/tmp/pgx-build.sock tsp -k ${id} || true; TS_SOCKET=/tmp/pgx-build.sock tsp -r ${id}`);
  }

  if (command === "flush") {
    return runManagedQueue(runner, output, config, "queue-flush", "TS_SOCKET=/tmp/pgx-build.sock tsp -C && TS_SOCKET=/tmp/pgx-check.sock tsp -C");
  }

  output.stderr += "Usage: queue <status|flush|tail <id>|cancel <id>>\n";
  return 1;
}

function isNumericId(value: string | undefined): value is string {
  return value !== undefined && /^[0-9]+$/.test(value);
}

async function runManagedQueue(
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: ManagedOperationConfig,
  commandName: string,
  shellCommand: string,
): Promise<number> {
  const result = await runManagedRemoteShell({
    runner,
    output,
    config,
    commandName,
    shellCommand,
    requireMutagenProof: true
  });
  return result.workflowExitCode;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }

  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
