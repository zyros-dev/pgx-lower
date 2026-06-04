import type { CommandRunner } from "./commands.js";

export type OperationOutput = {
  stdout: string;
  stderr: string;
};

export type OperationConfig = {
  mutagenSession: string;
  sshHost: string;
  remoteProjectPath: string;
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

  output.stderr += "Usage: sync <status|flush>\n";
  return 1;
}

export async function runThorCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig
): Promise<number> {
  const [command, ...rest] = args;

  if (command === "shell") {
    const separatorIndex = rest.indexOf("--");
    if (!rest.includes("--dangerous") || separatorIndex === -1 || separatorIndex === rest.length - 1) {
      output.stderr += "Usage: thor shell --dangerous -- <cmd...>\n";
      return 1;
    }

    const flush = await flushMutagen(runner, output, config);
    if (flush !== 0) {
      return flush;
    }

    return runRemote(runner, output, config, rest.slice(separatorIndex + 1));
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
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig
): Promise<number> {
  const [command, id] = args;

  if (command === "status") {
    return runRemoteWithFlush(
      runner,
      output,
      config,
      'for q in pgx-build pgx-check; do echo "=== ${q} queue ==="; TS_SOCKET=/tmp/${q}.sock tsp; done',
      "queue status"
    );
  }

  if (command === "tail" && isNumericId(id)) {
    return runRemoteWithFlush(
      runner,
      output,
      config,
      `TS_SOCKET=/tmp/pgx-build.sock tsp -t ${id}`,
      `queue tail ${id}`
    );
  }

  if (command === "cancel" && isNumericId(id)) {
    return runRemoteWithFlush(
      runner,
      output,
      config,
      `TS_SOCKET=/tmp/pgx-build.sock tsp -k ${id} || true; TS_SOCKET=/tmp/pgx-build.sock tsp -r ${id}`,
      `queue cancel ${id}`
    );
  }

  if (command === "flush") {
    return runRemoteWithFlush(
      runner,
      output,
      config,
      "TS_SOCKET=/tmp/pgx-build.sock tsp -C && TS_SOCKET=/tmp/pgx-check.sock tsp -C",
      "queue flush"
    );
  }

  output.stderr += "Usage: queue <status|flush|tail <id>|cancel <id>>\n";
  return 1;
}

function isNumericId(value: string | undefined): value is string {
  return value !== undefined && /^[0-9]+$/.test(value);
}

async function flushMutagen(runner: CommandRunner, output: OperationOutput, config: OperationConfig): Promise<number> {
  if (config.runningOnRemote) {
    output.stdout += "mutagen: skipped (already on thor)\n";
    return 0;
  }
  output.stdout += `Flushing Mutagen session: ${config.mutagenSession}\n`;
  const result = await runner.run("mutagen", ["sync", "flush", config.mutagenSession]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

async function runRemoteWithFlush(
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig,
  shellCommand: string,
  displayCommand: string
): Promise<number> {
  const flush = await flushMutagen(runner, output, config);
  if (flush !== 0) {
    return flush;
  }

  return runRemoteShell(runner, output, config, shellCommand, displayCommand);
}

async function runRemote(
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig,
  command: string[]
): Promise<number> {
  const remoteCommand = `cd ${quoteShell(config.remoteProjectPath)} && ${command
    .map(quoteShell)
    .join(" ")}`;
  const remoteShell = `export PATH=$HOME/.local/bin:$PATH && ${remoteCommand}`;
  output.stdout += `thor: ${config.sshHost}:${config.remoteProjectPath}\n`;
  output.stdout += `$ ${command.join(" ")}\n`;
  const result = config.runningOnRemote
    ? await runner.run("bash", ["-lc", remoteShell])
    : await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(remoteShell)]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

async function runRemoteShell(
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig,
  shellCommand: string,
  displayCommand: string
): Promise<number> {
  const remoteShell = `export PATH=$HOME/.local/bin:$PATH && cd ${quoteShell(config.remoteProjectPath)} && ${shellCommand}`;
  output.stdout += `thor: ${config.sshHost}:${config.remoteProjectPath}\n`;
  output.stdout += `$ ${displayCommand}\n`;
  const result = config.runningOnRemote
    ? await runner.run("bash", ["-lc", remoteShell])
    : await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(remoteShell)]);
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
