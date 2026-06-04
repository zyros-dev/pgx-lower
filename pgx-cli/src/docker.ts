import type { CommandRunner } from "./commands.js";
import type { OperationConfig, OperationOutput } from "./operations.js";

export type DockerConfig = OperationConfig & {
  dockerContainer: string;
};

export async function runDockerCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: DockerConfig
): Promise<number> {
  const [command, subcommand] = args;
  if (command === "status") {
    return runRemoteShell(
      runner,
      output,
      config,
      `docker ps --format '{{.Names}}' | grep -E '^${quoteRegex(config.dockerContainer)}$'`
    );
  }
  if (command === "build" && subcommand === "ptest") {
    return runRemoteShell(runner, output, config, ptestCommand(config.dockerContainer));
  }
  if (command === "build" && subcommand === "release") {
    return runRemoteShell(runner, output, config, releaseCommand(config.dockerContainer));
  }

  output.stderr += "Usage: docker <status|build <ptest|release>>\n";
  return 1;
}

function ptestCommand(container: string): string {
  return `docker exec ${quoteShell(container)} bash -lc ${quoteShell(
    [
      "cd /workspace",
      "rm -f CMakeCache.txt",
      "mkdir -p build-artifacts/docker/ptest",
      "cd build-artifacts/docker/ptest",
      "rm -f CMakeCache.txt",
      "cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON /workspace",
      "cmake --build .",
      "cmake --install .",
      "ctest --output-on-failure"
    ].join(" && ")
  )}`;
}

function releaseCommand(container: string): string {
  return `docker exec ${quoteShell(container)} bash -lc ${quoteShell(
    [
      "cd /workspace",
      "rm -rf build-artifacts/docker/ptest-release",
      "mkdir -p build-artifacts/docker/ptest-release",
      "cd build-artifacts/docker/ptest-release",
      "cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_ONLY_EXTENSION=ON /workspace",
      "ninja",
      "strip --strip-debug extension/pgx_lower.so"
    ].join(" && ")
  )}`;
}

async function runRemoteShell(
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig,
  shellCommand: string
): Promise<number> {
  const remoteShell = `cd ${quoteShell(config.remoteProjectPath)} && ${shellCommand}`;
  const result = config.runningOnRemote
    ? await runner.run("bash", ["-lc", remoteShell])
    : await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(remoteShell)]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

function quoteRegex(value: string): string {
  return value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
