import type { StreamingCommandRunner } from "./commands.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";
import type { OperationConfig, OperationOutput } from "./operations.js";

export type DockerConfig = ManagedOperationConfig & {
  dockerContainer: string;
};

export async function runDockerCommand(
  args: string[],
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DockerConfig
): Promise<number> {
  const [command, subcommand] = args;
  if (command === "status") {
    return runManagedDocker(runner, output, config, "docker-status", `docker ps --format '{{.Names}}' | grep -E '^${quoteRegex(config.dockerContainer)}$'`);
  }
  if (command === "build" && subcommand === "ptest") {
    return runManagedDocker(runner, output, config, "docker-build-ptest", ptestCommand(config.dockerContainer));
  }
  if (command === "build" && subcommand === "release") {
    return runManagedDocker(runner, output, config, "docker-build-release", releaseCommand(config.dockerContainer));
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

async function runManagedDocker(
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DockerConfig,
  commandName: string,
  shellCommand: string
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

function quoteRegex(value: string): string {
  return value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
