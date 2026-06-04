import { renderBuildExplain, renderConfigureCommand } from "./build-profile.js";
import type { CommandRunner } from "./commands.js";
import type { OperationConfig, OperationOutput } from "./operations.js";
import type { ResolvedProfileConfig } from "./project-config.js";

export type DevBuildConfig = OperationConfig & {
  profileName?: string;
  profile?: ResolvedProfileConfig;
  dockerContainer?: string;
};

export async function runDevBuildCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: DevBuildConfig
): Promise<number> {
  if (args[0] === "explain") {
    if (!config.profileName || !config.profile) {
      output.stderr += "Usage: dev build explain --profile <name>\n";
      return 1;
    }
    output.stdout += renderBuildExplain(config.profileName, config.profile);
    return 0;
  }

  const action = args[0] ?? "";
  if (["configure", "compile", "install", "clean", "reconfigure"].includes(action)) {
    if (!config.profile || !config.dockerContainer) {
      output.stderr += `Usage: dev build ${action} --profile <name>\n`;
      return 1;
    }
    const command = buildProfileCommand(action, config.profile, config.dockerContainer);
    const flush = await flushMutagen(runner, output, config);
    if (flush !== 0) return flush;
    return runRemoteShell(runner, output, config, command, `dev build ${action} --profile ${config.profileName ?? "unknown"}`);
  }

  output.stderr += "Usage: dev build <explain|configure|compile|install|clean|reconfigure> --profile <name>\n";
  return 1;
}

async function flushMutagen(runner: CommandRunner, output: OperationOutput, config: OperationConfig): Promise<number> {
  if (config.runningOnRemote) {
    output.stdout += "mutagen: skipped (already on thor)\n";
    return 0;
  }
  const result = await runner.run("mutagen", ["sync", "flush", config.mutagenSession]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

async function runRemoteShell(
  runner: CommandRunner,
  output: OperationOutput,
  config: OperationConfig,
  command: string,
  summary: string
): Promise<number> {
  const remoteShell = `cd ${quoteShell(config.remoteProjectPath)} && ${command}`;
  const result = config.runningOnRemote
    ? await runner.run("bash", ["-lc", remoteShell])
    : await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(remoteShell)]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  output.stdout += `${summary}: exit ${result.exitCode}\n`;
  return result.exitCode;
}

function buildProfileCommand(action: string, profile: ResolvedProfileConfig, container: string): string {
  const configure = renderConfigureCommand(profile, "/workspace").map(quoteShell).join(" ");
  if (action === "configure") return `docker exec ${quoteShell(container)} ${configure}`;
  if (action === "compile") return `docker exec ${quoteShell(container)} cmake --build ${quoteShell(profile.build.build_dir)}`;
  if (action === "install") return `docker exec ${quoteShell(container)} cmake --install ${quoteShell(profile.build.build_dir)}`;
  if (action === "clean") return `docker exec ${quoteShell(container)} cmake --build ${quoteShell(profile.build.build_dir)} --target clean`;
  if (action === "reconfigure") return `docker exec ${quoteShell(container)} ${configure}`;
  throw new Error(`unknown build profile action: ${action}`);
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
