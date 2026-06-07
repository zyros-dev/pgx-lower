import { renderBuildExplain, renderConfigureCommand } from "./build-profile.js";
import type { StreamingCommandRunner } from "./commands.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";
import type { OperationConfig, OperationOutput } from "./operations.js";
import type { ResolvedProfileConfig } from "./project-config.js";

export type DevBuildConfig = ManagedOperationConfig & {
  profileName?: string;
  profile?: ResolvedProfileConfig;
  dockerContainer?: string;
};

export async function runDevBuildCommand(
  args: string[],
  runner: StreamingCommandRunner,
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
    const result = await runManagedRemoteShell({
      runner,
      output,
      config: { ...config, dockerContainer: config.dockerContainer },
      commandName: `dev-build-${action}-${config.profileName ?? "unknown"}`,
      shellCommand: command,
      requireMutagenProof: true,
      metadata: {
        profile: {
          name: config.profileName,
          buildDir: config.profile.build.build_dir
        }
      },
      artifactPaths: [config.profile.build.build_dir]
    });
    return result.workflowExitCode;
  }

  output.stderr += "Usage: dev build <explain|configure|compile|install|clean|reconfigure> --profile <name>\n";
  return 1;
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
