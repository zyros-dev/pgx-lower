import { renderBuildExplain, renderConfigureCommand } from "./build-profile.js";
import type { StreamingCommandRunner } from "./commands.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";
import type { OperationConfig, OperationOutput } from "./operations.js";
import type { ResolvedProfileConfig } from "./project-config.js";
import { satisfyGateFailureState } from "./gate-memory.js";

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
    await clearMatchingGateMemory(["build", ...args], result.workflowExitCode, runner, output, config);
    return result.workflowExitCode;
  }

  output.stderr += "Usage: dev build <explain|configure|compile|install|clean|reconfigure> --profile <name>\n";
  return 1;
}

async function clearMatchingGateMemory(
  argv: string[],
  exitCode: number,
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DevBuildConfig
): Promise<void> {
  if (exitCode !== 0) return;
  const currentHead = await readCurrentHead(runner, config);
  const previousFailure = satisfyGateFailureState({ root: config.localProjectPath, currentHead, argv });
  if (!previousFailure) return;
  output.stdout += `gate memory: focused reproducer passed; cleared review gate block for ${previousFailure.stepName}\n`;
}

async function readCurrentHead(runner: StreamingCommandRunner, config: DevBuildConfig): Promise<string> {
  const result = await runner.run("git", ["-C", config.localProjectPath, "rev-parse", "HEAD"]);
  return result.exitCode === 0 && result.stdout.trim() ? result.stdout.trim() : "unknown";
}

function buildProfileCommand(action: string, profile: ResolvedProfileConfig, container: string): string {
  const configure = renderConfigureCommand(profile, "/workspace").map(quoteShell).join(" ");
  const configureShell = `${cleanWorkspaceCmakeArtifactsCommand()} && ${configure}`;
  if (action === "configure") return `${buildCliCommand()} && docker exec ${quoteShell(container)} bash -lc ${quoteShell(configureShell)}`;
  if (action === "compile") return `docker exec ${quoteShell(container)} cmake --build ${quoteShell(profile.build.build_dir)}`;
  if (action === "install") return `docker exec ${quoteShell(container)} cmake --install ${quoteShell(profile.build.build_dir)}`;
  if (action === "clean") return `docker exec ${quoteShell(container)} cmake --build ${quoteShell(profile.build.build_dir)} --target clean`;
  if (action === "reconfigure") return `${buildCliCommand()} && docker exec ${quoteShell(container)} bash -lc ${quoteShell(configureShell)}`;
  throw new Error(`unknown build profile action: ${action}`);
}

function buildCliCommand(): string {
  return "npm --prefix pgx-cli install && npm --prefix pgx-cli run build";
}

function cleanWorkspaceCmakeArtifactsCommand(): string {
  return [
    "rm -rf /workspace/CMakeFiles /workspace/include/runtime-defs",
    "rm -f /workspace/CMakeCache.txt /workspace/build.ninja /workspace/.ninja_deps /workspace/.ninja_log /workspace/tablegen_compile_commands.yml /workspace/CTestTestfile.cmake /workspace/cmake_install.cmake",
    "find /workspace/src/lingodb/mlir \\( -name CMakeFiles -o -name CTestTestfile.cmake -o -name cmake_install.cmake -o -name '*.inc' -o -name '*.inc.d' -o -name '*.o' -o -name '*.a' \\) -exec rm -rf {} +"
  ].join(" && ");
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
