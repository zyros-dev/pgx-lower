#!/usr/bin/env node
import { loadConfig } from "./config.js";
import { DEFAULT_CONFIG_PATH, writeConfig } from "./config.js";
import { renderBuildExplain } from "./build-profile.js";
import { helpText, runCli } from "./cli.js";
import { NodeCommandRunner } from "./commands.js";
import { runBenchCommand } from "./bench.js";
import { runDevBuildCommand } from "./dev-build.js";
import { runDevCommand } from "./dev.js";
import { runDockerCommand } from "./docker.js";
import { runLogsCommand } from "./logs.js";
import { connectMcp } from "./mcp.js";
import { runRouteCheckCommand } from "./pg-regress-routes.js";
import { runPsqlRegressionBurndownCommand } from "./psql-regression-burndown.js";
import { runUnitSqlCommand } from "./unit-sql.js";
import { runGatewayCommand } from "./run.js";
import { runCodexPolicyCommand } from "./codex-policy.js";
import {
  flushRemainingOutput,
  runQueueCommand,
  runSetupCommand,
  runSyncCommand,
  runThorCommand
} from "./operations.js";
import type { OperationOutput } from "./operations.js";
import { loadProjectConfig, resolveProfile } from "./project-config.js";
import { DEFAULT_REQUEST_DIR, writeRequest } from "./requests.js";
import { runRepoCommand } from "./repo-audit.js";
import { DEFAULT_USAGE_PATH, incrementUsage } from "./usage.js";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const { url, argv } = parseGlobalArgs(process.argv.slice(2));
const config = loadConfig({ env: process.env, argvUrl: url });
incrementUsage(DEFAULT_USAGE_PATH, argv);
const packageDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const io: OperationOutput = {
  stdout: "",
  stderr: "",
  liveStdout: (text) => process.stdout.write(text),
  liveStderr: (text) => process.stderr.write(text)
};

try {
  if (!argv[0] || argv[0] === "help" || argv[0] === "--help" || argv[0] === "-h") {
    process.stdout.write(helpText());
    process.exitCode = 0;
    process.exit();
  }

  const configExitCode = handleConfigCommand(argv, config.url);
  if (configExitCode !== undefined) {
    process.exitCode = configExitCode;
    process.exit();
  }

  const tunnelExitCode = await handleTunnelCommand(argv, config);
  if (tunnelExitCode !== undefined) {
    process.exitCode = tunnelExitCode;
    process.exit();
  }

  const runner = new NodeCommandRunner();
  if (argv[0] === "setup") {
    process.exitCode = await runSetupCommand(argv.slice(1), runner, io, {
      packageDir,
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      dockerContainer: config.dockerContainer,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "sync") {
    process.exitCode = await runSyncCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "thor") {
    process.exitCode = await runThorCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "queue") {
    process.exitCode = await runQueueCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "docker") {
    process.exitCode = await runDockerCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      dockerContainer: config.dockerContainer,
      localProjectPath: config.localProjectPath,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "logs") {
    process.exitCode = await runLogsCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      dockerContainer: config.dockerContainer,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "run") {
    process.exitCode = await runGatewayCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      dockerContainer: config.dockerContainer,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "bench") {
    process.exitCode = await runBenchCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      dockerContainer: config.dockerContainer,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "codex-policy") {
    process.exitCode = runCodexPolicyCommand(argv.slice(1), io);
    flushIo();
    process.exit();
  }

  if (argv[0] === "repo") {
    process.exitCode = await runRepoCommand(argv.slice(1), runner, io, {
      localProjectPath: config.localProjectPath
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "test" && argv[1] === "route-check") {
    process.exitCode = await runRouteCheckCommand(argv.slice(2), runner, io);
    flushIo();
    process.exit();
  }

  if (argv[0] === "test" && argv[1] === "psql-regression-burndown") {
    process.exitCode = await runPsqlRegressionBurndownCommand(argv.slice(2), runner, io);
    flushIo();
    process.exit();
  }

  if (argv[0] === "test" && argv[1] === "unit-sql") {
    process.exitCode = runUnitSqlCommand(argv.slice(2), io);
    flushIo();
    process.exit();
  }

  if (argv[0] === "dev" && argv[1] === "build") {
    const devBuildArgs = argv.slice(2);
    const profileName = profileNameFromArgs(devBuildArgs);
    const projectConfig = profileName ? loadProjectConfig() : undefined;
    const profile = projectConfig && profileName ? resolveProfile(projectConfig, profileName) : undefined;
    process.exitCode = await runDevBuildCommand(devBuildArgs, runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      profileName,
      profile,
      dockerContainer: config.dockerContainer,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "dev") {
    process.exitCode = await runDevCommand(argv.slice(1), runner, io, {
      mutagenSession: config.mutagenSession,
      sshHost: config.sshHost,
      remoteProjectPath: config.remoteProjectPath,
      localProjectPath: config.localProjectPath,
      dockerContainer: config.dockerContainer,
      buildQueue: config.buildQueue,
      checkQueue: config.checkQueue,
      sync: config.sync,
      output: config.output,
      runningOnRemote: config.runningOnRemote
    });
    flushIo();
    process.exit();
  }

  if (argv[0] === "request") {
    const kind = argv[1];
    if (kind !== "feature" && kind !== "complaint") {
      process.stderr.write("Usage: pgx-cli request <feature|complaint> <message...>\n");
      process.exit(1);
    }

    const path = writeRequest(DEFAULT_REQUEST_DIR, kind, argv.slice(2));
    process.stdout.write(`Wrote ${path}\n`);
    process.exit(0);
  }

  const client = await connect(config.url);
  const exitCode = await runCli(argv, client, io, {
    url: config.url,
    remoteUrl: config.remoteUrl,
    sshHost: config.sshHost,
    projectPath: config.projectPath
  });
  flushIo();
  process.exitCode = exitCode;
} catch (error) {
  flushIo();
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}

function flushIo(): void {
  flushRemainingOutput(
    io,
    (text) => process.stdout.write(text),
    (text) => process.stderr.write(text)
  );
}

function parseGlobalArgs(args: string[]): { url?: string; argv: string[] } {
  const argv = [...args];
  const urlIndex = argv.findIndex((arg) => arg === "--url");
  if (urlIndex === -1) {
    return { argv };
  }

  const url = argv[urlIndex + 1];
  if (!url) {
    throw new Error("Usage: pgx-cli --url <mcp-url> <command>");
  }

  argv.splice(urlIndex, 2);
  return { url, argv };
}

async function connect(url: string) {
  try {
    return await connectMcp(url);
  } catch (error) {
    throw new Error(
      `Could not connect to CLion MCP at ${url}. Is CLion running with the MCP server enabled?\n${error instanceof Error ? error.message : String(error)}`
    );
  }
}

function handleConfigCommand(argv: string[], currentUrl: string): number | undefined {
  if (argv[0] !== "config") {
    return undefined;
  }

  if (argv[1] === "validate") {
    const projectConfig = loadProjectConfig();
    if (!projectConfig) {
      process.stderr.write("No pgx-cli.yaml found\n");
      return 1;
    }
    process.stdout.write(`Config valid: ${projectConfig.configPath}\n`);
    return 0;
  }

  if (argv[1] === "path") {
    process.stdout.write(`${DEFAULT_CONFIG_PATH}\n`);
    return 0;
  }

  if (argv[1] === "show" && argv.includes("--profile")) {
    const profileName = argv[argv.indexOf("--profile") + 1];
    if (!profileName) {
      process.stderr.write("Usage: pgx-cli config show --profile <name>\n");
      return 1;
    }
    const projectConfig = loadProjectConfig();
    if (!projectConfig) {
      process.stderr.write("No pgx-cli.yaml found\n");
      return 1;
    }
    process.stdout.write(renderBuildExplain(profileName, resolveProfile(projectConfig, profileName)));
    return 0;
  }

  if (argv[1] === "show" && argv.includes("--sources")) {
    const projectConfig = loadProjectConfig();
    process.stdout.write(`${JSON.stringify({
      personalConfig: DEFAULT_CONFIG_PATH,
      projectConfig: projectConfig?.configPath,
      localProjectConfig: projectConfig?.localConfigPath
    }, null, 2)}\n`);
    return 0;
  }

  if (argv[1] === "show") {
    process.stdout.write(`${JSON.stringify({ ...config, url: currentUrl }, null, 2)}\n`);
    return 0;
  }

  if (argv[1] === "set-url") {
    const nextUrl = argv[2];
    if (!nextUrl) {
      process.stderr.write("Usage: pgx-cli config set-url <mcp-url>\n");
      return 1;
    }

    writeConfig(DEFAULT_CONFIG_PATH, { ...config, url: nextUrl });
    process.stdout.write(`Wrote ${DEFAULT_CONFIG_PATH}\n`);
    return 0;
  }

  if (argv[1] === "set-project") {
    const projectPath = argv[2];
    if (!projectPath) {
      process.stderr.write("Usage: pgx-cli config set-project <project-path>\n");
      return 1;
    }

    writeConfig(DEFAULT_CONFIG_PATH, { ...config, projectPath });
    process.stdout.write(`Wrote ${DEFAULT_CONFIG_PATH}\n`);
    return 0;
  }

  if (argv[1] === "set-ssh-host") {
    const sshHost = argv[2];
    if (!sshHost) {
      process.stderr.write("Usage: pgx-cli config set-ssh-host <ssh-host>\n");
      return 1;
    }

    writeConfig(DEFAULT_CONFIG_PATH, { ...config, sshHost });
    process.stdout.write(`Wrote ${DEFAULT_CONFIG_PATH}\n`);
    return 0;
  }

  process.stderr.write("Usage: pgx-cli config <path|show|set-url|set-project|set-ssh-host>\n");
  return 1;
}

function profileNameFromArgs(args: string[]): string | undefined {
  const index = args.indexOf("--profile");
  return index === -1 ? undefined : args[index + 1];
}

async function handleTunnelCommand(
  argv: string[],
  currentConfig: typeof config
): Promise<number | undefined> {
  if (argv[0] !== "tunnel") {
    return undefined;
  }

  if (await isReachable(currentConfig.url)) {
    process.stdout.write(`MCP URL is already reachable: ${currentConfig.url}\n`);
    return 0;
  }

  const local = new URL(currentConfig.url);
  const remote = new URL(currentConfig.remoteUrl);
  const localPort = local.port || (local.protocol === "https:" ? "443" : "80");
  const remotePort = remote.port || (remote.protocol === "https:" ? "443" : "80");
  const args = [
    "-f",
    "-N",
    "-L",
    `${local.hostname}:${localPort}:${remote.hostname}:${remotePort}`,
    currentConfig.sshHost
  ];

  const result = spawnSync("ssh", args, { encoding: "utf8" });
  if (result.status !== 0) {
    process.stderr.write(result.stderr || result.stdout || "ssh tunnel failed\n");
    return result.status ?? 1;
  }

  process.stdout.write(
    `Forwarding ${currentConfig.url} -> ${currentConfig.sshHost}:${currentConfig.remoteUrl}\n`
  );
  return 0;
}

async function isReachable(url: string): Promise<boolean> {
  try {
    await fetch(url, { signal: AbortSignal.timeout(2000) });
    return true;
  } catch {
    return false;
  }
}
