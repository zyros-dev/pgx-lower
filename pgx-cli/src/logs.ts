import type { CommandRunner } from "./commands.js";
import type { OperationConfig, OperationOutput } from "./operations.js";

export type LogsConfig = OperationConfig & {
  dockerContainer: string;
};

const defaultLines = 50;
const pgxErrorsPath = "/tmp/pgx_errors.log";

export async function runLogsCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: LogsConfig
): Promise<number> {
  const parsed = parseArgs(args);
  if (!parsed) {
    output.stderr += "Usage: logs <errors|docker|file <path>> [--lines N]\n";
    return 1;
  }

  const command =
    parsed.kind === "docker"
      ? `docker logs --tail ${parsed.lines} ${quoteShell(config.dockerContainer)}`
      : `docker exec ${quoteShell(config.dockerContainer)} tail -n ${parsed.lines} ${quoteShell(parsed.path)}`;
  const remoteShell = `cd ${quoteShell(config.remoteProjectPath)} && ${command}`;
  const result = config.runningOnRemote
    ? await runner.run("bash", ["-lc", remoteShell])
    : await runner.run("ssh", [config.sshHost, "bash", "-lc", quoteShell(remoteShell)]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

type ParsedArgs = { kind: "docker"; lines: number } | { kind: "file"; path: string; lines: number };

function parseArgs(args: string[]): ParsedArgs | undefined {
  const [command, ...rest] = args;
  let path: string | undefined;
  let options: string[];

  if (command === "errors") {
    path = pgxErrorsPath;
    options = rest;
  } else if (command === "docker") {
    const lines = parseLines(rest);
    return lines === undefined ? undefined : { kind: "docker", lines };
  } else if (command === "file") {
    path = rest[0];
    options = rest.slice(1);
  } else {
    return undefined;
  }

  if (!path) {
    return undefined;
  }

  const lines = parseLines(options);
  return lines === undefined ? undefined : { kind: "file", path, lines };
}

function parseLines(args: string[]): number | undefined {
  if (args.length === 0) {
    return defaultLines;
  }
  if (args.length !== 2 || (args[0] !== "--lines" && args[0] !== "-n")) {
    return undefined;
  }

  const parsed = Number(args[1]);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }

  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
