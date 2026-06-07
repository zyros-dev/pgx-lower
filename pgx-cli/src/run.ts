import { existsSync, statSync } from "node:fs";
import { isAbsolute, posix, relative, resolve } from "node:path";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationRunner } from "./managed-operations.js";
import type { OperationConfig, OperationOutput } from "./operations.js";
import type { ResolvedOutputConfig, ResolvedSyncConfig } from "./project-config.js";
import type { PreviewSelection } from "./managed-runner.js";

export type RunGatewayConfig = OperationConfig & {
  localProjectPath: string;
  dockerContainer: string;
  sync: ResolvedSyncConfig;
  output: ResolvedOutputConfig;
};

export async function runGatewayCommand(
  args: string[],
  runner: ManagedOperationRunner,
  output: OperationOutput,
  config: RunGatewayConfig
): Promise<number> {
  const [command, ...rest] = args;
  if (command === "thor") {
    const parsed = parsePayloadCommand(rest);
    if (!parsed) {
      output.stderr += "Usage: run thor [--head N|--tail N|--full] -- <cmd...>\n";
      return 1;
    }
    const result = await runManagedRemoteShell({
      runner,
      output,
      config,
      commandName: `run-thor-${parsed.payload[0] ?? "command"}`,
      shellCommand: parsed.payload.map(quoteShell).join(" "),
      requireMutagenProof: true,
      fullOutput: parsed.fullOutput,
      preview: parsed.preview
    });
    return result.workflowExitCode;
  }

  if (command === "docker") {
    const parsed = parsePayloadCommand(rest);
    if (!parsed) {
      output.stderr += "Usage: run docker [--head N|--tail N|--full] -- <cmd...>\n";
      return 1;
    }
    const result = await runManagedRemoteShell({
      runner,
      output,
      config,
      commandName: `run-docker-${parsed.payload[0] ?? "command"}`,
      shellCommand: `docker exec ${quoteShell(config.dockerContainer)} ${parsed.payload.map(quoteShell).join(" ")}`,
      requireMutagenProof: true,
      fullOutput: parsed.fullOutput,
      preview: parsed.preview
    });
    return result.workflowExitCode;
  }

  if (command === "psql") {
    const { args: psqlArgs, fullOutput, preview } = parseExpansionFlags(rest);
    const psqlCommand = psqlShellCommand(psqlArgs, config, output);
    if (!psqlCommand) {
      return 1;
    }
    const result = await runManagedRemoteShell({
      runner,
      output,
      config,
      commandName: "run-psql",
      shellCommand: psqlCommand,
      requireMutagenProof: true,
      fullOutput,
      preview
    });
    return result.workflowExitCode;
  }

  output.stderr += "Usage: run <thor|docker|psql> ...\n";
  return 1;
}

function parsePayloadCommand(args: string[]): {
  payload: string[];
  fullOutput: boolean;
  preview?: PreviewSelection;
} | undefined {
  const separator = args.indexOf("--");
  if (separator === -1 || separator === args.length - 1) {
    return undefined;
  }
  const { fullOutput, preview } = parseExpansionFlags(args.slice(0, separator));
  return { payload: args.slice(separator + 1), fullOutput, preview };
}

function psqlShellCommand(args: string[], config: RunGatewayConfig, output: OperationOutput): string | undefined {
  const queryIndex = args.indexOf("--query");
  if (queryIndex !== -1) {
    const query = args[queryIndex + 1];
    if (!query) {
      output.stderr += "Usage: run psql --query <sql>\n";
      return undefined;
    }
    return dockerPsql(config, `-c ${quoteShell(query)}`);
  }

  const fileIndex = args.indexOf("--file");
  if (fileIndex !== -1) {
    const path = args[fileIndex + 1];
    if (!path) {
      output.stderr += "Usage: run psql --file <repo-local.sql>\n";
      return undefined;
    }
    const resolved = resolveRepoLocalPath(path, config.localProjectPath);
    if (!resolved.ok) {
      output.stderr += `${resolved.reason}\n`;
      return undefined;
    }
    const workspacePath = `/workspace/${resolved.relativePath}`;
    return `${readableWorkspaceFileCommand(config, workspacePath)} && ${dockerPsql(config, `-f ${quoteShell(workspacePath)}`)}`;
  }

  output.stderr += "Usage: run psql <--query <sql>|--file <repo-local.sql>>\n";
  return undefined;
}

function dockerPsql(config: RunGatewayConfig, psqlArgs: string): string {
  const command = `/usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression ${psqlArgs}`;
  return `docker exec ${quoteShell(config.dockerContainer)} su postgres -c ${quoteShell(command)}`;
}

function readableWorkspaceFileCommand(config: RunGatewayConfig, workspacePath: string): string {
  const dirs = parentDirs(workspacePath);
  const command = `chmod o+x ${dirs.map(quoteShell).join(" ")} 2>/dev/null || true; chmod o+r ${quoteShell(workspacePath)} 2>/dev/null || true`;
  return `docker exec ${quoteShell(config.dockerContainer)} bash -lc ${quoteShell(command)}`;
}

function parentDirs(path: string): string[] {
  const dirs = ["/workspace"];
  let current = posix.dirname(path);
  const stack: string[] = [];
  while (current && current !== "/" && current !== "/workspace") {
    stack.push(current);
    current = posix.dirname(current);
  }
  return [...dirs, ...stack.reverse()];
}

function resolveRepoLocalPath(path: string, root: string): { ok: true; relativePath: string } | { ok: false; reason: string } {
  const rootPath = resolve(root);
  const candidates = isAbsolute(path) ? [resolve(path)] : [resolve(process.cwd(), path), resolve(rootPath, path)];
  const resolved = candidates.find((candidate) => isWithin(rootPath, candidate));
  if (!resolved) {
    return { ok: false, reason: `run psql --file rejected ${path}: outside the configured local checkout` };
  }
  if (!existsSync(resolved)) {
    return { ok: false, reason: `run psql --file rejected ${path}: file does not exist` };
  }
  const stat = statSync(resolved);
  if (!stat.isFile() || !resolved.endsWith(".sql")) {
    return { ok: false, reason: `run psql --file rejected ${path}: expected a repo-local .sql file` };
  }
  return { ok: true, relativePath: relative(rootPath, resolved) };
}

function parseExpansionFlags(args: string[]): { args: string[]; fullOutput: boolean; preview?: PreviewSelection } {
  const filtered = [...args];
  let preview: PreviewSelection | undefined;
  const fullIndex = filtered.indexOf("--full");
  if (fullIndex !== -1) {
    filtered.splice(fullIndex, 1);
    return { args: filtered, fullOutput: true };
  }
  for (const mode of ["head", "tail"] as const) {
    const index = filtered.indexOf(`--${mode}`);
    if (index === -1) continue;
    const lines = Number(filtered[index + 1]);
    if (Number.isFinite(lines) && lines > 0) {
      preview = { mode, lines: Math.floor(lines) };
    }
    filtered.splice(index, 2);
    break;
  }
  return { args: filtered, fullOutput: false, preview };
}

function isWithin(parent: string, child: string): boolean {
  const rel = relative(parent, child);
  return rel === "" || (!!rel && !rel.startsWith("..") && !isAbsolute(rel));
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
