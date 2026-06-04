import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join } from "node:path";
import type { CommandRunner } from "./commands.js";
import {
  assertRoutes,
  parseSqlManifest,
  validateGlobalIds,
  validRoutes,
  writeRouteSummary
} from "./route-assertions.js";
import type { RouteExecutionMode, RouteExpectation } from "./route-assertions.js";

export type RouteCheckOptions = {
  runName: string;
  profile: string;
  executionMode: RouteExecutionMode;
  sqlDir: string;
  outputDir: string;
  summaryPath: string;
  defaultAutoShouldRouteTo: RouteExpectation;
  requireRouteDirectives: boolean;
  pgRegressCommand: string[] | undefined;
};

type Io = {
  stdout: string;
  stderr: string;
};

const validExecutionModes = ["stock", "extension-auto", "force-fallback", "force-lower"] as const;

export function parseRouteCheckArgs(args: readonly string[]): RouteCheckOptions {
  if (args.includes("--help") || args.includes("-h")) {
    throw new Error(routeCheckUsage());
  }

  const values = new Map<string, string>();
  let requireRouteDirectives = false;
  let pgRegressCommand: string[] | undefined;
  let i = 0;
  while (i < args.length) {
    const arg = args[i];
    if (arg === "--require-route-directives") {
      requireRouteDirectives = true;
      i++;
      continue;
    }
    if (arg === "--pg-regress") {
      if (args[i + 1] !== "--") {
        throw new Error("Usage: --pg-regress -- <pg_regress-command> <args...>");
      }
      pgRegressCommand = [...args.slice(i + 2)];
      if (pgRegressCommand.length === 0) {
        throw new Error("Usage: --pg-regress -- <pg_regress-command> <args...>");
      }
      break;
    }
    if (!arg?.startsWith("--")) {
      throw new Error(routeCheckUsage());
    }
    const value = args[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${arg}`);
    }
    values.set(arg, value);
    i += 2;
  }

  const executionMode = requiredChoice(values, "--execution-mode", validExecutionModes);
  const defaultAutoShouldRouteTo = requiredChoice(values, "--default-auto-should-route-to", validRoutes);

  return {
    runName: required(values, "--run-name"),
    profile: required(values, "--profile"),
    executionMode,
    sqlDir: required(values, "--sql-dir"),
    outputDir: required(values, "--output-dir"),
    summaryPath: required(values, "--summary"),
    defaultAutoShouldRouteTo,
    requireRouteDirectives,
    pgRegressCommand
  };
}

export async function runRouteCheckCommand(
  args: readonly string[],
  runner: CommandRunner,
  io: Io
): Promise<number> {
  let options: RouteCheckOptions;
  try {
    options = parseRouteCheckArgs(args);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message.startsWith("Usage:")) {
      io.stdout += `${message}\n`;
      return 0;
    }
    io.stderr += `${message}\n`;
    return 1;
  }

  const manifests = readSqlManifests(options);
  validateGlobalIds(manifests);

  let pgRegressExitCode = 0;
  if (options.pgRegressCommand) {
    const [command, ...commandArgs] = options.pgRegressCommand;
    if (!command) {
      throw new Error("pg_regress command is empty");
    }
    const result = await runner.run(command, commandArgs);
    pgRegressExitCode = result.exitCode;
    io.stdout += result.stdout;
    io.stderr += result.stderr;
  }

  const outputsByPath = readOutputs(manifests, options.outputDir);
  const report = assertRoutes({
    runName: options.runName,
    profile: options.profile,
    executionMode: options.executionMode,
    manifests,
    outputsByPath
  });

  mkdirSync(dirname(options.summaryPath), { recursive: true });
  writeFileSync(options.summaryPath, writeRouteSummary(report));

  if (report.failures.length > 0) {
    io.stderr += `FAIL: route assertions failed. Summary: ${options.summaryPath}\n`;
    return 1;
  }

  if (pgRegressExitCode !== 0) {
    io.stderr += `pg_regress exited ${pgRegressExitCode}. Summary: ${options.summaryPath}\n`;
    return pgRegressExitCode;
  }

  io.stdout += `OK: route assertions passed. Summary: ${options.summaryPath}\n`;
  return 0;
}

function readSqlManifests(options: RouteCheckOptions) {
  return readdirSync(options.sqlDir)
    .filter((file) => file.endsWith(".sql"))
    .sort()
    .map((file) => {
      const path = join(options.sqlDir, file);
      return parseSqlManifest({
        path,
        sql: readFileSync(path, "utf8"),
        defaultRoute: options.defaultAutoShouldRouteTo,
        requireRouteDirectives: options.requireRouteDirectives
      });
    });
}

function readOutputs(manifests: ReturnType<typeof readSqlManifests>, outputDir: string): Map<string, string> {
  const outputs = new Map<string, string>();
  for (const manifest of manifests) {
    const stem = basename(manifest.path, extname(manifest.path));
    outputs.set(manifest.path, readFileSync(join(outputDir, `${stem}.out`), "utf8"));
  }
  return outputs;
}

function required(values: ReadonlyMap<string, string>, key: string): string {
  const value = values.get(key);
  if (!value) {
    throw new Error(routeCheckUsage());
  }
  return value;
}

function requiredChoice<const T extends readonly string[]>(
  values: ReadonlyMap<string, string>,
  key: string,
  choices: T
): T[number] {
  const value = required(values, key);
  if (!choices.includes(value)) {
    throw new Error(`${key} must be one of: ${choices.join(", ")}`);
  }
  return value;
}

function routeCheckUsage(): string {
  return [
    "Usage: pgx-cli test route-check",
    "  --run-name <name>",
    "  --profile <profile-name>",
    "  --execution-mode <stock|extension-auto|force-fallback|force-lower>",
    "  --sql-dir <dir>",
    "  --output-dir <dir>",
    "  --summary <path>",
    "  --default-auto-should-route-to <lower|fallback|ignore|not_asserted>",
    "  [--require-route-directives]",
    "  [--pg-regress -- <pg_regress-command> <arg>]"
  ].join("\n");
}
