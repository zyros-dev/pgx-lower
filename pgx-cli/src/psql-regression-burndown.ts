import { copyFileSync, createWriteStream, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { finished } from "node:stream/promises";
import type { StreamingCommandRunner } from "./commands.js";
import type { RunResult } from "./commands.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";
import { runRouteCheckCommand } from "./pg-regress-routes.js";

export type PsqlRegressionStatus = {
  passing: Set<string>;
  failing: Set<string>;
};

export type PsqlRegressionDelta = {
  passing: Set<string>;
  failing: Set<string>;
  baseline: Set<string>;
  stillFailing: Set<string>;
  newFailures: Set<string>;
  nowPassing: Set<string>;
};

export type PsqlRegressionBurndownOptions = {
  source: string;
  baseline: string;
  outputDir: string;
  summary: string;
  routeSummary: string;
  pgRegress: string;
  bindir: string;
  dlpath: string;
  schedule: string;
  loadExtension: string;
  record: boolean;
};

type Io = {
  stdout: string;
  stderr: string;
};

type InternalEnvironmentOptions = {
  allowUnsafeLocalForTests?: boolean;
};

const defaultPgRegressPath = "/usr/local/pgsql/lib/pgxs/src/test/regress/pg_regress";
const managedRunnerFlag = "--from-managed-runner";
const statusLineRe = /^\s*test\s+([^\s]+)\s+\.\.\.\s+(ok|FAILED)\b/;
const tapStatusLineRe = /^(?:\d+:\s+)?(not\s+ok|ok)\s+\d+\s*[-+]\s*([^\s]+)(?:\s|$)/;

export function parsePgRegressStatusLines(text: string): PsqlRegressionStatus {
  const passing = new Set<string>();
  const failing = new Set<string>();

  for (const line of text.split("\n")) {
    const match = statusLineRe.exec(line);
    if (!match) {
      continue;
    }

    const name = match[1];
    const status = match[2];
    if (!status || !name) {
      continue;
    }

    if (status === "ok") {
      passing.add(name);
    } else {
      failing.add(name);
    }
  }

  if (passing.size > 0 || failing.size > 0) {
    return { passing, failing };
  }

  for (const line of text.split("\n")) {
    const match = tapStatusLineRe.exec(line.trim());
    if (!match) {
      continue;
    }

    const status = match[1];
    const name = match[2];
    if (!status || !name) {
      continue;
    }

    if (status === "ok") {
      passing.add(name);
    } else {
      failing.add(name);
    }
  }

  return { passing, failing };
}

export function classifyPsqlRegressionDelta(options: {
  passing: Set<string>;
  failing: Set<string>;
  baseline: Set<string>;
}): PsqlRegressionDelta {
  return {
    passing: options.passing,
    failing: options.failing,
    baseline: options.baseline,
    stillFailing: intersection(options.failing, options.baseline),
    newFailures: difference(options.failing, options.baseline),
    nowPassing: intersection(options.baseline, options.passing)
  };
}

export function psqlRegressionDeltaExitCode(delta: PsqlRegressionDelta): number {
  return delta.newFailures.size > 0 || delta.nowPassing.size > 0 ? 1 : 0;
}

export function readPsqlRegressionBaseline(text: string): Set<string> {
  return new Set(
    text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#"))
  );
}

export function renderPsqlRegressionBaseline(failures: ReadonlySet<string>): string {
  return `${[...failures].sort().join("\n")}\n`;
}

export function missingPsqlRegressionSourceMessage(source: string): string {
  return [
    `Missing PostgreSQL regression source under ${source}.`,
    "Expected PostgreSQL 17.6 src/test/regress files: sql/, expected/, data/, parallel_schedule, and resultmap.",
    "",
    "Bootstrap PostgreSQL 17.6 regression sources:",
    "mkdir -p build-artifacts/psql-regression",
    "curl -L https://ftp.postgresql.org/pub/source/v17.6/postgresql-17.6.tar.bz2 \\",
    "  -o build-artifacts/psql-regression/postgresql-17.6.tar.bz2",
    "tar -xjf build-artifacts/psql-regression/postgresql-17.6.tar.bz2 \\",
    "  -C build-artifacts/psql-regression",
    "",
    "Then copy from build-artifacts/psql-regression/postgresql-17.6/src/test/regress."
  ].join("\n");
}

export function validatePsqlRegressionSource(source: string, exists: (path: string) => boolean): void {
  const requiredPaths = [
    join(source, "sql"),
    join(source, "expected"),
    join(source, "data"),
    join(source, "resultmap"),
    join(source, "parallel_schedule")
  ];
  const missing = requiredPaths.filter((path) => !exists(path));
  if (missing.length > 0) {
    throw new Error(`${missingPsqlRegressionSourceMessage(source)}\n\nMissing path: ${missing[0]}`);
  }
}

export function renderPsqlRegressionSummary(options: {
  runName: string;
  source: string;
  delta: PsqlRegressionDelta;
  routeSummaryPath: string;
}): string {
  const { delta } = options;
  return [
    `# PostgreSQL Regression Burndown: ${options.runName}`,
    "",
    `Source: \`${options.source}\``,
    `Route summary: \`${options.routeSummaryPath}\``,
    "",
    `Passing: ${delta.passing.size}`,
    `Failing: ${delta.failing.size}`,
    `Baseline known-failing: ${delta.baseline.size}`,
    `Still failing: ${delta.stillFailing.size}`,
    `Newly failing: ${delta.newFailures.size}`,
    `Newly passing: ${delta.nowPassing.size}`,
    "",
    "## Newly Failing",
    renderNameList(delta.newFailures),
    "",
    "## Newly Passing",
    renderNameList(delta.nowPassing),
    "",
    "## Still Failing",
    renderNameList(delta.stillFailing),
    ""
  ].join("\n");
}

export function parsePsqlRegressionBurndownArgs(args: readonly string[]): PsqlRegressionBurndownOptions {
  if (args.includes("--help") || args.includes("-h")) {
    throw new Error(psqlRegressionBurndownUsage());
  }

  const values = new Map<string, string>();
  const valueOptions = new Set([
    "--source",
    "--baseline",
    "--output-dir",
    "--summary",
    "--route-summary",
    "--pg-regress",
    "--bindir",
    "--dlpath",
    "--schedule",
    "--load-extension"
  ]);
  let record = false;
  let i = 0;
  while (i < args.length) {
    const arg = args[i];
    if (arg === "--record") {
      record = true;
      i++;
      continue;
    }
    if (!arg?.startsWith("--")) {
      throw new Error(psqlRegressionBurndownUsage());
    }
    if (!valueOptions.has(arg)) {
      throw new Error(`Unknown option ${arg}`);
    }
    const value = args[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${arg}`);
    }
    values.set(arg, value);
    i += 2;
  }

  return {
    source: values.get("--source") ?? "tests/psql-regression",
    baseline: values.get("--baseline") ?? "tests/psql-regression/baselines/current.txt",
    outputDir: values.get("--output-dir") ?? "tests/psql-regression/results",
    summary: values.get("--summary") ?? "build-artifacts/test-runs/psql-regression-burndown/summary.md",
    routeSummary: values.get("--route-summary")
      ?? "build-artifacts/test-runs/psql-regression-burndown/route-summary.md",
    pgRegress: values.get("--pg-regress") ?? defaultPgRegressPath,
    bindir: values.get("--bindir") ?? "/usr/local/pgsql/bin",
    dlpath: values.get("--dlpath") ?? "/usr/local/pgsql/lib",
    schedule: values.get("--schedule") ?? "parallel_schedule",
    loadExtension: values.get("--load-extension") ?? "pgx_lower",
    record
  };
}

export function buildPsqlRegressionPgRegressCommand(options: PsqlRegressionBurndownOptions): string[] {
  const schedule = isAbsolute(options.schedule) ? options.schedule : join(options.source, options.schedule);
  return [
    options.pgRegress,
    `--bindir=${options.bindir}`,
    `--dlpath=${options.dlpath}`,
    `--inputdir=${options.source}`,
    `--outputdir=${options.outputDir}`,
    `--schedule=${schedule}`,
    `--load-extension=${options.loadExtension}`
  ];
}

export async function runPsqlRegressionBurndownCliCommand(
  args: readonly string[],
  runner: StreamingCommandRunner,
  io: Io,
  config: ManagedOperationConfig,
  environment: InternalEnvironmentOptions = {}
): Promise<number> {
  const { normalizedArgs, fromManagedRunner } = stripManagedRunnerFlag(args);
  if (normalizedArgs.includes("--help") || normalizedArgs.includes("-h")) {
    io.stdout += `${psqlRegressionBurndownUsage()}\n`;
    return 0;
  }

  let options: PsqlRegressionBurndownOptions;
  try {
    options = parsePsqlRegressionBurndownArgs(normalizedArgs);
  } catch (error) {
    io.stderr += `${error instanceof Error ? error.message : String(error)}\n`;
    return 1;
  }

  if (fromManagedRunner) {
    if (!environment.allowUnsafeLocalForTests && process.env.PGX_PSQL_REGRESSION_INTERNAL !== "1") {
      io.stderr += "psql-regression-burndown requires PGX_PSQL_REGRESSION_INTERNAL=1 when --from-managed-runner is used\n";
      return 1;
    }
    return runPsqlRegressionBurndownCommand(normalizedArgs, runner, io);
  }

  if (shouldRunPsqlRegressionInPlace(options.source)) {
    return runPsqlRegressionBurndownCommand(normalizedArgs, runner, io);
  }

  try {
    validatePsqlRegressionSource(options.source, existsSync);
  } catch (error) {
    io.stderr += `${error instanceof Error ? error.message : String(error)}\n`;
    return 1;
  }

  const shellCommand = buildManagedPsqlRegressionShellCommand(options, config);
  const managedIo = {
    stdout: io.stdout,
    stderr: io.stderr,
    liveStdout: () => undefined,
    liveStderr: () => undefined
  };
  const result = await runManagedRemoteShell({
    runner,
    output: managedIo,
    config,
    commandName: "test-psql-regression-burndown",
    shellCommand,
    requireMutagenProof: true,
    artifactPaths: [
      workspacePathForManagedRun(options.outputDir, config.localProjectPath),
      workspacePathForManagedRun(options.summary, config.localProjectPath),
      workspacePathForManagedRun(options.routeSummary, config.localProjectPath)
    ]
  });
  io.stdout = managedIo.stdout;
  io.stderr = managedIo.stderr;
  return result.workflowExitCode;
}

export async function runPsqlRegressionBurndownCommand(
  args: readonly string[],
  runner: StreamingCommandRunner,
  io: Io
): Promise<number> {
  let options: PsqlRegressionBurndownOptions;
  try {
    options = parsePsqlRegressionBurndownArgs(args);
    validatePsqlRegressionSource(options.source, existsSync);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message.startsWith("Usage:")) {
      io.stdout += `${message}\n`;
      return 0;
    }
    io.stderr += `${message}\n`;
    return 1;
  }

  mkdirSync(options.outputDir, { recursive: true });
  mkdirSync(dirname(options.summary), { recursive: true });
  mkdirSync(dirname(options.routeSummary), { recursive: true });

  const pgRegressCommand = buildPsqlRegressionPgRegressCommand(options);
  const [command, ...commandArgs] = pgRegressCommand;
  if (!command) {
    io.stderr += "ERROR: pg_regress command is empty\n";
    return 1;
  }

  const pgRegressLog = join(options.outputDir, "pg_regress.log");
  await runStreamingToLog(runner, command, commandArgs, pgRegressLog, options.outputDir);
  const pgRegressTranscript = readFileSync(pgRegressLog, "utf8");

  const parsed = parsePgRegressStatusLines(pgRegressTranscript);
  if (parsed.passing.size === 0 && parsed.failing.size === 0) {
    io.stderr += `ERROR: no pg_regress status lines found. Inspect ${pgRegressLog}\n`;
    return 2;
  }

  const routeCheckLog = join(options.outputDir, "route-check.log");
  const routeCheckOutputDir = pgRegressResultsDir(options);
  const routeCheckSqlDir = prepareRouteCheckSqlDir(options, routeCheckOutputDir);
  const routeCheckIo = { stdout: "", stderr: "" };
  let routeCheckExitCode = 0;
  try {
    routeCheckExitCode = await runRouteCheckCommand(
      [
        "--run-name",
        "psql-regression-burndown",
        "--profile",
        "debug",
        "--execution-mode",
        "extension-auto",
        "--sql-dir",
        routeCheckSqlDir,
        "--output-dir",
        routeCheckOutputDir,
        "--summary",
        options.routeSummary,
        "--default-auto-should-route-to",
        "not_asserted"
      ],
      runner,
      routeCheckIo
    );
  } catch (error) {
    routeCheckExitCode = 1;
    routeCheckIo.stderr += `${error instanceof Error ? error.message : String(error)}\n`;
  }
  writeFileSync(routeCheckLog, `${routeCheckIo.stdout}${routeCheckIo.stderr}`);
  if (routeCheckExitCode !== 0) {
    io.stderr += `INFO: route observation failed; inspect ${options.routeSummary} and ${routeCheckLog}\n`;
  }

  if (options.record) {
    writeFileSync(options.baseline, renderPsqlRegressionBaseline(parsed.failing));
    io.stdout += `Recorded ${parsed.failing.size} failing upstream PostgreSQL tests to ${options.baseline}\n`;
    return 0;
  }

  const baseline = readPsqlRegressionBaseline(readFileSync(options.baseline, "utf8"));
  const delta = classifyPsqlRegressionDelta({
    passing: parsed.passing,
    failing: parsed.failing,
    baseline
  });
  writeFileSync(options.summary, renderPsqlRegressionSummary({
    runName: "psql-regression-burndown",
    source: options.source,
    delta,
    routeSummaryPath: options.routeSummary
  }));

  const exitCode = psqlRegressionDeltaExitCode(delta);
  if (exitCode === 0) {
    io.stdout += `OK: PostgreSQL regression delta matches baseline. Summary: ${options.summary}\n`;
  } else {
    io.stderr += `FAIL: PostgreSQL regression delta changed. Summary: ${options.summary}\n`;
  }
  return exitCode;
}

async function runStreamingToLog(
  runner: StreamingCommandRunner,
  command: string,
  args: string[],
  logPath: string,
  outputDir: string
): Promise<number> {
  const log = createWriteStream(logPath, { flags: "w" });
  const preflight = await ensurePgRegressOutputDirWritable(runner, outputDir);
  if (preflight && preflight.exitCode !== 0) {
    log.write(preflight.stdout);
    log.write(preflight.stderr);
    log.end();
    await finished(log);
    return preflight.exitCode;
  }
  const execution = buildPsqlRegressionExecutionCommand(command, args);
  const result = await runner.runStreaming(execution.command, execution.args, { stdout: log, stderr: log });
  log.end();
  await finished(log);
  return result.childExitCode;
}

function renderNameList(names: ReadonlySet<string>): string {
  if (names.size === 0) {
    return "- none";
  }
  return [...names].sort().map((name) => `- ${name}`).join("\n");
}

function difference(left: ReadonlySet<string>, right: ReadonlySet<string>): Set<string> {
  return new Set([...left].filter((value) => !right.has(value)));
}

function intersection(left: ReadonlySet<string>, right: ReadonlySet<string>): Set<string> {
  return new Set([...left].filter((value) => right.has(value)));
}

function pgRegressResultsDir(options: PsqlRegressionBurndownOptions): string {
  const nestedResultsDir = join(options.outputDir, "results");
  return existsSync(nestedResultsDir) ? nestedResultsDir : options.outputDir;
}

function prepareRouteCheckSqlDir(options: PsqlRegressionBurndownOptions, outputDir: string): string {
  const sourceSqlDir = join(options.source, "sql");
  const routeCheckSqlDir = join(options.outputDir, "route-check-sql");
  rmSync(routeCheckSqlDir, { recursive: true, force: true });
  mkdirSync(routeCheckSqlDir, { recursive: true });

  for (const file of readdirSync(sourceSqlDir).filter((item) => item.endsWith(".sql")).sort()) {
    const stem = basename(file, extname(file));
    if (!existsSync(join(outputDir, `${stem}.out`))) {
      continue;
    }
    copyFileSync(join(sourceSqlDir, file), join(routeCheckSqlDir, file));
  }

  return routeCheckSqlDir;
}

function psqlRegressionBurndownUsage(): string {
  return [
    "Usage: pgx-cli test psql-regression-burndown",
    "  [--source <tests/psql-regression>]",
    "  [--baseline <baseline-file>]",
    "  [--output-dir <results-dir>]",
    "  [--summary <summary.md>]",
    "  [--route-summary <route-summary.md>]",
    `  [--pg-regress <pg_regress>] default: ${defaultPgRegressPath}`,
    "  [--bindir <postgres-bindir>]",
    "  [--dlpath <postgres-libdir>]",
    "  [--schedule <schedule-file>]",
    "  [--load-extension <extension-name>]",
    "  [--record]"
  ].join("\n");
}

export function buildPsqlRegressionExecutionCommand(
  command: string,
  args: readonly string[],
  getuid: (() => number | undefined) | undefined = defaultGetuid()
): { command: string; args: string[] } {
  const uid = getuid?.();
  if (uid === 0) {
    return {
      command: "su",
      args: ["postgres", "-c", [command, ...args].map(quoteShell).join(" ")]
    };
  }
  return { command, args: [...args] };
}

export function buildPsqlRegressionOutputDirPreflightCommand(
  outputDir: string,
  getuid: (() => number | undefined) | undefined = defaultGetuid()
): { command: string; args: string[] } | undefined {
  const uid = getuid?.();
  if (uid !== 0) {
    return undefined;
  }
  return {
    command: "sh",
    args: [
      "-c",
      `mkdir -p ${quoteShell(outputDir)} && chown -R postgres:postgres ${quoteShell(outputDir)} && chmod 0775 ${quoteShell(outputDir)}`
    ]
  };
}

function defaultGetuid(): (() => number | undefined) | undefined {
  const getuid = process.getuid;
  if (typeof getuid !== "function") {
    return undefined;
  }
  return () => getuid.call(process);
}

async function ensurePgRegressOutputDirWritable(
  runner: StreamingCommandRunner,
  outputDir: string
): Promise<RunResult | undefined> {
  const preflight = buildPsqlRegressionOutputDirPreflightCommand(outputDir);
  if (!preflight) {
    return undefined;
  }
  return runner.run(preflight.command, preflight.args);
}

function stripManagedRunnerFlag(args: readonly string[]): { normalizedArgs: string[]; fromManagedRunner: boolean } {
  const normalizedArgs: string[] = [];
  let fromManagedRunner = false;
  for (const arg of args) {
    if (arg === managedRunnerFlag) {
      fromManagedRunner = true;
      continue;
    }
    normalizedArgs.push(arg);
  }
  return { normalizedArgs, fromManagedRunner };
}

function shouldRunPsqlRegressionInPlace(source: string): boolean {
  return existsSync("/.dockerenv") && isPathWithin("/workspace", source);
}

function buildManagedPsqlRegressionShellCommand(
  options: PsqlRegressionBurndownOptions,
  config: ManagedOperationConfig
): string {
  const internalArgs = [
    "/workspace/pgx-cli/dist/index.js",
    "test",
    "psql-regression-burndown",
    managedRunnerFlag,
    "--source",
    workspacePathForManagedRun(options.source, config.localProjectPath),
    "--baseline",
    workspacePathForManagedRun(options.baseline, config.localProjectPath),
    "--output-dir",
    workspacePathForManagedRun(options.outputDir, config.localProjectPath),
    "--summary",
    workspacePathForManagedRun(options.summary, config.localProjectPath),
    "--route-summary",
    workspacePathForManagedRun(options.routeSummary, config.localProjectPath),
    "--pg-regress",
    options.pgRegress,
    "--bindir",
    options.bindir,
    "--dlpath",
    options.dlpath,
    "--schedule",
    managedSchedulePath(options.schedule, config.localProjectPath),
    "--load-extension",
    options.loadExtension,
    ...(options.record ? ["--record"] : [])
  ];
  const dockerCommand = `docker exec ${quoteShell(config.dockerContainer ?? "pgx-lower-dev")} bash -lc ${quoteShell(
    [
      buildAndInstallExtensionCommand("debug", "psql-regression-burndown"),
      "chmod -R o+rX /workspace",
      `PGX_PSQL_REGRESSION_INTERNAL=1 ${internalArgs.map(quoteShell).join(" ")}`
    ].join(" && ")
  )}`;
  return [
    "npm --prefix pgx-cli install",
    "npm --prefix pgx-cli run build",
    dockerCommand
  ].join(" && ");
}

function managedSchedulePath(schedule: string, localRoot: string): string {
  return isAbsolute(schedule) ? workspacePathForManagedRun(schedule, localRoot) : schedule;
}

function workspacePathForManagedRun(path: string, localRoot: string): string {
  if (!isAbsolute(path)) {
    return `/workspace/${path.replaceAll(sep, "/")}`;
  }
  const rel = relative(resolve(localRoot), resolve(path));
  if (rel.startsWith("..") || isAbsolute(rel)) {
    throw new Error(`${path} must be inside ${localRoot} for managed execution`);
  }
  return `/workspace/${rel.replaceAll(sep, "/")}`;
}

function buildAndInstallExtensionCommand(profile: string, runName: string): string {
  const buildType = profile === "release" ? "Release" : "Debug";
  const buildDir = `/workspace/build-artifacts/compare-postgres/${safeName(runName)}/${profile}`;
  return [
    cleanWorkspaceCmakeArtifactsCommand(),
    `mkdir -p ${buildDir}`,
    `cd ${buildDir}`,
    `([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=${buildType} -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache /workspace)`,
    "cmake --build .",
    "cmake --install ."
  ].join(" && ");
}

function cleanWorkspaceCmakeArtifactsCommand(): string {
  return [
    "rm -rf /workspace/CMakeFiles /workspace/include/runtime-defs",
    "rm -f /workspace/CMakeCache.txt /workspace/build.ninja /workspace/.ninja_deps /workspace/.ninja_log /workspace/tablegen_compile_commands.yml /workspace/CTestTestfile.cmake /workspace/cmake_install.cmake",
    "find /workspace/src/lingodb/mlir \\( -name CMakeFiles -o -name CTestTestfile.cmake -o -name cmake_install.cmake -o -name '*.inc' -o -name '*.inc.d' -o -name '*.o' -o -name '*.a' \\) -exec rm -rf {} +"
  ].join(" && ");
}

function safeName(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "run";
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}

function isPathWithin(parent: string, child: string): boolean {
  const rel = relative(resolve(parent), resolve(child));
  return rel === "" || (!!rel && !rel.startsWith("..") && !isAbsolute(rel));
}
