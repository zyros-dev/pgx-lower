import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash, randomBytes } from "node:crypto";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import type { StreamingCommandRunner } from "./commands.js";
import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig } from "./managed-operations.js";
import { satisfyGateFailureState } from "./gate-memory.js";
import {
  compareResultBlocks,
  parseCompareDiagnostics,
  parseResultBlocks
} from "./compare-postgres-results.js";
import type { ComparePostgresFailure, ResultBlock } from "./compare-postgres-results.js";
import {
  renderComparePostgresScripts,
  setupSqlConfiguresPreloadLibraries,
  setupSqlReferencesPgxLower,
  setupSqlUsesDoBlock,
  setupSqlUsesLoad,
  setupSqlUsesPsqlMetaCommand
} from "./compare-postgres-render.js";
import type { RenderedComparePostgresFile, RenderedComparePostgresQuery } from "./compare-postgres-render.js";
import { parseSqlManifest, readComparePostgresManifests, resolveComparePostgresTarget } from "./sql-manifest.js";

export type ComparePostgresOptions = {
  workload: string;
  root: string;
  outputDir?: string;
  runName?: string;
  summaryPath?: string;
  jsonPath?: string;
  fromManagedRunner: boolean;
};

type InternalEnvironmentOptions = {
  allowUnsafeLocalForTests?: boolean;
};

type Io = {
  stdout: string;
  stderr: string;
};

type ScriptRun = {
  stdoutPath: string;
  stderrPath: string;
  blocks: ResultBlock[];
  exitCode: number;
  diagnostics: ReturnType<typeof parseCompareDiagnostics>;
  parseError?: string;
};

type SetupRun = {
  stdoutPath: string;
  stderrPath: string;
  exitCode: number;
  diagnostics: ReturnType<typeof parseCompareDiagnostics>;
};

type RenderedRunFile = {
  file: RenderedComparePostgresFile;
  stockScript: string;
  extensionScript: string;
};

type ScriptDiagnostics = {
  sourceFile: string;
  variant: "stock" | "extension";
  routeNotices: string[];
  pgxNotices: string[];
  warnings: string[];
  errors: string[];
};

type ComparePostgresQueryComparison = {
  workload: string;
  profile: string;
  sourceFile: string;
  queryId: string;
  statementIndex: number;
  comparisonMode: RenderedComparePostgresQuery["comparisonMode"];
  routeExpectation: RenderedComparePostgresQuery["routeExpectation"];
  stockRowCount: number;
  extensionRowCount: number;
  mismatchKind: string | null;
  preview: string | null;
};

export function parseComparePostgresArgs(args: readonly string[]): ComparePostgresOptions {
  if (args.includes("--help") || args.includes("-h")) {
    throw new Error(comparePostgresUsage());
  }

  const values = new Map<string, string>();
  let fromManagedRunner = false;
  let i = 0;
  const valueOptions = new Set(["--workload", "--root", "--output-dir", "--run-name", "--summary", "--json"]);
  while (i < args.length) {
    const arg = args[i];
    if (arg === "--from-managed-runner") {
      fromManagedRunner = true;
      i++;
      continue;
    }
    if (!arg?.startsWith("--")) {
      throw new Error(comparePostgresUsage());
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

  const workload = values.get("--workload");
  if (!workload) {
    throw new Error(comparePostgresUsage());
  }

  return {
    workload,
    root: values.get("--root") ?? findNearestRepoRoot(process.cwd()),
    outputDir: values.get("--output-dir"),
    runName: values.get("--run-name"),
    summaryPath: values.get("--summary"),
    jsonPath: values.get("--json"),
    fromManagedRunner
  };
}

export async function runComparePostgresCliCommand(
  args: readonly string[],
  runner: StreamingCommandRunner,
  io: Io,
  config: ManagedOperationConfig
): Promise<number> {
  if (args.includes("--help") || args.includes("-h")) {
    io.stdout += `${comparePostgresUsage()}\n`;
    return 0;
  }

  let options: ComparePostgresOptions;
  try {
    options = parseComparePostgresArgs(args);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr += `${message}\n`;
    return 1;
  }

  options = normalizeRunName(options);
  let targetProfile: string;
  try {
    targetProfile = resolveComparePostgresTarget({ root: options.root, workload: options.workload }).profile;
  } catch (error) {
    io.stderr += `${errorMessage(error)}\n`;
    return 1;
  }
  const shellCommand = buildManagedComparePostgresShellCommand(options, config, targetProfile);
  const result = await runManagedRemoteShell({
    runner,
    output: io,
    config,
    commandName: `test-compare-postgres-${options.workload}`,
    shellCommand,
    requireMutagenProof: true,
    artifactPaths: [managedOutputDir(options)]
  });
  await clearMatchingGateMemory(["test", "compare-postgres", ...args], result.workflowExitCode, runner, io, config);
  return result.workflowExitCode;
}

async function clearMatchingGateMemory(
  argv: string[],
  exitCode: number,
  runner: StreamingCommandRunner,
  output: Io,
  config: ManagedOperationConfig
): Promise<void> {
  if (exitCode !== 0) return;
  const currentHead = await readCurrentHead(runner, config);
  const previousFailure = satisfyGateFailureState({ root: config.localProjectPath, currentHead, argv });
  if (!previousFailure) return;
  output.stdout += `gate memory: focused reproducer passed; cleared review gate block for ${previousFailure.stepName}\n`;
}

async function readCurrentHead(runner: StreamingCommandRunner, config: ManagedOperationConfig): Promise<string> {
  const result = await runner.run("git", ["-C", config.localProjectPath, "rev-parse", "HEAD"]);
  return result.exitCode === 0 && result.stdout.trim() ? result.stdout.trim() : "unknown";
}

export async function runComparePostgresInternalCommand(
  args: readonly string[],
  runner: StreamingCommandRunner,
  io: Io,
  environment: InternalEnvironmentOptions = {}
): Promise<number> {
  if (args.includes("--help") || args.includes("-h")) {
    io.stdout += `${comparePostgresUsage()}\n`;
    return 0;
  }

  let options: ComparePostgresOptions;
  try {
    options = parseComparePostgresArgs(args);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr += `${message}\n`;
    return 1;
  }

  if (!options.fromManagedRunner) {
    io.stderr += "compare-postgres-internal requires --from-managed-runner\n";
    return 1;
  }
  if (process.env.PGX_COMPARE_POSTGRES_INTERNAL !== "1") {
    io.stderr += "compare-postgres-internal requires PGX_COMPARE_POSTGRES_INTERNAL=1\n";
    return 1;
  }
  if (!environment.allowUnsafeLocalForTests && !isContainerWorkspace(options.root)) {
    io.stderr += "compare-postgres-internal requires the /workspace container environment\n";
    return 1;
  }

  try {
    return await runComparePostgresInternal(normalizeRunName(options), runner, io);
  } catch (error) {
    io.stderr += `${errorMessage(error)}\n`;
    return 1;
  }
}

async function runComparePostgresInternal(
  options: ComparePostgresOptions,
  runner: StreamingCommandRunner,
  io: Io
): Promise<number> {
  const target = resolveComparePostgresTarget({ root: options.root, workload: options.workload });
  validateSetupFileDoesNotLoadPgxLower(target);
  const manifests = readComparePostgresManifests(target);
  if (!target.scratchDatabase) {
    validateNoInlineSetupForSharedDatabase(manifests);
  }
  const rendered = renderComparePostgresScripts({
    target,
    manifests
  });
  const outputDir = options.outputDir ?? join(options.root, defaultOutputDir(options));
  const summaryPath = options.summaryPath ?? join(outputDir, "compare-postgres-summary.md");
  const jsonPath = options.jsonPath ?? join(outputDir, "compare-postgres-diff.json");
  const scriptsDir = join(outputDir, "scripts");
  const outputsDir = join(outputDir, "outputs");
  mkdirSync(scriptsDir, { recursive: true });
  mkdirSync(outputsDir, { recursive: true });

  const stockDb = target.scratchDatabase ? scratchDbName(options, "stock") : "postgres";
  const extensionDb = target.scratchDatabase ? scratchDbName(options, "extension") : "postgres";
  const failures: ComparePostgresFailure[] = [];
  const allQueries: RenderedComparePostgresQuery[] = [];
  const comparisons: ComparePostgresQueryComparison[] = [];
  const artifactPaths: string[] = [];
  const diagnostics: ScriptDiagnostics[] = [];
  const runFiles: RenderedRunFile[] = rendered.files.map((file) => {
    const stockScript = writeScript(scriptsDir, "stock", file);
    const extensionScript = writeScript(scriptsDir, "extension", file);
    artifactPaths.push(stockScript, extensionScript);
    return { file, stockScript, extensionScript };
  });

  if (rendered.comparableCount === 0) {
    failures.push(noComparableQueriesFailure(target.name, target.profile));
  }

  if (
    target.expectedComparableCount !== undefined &&
    rendered.comparableCount !== target.expectedComparableCount
  ) {
    failures.push(expectedComparableCountFailure(target.name, target.profile, target.expectedComparableCount, rendered.comparableCount));
  }

  if (target.scratchDatabase) {
    const stockReset = await resetDatabase(runner, stockDb, target.name, target.profile, "stock");
    if (stockReset) {
      failures.push(stockReset);
    }
    const extensionReset = await resetDatabase(runner, extensionDb, target.name, target.profile, "extension");
    if (extensionReset) {
      failures.push(extensionReset);
    }
  }

  if (failures.length === 0 && rendered.setupFile) {
    const stockSetup = await runAndCaptureSetup(runner, stockDb, rendered.setupFile, outputsDir, "stock");
    const extensionSetup = await runAndCaptureSetup(runner, extensionDb, rendered.setupFile, outputsDir, "extension");
    artifactPaths.push(stockSetup.stdoutPath, stockSetup.stderrPath, extensionSetup.stdoutPath, extensionSetup.stderrPath);
    diagnostics.push(setupDiagnostics(rendered.setupFile, "stock", stockSetup), setupDiagnostics(rendered.setupFile, "extension", extensionSetup));
    failures.push(...stockSetupPgxLowerDiagnosticFailures(target.name, target.profile, rendered.setupFile, stockSetup));
    if (stockSetup.exitCode !== 0) {
      failures.push(setupFailure(target.name, target.profile, rendered.setupFile, "setup_run_failed", stockSetup.exitCode));
    }
    if (extensionSetup.exitCode !== 0) {
      failures.push(setupFailure(target.name, target.profile, rendered.setupFile, "setup_run_failed", extensionSetup.exitCode));
    }
  }

  for (const { file, stockScript, extensionScript } of failures.length === 0 ? runFiles : []) {
    allQueries.push(...file.queries);
    const stock = await runAndCaptureScript(runner, stockDb, stockScript, outputsDir, "stock", file);
    const extension = await runAndCaptureScript(runner, extensionDb, extensionScript, outputsDir, "extension", file);
    artifactPaths.push(stock.stdoutPath, stock.stderrPath, extension.stdoutPath, extension.stderrPath);
    diagnostics.push(scriptDiagnostics(file, "stock", stock), scriptDiagnostics(file, "extension", extension));
    failures.push(...stockPgxLowerDiagnosticFailures(target.name, target.profile, file, stock));
    failures.push(...extensionForcedFallbackDiagnosticFailures(target.name, target.profile, file, extension));
    const fileComparison = compareFileBlocks(target.name, target.profile, file, stock, extension);
    comparisons.push(...fileComparison.comparisons);
    failures.push(...fileComparison.failures);
    if (stock.exitCode !== 0) {
      failures.push(runFailure(target.name, target.profile, file, "stock_run_failed", stock.exitCode));
    }
    if (extension.exitCode !== 0) {
      failures.push(runFailure(target.name, target.profile, file, "extension_run_failed", extension.exitCode));
    }
    for (const diagnostic of [...stock.diagnostics.errors, ...extension.diagnostics.errors]) {
      failures.push(runFailure(target.name, target.profile, file, diagnostic, 1));
    }
  }

  if (rendered.nonComparableQueries.length > 0) {
    for (const query of rendered.nonComparableQueries) {
      failures.push({
        workload: target.name,
        profile: target.profile,
        sourceFile: query.sourceFile,
        queryId: query.id,
        statementIndex: query.statementIndex,
        comparisonMode: "multiset",
        routeExpectation: "not_asserted",
        stockRowCount: 0,
        extensionRowCount: 0,
        kind: query.reason,
        preview: query.reason
      });
    }
  }

  if (target.scratchDatabase) {
    const stockCleanup = await cleanupDatabase(runner, stockDb, target.name, target.profile, "stock");
    if (stockCleanup) {
      failures.push(stockCleanup);
    }
    const extensionCleanup = await cleanupDatabase(runner, extensionDb, target.name, target.profile, "extension");
    if (extensionCleanup) {
      failures.push(extensionCleanup);
    }
  }

  const diff = {
    workload: target.name,
    profile: target.profile,
    comparedQueries: allQueries.length,
    orderedQueries: allQueries.filter((query) => query.comparisonMode === "ordered").length,
    multisetQueries: allQueries.filter((query) => query.comparisonMode === "multiset").length,
    comparableSourceFiles: rendered.files.map((file) => file.sourceFile),
    comparableQueries: rendered.files.flatMap((file) => file.queries),
    comparisons,
    setupFile: rendered.setupFile,
    skippedSetupFile: rendered.setupFile,
    scratchDatabase: target.scratchDatabase,
    excludedFiles: target.excludedFiles,
    nonComparableQueries: rendered.nonComparableQueries,
    diagnostics,
    artifactPaths,
    failures
  };
  mkdirSync(dirname(summaryPath), { recursive: true });
  mkdirSync(dirname(jsonPath), { recursive: true });
  writeFileSync(jsonPath, `${JSON.stringify(diff, null, 2)}\n`);
  writeFileSync(summaryPath, renderSummary(diff));

  if (failures.length > 0) {
    io.stderr += `FAIL: compare-postgres mismatch. Summary: ${summaryPath}\n`;
    io.stderr += `Diff: ${jsonPath}\n`;
    io.stderr += failures.slice(0, 5).map((failure) => `${failure.queryId}: ${failure.kind} ${failure.preview}`).join("\n");
    io.stderr += "\n";
    return 1;
  }

  io.stdout += `OK: compare-postgres passed. Summary: ${summaryPath}\n`;
  io.stdout += `Diff: ${jsonPath}\n`;
  return 0;
}

function validateNoInlineSetupForSharedDatabase(manifests: ReturnType<typeof readComparePostgresManifests>): void {
  const hasComparableQuery = manifests.some((manifest) => manifest.statements.some((statement) => statement.comparable));
  if (!hasComparableQuery) {
    return;
  }
  for (const manifest of manifests) {
    for (const statement of manifest.statements) {
      if (statement.statementKind === "setup") {
        throw new Error(`${manifest.path}: scratch_database: false cannot be used with inline setup SQL before comparable queries`);
      }
    }
  }
}

function validateSetupFileDoesNotLoadPgxLower(target: ReturnType<typeof resolveComparePostgresTarget>): void {
  if (!target.setupFile) {
    return;
  }
  const manifest = parseSqlManifest({
    path: target.setupFile,
    sql: readFileSync(target.setupFile, "utf8"),
    defaultRoute: "ignore",
    requireRouteDirectives: false
  });
  for (const statement of manifest.statements) {
    if (statement.statementKind !== "setup") {
      throw new Error(`${target.setupFile}: workload setup SQL cannot contain row-producing statements`);
    }
    if (setupSqlUsesPsqlMetaCommand(statement.sql)) {
      throw new Error(`${target.setupFile}: workload setup SQL cannot use psql meta commands`);
    }
    if (setupSqlUsesDoBlock(statement.sql)) {
      throw new Error(`${target.setupFile}: workload setup SQL cannot use DO blocks`);
    }
    if (setupSqlReferencesPgxLower(statement.sql)) {
      throw new Error(`${target.setupFile}: workload setup SQL cannot load or configure pgx_lower`);
    }
    if (setupSqlUsesLoad(statement.sql)) {
      throw new Error(`${target.setupFile}: workload setup SQL cannot use LOAD`);
    }
    if (setupSqlConfiguresPreloadLibraries(statement.sql)) {
      throw new Error(`${target.setupFile}: workload setup SQL cannot configure preload libraries`);
    }
  }
}

function buildManagedComparePostgresShellCommand(
  options: ComparePostgresOptions,
  config: ManagedOperationConfig,
  targetProfile: string
): string {
  const internalArgs = [
    "/workspace/pgx-cli/dist/index.js",
    "test",
    "compare-postgres-internal",
    "--workload",
    options.workload,
    "--root",
    "/workspace",
    "--output-dir",
    managedOutputDir(options),
    "--from-managed-runner"
  ];
  if (options.runName) {
    internalArgs.push("--run-name", options.runName);
  }
  if (options.summaryPath) {
    internalArgs.push("--summary", options.summaryPath);
  }
  if (options.jsonPath) {
    internalArgs.push("--json", options.jsonPath);
  }
  const dockerCommand = `docker exec ${quoteShell(config.dockerContainer ?? "pgx-lower-dev")} bash -lc ${quoteShell(
    [
      buildAndInstallExtensionCommand(targetProfile, options.runName ?? safeName(options.workload)),
      "chmod -R o+rX /workspace",
      `PGX_COMPARE_POSTGRES_INTERNAL=1 ${internalArgs.map(quoteShell).join(" ")}`
    ].join(" && ")
  )}`;
  return [
    "npm --prefix pgx-cli install",
    "npm --prefix pgx-cli run build",
    dockerCommand
  ].join(" && ");
}

function writeScript(
  scriptsDir: string,
  variant: "stock" | "extension",
  file: RenderedComparePostgresFile
): string {
  const dir = join(scriptsDir, variant);
  const path = join(dir, file.stem);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, variant === "stock" ? file.stockSql : file.extensionSql);
  return path;
}

async function runAndCaptureScript(
  runner: StreamingCommandRunner,
  database: string,
  scriptPath: string,
  outputsDir: string,
  variant: "stock" | "extension",
  file: RenderedComparePostgresFile
): Promise<ScriptRun> {
  const dir = join(outputsDir, variant);
  const relativeStem = file.stem.replace(/\.sql$/u, "");
  const stdoutPath = join(dir, `${relativeStem}.stdout`);
  const stderrPath = join(dir, `${relativeStem}.stderr`);
  mkdirSync(dirname(stdoutPath), { recursive: true });
  const result = await runPsqlFile(runner, database, scriptPath);
  writeFileSync(stdoutPath, result.stdout);
  writeFileSync(stderrPath, result.stderr);
  let blocks: ResultBlock[] = [];
  let parseError: string | undefined;
  try {
    blocks = parseResultBlocks(result.stdout);
  } catch (error) {
    parseError = errorMessage(error);
  }
  return {
    stdoutPath,
    stderrPath,
    blocks,
    exitCode: result.exitCode,
    diagnostics: parseCompareDiagnostics(result.stderr),
    ...(parseError ? { parseError } : {})
  };
}

async function runAndCaptureSetup(
  runner: StreamingCommandRunner,
  database: string,
  setupPath: string,
  outputsDir: string,
  variant: "stock" | "extension"
): Promise<SetupRun> {
  const dir = join(outputsDir, variant);
  mkdirSync(dir, { recursive: true });
  const stdoutPath = join(dir, "setup.stdout");
  const stderrPath = join(dir, "setup.stderr");
  const result = await runPsqlFile(runner, database, setupPath);
  writeFileSync(stdoutPath, result.stdout);
  writeFileSync(stderrPath, result.stderr);
  return { stdoutPath, stderrPath, exitCode: result.exitCode, diagnostics: parseCompareDiagnostics(result.stderr) };
}

async function resetDatabase(
  runner: StreamingCommandRunner,
  database: string,
  workload: string,
  profile: string,
  variant: "stock" | "extension"
): Promise<ComparePostgresFailure | undefined> {
  const drop = await runAsPostgres(runner, `/usr/local/pgsql/bin/dropdb --if-exists ${quoteShell(database)}`);
  if (drop.exitCode !== 0) {
    return databaseResetFailure(workload, profile, variant, "dropdb", drop);
  }
  const create = await runAsPostgres(runner, `/usr/local/pgsql/bin/createdb ${quoteShell(database)}`);
  if (create.exitCode !== 0) {
    return databaseResetFailure(workload, profile, variant, "createdb", create);
  }
  return undefined;
}

async function cleanupDatabase(
  runner: StreamingCommandRunner,
  database: string,
  workload: string,
  profile: string,
  variant: "stock" | "extension"
): Promise<ComparePostgresFailure | undefined> {
  const drop = await runAsPostgres(runner, `/usr/local/pgsql/bin/dropdb --if-exists ${quoteShell(database)}`);
  if (drop.exitCode !== 0) {
    return databaseCleanupFailure(workload, profile, variant, drop);
  }
  return undefined;
}

async function runPsqlFile(
  runner: StreamingCommandRunner,
  database: string,
  path: string
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  return runAsPostgres(
    runner,
    `env -u PGOPTIONS /usr/local/pgsql/bin/psql -X -v ON_ERROR_STOP=on -d ${quoteShell(database)} -f ${quoteShell(path)}`
  );
}

async function runAsPostgres(
  runner: StreamingCommandRunner,
  command: string
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  return runningAsRoot() ? runner.run("su", ["postgres", "-c", command]) : runner.run("sh", ["-c", command]);
}

function runningAsRoot(): boolean {
  return typeof process.getuid === "function" && process.getuid() === 0;
}

function compareFileBlocks(
  workload: string,
  profile: string,
  file: RenderedComparePostgresFile,
  stock: ScriptRun,
  extension: ScriptRun
): { comparisons: ComparePostgresQueryComparison[]; failures: ComparePostgresFailure[] } {
  const stockBlocks = new Map(stock.blocks.map((block) => [block.id, block]));
  const extensionBlocks = new Map(extension.blocks.map((block) => [block.id, block]));
  const failures: ComparePostgresFailure[] = [];
  const comparisons: ComparePostgresQueryComparison[] = [];
  if (stock.parseError) {
    failures.push(scriptFailure(workload, profile, file, "stock_result_parse_failed", stock.parseError));
  }
  if (extension.parseError) {
    failures.push(scriptFailure(workload, profile, file, "extension_result_parse_failed", extension.parseError));
  }
  if (failures.length > 0) {
    return { comparisons, failures };
  }

  const expectedIds = new Set(file.queries.map((query) => query.id));
  for (const block of stock.blocks) {
    if (!expectedIds.has(block.id)) {
      failures.push(unexpectedResultBlockFailure(workload, profile, file, "stock", block.id));
    }
  }
  for (const block of extension.blocks) {
    if (!expectedIds.has(block.id)) {
      failures.push(unexpectedResultBlockFailure(workload, profile, file, "extension", block.id));
    }
  }
  if (failures.length > 0) {
    return { comparisons, failures };
  }

  for (const query of file.queries) {
    const stockBlock = stockBlocks.get(query.id);
    const extensionBlock = extensionBlocks.get(query.id);
    if (!stockBlock || !extensionBlock) {
      const missingBlockFailure = {
        workload,
        profile,
        sourceFile: query.sourceFile,
        queryId: query.id,
        statementIndex: query.statementIndex,
        comparisonMode: query.comparisonMode,
        routeExpectation: query.routeExpectation,
        stockRowCount: 0,
        extensionRowCount: 0,
        kind: "missing_result_block",
        preview: `stock=${!!stockBlock} extension=${!!extensionBlock}`
      };
      comparisons.push(queryComparison(workload, profile, query, 0, 0, missingBlockFailure));
      failures.push(missingBlockFailure);
      continue;
    }
    const comparison = compareResultBlocks({
      workload,
      profile,
      sourceFile: query.sourceFile,
      statementIndex: query.statementIndex,
      routeExpectation: query.routeExpectation,
      stock: stockBlock,
      extension: extensionBlock
    });
    comparisons.push(
      queryComparison(
        workload,
        profile,
        query,
        comparison.stockRowCount,
        comparison.extensionRowCount,
        comparison.failure
      )
    );
    if (comparison.failure) {
      failures.push(comparison.failure);
    }
  }
  return { comparisons, failures };
}

function queryComparison(
  workload: string,
  profile: string,
  query: RenderedComparePostgresQuery,
  stockRowCount: number,
  extensionRowCount: number,
  failure?: ComparePostgresFailure
): ComparePostgresQueryComparison {
  return {
    workload,
    profile,
    sourceFile: query.sourceFile,
    queryId: query.id,
    statementIndex: query.statementIndex,
    comparisonMode: query.comparisonMode,
    routeExpectation: query.routeExpectation,
    stockRowCount,
    extensionRowCount,
    mismatchKind: failure?.kind ?? null,
    preview: failure?.preview ?? null
  };
}

function unexpectedResultBlockFailure(
  workload: string,
  profile: string,
  file: RenderedComparePostgresFile,
  variant: "stock" | "extension",
  blockId: string
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: file.sourceFile,
    queryId: blockId,
    statementIndex: 0,
    comparisonMode: "multiset",
    routeExpectation: "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind: "unexpected_result_block",
    preview: `${variant} emitted unexpected result block ${blockId}`
  };
}

function scriptDiagnostics(
  file: RenderedComparePostgresFile,
  variant: "stock" | "extension",
  run: ScriptRun
): ScriptDiagnostics {
  return {
    sourceFile: file.sourceFile,
    variant,
    routeNotices: run.diagnostics.routeNotices,
    pgxNotices: run.diagnostics.pgxNotices,
    warnings: run.diagnostics.warnings,
    errors: run.diagnostics.errors
  };
}

function setupDiagnostics(setupFile: string, variant: "stock" | "extension", run: SetupRun): ScriptDiagnostics {
  return {
    sourceFile: setupFile,
    variant,
    routeNotices: run.diagnostics.routeNotices,
    pgxNotices: run.diagnostics.pgxNotices,
    warnings: run.diagnostics.warnings,
    errors: run.diagnostics.errors
  };
}

function stockPgxLowerDiagnosticFailures(
  workload: string,
  profile: string,
  file: RenderedComparePostgresFile,
  stock: ScriptRun
): ComparePostgresFailure[] {
  const diagnostics = stock.diagnostics.pgxDiagnostics;
  if (diagnostics.length === 0) {
    return [];
  }
  return [scriptFailure(workload, profile, file, "stock_pgx_lower_diagnostic", boundedText(diagnostics[0] ?? ""))];
}

function extensionForcedFallbackDiagnosticFailures(
  workload: string,
  profile: string,
  file: RenderedComparePostgresFile,
  extension: ScriptRun
): ComparePostgresFailure[] {
  const diagnostic = extension.diagnostics.pgxDiagnostics.find((line) => /force_fallback/iu.test(line));
  if (!diagnostic) {
    return [];
  }
  return [scriptFailure(workload, profile, file, "extension_forced_fallback_diagnostic", boundedText(diagnostic))];
}

function stockSetupPgxLowerDiagnosticFailures(
  workload: string,
  profile: string,
  setupFile: string,
  stock: SetupRun
): ComparePostgresFailure[] {
  const diagnostic = stock.diagnostics.pgxDiagnostics[0];
  if (!diagnostic) {
    return [];
  }
  return [
    {
      workload,
      profile,
      sourceFile: setupFile,
      queryId: "setup",
      statementIndex: 0,
      comparisonMode: "multiset",
      routeExpectation: "not_asserted",
      stockRowCount: 0,
      extensionRowCount: 0,
      kind: "stock_pgx_lower_diagnostic",
      preview: boundedText(diagnostic)
    }
  ];
}

function runFailure(
  workload: string,
  profile: string,
  file: RenderedComparePostgresFile,
  kind: string,
  exitCode: number
): ComparePostgresFailure {
  return scriptFailure(workload, profile, file, kind, `exit ${exitCode}`);
}

function scriptFailure(
  workload: string,
  profile: string,
  file: RenderedComparePostgresFile,
  kind: string,
  preview: string
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: file.sourceFile,
    queryId: file.queries[0]?.id ?? file.stem,
    statementIndex: file.queries[0]?.statementIndex ?? 0,
    comparisonMode: file.queries[0]?.comparisonMode ?? "multiset",
    routeExpectation: file.queries[0]?.routeExpectation ?? "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind,
    preview
  };
}

function setupFailure(
  workload: string,
  profile: string,
  setupFile: string,
  kind: string,
  exitCode: number
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: setupFile,
    queryId: "setup",
    statementIndex: 0,
    comparisonMode: "multiset",
    routeExpectation: "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind,
    preview: `exit ${exitCode}`
  };
}

function databaseResetFailure(
  workload: string,
  profile: string,
  variant: "stock" | "extension",
  operation: "dropdb" | "createdb",
  result: { exitCode: number; stdout: string; stderr: string }
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: "database",
    queryId: `${variant}-database`,
    statementIndex: 0,
    comparisonMode: "multiset",
    routeExpectation: "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind: "database_reset_failed",
    preview: `${operation} exit ${result.exitCode}: ${boundedText(result.stderr || result.stdout)}`
  };
}

function databaseCleanupFailure(
  workload: string,
  profile: string,
  variant: "stock" | "extension",
  result: { exitCode: number; stdout: string; stderr: string }
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: "database",
    queryId: `${variant}-database`,
    statementIndex: 0,
    comparisonMode: "multiset",
    routeExpectation: "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind: "database_cleanup_failed",
    preview: `dropdb exit ${result.exitCode}: ${boundedText(result.stderr || result.stdout)}`
  };
}

function expectedComparableCountFailure(
  workload: string,
  profile: string,
  expected: number,
  actual: number
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: "manifest",
    queryId: "manifest",
    statementIndex: 0,
    comparisonMode: "multiset",
    routeExpectation: "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind: "expected_comparable_count_mismatch",
    preview: `expected ${expected} comparable statements, found ${actual}`
  };
}

function noComparableQueriesFailure(
  workload: string,
  profile: string
): ComparePostgresFailure {
  return {
    workload,
    profile,
    sourceFile: "manifest",
    queryId: "manifest",
    statementIndex: 0,
    comparisonMode: "multiset",
    routeExpectation: "not_asserted",
    stockRowCount: 0,
    extensionRowCount: 0,
    kind: "no_comparable_queries",
    preview: "no comparable SQL statements found"
  };
}

function renderSummary(diff: {
  workload: string;
  profile: string;
  comparedQueries: number;
  orderedQueries: number;
  multisetQueries: number;
  comparableSourceFiles: string[];
  comparableQueries: RenderedComparePostgresQuery[];
  setupFile?: string;
  scratchDatabase: boolean;
  excludedFiles: Array<{ glob: string; reason: string; matchedFiles?: string[] }>;
  nonComparableQueries: Array<{ id: string; sourceFile: string; statementIndex: number; reason: string }>;
  diagnostics: ScriptDiagnostics[];
  artifactPaths: string[];
  failures: ComparePostgresFailure[];
}): string {
  return [
    "# Compare-Postgres Summary",
    "",
    `Workload: ${diff.workload}`,
    `Profile: ${diff.profile}`,
    `Compared queries: ${diff.comparedQueries}`,
    `Ordered queries: ${diff.orderedQueries}`,
    `Multiset queries: ${diff.multisetQueries}`,
    `Scratch databases: ${diff.scratchDatabase ? "enabled" : "disabled"}`,
    `Setup file skipped as source: ${diff.setupFile ?? "none"}`,
    `Non-comparable queries: ${diff.nonComparableQueries.length}`,
    `Failures: ${diff.failures.length}`,
    "",
    "## Source Files",
    ...listOrNone(diff.comparableSourceFiles),
    "",
    "## Comparable Queries",
    ...listOrNone(diff.comparableQueries.map(queryLine)),
    "",
    "## Excluded Files",
    ...listOrNone(excludedFileLines(diff.excludedFiles)),
    "",
    "## Non-Comparable Queries",
    ...listOrNone(diff.nonComparableQueries.map(nonComparableQueryLine)),
    "",
    "## First Failures",
    ...listOrNone(diff.failures.slice(0, 10).map((failure) => `${failure.queryId}: ${failure.kind} ${failure.preview}`)),
    "",
    "## Diagnostics",
    ...listOrNone(diagnosticLines(diff.diagnostics)),
    "",
    "## Artifacts",
    ...listOrNone(diff.artifactPaths),
    ""
  ].join("\n");
}

function queryLine(query: RenderedComparePostgresQuery): string {
  return `${query.id}: ${query.sourceFile}:${query.statementIndex} ${query.comparisonMode} route=${query.routeExpectation}`;
}

function excludedFileLines(excludedFiles: Array<{ glob: string; reason: string; matchedFiles?: string[] }>): string[] {
  const lines: string[] = [];
  for (const file of excludedFiles) {
    if (!file.matchedFiles || file.matchedFiles.length === 0) {
      lines.push(`${file.glob}: ${file.reason}`);
      continue;
    }
    for (const matched of file.matchedFiles) {
      lines.push(`${matched}: ${file.reason} (${file.glob})`);
    }
  }
  return lines;
}

function nonComparableQueryLine(query: { id: string; sourceFile: string; statementIndex: number; reason: string }): string {
  return `${query.id}: ${query.reason} ${query.sourceFile}:${query.statementIndex}`;
}

function diagnosticLines(diagnostics: readonly ScriptDiagnostics[]): string[] {
  const lines: string[] = [];
  for (const diagnostic of diagnostics) {
    for (const [name, messages] of [
      ["route", diagnostic.routeNotices],
      ["notice", diagnostic.pgxNotices],
      ["warning", diagnostic.warnings],
      ["error", diagnostic.errors]
    ] as const) {
      for (const message of messages) {
        lines.push(`${diagnostic.variant} ${diagnostic.sourceFile} ${name}: ${message}`);
      }
    }
  }
  return lines;
}

function listOrNone(items: readonly string[]): string[] {
  return items.length > 0 ? items.map((item) => `- ${item}`) : ["- none"];
}

function scratchDbName(options: ComparePostgresOptions, suffix: string): string {
  const base = safeName(options.runName ?? options.workload);
  const hash = createHash("sha1").update(base).digest("hex").slice(0, 8);
  const prefix = "pgx_compare_";
  const suffixPart = `_${suffix}`;
  const baseLimit = 63 - prefix.length - 1 - hash.length - suffixPart.length;
  return `${prefix}${base.slice(0, Math.max(1, baseLimit))}_${hash}${suffixPart}`;
}

function defaultOutputDir(options: ComparePostgresOptions): string {
  const runName = options.runName ?? timestampRunName(options.workload);
  return join("build-artifacts", "test-runs", runName, "compare-postgres");
}

function findNearestRepoRoot(start: string): string {
  let dir = resolve(start);
  while (true) {
    if (existsSync(join(dir, "tests", "workloads.yaml"))) {
      return dir;
    }
    const parent = dirname(dir);
    if (parent === dir) {
      return resolve(start);
    }
    dir = parent;
  }
}

function normalizeRunName(options: ComparePostgresOptions): ComparePostgresOptions {
  return { ...options, runName: options.runName ? safeName(options.runName) : timestampRunName(options.workload) };
}

function timestampRunName(workload: string, now = new Date()): string {
  const timestamp = [
    now.getUTCFullYear().toString().padStart(4, "0"),
    (now.getUTCMonth() + 1).toString().padStart(2, "0"),
    now.getUTCDate().toString().padStart(2, "0"),
    "-",
    now.getUTCHours().toString().padStart(2, "0"),
    now.getUTCMinutes().toString().padStart(2, "0"),
    now.getUTCSeconds().toString().padStart(2, "0")
  ].join("");
  return `${timestamp}-${randomBytes(3).toString("hex")}-${safeName(workload)}`;
}

function managedOutputDir(options: ComparePostgresOptions): string {
  const outputDir = options.outputDir ?? defaultOutputDir(options);
  return isAbsolute(outputDir) ? outputDir : `/workspace/${outputDir}`;
}

function safeName(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "run";
}

function isContainerWorkspace(root: string): boolean {
  return existsSync("/.dockerenv") && isPathWithin("/workspace", root);
}

function isPathWithin(parent: string, child: string): boolean {
  const rel = relative(resolve(parent), resolve(child));
  return rel === "" || (!!rel && !rel.startsWith("..") && !isAbsolute(rel));
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

function boundedText(text: string): string {
  return text.trim().replace(/\s+/g, " ").slice(0, 500);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function comparePostgresUsage(): string {
  return [
    "Usage: pgx-cli test compare-postgres --workload <name>",
    "  [--root <repo-root>]",
    "  [--output-dir <path>]",
    "  [--run-name <name>]",
    "  [--summary <path>]",
    "  [--json <path>]"
  ].join("\n");
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
