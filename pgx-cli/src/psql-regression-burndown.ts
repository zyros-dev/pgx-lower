import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, isAbsolute, join } from "node:path";
import type { CommandRunner } from "./commands.js";
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

const statusLineRe = /^(?:\d+:\s+)?(not\s+ok|ok)\s+\d+\s*-\s*([^\s]+)(?:\s|$)/;

export function parsePgRegressStatusLines(text: string): PsqlRegressionStatus {
  const passing = new Set<string>();
  const failing = new Set<string>();

  for (const line of text.split("\n")) {
    const match = statusLineRe.exec(line.trim());
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
    "Expected PostgreSQL 17.6 src/test/regress files: sql/, expected/, and parallel_schedule.",
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
    pgRegress: values.get("--pg-regress") ?? "pg_regress",
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

export async function runPsqlRegressionBurndownCommand(
  args: readonly string[],
  runner: CommandRunner,
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

  const pgRegressResult = await runner.run(command, commandArgs);
  const pgRegressTranscript = `${pgRegressResult.stdout}${pgRegressResult.stderr}`;
  const pgRegressLog = join(options.outputDir, "pg_regress.log");
  writeFileSync(pgRegressLog, pgRegressTranscript);

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
    "  [--pg-regress <pg_regress>]",
    "  [--bindir <postgres-bindir>]",
    "  [--dlpath <postgres-libdir>]",
    "  [--schedule <schedule-file>]",
    "  [--load-extension <extension-name>]",
    "  [--record]"
  ].join("\n");
}
