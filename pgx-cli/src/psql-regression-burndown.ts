import { join } from "node:path";

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
