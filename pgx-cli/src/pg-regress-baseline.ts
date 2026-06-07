export type PgRegressBaselineResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
};

const lineRe = /^(?:\d+:\s+)?(not\s+ok|ok)\s+\d+\s*-\s*(\S+)/;
const bailOutRe = /(?:^|\s)Bail out!/m;
const missingExpectedRe = /diff:\s+(?:\S*\/)?tests\/expected\/([^/\s:]+?)\.out:\s*No such file or directory/g;

export function evaluatePgRegressBaseline(raw: string, baselineText: string): PgRegressBaselineResult {
  const { passing, failing } = parsePgRegressOutput(raw);
  const bailOut = bailOutRe.test(raw);
  const missingExpected = [...raw.matchAll(missingExpectedRe)].map((match) => match[1] ?? "").filter(Boolean).sort();

  if (passing.size === 0 && failing.size === 0 && !bailOut) {
    return {
      exitCode: 2,
      stdout: "",
      stderr: "ERROR: no TAP-ish lines found in input - is ctest actually running with --output-on-failure?\n"
    };
  }

  const baseline = readBaseline(baselineText);
  const newFailures = difference(failing, baseline);
  const nowPassing = intersection(baseline, passing);
  const stillFailing = intersection(failing, baseline);
  const stdout: string[] = [
    "=== pg_regress delta vs baseline ===",
    `Baseline known-failing: ${baseline.size}`,
    `This run:  passing=${passing.size}  failing=${failing.size}`,
    `  still failing (expected): ${stillFailing.size}`,
    `  newly failing (regression): ${newFailures.size}`,
    `  newly passing (improvement): ${nowPassing.size}`
  ];

  if (nowPassing.size > 0) {
    stdout.push("", "NEWLY PASSING - remove these from the baseline file:");
    for (const name of [...nowPassing].sort()) {
      stdout.push(`  + ${name}`);
    }
  }

  if (bailOut) {
    stdout.push("");
    if (missingExpected.length > 0) {
      for (const name of missingExpected) {
        stdout.push(`NEW TEST NEEDS EXPECTED FILE: ${name}`);
      }
      stdout.push(
        "",
        `RED: pg_regress bailed out - expected-output file(s) missing for: ${missingExpected.join(", ")}.`,
        "     Copy the generated results/<name>.out to tests/expected/<name>.out after the RED run,",
        "     then re-run `pgx-cli dev gate review` to confirm green."
      );
    } else {
      stdout.push(
        "NEW TEST NEEDS EXPECTED FILE: <unknown - pg_regress bailed without a 'diff: .../expected/<name>.out' line>",
        "",
        "RED: pg_regress printed 'Bail out!' but the path could not be parsed from the transcript."
      );
    }
    return { exitCode: 1, stdout: `${stdout.join("\n")}\n`, stderr: "" };
  }

  if (newFailures.size > 0) {
    stdout.push("", "REGRESSIONS - these tests used to pass and now fail:");
    for (const name of [...newFailures].sort()) {
      stdout.push(`  x ${name}`);
    }
    stdout.push(
      "",
      "FAIL: regressions detected. Fix the test(s) or, if the failure is genuinely expected, update the baseline and justify it in the PR body."
    );
    return { exitCode: 1, stdout: `${stdout.join("\n")}\n`, stderr: "" };
  }

  stdout.push("", "OK: no new regressions vs baseline.");
  return { exitCode: 0, stdout: `${stdout.join("\n")}\n`, stderr: "" };
}

export function hasPgRegressBaselineInput(raw: string): boolean {
  const parsed = parsePgRegressOutput(raw);
  return parsed.passing.size > 0 || parsed.failing.size > 0 || bailOutRe.test(raw);
}

export function summarizePgRegressBaseline(raw: string, baselineText: string): {
  workflowExitCode: number;
  lines: string[];
} {
  const evaluated = evaluatePgRegressBaseline(raw, baselineText);
  const { passing, failing } = parsePgRegressOutput(raw);
  const baseline = readBaseline(baselineText);
  const newFailures = difference(failing, baseline);
  const stillFailing = intersection(failing, baseline);
  const nowPassing = intersection(baseline, passing);
  const lines: string[] = [
    `workflow exit: ${evaluated.exitCode}`,
    `passing: ${passing.size}`,
    `failing: ${failing.size}`
  ];

  for (const name of [...failing].sort()) {
    lines.push(`failed test: ${name}`);
  }
  for (const diff of parseDiffPaths(raw)) {
    lines.push(`diff: ${diff}`);
  }
  if (newFailures.size > 0) {
    lines.push(`new regression: ${[...newFailures].sort().join(", ")}`);
  }
  if (stillFailing.size > 0) {
    lines.push(`still failing: ${[...stillFailing].sort().join(", ")}`);
  }
  if (nowPassing.size > 0) {
    lines.push(`newly passing: ${[...nowPassing].sort().join(", ")}`);
  }
  return { workflowExitCode: evaluated.exitCode, lines };
}

export function detectCtestFailure(raw: string): string | undefined {
  const lines = raw.split("\n").map((line) => line.trim()).filter(Boolean);
  const summary = lines.find((line) => {
    const match = /\d+%\s+tests passed,\s+(\d+)\s+tests failed out of\s+\d+/i.exec(line);
    return !!match && Number(match[1]) > 0;
  });
  const errors = lines.find((line) => /Errors while running CTest/i.test(line));
  const failedStart = lines.findIndex((line) => /^The following tests FAILED:/i.test(line));
  const failedTests = failedStart === -1
    ? []
    : lines.slice(failedStart + 1).filter((line) => /^\d+\s+-\s+\S+/.test(line));

  if (!summary && !errors && failedTests.length === 0) {
    return undefined;
  }

  return ["CTest failed", summary, ...failedTests].filter(Boolean).join("\n");
}

function parsePgRegressOutput(raw: string): { passing: Set<string>; failing: Set<string> } {
  const passing = new Set<string>();
  const failing = new Set<string>();
  for (const line of raw.split("\n")) {
    const match = lineRe.exec(line.trim());
    if (!match) continue;
    const name = match[2];
    if (!name) continue;
    if (match[1] === "ok") {
      passing.add(name);
    } else {
      failing.add(name);
    }
  }
  return { passing, failing };
}

function readBaseline(text: string): Set<string> {
  return new Set(
    text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#"))
  );
}

function parseDiffPaths(raw: string): string[] {
  const paths: string[] = [];
  for (const line of raw.split("\n")) {
    const match = /diffs:\s+(\S+)|(\S+\.diff)\b/.exec(line.trim());
    const path = match?.[1] ?? match?.[2];
    if (path) paths.push(path);
  }
  return [...new Set(paths)].sort();
}

function difference(left: Set<string>, right: Set<string>): Set<string> {
  return new Set([...left].filter((value) => !right.has(value)));
}

function intersection(left: Set<string>, right: Set<string>): Set<string> {
  return new Set([...left].filter((value) => right.has(value)));
}
