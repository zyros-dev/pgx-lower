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

function difference(left: Set<string>, right: Set<string>): Set<string> {
  return new Set([...left].filter((value) => !right.has(value)));
}

function intersection(left: Set<string>, right: Set<string>): Set<string> {
  return new Set([...left].filter((value) => right.has(value)));
}
