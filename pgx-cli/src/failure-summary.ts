import { streamSampleToText } from "./output.js";
import type { StreamSample } from "./output.js";
import { summarizePgRegressBaseline } from "./pg-regress-baseline.js";

export type FailureSummaryInput = {
  commandName: string;
  stdoutSample: StreamSample | string;
  stderrSample: StreamSample | string;
  childExitCode: number;
  workflowExitCode?: number;
  postprocessedFailure?: string;
  transcriptPaths?: {
    stdoutPath?: string;
    stderrPath?: string;
    combinedPath?: string;
  };
};

export type FailureSummary = {
  kind: string;
  lines: string[];
};

export function summarizeFailure(input: FailureSummaryInput): FailureSummary | undefined {
  const workflowExitCode = input.workflowExitCode ?? input.childExitCode;
  if (input.childExitCode === 0 && workflowExitCode === 0) {
    return undefined;
  }

  if (input.postprocessedFailure) {
    return {
      kind: "postprocessed",
      lines: capLines(["postprocessed-failure: " + input.postprocessedFailure])
    };
  }

  const text = `${streamSampleToText(input.stdoutSample)}\n${streamSampleToText(input.stderrSample)}`;
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  const lowerCommand = input.commandName.toLowerCase();

  const route = firstMatching(lines, [/route assertion failed/i]);
  if (route) return summary("route", [`route: ${route}`], input);

  const unitSqlDiagnosticPattern = /tests\/unit-tests\/sql\/.*(?:ERROR:|WARNING:.*\[PROBLEM:ERROR_LEVEL\])|^unit-sql:/i;
  const unitSqlDiagnostics = matching(lines, [unitSqlDiagnosticPattern]);
  const unitSql = unitSqlDiagnostics[0] ?? firstMatching(lines, [/^--- .*\.sql ---$/i]);
  if (unitSql) {
    const evidence = unitSqlDiagnostics.length > 0 ? unitSqlDiagnostics : matching(lines, [/^--- .*\.sql ---$/i]);
    return summary("unit-sql", evidence.map((line) => `unit-sql: ${line}`), input);
  }

  const psql = firstMatching(lines, [/^psql:.*(?:ERROR|WARNING|NOTICE):/i]);
  if (psql) {
    return summary("psql", matching(lines, [/^psql:.*(?:ERROR|WARNING|NOTICE):/i]).map((line) => `psql: ${line}`), input);
  }

  const mutagen = firstMatching(lines, [/paused/i, /alpha disconnected/i, /beta disconnected/i, /conflict/i, /problem/i, /unsafe status/i]);
  if (mutagen || lowerCommand.includes("mutagen") || lowerCommand.includes("sync-preflight")) {
    return summary("mutagen", mutagen ? [`mutagen: ${mutagen}`] : ["mutagen: preflight failed"], input);
  }

  const regress = firstMatching(lines, [/^not ok \d+/i, /test .* \.\.\. FAILED/i, /^diffs:/i]);
  if (regress || lowerCommand.includes("pg_regress") || lowerCommand.includes("tpch")) {
    const pgRegress = summarizePgRegressBaseline(text, "");
    return summary("pg_regress", pgRegress.lines.map((line) => `pg_regress: ${line}`), input);
  }

  const tidy = firstMatching(lines, [/\[[a-z0-9_.-]+-[a-z0-9_.-]+\]$/i, /clang-tidy/i]);
  if (tidy || lowerCommand.includes("lint")) {
    const evidence = matching(lines, [/\[[a-z0-9_.-]+-[a-z0-9_.-]+\]$/i, /warning:|error:/i]);
    return summary("clang-tidy", evidence.map((line) => `clang-tidy: ${line}`), input);
  }

  const mlir = firstMatching(lines, [/Phase \S+ failed/i, /\/tmp\/pgx_ir\/.*\.mlir/i, /MLIR lowering pipeline failed/i]);
  if (mlir) {
    return summary("tpch-debug", matching(lines, [/Phase \S+ failed/i, /\/tmp\/pgx_ir\/.*\.mlir/i, /MLIR lowering pipeline failed/i]).map((line) => `tpch-debug: ${line}`), input);
  }

  const failedTarget = firstMatching(lines, [/^FAILED:/i]);
  const compileError = firstMatching(lines, [/:\d+:\d+:\s+(?:fatal )?error:/i, /ld: error:/i]);
  if (failedTarget || compileError || lowerCommand.includes("build") || lowerCommand.includes("compile")) {
    const evidence = [
      ...(failedTarget ? [`cmake/ninja: ${failedTarget}`] : []),
      ...(compileError ? [`compile-error: ${compileError}`] : [])
    ];
    return summary("compile-error", evidence.length > 0 ? evidence : ["compile-error: build failed"], input);
  }

  const spooler = firstMatching(lines, [/\[job \d+ queued/i, /Exit status:\s*\d+/i]);
  if (spooler || lowerCommand.includes("queue")) {
    const evidence = matching(lines, [/\[job \d+ queued/i, /Exit status:\s*\d+/i]);
    return summary("task-spooler", evidence.map((line) => `task-spooler: ${line}`), input);
  }

  return summary("generic", lines.slice(-6).map((line) => `failure: ${line}`), input);
}

function summary(kind: string, lines: string[], input: FailureSummaryInput): FailureSummary {
  const result = lines.length > 0 ? lines : [`${kind}: command failed`];
  if (input.transcriptPaths?.combinedPath) {
    result.push(`transcript: ${input.transcriptPaths.combinedPath}`);
  }
  return { kind, lines: capLines(result) };
}

function matching(lines: string[], patterns: RegExp[]): string[] {
  return lines.filter((line) => patterns.some((pattern) => pattern.test(line)));
}

function firstMatching(lines: string[], patterns: RegExp[]): string | undefined {
  return lines.find((line) => patterns.some((pattern) => pattern.test(line)));
}

function capLines(lines: string[]): string[] {
  return lines.filter(Boolean).slice(0, 12);
}
