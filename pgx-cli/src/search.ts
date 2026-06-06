import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { CommandRunner } from "./commands.js";
import type { OperationOutput } from "./operations.js";

export type SearchConfig = {
  localProjectPath: string;
};

type ParsedArgs = {
  lines: number;
  pattern: string;
  paths: string[];
  globs: string[];
};

const defaultLines = 50;

export async function runRgCommand(
  args: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: SearchConfig
): Promise<number> {
  const parsed = parseArgs(args);
  if (!parsed) {
    output.stderr += "Usage: rg [--lines N|-n N] [--glob G|-g G] <pattern> [path...]\n";
    return 1;
  }

  const transcriptPath = createTranscriptPath();
  const rgArgs = [
    "-n",
    "--color",
    "never",
    ...parsed.globs.flatMap((glob) => ["--glob", glob]),
    parsed.pattern,
    ...(parsed.paths.length > 0 ? parsed.paths : ["."])
  ];
  const command = [
    `cd ${quoteShell(config.localProjectPath)}`,
    `${["rg", ...rgArgs].map(quoteShell).join(" ")} > ${quoteShell(transcriptPath)} 2>&1`,
    "status=$?",
    `sed -n '1,${parsed.lines}p' ${quoteShell(transcriptPath)}`,
    `total=$(wc -l < ${quoteShell(transcriptPath)} | tr -d ' ')`,
    `if [ "$total" -gt ${parsed.lines} ]; then`,
    `  printf '%s\\n' 'pgx-cli rg: output truncated to ${parsed.lines} lines; full transcript: ${transcriptPath}' >&2`,
    "fi",
    "exit $status"
  ].join("\n");
  const result = await runner.run("bash", ["-lc", command]);
  output.stdout += result.stdout;
  output.stderr += result.stderr;
  return result.exitCode;
}

function parseArgs(args: string[]): ParsedArgs | undefined {
  let lines = defaultLines;
  const globs: string[] = [];
  const rest: string[] = [];

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--lines" || arg === "-n") {
      const next = args[index + 1];
      const parsed = Number(next);
      if (!Number.isInteger(parsed) || parsed <= 0) {
        return undefined;
      }
      lines = parsed;
      index += 1;
      continue;
    }
    if (arg === "--glob" || arg === "-g") {
      const next = args[index + 1];
      if (!next) {
        return undefined;
      }
      globs.push(next);
      index += 1;
      continue;
    }
    rest.push(arg);
  }

  const [pattern, ...paths] = rest;
  if (!pattern) {
    return undefined;
  }
  return { lines, pattern, paths, globs };
}

function createTranscriptPath(): string {
  const dir = join(tmpdir(), "pgx-cli-transcripts");
  mkdirSync(dir, { recursive: true });
  return join(dir, `pgx-cli-rg-${new Date().toISOString().replace(/[:.]/g, "-")}-${process.pid}.log`);
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }

  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
