import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationRunner } from "./managed-operations.js";
import type { OperationOutput } from "./operations.js";
import type { ResolvedOutputConfig, ResolvedSyncConfig } from "./project-config.js";

export type IrConfig = {
  localProjectPath: string;
  output: ResolvedOutputConfig;
  mutagenSession?: string;
  sshHost?: string;
  remoteProjectPath?: string;
  sync?: ResolvedSyncConfig;
  runningOnRemote?: boolean;
};

type ParsedIrArgs = {
  target: string;
  head?: number;
  tail?: number;
  pattern?: string;
  full: boolean;
};

const defaultTailLines = 80;

export async function runIrCommand(
  args: string[],
  runner: ManagedOperationRunner,
  output: OperationOutput,
  config: IrConfig
): Promise<number> {
  const parsed = parseIrArgs(args);
  if (!parsed) {
    output.stderr += irUsage();
    return 1;
  }
  if (!isRemoteIrConfig(config)) {
    output.stderr += "ir inspect requires remote pgx-cli config with ssh host, remote path, mutagen session, and sync settings\n";
    return 1;
  }

  const shellCommand = buildIrShellCommand(parsed);
  const result = await runManagedRemoteShell({
    runner,
    output,
    config,
    commandName: "ir-inspect",
    shellCommand,
    requireMutagenProof: false,
    mutagenProofSkipReason: "remote-only IR inspection",
    fullOutput: parsed.full,
    preview: parsed.full
      ? undefined
      : { mode: parsed.head ? "head" : "tail", lines: parsed.head ?? parsed.tail ?? defaultTailLines }
  });
  return result.workflowExitCode;
}

function parseIrArgs(args: string[]): ParsedIrArgs | undefined {
  const [command, target, ...options] = args;
  if (command !== "inspect" || !target) {
    return undefined;
  }
  let head: number | undefined;
  let tail: number | undefined;
  let pattern: string | undefined;
  let full = false;

  for (let index = 0; index < options.length; index += 1) {
    const option = options[index];
    if (option === "--full") {
      full = true;
      continue;
    }
    if (option === "--head" || option === "--tail") {
      const value = parsePositiveInt(options[index + 1]);
      if (value === undefined) return undefined;
      if (option === "--head") head = value;
      else tail = value;
      index += 1;
      continue;
    }
    if (option === "--pattern") {
      const value = options[index + 1];
      if (!value) return undefined;
      pattern = value;
      index += 1;
      continue;
    }
    return undefined;
  }

  if ([head !== undefined, tail !== undefined, pattern !== undefined].filter(Boolean).length > 1) {
    return undefined;
  }

  return { target, head, tail, pattern, full };
}

function buildIrShellCommand(args: ParsedIrArgs): string {
  const fileExpression = args.target === "latest"
    ? [
        "latest=$(find /tmp/pgx_ir -type f -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)",
        "if [ -z \"$latest\" ]; then printf '%s\\n' 'missing /tmp/pgx_ir or no IR files found' 'next: run a pgx-lower workflow that emits IR, then retry pgx-cli ir inspect latest' >&2; exit 1; fi",
        "file=$latest"
      ].join("\n")
    : `file=${quoteShell(args.target)}`;
  const command = args.pattern
    ? `rg -n --color never ${quoteShell(args.pattern)} "$file" | sed -n '1,${defaultTailLines}p'`
    : args.head
      ? `sed -n '1,${args.head}p' "$file"`
      : args.full
        ? "sed -n '1,$p' \"$file\""
        : `tail -n ${args.tail ?? defaultTailLines} "$file"`;

  return [
    "if [ ! -d /tmp/pgx_ir ]; then printf '%s\\n' 'missing /tmp/pgx_ir' 'next: run a pgx-lower workflow that emits IR, then retry pgx-cli ir inspect latest' >&2; exit 1; fi",
    fileExpression,
    "if [ ! -f \"$file\" ]; then printf '%s\\n' \"missing IR file: $file\" 'next: pass latest or an existing /tmp/pgx_ir path' >&2; exit 1; fi",
    "printf 'ir file: %s\\n' \"$file\"",
    command
  ].join("\n");
}

function parsePositiveInt(value: string | undefined): number | undefined {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function irUsage(): string {
  return "Usage: ir inspect <latest|path> [--head N|--tail N|--pattern P]\n";
}

function isRemoteIrConfig(config: IrConfig): config is IrConfig & {
  mutagenSession: string;
  sshHost: string;
  remoteProjectPath: string;
  sync: ResolvedSyncConfig;
} {
  return !!config.mutagenSession && !!config.sshHost && !!config.remoteProjectPath && !!config.sync;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
