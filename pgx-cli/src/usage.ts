import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

export const DEFAULT_USAGE_PATH = join(homedir(), ".config", "pgx-cli", "usage.json");

export function normalizeCommandKey(argv: string[]): string {
  const [command, subcommand] = argv;

  if (!command) {
    return "help";
  }
  if (command === "thor" && subcommand === "shell") {
    return "thor shell";
  }
  if (command === "dev" && subcommand) {
    return `dev ${subcommand}`;
  }
  if (command === "sync" && subcommand) {
    return `sync ${subcommand}`;
  }
  if (command === "clion" && subcommand) {
    return `clion ${subcommand}`;
  }
  if (command === "config" && subcommand) {
    return `config ${subcommand}`;
  }

  return command;
}

export function incrementUsage(path: string, argv: string[]): void {
  try {
    const key = normalizeCommandKey(argv);
    const current = existsSync(path)
      ? (JSON.parse(readFileSync(path, "utf8")) as Record<string, number>)
      : {};
    current[key] = (current[key] ?? 0) + 1;

    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(current, null, 2)}\n`);
  } catch {
    // Usage counters are best-effort and must never fail the primary command.
  }
}
