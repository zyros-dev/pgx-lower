import { mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const DEFAULT_REQUEST_DIR = join(homedir(), ".config", "pgx-cli", "requests");

export type RequestKind = "feature" | "complaint";

export function writeRequest(
  dir: string,
  kind: RequestKind,
  words: string[],
  now = new Date()
): string {
  if (words.length === 0) {
    throw new Error("Usage: request <feature|complaint> <message...>");
  }

  mkdirSync(dir, { recursive: true });
  const timestamp = now.toISOString();
  const stamp = timestamp.replaceAll(":", "").replaceAll(".", "");
  const path = join(dir, `${stamp}-${kind}.md`);
  const message = words.join(" ");
  writeFileSync(
    path,
    `# pgx-cli ${kind}\n\nTimestamp: ${timestamp}\nKind: ${kind}\n\n## Message\n\n${message}\n`
  );
  return path;
}
