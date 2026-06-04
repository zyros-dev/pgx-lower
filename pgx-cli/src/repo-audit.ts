import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { CommandRunner } from "./commands.js";
import type { OperationOutput } from "./operations.js";

export type RepoAuditConfig = {
  localProjectPath: string;
};

const ignoredDirs = new Set([".git", "build-artifacts", "node_modules"]);
const allowedExact = new Set([
  ".githooks/pre-commit",
  ".githooks/pre-push",
  "docker/docker-compose.yml",
  "pgx-cli.yaml",
  "pgx-cli/dist/index.js",
  "tools/clion/container-clang",
  "tools/clion/container-clang++",
  "tools/clion/container-compiler"
]);

export async function runRepoCommand(
  args: string[],
  _runner: CommandRunner,
  output: OperationOutput,
  config: RepoAuditConfig
): Promise<number> {
  const [command] = args;
  if (command !== "audit-tools") {
    output.stderr += "Usage: repo audit-tools\n";
    return 1;
  }

  const unexpected = auditToolSurface(config.localProjectPath);
  if (unexpected.length === 0) {
    output.stdout += "repo audit-tools: ok\n";
    return 0;
  }

  output.stderr += "Unexpected loose tool entrypoints:\n";
  for (const path of unexpected) {
    output.stderr += `- ${path}\n`;
  }
  return 1;
}

export function auditToolSurface(root: string): string[] {
  return walk(root, "").filter((relativePath) => isUnexpectedTool(root, relativePath)).sort();
}

function walk(root: string, relativeDir: string): string[] {
  const dir = join(root, relativeDir);
  const results: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const relativePath = relativeDir ? `${relativeDir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      if (ignoredDirs.has(entry.name) || relativePath === "pgx-cli/node_modules") continue;
      results.push(...walk(root, relativePath));
      continue;
    }
    if (entry.isFile()) {
      results.push(relativePath);
    }
  }
  return results;
}

function isUnexpectedTool(root: string, relativePath: string): boolean {
  if (!isToolLike(root, relativePath)) return false;
  if (allowedExact.has(relativePath)) return false;
  if (/^docker\/[^/]+\/(?:docker-)?entrypoint\.sh$/.test(relativePath)) return false;
  if (/^docker\/[^/]+\/Dockerfile$/.test(relativePath)) return false;
  if (relativePath === "docker/entrypoint.sh") return false;
  return true;
}

function isToolLike(root: string, relativePath: string): boolean {
  if (relativePath === "Makefile" || relativePath.endsWith("/Makefile")) return true;
  if (relativePath.endsWith(".sh") || relativePath.endsWith(".py")) return true;
  if (relativePath.endsWith(".yml") || relativePath.endsWith(".yaml")) return true;
  if (relativePath.endsWith("Dockerfile")) return true;
  if (relativePath.startsWith("pgx-cli/node_modules/")) return false;

  const mode = statSync(join(root, relativePath)).mode;
  return (mode & 0o111) !== 0;
}
