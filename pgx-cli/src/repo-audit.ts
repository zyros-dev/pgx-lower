import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { CommandRunner } from "./commands.js";
import type { OperationOutput } from "./operations.js";

export type RepoAuditConfig = {
  localProjectPath: string;
};

const ignoredDirs = new Set([".git", "benchmark", "build-artifacts", "build-docker-lint", "node_modules"]);
const allowedExact = new Set([
  ".githooks/pre-commit",
  ".githooks/pre-push",
  "docker/docker-compose.yml",
  "pgx-cli.yaml",
  "pgx-cli/clion-wrappers/container-clang",
  "pgx-cli/clion-wrappers/container-clang++",
  "pgx-cli/clion-wrappers/container-compiler",
  "pgx-cli/dist/index.js",
  "tests/workloads.yaml",
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
      if (shouldIgnoreDir(entry.name, relativePath)) continue;
      results.push(...walk(root, relativePath));
      continue;
    }
    if (entry.isFile()) {
      results.push(relativePath);
    }
  }
  return results;
}

function shouldIgnoreDir(name: string, relativePath: string): boolean {
  if (ignoredDirs.has(name) || relativePath === "pgx-cli/node_modules") return true;
  if (name === "_deps" || name.startsWith("cmake-build-")) return true;
  if (name === "benchmark" || name === "build" || name.startsWith("build-docker-")) return true;
  return false;
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
