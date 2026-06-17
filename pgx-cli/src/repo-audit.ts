import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { CommandRunner } from "./commands.js";
import type { OperationOutput } from "./operations.js";

export type RepoAuditConfig = {
  localProjectPath: string;
};

export type ToolClassification = "allowed" | "internal" | "generated" | "migrated" | "deferred" | "unexpected";

type ToolInventoryEntry = {
  path: string;
  kind: string;
  classification: ToolClassification;
};

const ignoredDirs = new Set([".git", "build-artifacts", "build-docker-lint", "node_modules"]);
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
const migratedBenchmarkHelpers = new Set([
  "benchmark/tpch/aggregate.py",
  "benchmark/tpch/fxt_to_flamegraph.py",
  "benchmark/tpch/metrics_collector.py",
  "benchmark/tpch/run.py"
]);
const internalExact = new Set([
  // Benchmark SQL/data fixtures are consumed by pgx-cli-managed workflows; they are not helper entrypoints.
  "benchmark/tpch_init.sql"
]);
const deferredExact = new Set<string>();
const toolReferencePattern =
  /(?:^|[^A-Za-z0-9_./-])((?:\.\/)?(?:scripts\/[A-Za-z0-9_./-]+\.(?:sh|py)|tools\/scripts\/[A-Za-z0-9_./-]+\.(?:sh|py)|benchmark\/[A-Za-z0-9_./-]+\.py|bench_runner(?:_v2)?\.py|overnight_bench(?:_v2)?\.sh|benchmark-config\.ya?ml|docker\/Makefile|justfile|Justfile|Makefile))(?=$|[^A-Za-z0-9_/-])/g;

export async function runRepoCommand(
  args: string[],
  _runner: CommandRunner,
  output: OperationOutput,
  config: RepoAuditConfig
): Promise<number> {
  const [command, ...rest] = args;
  if (command !== "audit-tools" || rest.some((arg) => arg !== "--inventory") || rest.length > 1) {
    output.stderr += "Usage: repo audit-tools [--inventory]\n";
    return 1;
  }

  if (rest.includes("--inventory")) {
    const inventory = auditToolInventory(config.localProjectPath);
    output.stdout += renderInventory(inventory);
    const unexpected = inventory.filter((entry) => entry.classification === "unexpected");
    if (unexpected.length > 0) {
      output.stderr += "unexpected inventory entries:\n";
      for (const entry of unexpected) {
        output.stderr += `- ${entry.path}\n`;
      }
      return 1;
    }
    return 0;
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
  return auditToolInventory(root)
    .filter((entry) => entry.kind !== "reference" && entry.classification === "unexpected")
    .map((entry) => entry.path);
}

export function auditToolInventory(root: string): ToolInventoryEntry[] {
  const paths = walk(root, "");
  const inventory = paths
    .filter((relativePath) => isInventoryEntry(root, relativePath))
    .map((relativePath) => ({
      path: relativePath,
      kind: classifyKind(relativePath),
      classification: classifyTool(root, relativePath)
    }));
  return [...inventory, ...scanToolReferences(root, paths)].sort((left, right) => left.path.localeCompare(right.path));
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
  if (name === "build" || name.startsWith("build-docker-")) return true;
  return false;
}

function isInventoryEntry(root: string, relativePath: string): boolean {
  return isTargetedInventorySurface(relativePath) || isToolLike(root, relativePath);
}

function isTargetedInventorySurface(relativePath: string): boolean {
  return (
    relativePath === "justfile" ||
    relativePath === "Justfile" ||
    relativePath.startsWith("scripts/") ||
    relativePath.startsWith("tools/scripts/") ||
    relativePath.startsWith("benchmark/") ||
    relativePath.startsWith("docs/") ||
    relativePath.startsWith(".codex/rules/") ||
    isCMakeFile(relativePath)
  );
}

function classifyTool(_root: string, relativePath: string): ToolClassification {
  if (isGenerated(relativePath)) return "generated";
  if (isInternal(relativePath)) return "internal";
  if (isAllowed(relativePath)) return "allowed";
  if (migratedBenchmarkHelpers.has(relativePath)) return "migrated";
  if (isDeferred(relativePath)) return "deferred";
  return "unexpected";
}

function classifyKind(relativePath: string): string {
  if (relativePath === "justfile" || relativePath === "Justfile") return "justfile";
  if (relativePath.startsWith(".codex/rules/")) return "codex-rule";
  if (isGenerated(relativePath)) return "generated";
  if (relativePath.startsWith("benchmark/")) return "benchmark";
  if (relativePath.startsWith("docs/")) return "docs";
  if (isCMakeFile(relativePath)) return "cmake";
  if (relativePath.startsWith(".githooks/")) return "hook";
  if (relativePath.startsWith("docker/")) return "docker";
  if (relativePath.startsWith("pgx-cli/clion-wrappers/")) return "wrapper";
  if (relativePath.endsWith(".yaml") || relativePath.endsWith(".yml")) return "config";
  if (relativePath === "Makefile" || relativePath.endsWith("/Makefile")) return "makefile";
  if (relativePath.endsWith(".sh") || relativePath.endsWith(".py")) return "script";
  return "tool";
}

function isAllowed(relativePath: string): boolean {
  if (allowedExact.has(relativePath)) return true;
  if (/^docker\/[^/]+\/(?:docker-)?entrypoint\.sh$/.test(relativePath)) return true;
  if (/^docker\/[^/]+\/Dockerfile$/.test(relativePath)) return true;
  if (relativePath === "docker/entrypoint.sh") return true;
  return false;
}

function isInternal(relativePath: string): boolean {
  return (
    internalExact.has(relativePath) ||
    relativePath.startsWith(".codex/rules/") ||
    relativePath.startsWith("pgx-cli/clion-wrappers/") ||
    relativePath.startsWith("docs/") ||
    isCMakeFile(relativePath)
  );
}

function isGenerated(relativePath: string): boolean {
  return (
    relativePath.startsWith("pgx-cli/dist/") ||
    relativePath === "pgx-cli/cmake-tools/runtime-header-tool" ||
    relativePath === "extension/pgx_lower.so" ||
    relativePath.includes("/__pycache__/") ||
    relativePath.endsWith(".pyc") ||
    relativePath.endsWith(".so") ||
    relativePath.endsWith("/CTestTestfile.cmake") ||
    relativePath.endsWith("/cmake_install.cmake")
  );
}

function isDeferred(relativePath: string): boolean {
  return deferredExact.has(relativePath);
}

function isCMakeFile(relativePath: string): boolean {
  return relativePath === "CMakeLists.txt" || relativePath.endsWith("/CMakeLists.txt") || relativePath.endsWith(".cmake");
}

function isToolLike(root: string, relativePath: string): boolean {
  if (relativePath === "justfile" || relativePath === "Justfile") return true;
  if (relativePath === "Makefile" || relativePath.endsWith("/Makefile")) return true;
  if (relativePath.endsWith(".sh") || relativePath.endsWith(".py")) return true;
  if (relativePath.endsWith(".yml") || relativePath.endsWith(".yaml")) return true;
  if (relativePath.endsWith("Dockerfile")) return true;
  if (relativePath.startsWith("pgx-cli/node_modules/")) return false;

  const mode = statSync(join(root, relativePath)).mode;
  return (mode & 0o111) !== 0;
}

function renderInventory(inventory: ToolInventoryEntry[]): string {
  const rows = [
    { path: "path", kind: "kind", classification: "classification" },
    ...inventory
  ];
  const pathWidth = Math.max(...rows.map((row) => row.path.length));
  const kindWidth = Math.max(...rows.map((row) => row.kind.length));
  return `${rows.map((row) => `${row.path.padEnd(pathWidth)}  ${row.kind.padEnd(kindWidth)}  ${row.classification}`).join("\n")}\n`;
}

function scanToolReferences(root: string, paths: string[]): ToolInventoryEntry[] {
  const references: ToolInventoryEntry[] = [];
  const seen = new Set<string>();
  for (const relativePath of paths) {
    if (!isReferenceScanFile(relativePath)) continue;
    const lines = readFileSync(join(root, relativePath), "utf8").split(/\r?\n/);
    lines.forEach((line, index) => {
      for (const match of line.matchAll(toolReferencePattern)) {
        const target = normalizeReference(match[1]);
        const path = `${relativePath}:${index + 1}:${target}`;
        if (seen.has(path)) continue;
        seen.add(path);
        references.push({
          path,
          kind: "reference",
          classification: classifyReferenceTarget(target)
        });
      }
    });
  }
  return references;
}

function isReferenceScanFile(relativePath: string): boolean {
  const name = relativePath.split("/").at(-1) ?? "";
  return (
    relativePath === "AGENTS.md" ||
    relativePath === "CLAUDE.md" ||
    name === "README.md" ||
    (relativePath.startsWith("docs/") && relativePath.endsWith(".md")) ||
    (relativePath.startsWith("specs/") && relativePath.endsWith(".md")) ||
    (relativePath.startsWith("wiki/specs/") && relativePath.endsWith(".md")) ||
    isCMakeFile(relativePath)
  );
}

function normalizeReference(reference: string): string {
  return reference.startsWith("./") ? reference.slice(2) : reference;
}

function classifyReferenceTarget(relativePath: string): ToolClassification {
  if (isGenerated(relativePath)) return "generated";
  if (isInternal(relativePath)) return "internal";
  if (isAllowed(relativePath)) return "allowed";
  if (migratedBenchmarkHelpers.has(relativePath)) return "migrated";
  if (isDeferred(relativePath)) return "deferred";
  return "unexpected";
}
