import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import type { StreamingCommandRunner } from "./commands.js";
import type { DevConfig } from "./dev.js";
import { runMutagenPreflight } from "./mutagen-preflight.js";
import type { OperationOutput } from "./operations.js";

export type PreflightCheck = {
  name: string;
  ok: boolean;
  detail: string;
  next?: string;
};

export type PreflightResult = {
  ok: boolean;
  checks: PreflightCheck[];
};

type PreflightOptions = {
  strict: boolean;
  unitSql: boolean;
  sqlRuntime: boolean;
};

export async function runDevPreflightCommand(
  args: string[],
  runner: StreamingCommandRunner,
  output: OperationOutput,
  config: DevConfig
): Promise<number> {
  const options = parseOptions(args);
  if (!options) {
    output.stderr += "Usage: dev preflight [--strict] [--unit-sql] [--sql-runtime]\n";
    return 1;
  }

  const result = await runDevPreflight(runner, config, options);
  output.stdout += renderPreflightSummary(result);
  const failures = result.checks.filter((check) => !check.ok);
  if (failures.length > 0) {
    output.stderr += renderPreflightFailures(failures, options.strict);
  }
  return options.strict && !result.ok ? 1 : 0;
}

async function runDevPreflight(
  runner: StreamingCommandRunner,
  config: DevConfig,
  options: PreflightOptions
): Promise<PreflightResult> {
  const checks: PreflightCheck[] = [];
  const localBranch = await runLocalGitCommand(runner, config, ["branch", "--show-current"]);
  const localStatus = await runLocalGitCommand(runner, config, ["status", "--short"]);
  const localHead = await runLocalGitCommand(runner, config, ["rev-parse", "HEAD"]);
  checks.push(branchCommandCheck("local branch", localBranch));
  checks.push(statusCommandCheck("local worktree", localStatus));
  checks.push(revisionCommandCheck("local HEAD", localHead));

  const thorBranch = await runThorGitCommand(runner, config, "git branch --show-current");
  const thorStatus = await runThorGitCommand(runner, config, "git status --short");
  const thorHead = await runThorGitCommand(runner, config, "git rev-parse HEAD");
  checks.push(compareThorBranch(localBranch.stdout.trim(), thorBranch));
  checks.push(statusCommandCheck("thor worktree", thorStatus));
  checks.push(compareThorHead(localHead.stdout.trim(), thorHead));

  checks.push(await mutagenCheck(runner, config, options.strict));
  checks.push(cliDistCheck(config));
  if (options.unitSql) {
    checks.push(generatedUnitSqlCheck(config));
  }
  if (options.sqlRuntime) {
    checks.push(installedExtensionCheck(config, localHead.stdout.trim()));
  }

  return { ok: checks.every((check) => check.ok), checks };
}

function parseOptions(args: string[]): PreflightOptions | undefined {
  const known = new Set(["--strict", "--unit-sql", "--sql-runtime"]);
  if (args.some((arg) => !known.has(arg))) return undefined;
  const strict = args.includes("--strict");
  return {
    strict,
    unitSql: strict || args.includes("--unit-sql"),
    sqlRuntime: strict || args.includes("--sql-runtime")
  };
}

async function runLocalGitCommand(
  runner: StreamingCommandRunner,
  config: DevConfig,
  args: string[]
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  return runner.run("git", ["-C", config.localProjectPath, ...args]);
}

async function runThorGitCommand(
  runner: StreamingCommandRunner,
  config: DevConfig,
  shellCommand: string
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  if (config.runningOnRemote) {
    return runner.run("bash", ["-lc", shellCommand]);
  }
  return runner.run("ssh", [
    config.sshHost,
    "bash",
    "-lc",
    quoteShell(`cd ${quoteShell(config.remoteProjectPath)} && ${shellCommand}`)
  ]);
}

function branchCommandCheck(name: string, result: { exitCode: number; stdout: string; stderr: string }): PreflightCheck {
  if (result.exitCode !== 0) {
    return {
      name,
      ok: false,
      detail: `${name} unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`
    };
  }
  const branch = result.stdout.trim();
  return {
    name,
    ok: branch.length > 0,
    detail: branch.length > 0 ? `${name} ${branch}` : `${name} unavailable`
  };
}

function revisionCommandCheck(name: string, result: { exitCode: number; stdout: string; stderr: string }): PreflightCheck {
  if (result.exitCode !== 0) {
    return {
      name,
      ok: false,
      detail: `${name} unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`
    };
  }
  const revision = result.stdout.trim();
  return {
    name,
    ok: revision.length > 0,
    detail: revision.length > 0 ? `${name} ${shortSha(revision)}` : `${name} unavailable`
  };
}

function statusCommandCheck(name: string, result: { exitCode: number; stdout: string; stderr: string }): PreflightCheck {
  if (result.exitCode !== 0) {
    return {
      name,
      ok: false,
      detail: `${name} status unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`
    };
  }
  const status = result.stdout.trim();
  return {
    name,
    ok: status.length === 0,
    detail: status.length === 0 ? `${name} clean` : `${name} dirty: ${oneLine(status)}`,
    next: status.length === 0 ? undefined : `review ${name} changes before running expensive gates`
  };
}

function compareThorBranch(
  localBranch: string,
  thorBranch: { exitCode: number; stdout: string; stderr: string }
): PreflightCheck {
  if (thorBranch.exitCode !== 0) {
    return {
      name: "thor branch",
      ok: false,
      detail: `thor branch unavailable${thorBranch.stderr.trim() ? `: ${oneLine(thorBranch.stderr)}` : ""}`
    };
  }
  const remoteBranch = thorBranch.stdout.trim();
  const ok = localBranch.length > 0 && remoteBranch === localBranch;
  return {
    name: "thor branch",
    ok,
    detail: ok
      ? `thor branch ${remoteBranch} matches local`
      : `thor branch differs: local ${localBranch || "(unknown)"}, thor ${remoteBranch || "(unknown)"}`,
    next: ok ? undefined : "switch thor to the local branch or refresh the managed checkout"
  };
}

function compareThorHead(
  localHead: string,
  thorHead: { exitCode: number; stdout: string; stderr: string }
): PreflightCheck {
  if (thorHead.exitCode !== 0) {
    return {
      name: "thor HEAD",
      ok: false,
      detail: `thor HEAD unavailable${thorHead.stderr.trim() ? `: ${oneLine(thorHead.stderr)}` : ""}`
    };
  }
  const remoteHead = thorHead.stdout.trim();
  const ok = localHead.length > 0 && remoteHead === localHead;
  return {
    name: "thor HEAD",
    ok,
    detail: ok
      ? `thor HEAD ${shortSha(remoteHead)} matches local`
      : `thor HEAD differs: local ${shortSha(localHead) || "(unknown)"}, thor ${shortSha(remoteHead) || "(unknown)"}`,
    next: ok ? undefined : "sync or update thor checkout to the same revision before running expensive gates"
  };
}

async function mutagenCheck(runner: StreamingCommandRunner, config: DevConfig, strict: boolean): Promise<PreflightCheck> {
  const result = await runMutagenPreflight({
    runner,
    sessionName: config.mutagenSession,
    localProjectPath: config.localProjectPath,
    remoteProjectPath: config.remoteProjectPath,
    sshHost: config.sshHost,
    runId: "dev-preflight",
    flushTimeoutSeconds: config.sync.flush_timeout_seconds,
    runningOnRemote: config.runningOnRemote,
    requireProof: strict,
    proofPath: config.sync.proof.path,
    proofSkipReason: strict ? undefined : "dev preflight non-strict health check"
  });
  return {
    name: "mutagen",
    ok: result.ok,
    detail: result.message,
    next: result.ok ? undefined : result.next.join("; ")
  };
}

function cliDistCheck(config: DevConfig): PreflightCheck {
  const distPath = join(config.localProjectPath, "pgx-cli", "dist", "index.js");
  if (!existsSync(distPath)) {
    return {
      name: "pgx-cli dist",
      ok: false,
      detail: "pgx-cli dist missing: pgx-cli/dist/index.js",
      next: "npm --prefix pgx-cli run build"
    };
  }
  const sourceRoot = join(config.localProjectPath, "pgx-cli", "src");
  const newestSource = newestMtimeMs(sourceRoot, (path) => path.endsWith(".ts"));
  const distMtime = statSync(distPath).mtimeMs;
  const ok = newestSource <= distMtime;
  return {
    name: "pgx-cli dist",
    ok,
    detail: ok
      ? "pgx-cli dist fresh"
      : "pgx-cli dist stale: source is newer than pgx-cli/dist/index.js",
    next: ok ? undefined : "npm --prefix pgx-cli run build"
  };
}

function generatedUnitSqlCheck(config: DevConfig): PreflightCheck {
  const sqlDir = join(config.localProjectPath, "tests", "unit-tests", "sql");
  const sqlFiles = new Set(existsSync(sqlDir)
    ? readdirSync(sqlDir, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.endsWith(".sql"))
      .map((entry) => entry.name)
    : []);
  const expectedSuites = expectedUnitSuites(config);
  const failures: string[] = [];
  if (sqlFiles.size === 0) {
    failures.push("generated unit SQL missing: tests/unit-tests/sql/*.sql");
  }
  for (const suite of expectedSuites) {
    const sqlName = `${suite.name}.sql`;
    const sqlPath = join(sqlDir, sqlName);
    if (!sqlFiles.has(sqlName)) {
      failures.push(`generated unit SQL missing: ${sqlName}`);
      continue;
    }
    if (statSync(sqlPath).mtimeMs < suite.sourceMtimeMs) {
      failures.push(`generated unit SQL stale: ${sqlName}`);
    }
  }
  if (expectedSuites.length === 0) {
    failures.push("generated unit SQL parity not proven: no src/pgx-lower/test/*_tests.cpp sources found");
  }
  return {
    name: "generated unit SQL",
    ok: failures.length === 0,
    detail: failures.length === 0
      ? `generated unit SQL fresh (${expectedSuites.length} expected suite${expectedSuites.length === 1 ? "" : "s"})`
      : failures.join("; "),
    next: failures.length === 0 ? undefined : "pgx-cli test unit-sql --root <repo>"
  };
}

function installedExtensionCheck(config: DevConfig, currentHead: string): PreflightCheck {
  const summary = latestInstallSummary(config);
  if (!summary) {
    return {
      name: "installed extension",
      ok: false,
      detail: "installed extension freshness not proven: no successful build/install summary found",
      next: installNextAction()
    };
  }
  if (!summary.gitHead) {
    return {
      name: "installed extension",
      ok: false,
      detail: `installed extension freshness not tied to current HEAD: summary ${summary.runId} has no gitHead`,
      next: installNextAction()
    };
  }
  if (summary.gitHead !== currentHead) {
    return {
      name: "installed extension",
      ok: false,
      detail: `installed extension built for ${shortSha(summary.gitHead)}, current HEAD ${shortSha(currentHead)}`,
      next: installNextAction()
    };
  }
  const sourceMtime = Math.max(
    newestMtimeMs(join(config.localProjectPath, "src", "pgx-lower"), () => true),
    newestMtimeMs(join(config.localProjectPath, "extension"), () => true),
    fileMtimeMs(join(config.localProjectPath, "CMakeLists.txt"))
  );
  const ok = sourceMtime <= summary.finishedAtMs;
  return {
    name: "installed extension",
    ok,
    detail: ok
      ? `installed extension fresh from ${summary.runId}`
      : `installed extension stale: source is newer than latest build/install summary ${summary.runId}`,
    next: ok ? undefined : installNextAction()
  };
}

function latestInstallSummary(config: DevConfig): { runId: string; finishedAtMs: number; gitHead?: string } | undefined {
  const runsRoot = join(config.localProjectPath, config.output.transcript_dir);
  if (!existsSync(runsRoot)) return undefined;
  let latest: { runId: string; finishedAtMs: number; gitHead?: string } | undefined;
  for (const entry of readdirSync(runsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const summaryPath = join(runsRoot, entry.name, "summary.json");
    if (!existsSync(summaryPath)) continue;
    const summary = readJsonObject(summaryPath);
    if (!summary || summary.workflowExitCode !== 0) continue;
    if (!summaryLooksLikeInstall(summary)) continue;
    const finishedAt = typeof summary.finishedAt === "string"
      ? Date.parse(summary.finishedAt)
      : statSync(summaryPath).mtimeMs;
    if (!Number.isFinite(finishedAt)) continue;
    if (!latest || finishedAt > latest.finishedAtMs) {
      latest = {
        runId: typeof summary.runId === "string" ? summary.runId : entry.name,
        finishedAtMs: finishedAt,
        gitHead: summaryGitHead(summary)
      };
    }
  }
  return latest;
}

function summaryLooksLikeInstall(summary: Record<string, unknown>): boolean {
  const command = Array.isArray(summary.command) ? summary.command.join(" ") : "";
  return command.includes("cmake --install");
}

function summaryGitHead(summary: Record<string, unknown>): string | undefined {
  if (typeof summary.gitHead === "string") return summary.gitHead;
  if (typeof summary.headSha === "string") return summary.headSha;
  const git = summary.git;
  if (git && typeof git === "object" && !Array.isArray(git)) {
    const gitRecord = git as Record<string, unknown>;
    if (typeof gitRecord.head === "string") return gitRecord.head;
    if (typeof gitRecord.headSha === "string") return gitRecord.headSha;
  }
  return undefined;
}

function expectedUnitSuites(config: DevConfig): Array<{ name: string; sourceMtimeMs: number }> {
  const testRoot = join(config.localProjectPath, "src", "pgx-lower", "test");
  if (!existsSync(testRoot)) return [];
  return walkFiles(testRoot)
    .filter((path) => path.endsWith("_tests.cpp"))
    .map((path) => ({
      name: path.slice(path.lastIndexOf("/") + 1, -"_tests.cpp".length),
      sourceMtimeMs: statSync(path).mtimeMs
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function readJsonObject(path: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  } catch {
    return undefined;
  }
}

function newestMtimeMs(root: string, include: (path: string) => boolean): number {
  if (!existsSync(root)) return 0;
  let newest = 0;
  for (const path of walkFiles(root)) {
    if (!include(path)) continue;
    newest = Math.max(newest, statSync(path).mtimeMs);
  }
  return newest;
}

function fileMtimeMs(path: string): number {
  return existsSync(path) ? statSync(path).mtimeMs : 0;
}

function walkFiles(root: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(path));
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files;
}

function renderPreflightSummary(result: PreflightResult): string {
  const lines = [`dev preflight: ${result.ok ? "ok" : "failed"}`];
  for (const check of result.checks) {
    lines.push(`- ${check.ok ? "ok" : "fail"} ${check.name}: ${check.detail}`);
  }
  return `${lines.join("\n")}\n`;
}

function renderPreflightFailures(failures: PreflightCheck[], strict: boolean): string {
  const prefix = strict ? "fail" : "warning";
  return failures.map((failure) => {
    const next = failure.next ? `\nnext: ${failure.next}` : "";
    return `preflight: ${prefix}: ${failure.detail}${next}`;
  }).join("\n") + "\n";
}

function oneLine(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function shortSha(value: string): string {
  return value.slice(0, 12);
}

function installNextAction(): string {
  return "pgx-cli dev build install --profile debug";
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
