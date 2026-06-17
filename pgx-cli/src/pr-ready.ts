import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { checkEvidenceFile } from "./agent-evidence.js";
import type { EvidenceFile } from "./agent-evidence.js";
import type { CommandRunner, RunResult } from "./commands.js";
import type { OperationOutput } from "./operations.js";

type PrReadyConfig = {
  localProjectPath: string;
  output: {
    transcript_dir?: string;
  };
};

type PrReadyOptions = {
  evidence: string;
  reviewRun?: string;
  allowNoPr: boolean;
};

type ReadyCheck = {
  name: string;
  ok: boolean;
  detail: string;
  next?: string;
};

type ReviewSummary = {
  runId: string;
  path: string;
  finishedAtMs: number;
  summary: Record<string, unknown>;
};

const reviewGateCommandNames = new Set(["dev-gate-review"]);

const mergeableStates = new Set(["CLEAN", "HAS_HOOKS"]);

export async function runPrReadyCommand(
  argv: string[],
  runner: CommandRunner,
  output: OperationOutput,
  config: PrReadyConfig
): Promise<number> {
  const parsed = parseOptions(argv);
  if (!parsed) {
    output.stderr += usage();
    return 1;
  }

  const checks: ReadyCheck[] = [];
  const status = await runner.run("git", ["-C", config.localProjectPath, "status", "--short"]);
  checks.push(localStatusCheck(status));

  const branchResult = await runner.run("git", ["-C", config.localProjectPath, "branch", "--show-current"]);
  const branch = branchResult.stdout.trim();
  checks.push(branchCheck(branchResult));

  const headResult = await runner.run("git", ["-C", config.localProjectPath, "rev-parse", "HEAD"]);
  const head = headResult.stdout.trim();
  checks.push(headCheck(headResult));

  checks.push(evidenceCheck(config, parsed.evidence));
  checks.push(reviewGateCheck(config, parsed.reviewRun, head));

  if (branch) {
    const pushed = await runner.run("git", ["-C", config.localProjectPath, "ls-remote", "--heads", "origin", branch]);
    checks.push(branchPushedCheck(branch, head, pushed));
  } else {
    checks.push({
      name: "branch pushed",
      ok: false,
      detail: "branch push cannot be checked without a current branch",
      next: "git branch --show-current"
    });
  }

  if (parsed.allowNoPr) {
    checks.push({
      name: "pull request",
      ok: true,
      detail: "skipped by --allow-no-pr",
      next: "gh pr create"
    });
  } else if (branch) {
    const pr = await runner.run("gh", [
      "pr",
      "view",
      "--head",
      branch,
      "--json",
      "url,mergeStateStatus,headRefName,baseRefName"
    ]);
    checks.push(prCheck(branch, pr));
  } else {
    checks.push({
      name: "pull request",
      ok: false,
      detail: "pull request cannot be checked without a current branch",
      next: "git branch --show-current"
    });
  }

  output.stdout += renderChecklist(checks);
  const failures = checks.filter((check) => !check.ok);
  if (failures.length > 0) {
    output.stderr += renderFailures(failures);
    return 1;
  }
  return 0;
}

function parseOptions(argv: string[]): PrReadyOptions | undefined {
  const [command, ...args] = argv;
  if (command !== "ready") return undefined;

  let evidence: string | undefined;
  let reviewRun: string | undefined;
  let allowNoPr = false;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--allow-no-pr") {
      allowNoPr = true;
      continue;
    }
    if (arg === "--evidence" || arg === "--review-run") {
      const value = args[index + 1];
      if (!value) return undefined;
      index += 1;
      if (arg === "--evidence") evidence = value;
      else reviewRun = value;
      continue;
    }
    return undefined;
  }

  return evidence ? { evidence, reviewRun, allowNoPr } : undefined;
}

function localStatusCheck(result: RunResult): ReadyCheck {
  if (result.exitCode !== 0) {
    return {
      name: "local worktree",
      ok: false,
      detail: `local worktree status unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`,
      next: "git status --short"
    };
  }
  const status = result.stdout.trim();
  return {
    name: "local worktree",
    ok: status.length === 0,
    detail: status.length === 0 ? "clean" : `local worktree dirty: ${oneLine(status)}`,
    next: status.length === 0 ? undefined : "git status --short"
  };
}

function branchCheck(result: RunResult): ReadyCheck {
  if (result.exitCode !== 0) {
    return {
      name: "branch",
      ok: false,
      detail: `branch unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`,
      next: "git branch --show-current"
    };
  }
  const branch = result.stdout.trim();
  return {
    name: "branch",
    ok: branch.length > 0,
    detail: branch.length > 0 ? branch : "current branch missing",
    next: branch.length > 0 ? undefined : "git switch <branch>"
  };
}

function headCheck(result: RunResult): ReadyCheck {
  if (result.exitCode !== 0) {
    return {
      name: "HEAD",
      ok: false,
      detail: `HEAD unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`,
      next: "git rev-parse HEAD"
    };
  }
  const head = result.stdout.trim();
  return {
    name: "HEAD",
    ok: head.length > 0,
    detail: head.length > 0 ? shortSha(head) : "HEAD missing",
    next: head.length > 0 ? undefined : "git rev-parse HEAD"
  };
}

function evidenceCheck(config: PrReadyConfig, evidence: string): ReadyCheck {
  const path = resolvePath(config, evidence);
  if (!existsSync(path)) {
    return {
      name: "evidence",
      ok: false,
      detail: `evidence file missing: ${displayPath(config, path)}`,
      next: `pgx-cli agent evidence check --file ${evidence}`
    };
  }

  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as EvidenceFile;
    const result = checkEvidenceFile(parsed);
    return {
      name: "evidence",
      ok: result.ok,
      detail: result.ok ? `complete: ${displayPath(config, path)}` : result.messages.join("; "),
      next: result.ok ? undefined : `pgx-cli agent evidence check --file ${evidence}`
    };
  } catch (error) {
    return {
      name: "evidence",
      ok: false,
      detail: `evidence file unreadable: ${error instanceof Error ? error.message : String(error)}`,
      next: `pgx-cli agent evidence check --file ${evidence}`
    };
  }
}

function reviewGateCheck(config: PrReadyConfig, reviewRun: string | undefined, currentHead: string): ReadyCheck {
  const summary = reviewRun
    ? readReviewSummary(config, reviewRun)
    : latestReviewSummary(config);
  if (!summary) {
    return {
      name: "review gate",
      ok: false,
      detail: reviewRun
        ? `review gate summary missing: ${reviewRun}`
        : "review gate summary missing",
      next: "pgx-cli dev gate review"
    };
  }
  if (!summaryLooksLikeReviewGate(summary.summary)) {
    return {
      name: "review gate",
      ok: false,
      detail: `review gate summary command mismatch: ${summary.runId}`,
      next: "pgx-cli dev gate review"
    };
  }

  if (summary.summary.workflowExitCode !== 0) {
    return {
      name: "review gate",
      ok: false,
      detail: `review gate failed in ${summary.runId}: workflowExitCode ${String(summary.summary.workflowExitCode)}`,
      next: "pgx-cli dev gate review"
    };
  }

  const recordedHead = summaryGitHead(summary.summary);
  if (!recordedHead) {
    return {
      name: "review gate",
      ok: false,
      detail: `review gate HEAD missing: summary ${summary.runId} has no gitHead/headSha/git.head`,
      next: "pgx-cli dev gate review"
    };
  }
  if (currentHead && recordedHead !== currentHead) {
    return {
      name: "review gate",
      ok: false,
      detail: `review gate HEAD mismatch: summary ${shortSha(recordedHead)}, current ${shortSha(currentHead)}`,
      next: "pgx-cli dev gate review"
    };
  }

  return {
    name: "review gate",
    ok: true,
    detail: `${summary.runId} at ${shortSha(recordedHead)}`
  };
}

function branchPushedCheck(branch: string, currentHead: string, result: RunResult): ReadyCheck {
  if (!currentHead) {
    return {
      name: "branch pushed",
      ok: false,
      detail: "branch push cannot be checked without current HEAD",
      next: "git rev-parse HEAD"
    };
  }
  if (result.exitCode !== 0) {
    return {
      name: "branch pushed",
      ok: false,
      detail: `branch push status unavailable${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`,
      next: `git push -u origin ${branch}`
    };
  }
  const expectedRef = `refs/heads/${branch}`;
  const remoteHead = result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim().split(/\s+/))
    .find((fields) => fields.length >= 2 && fields[1] === expectedRef)?.[0];
  const ok = remoteHead === currentHead;
  return {
    name: "branch pushed",
    ok,
    detail: ok
      ? `origin/${branch} at ${shortSha(remoteHead)}`
      : remoteHead
        ? `origin/${branch} stale: remote ${shortSha(remoteHead)}, local ${shortSha(currentHead)}`
        : `origin/${branch} missing`,
    next: ok ? undefined : `git push -u origin ${branch}`
  };
}

function prCheck(branch: string, result: RunResult): ReadyCheck {
  if (result.exitCode !== 0) {
    return {
      name: "pull request",
      ok: false,
      detail: `pull request missing${result.stderr.trim() ? `: ${oneLine(result.stderr)}` : ""}`,
      next: "gh pr create"
    };
  }

  const parsed = readJsonObject(result.stdout);
  if (!parsed) {
    return {
      name: "pull request",
      ok: false,
      detail: "pull request response unreadable",
      next: `gh pr view --head ${branch} --json url,mergeStateStatus,headRefName,baseRefName`
    };
  }

  const url = typeof parsed.url === "string" ? parsed.url : "";
  const headRefName = typeof parsed.headRefName === "string" ? parsed.headRefName : undefined;
  const baseRefName = typeof parsed.baseRefName === "string" ? parsed.baseRefName : undefined;
  const mergeStateStatus = typeof parsed.mergeStateStatus === "string" ? parsed.mergeStateStatus : "";
  if (!url) {
    return {
      name: "pull request",
      ok: false,
      detail: "pull request missing url",
      next: "gh pr create"
    };
  }
  if (!headRefName) {
    return {
      name: "pull request",
      ok: false,
      detail: "pull request headRefName missing",
      next: `gh pr view --head ${branch} --json url,mergeStateStatus,headRefName,baseRefName`
    };
  }
  if (!baseRefName) {
    return {
      name: "pull request",
      ok: false,
      detail: "pull request baseRefName missing",
      next: `gh pr view --head ${branch} --json url,mergeStateStatus,headRefName,baseRefName`
    };
  }
  if (headRefName !== branch) {
    return {
      name: "pull request",
      ok: false,
      detail: `pull request head mismatch: ${headRefName}, current ${branch}`,
      next: `gh pr view --head ${branch}`
    };
  }
  if (!mergeableStates.has(mergeStateStatus)) {
    return {
      name: "pull request",
      ok: false,
      detail: `pull request not mergeable: ${mergeStateStatus || "(unknown)"}`,
      next: `gh pr view ${url} --web`
    };
  }

  return {
    name: "pull request",
    ok: true,
    detail: `${url} mergeable ${mergeStateStatus}`
  };
}

function latestReviewSummary(config: PrReadyConfig): ReviewSummary | undefined {
  const root = runsRoot(config);
  if (!existsSync(root)) return undefined;
  let latest: ReviewSummary | undefined;
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const summary = readReviewSummary(config, entry.name);
    if (!summary || !summaryLooksLikeReviewGate(summary.summary)) continue;
    if (!latest || summary.finishedAtMs > latest.finishedAtMs) latest = summary;
  }
  return latest;
}

function readReviewSummary(config: PrReadyConfig, runId: string): ReviewSummary | undefined {
  const path = join(runsRoot(config), runId, "summary.json");
  if (!existsSync(path)) return undefined;
  const summary = readJsonObject(readFileSync(path, "utf8"));
  if (!summary) return undefined;
  return {
    runId: typeof summary.runId === "string" ? summary.runId : runId,
    path,
    finishedAtMs: summaryFinishedAtMs(summary, path),
    summary
  };
}

function summaryLooksLikeReviewGate(summary: Record<string, unknown>): boolean {
  const commandName = typeof summary.commandName === "string" ? summary.commandName : "";
  if (reviewGateCommandNames.has(commandName)) return true;
  if (commandName.includes("dev-gate-review")) return true;
  const command = Array.isArray(summary.command) ? summary.command.join(" ") : "";
  return command.includes("dev gate review");
}

function summaryFinishedAtMs(summary: Record<string, unknown>, path: string): number {
  const finishedAt = typeof summary.finishedAt === "string" ? Date.parse(summary.finishedAt) : NaN;
  return Number.isFinite(finishedAt) ? finishedAt : statSync(path).mtimeMs;
}

function summaryGitHead(summary: Record<string, unknown>): string | undefined {
  if (typeof summary.gitHead === "string") return summary.gitHead;
  if (typeof summary.headSha === "string") return summary.headSha;
  if (typeof summary.head === "string") return summary.head;
  const git = summary.git;
  if (git && typeof git === "object" && !Array.isArray(git)) {
    const gitRecord = git as Record<string, unknown>;
    if (typeof gitRecord.head === "string") return gitRecord.head;
    if (typeof gitRecord.headSha === "string") return gitRecord.headSha;
  }
  return undefined;
}

function renderChecklist(checks: ReadyCheck[]): string {
  const lines = ["PR readiness:"];
  for (const check of checks) {
    lines.push(`- ${check.ok ? "ok" : "fail"} ${check.name}: ${check.detail}`);
  }
  const next = checks.filter((check) => !check.ok && check.next);
  if (next.length > 0) {
    lines.push("Next commands:");
    for (const check of next) {
      lines.push(`- ${check.next}`);
    }
  }
  lines.push(`PR readiness result: ${checks.every((check) => check.ok) ? "ok" : "failed"}`);
  return `${lines.join("\n")}\n`;
}

function renderFailures(failures: ReadyCheck[]): string {
  return failures.map((failure) => `${failure.detail}\n`).join("");
}

function resolvePath(config: PrReadyConfig, path: string): string {
  return path.startsWith("/") ? path : resolve(config.localProjectPath, path);
}

function displayPath(config: PrReadyConfig, path: string): string {
  const prefix = `${resolve(config.localProjectPath)}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}

function runsRoot(config: PrReadyConfig): string {
  return join(config.localProjectPath, config.output.transcript_dir ?? ".pgx-cli/runs");
}

function readJsonObject(text: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(text) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : undefined;
  } catch {
    return undefined;
  }
}

function oneLine(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function shortSha(value: string): string {
  return value.slice(0, 12);
}

function usage(): string {
  return "Usage: pr ready --evidence <path> [--review-run <run-id>] [--allow-no-pr]\n";
}
