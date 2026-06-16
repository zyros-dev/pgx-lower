import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { afterEach, describe, expect, test } from "vitest";
import { renderDefaultRules } from "../src/codex-policy.js";

const packageRoot = resolve(new URL("..", import.meta.url).pathname);
const tsxBin = join(packageRoot, "node_modules", ".bin", process.platform === "win32" ? "tsx.cmd" : "tsx");
const indexPath = join(packageRoot, "src", "index.ts");
const children: Array<ReturnType<typeof spawn>> = [];

afterEach(async () => {
  await Promise.all(children.splice(0).map((child) => stopChild(child)));
});

describe("CLI live output", () => {
  test("codex-policy rules emits full generated rules through the top-level CLI", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-policy-rules-"));
    try {
      const result = await runCli(["codex-policy", "rules"], root);

      expect(result.exitCode).toBe(0);
      expect(result.stderr).toBe("");
      expect(result.stdout).not.toContain("[... omitted");
      expect(result.stdout).toBe(renderDefaultRules({ root }));
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("managed commands print start line before a sleeping payload completes", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-live-output-"));
    try {
      const child = spawn(tsxBin, [indexPath, "run", "thor", "--", "sleep", "2"], {
        cwd: root,
        env: {
          ...process.env,
          PGX_LOCAL_PROJECT_PATH: root,
          PGX_REMOTE_PROJECT_PATH: root
        },
        stdio: ["ignore", "pipe", "pipe"]
      });
      children.push(child);

      const stdout = await waitForStdout(child, /pgx-cli: starting run-thor-sleep[\s\S]*run id:/, 5000);

      expect(stdout).toContain("pgx-cli: starting run-thor-sleep");
      expect(stdout).toContain("run id:");
      expect(stdout).not.toContain("exit:");
    } finally {
      await Promise.all(children.splice(0).map((child) => stopChild(child)));
      rmSync(root, { recursive: true, force: true });
    }
  });
});

function runCli(args: string[], cwd: string): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  return new Promise((resolvePromise, reject) => {
    let stdout = "";
    let stderr = "";
    const child = spawn(tsxBin, [indexPath, ...args], {
      cwd,
      env: {
        ...process.env,
        PGX_LOCAL_PROJECT_PATH: cwd,
        PGX_REMOTE_PROJECT_PATH: cwd
      },
      stdio: ["ignore", "pipe", "pipe"]
    });
    children.push(child);
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      children.splice(children.indexOf(child), 1);
      resolvePromise({ exitCode: code ?? 1, stdout, stderr });
    });
  });
}

function waitForStdout(child: ReturnType<typeof spawn>, pattern: RegExp, timeoutMs: number): Promise<string> {
  let stdout = "";
  return new Promise((resolvePromise, reject) => {
    const timer = setTimeout(() => reject(new Error(`timed out waiting for stdout; saw: ${stdout}`)), timeoutMs);
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
      if (pattern.test(stdout)) {
        clearTimeout(timer);
        resolvePromise(stdout);
      }
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`process exited before start line; code ${code}; stdout: ${stdout}`));
    });
  });
}

async function stopChild(child: ReturnType<typeof spawn>): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  child.kill("SIGTERM");
  await new Promise<void>((resolvePromise) => child.once("close", () => resolvePromise()));
}
