import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, test } from "vitest";
import {
  classifyAgentCommand,
  discoverWorkflowScripts,
  renderDefaultRules,
  runCodexPolicyCommand
} from "../src/codex-policy.js";

describe("codex policy", () => {
  test.each([
    ["ssh", "pgx-cli run thor -- ..."],
    ["docker", "pgx-cli run docker -- ..."],
    ["psql", "pgx-cli run psql ..."],
    ["pg_regress", "pgx-cli dev test ..."],
    ["ctest", "pgx-cli dev test ..."],
    ["cmake", "pgx-cli dev build ..."],
    ["ninja", "pgx-cli dev build ..."],
    ["tsp", "pgx-cli queue ..."],
    ["mutagen", "pgx-cli sync ..."],
    ["just", "pgx-cli ..."]
  ])("forbids raw %s", (binary, replacement) => {
    expect(classifyAgentCommand([binary, "x"])).toMatchObject({
      decision: "forbidden",
      replacement
    });
  });

  test.each([
    [["bash", "-lc", "ssh comfy true"], "pgx-cli run thor -- ..."],
    [["zsh", "-c", "docker exec pgx-lower-dev true"], "pgx-cli run docker -- ..."],
    [["sh", "-c", "psql -c 'SELECT 1'"], "pgx-cli run psql ..."],
    [["bash", "-lc", "cat /tmp/pgx_errors.log"], "pgx-cli logs errors"],
    [["bash", "-lc", "cat /tmp/pgx_ir/latest.mlir"], "pgx-cli ir inspect"],
    [["bash", "-lc", "find /tmp/pgx_ir -type f -print"], "pgx-cli ir inspect"],
    [["cat", "/tmp/pgx_errors.log"], "pgx-cli logs errors"],
    [["tail", "-n", "200", "/tmp/pgx_ir/latest.mlir"], "pgx-cli ir inspect"],
    [["gh", "pr", "comment", "1", "--body", "`pgx-cli dev gate review`"], "gh pr comment --body-file"]
  ])("forbids opaque unsafe wrapper %#", (argv, replacement) => {
    expect(classifyAgentCommand(argv)).toMatchObject({
      decision: "forbidden",
      replacement
    });
  });

  test.each([
    ["pgx-cli", "run", "thor", "--", "true"],
    ["bash", "-lc", "pgx-cli run thor -- true"],
    ["rg", "needle"],
    ["sed", "-n", "1,5p", "file"],
    ["find", "."],
    ["ls"],
    ["pwd"],
    ["git", "status", "--short"],
    ["git", "diff"],
    ["git", "show"],
    ["nl", "-ba", "file"],
    ["wc", "-l", "file"]
  ])("allows local inspection or pgx-cli command %s", (...argv) => {
    expect(classifyAgentCommand(argv).decision).toBe("allow");
  });

  test("discovers workflow scripts and renders deterministic rules", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-policy-"));
    try {
      mkdirSync(join(root, "scripts"), { recursive: true });
      mkdirSync(join(root, "tools", "scripts"), { recursive: true });
      mkdirSync(join(root, "benchmark", "tpch"), { recursive: true });
      writeFileSync(join(root, "scripts", "a.sh"), "");
      writeFileSync(join(root, "tools", "scripts", "b.py"), "");
      writeFileSync(join(root, "benchmark", "c.py"), "");
      writeFileSync(join(root, "benchmark", "tpch", "run.py"), "");

      const scripts = discoverWorkflowScripts(root);
      expect(scripts).toEqual([
        "benchmark/c.py",
        "benchmark/tpch/aggregate.py",
        "benchmark/tpch/fxt_to_flamegraph.py",
        "benchmark/tpch/metrics_collector.py",
        "benchmark/tpch/run.py",
        "scripts/a.sh",
        "tools/scripts/b.py"
      ]);

      for (const script of scripts) {
        expect(classifyAgentCommand([script], { workflowScripts: scripts }).decision).toBe("forbidden");
        expect(classifyAgentCommand([`./${script}`], { workflowScripts: scripts }).decision).toBe("forbidden");
      }

      const rules = renderDefaultRules({ root, workflowScripts: scripts });
      expect(rules).toContain('pattern = ["ssh"]');
      expect(rules).toContain('match: ssh comfy true');
      expect(rules).toContain('pattern = ["bash", "-lc"]');
      expect(rules).toContain('pattern = ["bash", "-c"]');
      expect(rules).toContain('pattern = ["sh", "-c"]');
      expect(rules).toContain('pattern = ["zsh", "-lc"]');
      expect(rules).toContain('pattern = ["zsh", "-c"]');
      expect(rules).toContain('pattern = ["cat"]');
      expect(rules).toContain('pattern = ["benchmark/tpch/aggregate.py"]');
      expect(rules).toContain('pattern = ["./scripts/a.sh"]');
      expect(rules).not.toMatch(/\n\n$/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test.each([
    "phase3.mlir",
    "latest.mlir"
  ])("rendered rules block the raw IR dump class for %s", (filename) => {
    const rules = renderDefaultRules();

    expect(rules).toContain('pattern = ["bash", "-lc"]');
    expect(rules).not.toContain(`pattern = ["bash", "-lc", "cat /tmp/pgx_ir/${filename}"]`);
    expect(classifyAgentCommand(["bash", "-lc", `cat /tmp/pgx_ir/${filename}`])).toMatchObject({
      decision: "forbidden",
      replacement: "pgx-cli ir inspect"
    });
  });

  test("check command returns non-zero for forbidden commands", () => {
    const output = { stdout: "", stderr: "" };

    expect(runCodexPolicyCommand(["check", "--", "ssh", "comfy", "true"], output)).toBe(1);
    expect(output.stdout).toContain("forbidden");
  });

  test("committed rule file matches rendered policy", () => {
    const repoRoot = resolve(new URL("../../", import.meta.url).pathname);
    const rulesPath = join(repoRoot, ".codex", "rules", "default.rules");

    expect(readFileSync(rulesPath, "utf8")).toBe(renderDefaultRules({ root: repoRoot }));
  });
});
