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
    ["pgx-cli", "run", "thor", "--", "true"],
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
      expect(scripts).toEqual(["benchmark/c.py", "benchmark/tpch/run.py", "scripts/a.sh", "tools/scripts/b.py"]);

      for (const script of scripts) {
        expect(classifyAgentCommand([script], { workflowScripts: scripts }).decision).toBe("forbidden");
        expect(classifyAgentCommand([`./${script}`], { workflowScripts: scripts }).decision).toBe("forbidden");
      }

	      const rules = renderDefaultRules({ root, workflowScripts: scripts });
	      expect(rules).toContain('pattern = ["ssh"]');
	      expect(rules).toContain('match: ssh comfy true');
	      expect(rules).toContain('pattern = ["./scripts/a.sh"]');
	      expect(rules).not.toMatch(/\n\n$/);
	    } finally {
      rmSync(root, { recursive: true, force: true });
    }
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
