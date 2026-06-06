import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { runRgCommand } from "../src/search.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return { exitCode: 0, stdout: "match\n", stderr: "" };
  }
}

describe("rg command", () => {
  test("runs ripgrep through a bounded transcript wrapper", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runRgCommand(
      ["--lines", "25", "--glob", "*.md", "primitive type", "/tmp/wiki specs"],
      runner,
      output,
      { localProjectPath: "/Users/nickvandermerwe/repos/pgx-lower" }
    );

    expect(exitCode).toBe(0);
    expect(output.stdout).toBe("match\n");
    expect(runner.calls).toHaveLength(1);
    expect(runner.calls[0].command).toBe("bash");
    expect(runner.calls[0].args[0]).toBe("-lc");
    expect(runner.calls[0].args[1]).toContain("cd /Users/nickvandermerwe/repos/pgx-lower");
    expect(runner.calls[0].args[1]).toContain("rg -n --color never --glob '*.md' 'primitive type' '/tmp/wiki specs'");
    expect(runner.calls[0].args[1]).toContain("sed -n '1,25p'");
    expect(runner.calls[0].args[1]).toContain("full transcript:");
  });

  test("rejects invalid line counts", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runRgCommand(["--lines", "0", "pattern"], runner, output, {
      localProjectPath: "/repo"
    });

    expect(exitCode).toBe(1);
    expect(runner.calls).toEqual([]);
    expect(output.stderr).toContain("Usage: rg");
  });
});
