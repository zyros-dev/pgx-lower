import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { renderBufferedOutput } from "../src/output.js";

describe("renderBufferedOutput", () => {
  test("leaves short output unchanged", () => {
    const rendered = renderBufferedOutput(
      { stdout: "one\ntwo\n", stderr: "warn\n" },
      { maxLines: 5 }
    );

    expect(rendered).toEqual({ stdout: "one\ntwo\n", stderr: "warn\n" });
  });

  test("caps long streams and writes a full transcript", () => {
    const dir = mkdtempSync(join(tmpdir(), "pgx-cli-output-"));
    const transcriptPath = join(dir, "transcript.log");
    const stdout = Array.from({ length: 8 }, (_, index) => `out ${index + 1}`).join("\n") + "\n";
    const stderr = Array.from({ length: 7 }, (_, index) => `err ${index + 1}`).join("\n") + "\n";

    const rendered = renderBufferedOutput(
      { stdout, stderr },
      { maxLines: 5, transcriptPath }
    );

    expect(rendered.stdout).toBe("out 1\nout 2\n[... omitted 4 lines ...]\nout 7\nout 8\n");
    expect(rendered.stderr).toContain("err 1\nerr 2\n[... omitted 3 lines ...]\nerr 6\nerr 7\n");
    expect(rendered.stderr).toContain(`pgx-cli: output truncated to 5 lines per stream; full transcript: ${transcriptPath}`);
    expect(readFileSync(transcriptPath, "utf8")).toContain(stdout);
    expect(readFileSync(transcriptPath, "utf8")).toContain(stderr);
  });
});
