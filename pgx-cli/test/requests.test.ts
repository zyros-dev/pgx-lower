import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { writeRequest } from "../src/requests.js";

describe("request inbox", () => {
  test("writes feature request markdown", () => {
    const dir = mkdtempSync(join(tmpdir(), "pgx-cli-requests-"));
    const path = writeRequest(
      dir,
      "feature",
      ["make", "gate", "better"],
      new Date("2026-06-04T00:00:00Z")
    );

    const files = readdirSync(dir);
    expect(files).toHaveLength(1);
    expect(path).toContain("feature");
    const text = readFileSync(path, "utf8");
    expect(text).toContain("Timestamp: 2026-06-04T00:00:00.000Z");
    expect(text).toContain("Kind: feature");
    expect(text).toContain("make gate better");

    rmSync(dir, { recursive: true, force: true });
  });
});
