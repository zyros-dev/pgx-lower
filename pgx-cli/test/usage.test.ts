import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { incrementUsage, normalizeCommandKey } from "../src/usage.js";

describe("usage counters", () => {
  test("normalizes command keys", () => {
    expect(normalizeCommandKey(["doctor"])).toBe("doctor");
    expect(normalizeCommandKey(["clion", "doctor"])).toBe("clion doctor");
    expect(normalizeCommandKey(["thor", "just", "compile"])).toBe("thor just");
    expect(normalizeCommandKey(["sync", "status"])).toBe("sync status");
  });

  test("increments persisted counters", () => {
    const dir = mkdtempSync(join(tmpdir(), "pgx-cli-usage-"));
    const path = join(dir, "usage.json");

    incrementUsage(path, ["thor", "just", "compile"]);
    incrementUsage(path, ["thor", "just", "test"]);

    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({ "thor just": 2 });
    rmSync(dir, { recursive: true, force: true });
  });
});
