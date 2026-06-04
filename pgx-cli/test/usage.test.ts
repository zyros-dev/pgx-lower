import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { incrementUsage, normalizeCommandKey } from "../src/usage.js";

describe("usage counters", () => {
  test("normalizes command keys", () => {
    expect(normalizeCommandKey(["doctor"])).toBe("doctor");
    expect(normalizeCommandKey(["clion", "doctor"])).toBe("clion doctor");
    expect(normalizeCommandKey(["dev", "lint", "diff"])).toBe("dev lint");
    expect(normalizeCommandKey(["sync", "status"])).toBe("sync status");
  });

  test("increments persisted counters", () => {
    const dir = mkdtempSync(join(tmpdir(), "pgx-cli-usage-"));
    const path = join(dir, "usage.json");

    incrementUsage(path, ["dev", "lint", "diff"]);
    incrementUsage(path, ["dev", "lint", "file", "src/pgx-lower/example.cpp"]);

    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({ "dev lint": 2 });
    rmSync(dir, { recursive: true, force: true });
  });
});
