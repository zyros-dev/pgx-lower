import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { checkEvidenceFile, runAgentEvidenceCommand } from "../src/agent-evidence.js";

describe("agent evidence", () => {
  test("check fails when a claim has no green evidence and is not deferred", () => {
    const result = checkEvidenceFile({
      claims: [{ id: "bounded-output", claim: "bounded output", kind: "behavioral-test" }]
    });
    expect(result.ok).toBe(false);
    expect(result.messages).toContain("missing evidence: bounded-output");
  });

  test("check passes for green and explicitly deferred claims", () => {
    const result = checkEvidenceFile({
      claims: [
        {
          id: "bounded-output",
          claim: "bounded output",
          kind: "behavioral-test",
          greenEvidence: "npm test -- managed-runner passed"
        },
        {
          id: "hook-install",
          claim: "Codex hook installed",
          kind: "deferral",
          deferredReason: "blocked on Codex hook rollout"
        }
      ]
    });
    expect(result.ok).toBe(true);
  });

  test("check rejects duplicate, missing, and unknown claim metadata", () => {
    const result = checkEvidenceFile({
      claims: [
        { id: "dup", claim: "first", kind: "behavioral-test", greenEvidence: "passed" },
        { id: "dup", claim: "second", kind: "structural-test", greenEvidence: "built" },
        { id: "", claim: "missing id", kind: "manual-observation", greenEvidence: "observed" },
        { id: "missing-claim", claim: "", kind: "gate", greenEvidence: "review gate passed" },
        { id: "unknown-kind", claim: "bad kind", kind: "smoke-test", greenEvidence: "passed" }
      ]
    });

    expect(result.ok).toBe(false);
    expect(result.messages).toContain("duplicate id: dup");
    expect(result.messages).toContain("missing id");
    expect(result.messages).toContain("missing claim: missing-claim");
    expect(result.messages).toContain("unknown kind: unknown-kind");
  });

  test("check rejects deferrals without reasons", () => {
    const result = checkEvidenceFile({
      claims: [{ id: "manual-smoke", claim: "manual smoke", kind: "deferral" }]
    });

    expect(result.ok).toBe(false);
    expect(result.messages).toContain("missing deferral reason: manual-smoke");
  });

  test("check fails closed when the evidence matrix is empty", () => {
    const result = checkEvidenceFile({ claims: [] });

    expect(result.ok).toBe(false);
    expect(result.messages).toContain("missing evidence claims");
  });

  test("init writes the default ignored evidence file", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-evidence-"));
    try {
      const output = { stdout: "", stderr: "" };
      const exitCode = runAgentEvidenceCommand(
        ["evidence", "init", "--claim", "bounded-output:bounded output"],
        output,
        { localProjectPath: root }
      );
      expect(exitCode).toBe(0);
      expect(output.stdout).toContain(".pgx-cli/evidence/current.json");
      expect(readFileSync(join(root, ".pgx-cli/evidence/current.json"), "utf8")).toContain("bounded-output");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("add and defer update claims in the selected evidence file", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-evidence-"));
    try {
      const file = join(root, "evidence.json");
      const output = { stdout: "", stderr: "" };

      expect(runAgentEvidenceCommand([
        "evidence",
        "init",
        "--file",
        file,
        "--claim",
        "bounded-output:bounded output",
        "--claim",
        "hook-install:hook installed"
      ], output, { localProjectPath: root })).toBe(0);
      expect(runAgentEvidenceCommand([
        "evidence",
        "add",
        "--file",
        file,
        "--id",
        "bounded-output",
        "--green",
        "npm test -- agent-evidence passed",
        "--artifact",
        ".pgx-cli/runs/1/summary.json",
        "--command",
        "npm test -- agent-evidence"
      ], output, { localProjectPath: root })).toBe(0);
      expect(runAgentEvidenceCommand([
        "evidence",
        "defer",
        "--file",
        file,
        "--id",
        "hook-install",
        "--reason",
        "blocked on hook rollout"
      ], output, { localProjectPath: root })).toBe(0);
      expect(runAgentEvidenceCommand(["evidence", "check", "--file", file], output, { localProjectPath: root })).toBe(0);

      const evidence = JSON.parse(readFileSync(file, "utf8")) as {
        claims: Array<{ id: string; greenEvidence?: string; artifact?: string; command?: string; deferredReason?: string }>;
      };
      expect(evidence.claims).toEqual([
        {
          id: "bounded-output",
          claim: "bounded output",
          kind: "behavioral-test",
          greenEvidence: "npm test -- agent-evidence passed",
          artifact: ".pgx-cli/runs/1/summary.json",
          command: "npm test -- agent-evidence"
        },
        {
          id: "hook-install",
          claim: "hook installed",
          kind: "deferral",
          deferredReason: "blocked on hook rollout"
        }
      ]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("commands reject missing required arguments and unknown ids", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-evidence-"));
    try {
      const file = join(root, "evidence.json");
      const output = { stdout: "", stderr: "" };

      expect(runAgentEvidenceCommand(["evidence", "init", "--file", file], output, { localProjectPath: root })).toBe(1);
      expect(output.stderr).toContain("Usage: agent evidence init --claim id:text");

      output.stderr = "";
      expect(runAgentEvidenceCommand([
        "evidence",
        "init",
        "--file",
        file,
        "--claim",
        "bounded-output:bounded output"
      ], output, { localProjectPath: root })).toBe(0);
      expect(runAgentEvidenceCommand([
        "evidence",
        "add",
        "--file",
        file,
        "--id",
        "missing",
        "--green",
        "passed"
      ], output, { localProjectPath: root })).toBe(1);
      expect(output.stderr).toContain("Unknown evidence id: missing");

      output.stderr = "";
      expect(runAgentEvidenceCommand([
        "evidence",
        "defer",
        "--file",
        file,
        "--id",
        "bounded-output"
      ], output, { localProjectPath: root })).toBe(1);
      expect(output.stderr).toContain("Usage: agent evidence defer --id <id> --reason <reason>");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("check command fails closed when the evidence file has no claims", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-evidence-"));
    try {
      const file = join(root, "evidence.json");
      writeFileSync(file, `${JSON.stringify({ claims: [] })}\n`);
      const output = { stdout: "", stderr: "" };

      expect(runAgentEvidenceCommand(["evidence", "check", "--file", file], output, { localProjectPath: root })).toBe(1);
      expect(output.stderr).toContain("missing evidence claims");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
