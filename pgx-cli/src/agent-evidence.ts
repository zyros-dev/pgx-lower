import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import type { OperationOutput } from "./operations.js";

export type EvidenceKind = "behavioral-test" | "structural-test" | "manual-observation" | "gate" | "deferral";

export type EvidenceClaim = {
  id: string;
  claim: string;
  kind: EvidenceKind;
  redEvidence?: string;
  greenEvidence?: string;
  artifact?: string;
  command?: string;
  deferredReason?: string;
  notes?: string;
};

export type EvidenceFile = {
  contract?: string;
  requiredClaims?: string[];
  claims: EvidenceClaim[];
};

export type EvidenceCheckResult = {
  ok: boolean;
  messages: string[];
};

type AgentEvidenceConfig = {
  localProjectPath: string;
};

const evidenceKinds = new Set<string>([
  "behavioral-test",
  "structural-test",
  "manual-observation",
  "gate",
  "deferral"
]);

const defaultEvidencePath = ".pgx-cli/evidence/current.json";

type EvidenceContract = {
  name: string;
  claims: Array<Pick<EvidenceClaim, "id" | "claim" | "kind">>;
};

export const agentHardeningEvidenceContract: EvidenceContract = {
  name: "agent-hardening",
  claims: [
    { id: "policy-rules", claim: "Codex rules block raw high-risk commands and broad shell-wrapper bypasses", kind: "behavioral-test" },
    { id: "policy-wrapper-class", claim: "codex-policy classifier rejects non-latest raw log/IR wrapper variants", kind: "behavioral-test" },
    { id: "evidence-contract", claim: "Evidence and readiness checks enforce the agent hardening contract", kind: "behavioral-test" },
    { id: "strict-preflight", claim: "dev gate review records strict preflight before expensive steps", kind: "behavioral-test" },
    { id: "pgx-cli-tests", claim: "pgx-cli test suite passed", kind: "structural-test" },
    { id: "pgx-cli-build", claim: "pgx-cli build passed", kind: "structural-test" },
    { id: "lint-diff", claim: "pgx-cli dev lint diff passed", kind: "gate" },
    { id: "gate-batch", claim: "pgx-cli dev gate batch passed", kind: "gate" },
    { id: "gate-review", claim: "pgx-cli dev gate review passed", kind: "gate" },
    { id: "dev-qa-loop", claim: "dev/QA review loop completed or was explicitly deferred", kind: "manual-observation" },
    { id: "workflow-skills", claim: "repo-local workflow skills exist or are explicitly deferred", kind: "manual-observation" }
  ]
};

const evidenceContracts = new Map([
  [agentHardeningEvidenceContract.name, agentHardeningEvidenceContract]
]);

export function checkEvidenceFile(
  file: EvidenceFile,
  options: { requireRequiredClaims?: boolean; requiredClaims?: string[] } = {}
): EvidenceCheckResult {
  const messages: string[] = [];
  const seen = new Set<string>();

  if (!file || !Array.isArray(file.claims)) {
    return { ok: false, messages: ["invalid evidence file: claims must be an array"] };
  }
  const requiredClaims = normalizeRequiredClaims(file, options);
  const declaredRequiredClaims = normalizeRequiredClaims(file, {});
  if (options.requireRequiredClaims && declaredRequiredClaims.length === 0) {
    messages.push("evidence requiredClaims missing");
  }
  if (file.claims.length === 0) {
    messages.push("missing evidence claims");
  }

  for (const claim of file.claims) {
    const id = typeof claim.id === "string" ? claim.id.trim() : "";
    const claimText = typeof claim.claim === "string" ? claim.claim.trim() : "";
    const kind = typeof claim.kind === "string" ? claim.kind : "";

    if (!id) {
      messages.push("missing id");
    } else if (seen.has(id)) {
      messages.push(`duplicate id: ${id}`);
    } else {
      seen.add(id);
    }

    if (!claimText) {
      messages.push(`missing claim: ${id || "(missing id)"}`);
    }

    if (!evidenceKinds.has(kind)) {
      messages.push(`unknown kind: ${id || "(missing id)"}`);
      continue;
    }

    if (kind === "deferral") {
      if (!hasText(claim.deferredReason)) {
        messages.push(`missing deferral reason: ${id || "(missing id)"}`);
      }
      continue;
    }

    if (!hasText(claim.greenEvidence)) {
      messages.push(`missing evidence: ${id || "(missing id)"}`);
    }
  }

  for (const requiredId of requiredClaims) {
    if (options.requireRequiredClaims && !declaredRequiredClaims.includes(requiredId)) {
      messages.push(`missing required claim declaration: ${requiredId}`);
    }
    if (!seen.has(requiredId)) {
      messages.push(`missing required evidence: ${requiredId}`);
    }
  }

  return { ok: messages.length === 0, messages };
}

export function runAgentEvidenceCommand(
  argv: string[],
  output: OperationOutput,
  config: AgentEvidenceConfig
): number {
  const [namespace, command, ...rest] = argv;
  if (namespace !== "evidence") {
    output.stderr += evidenceUsage();
    return 1;
  }

  try {
    if (command === "init") {
      return runInit(rest, output, config);
    }
    if (command === "add") {
      return runAdd(rest, output, config);
    }
    if (command === "defer") {
      return runDefer(rest, output, config);
    }
    if (command === "check") {
      return runCheck(rest, output, config);
    }

    output.stderr += evidenceUsage();
    return 1;
  } catch (error) {
    output.stderr += `${error instanceof Error ? error.message : String(error)}\n`;
    return 1;
  }
}

function runInit(args: string[], output: OperationOutput, config: AgentEvidenceConfig): number {
  const parsed = parseOptions(args, new Set(["--file", "--claim", "--contract"]));
  if (!parsed) {
    output.stderr += "Usage: agent evidence init [--contract agent-hardening | --claim id:text ...] [--file <path>]\n";
    return 1;
  }
  const contractName = parsed.values.contract?.[0];
  if (contractName) {
    const contract = findEvidenceContract(contractName);
    if (!contract) {
      output.stderr += `Unknown evidence contract: ${contractName}\n`;
      return 1;
    }
    const evidence = evidenceFromContract(contract);
    const path = evidencePath(config, parsed.file);
    writeEvidence(path, evidence);
    output.stdout += `Wrote ${displayPath(config, path)}\n`;
    return 0;
  }

  const claimSpecs = parsed?.values.claim ?? [];
  if (!parsed || claimSpecs.length === 0) {
    output.stderr += "Usage: agent evidence init [--contract agent-hardening | --claim id:text ...] [--file <path>]\n";
    return 1;
  }

  const claims = claimSpecs.map(parseClaimSpec);
  if (claims.some((claim) => !claim)) {
    output.stderr += "Usage: agent evidence init [--contract agent-hardening | --claim id:text ...] [--file <path>]\n";
    return 1;
  }

  const evidence: EvidenceFile = {
    requiredClaims: claims.map((claim) => claim!.id),
    claims: claims.map((claim) => ({
      id: claim!.id,
      claim: claim!.claim,
      kind: "behavioral-test"
    }))
  };
  const check = checkEvidenceFile(evidence);
  const metadataErrors = check.messages.filter((message) => !message.startsWith("missing evidence: "));
  if (metadataErrors.length > 0) {
    output.stderr += `${metadataErrors.join("\n")}\n`;
    return 1;
  }

  const path = evidencePath(config, parsed.file);
  writeEvidence(path, evidence);
  output.stdout += `Wrote ${displayPath(config, path)}\n`;
  return 0;
}

function runAdd(args: string[], output: OperationOutput, config: AgentEvidenceConfig): number {
  const parsed = parseOptions(args, new Set(["--file", "--id", "--green", "--artifact", "--command"]));
  const id = parsed?.values.id?.[0];
  const greenEvidence = parsed?.values.green?.[0];
  if (!parsed || !hasText(id) || !hasText(greenEvidence)) {
    output.stderr += "Usage: agent evidence add --id <id> --green <evidence> [--artifact <path>] [--command <command>] [--file <path>]\n";
    return 1;
  }

  const path = evidencePath(config, parsed.file);
  const evidence = readEvidence(path);
  const claim = evidence.claims.find((candidate) => candidate.id === id);
  if (!claim) {
    output.stderr += `Unknown evidence id: ${id}\n`;
    return 1;
  }

  claim.greenEvidence = greenEvidence;
  claim.artifact = parsed.values.artifact?.[0] ?? claim.artifact;
  claim.command = parsed.values.command?.[0] ?? claim.command;
  writeEvidence(path, evidence);
  output.stdout += `Updated ${id} in ${displayPath(config, path)}\n`;
  return 0;
}

function runDefer(args: string[], output: OperationOutput, config: AgentEvidenceConfig): number {
  const parsed = parseOptions(args, new Set(["--file", "--id", "--reason"]));
  const id = parsed?.values.id?.[0];
  const reason = parsed?.values.reason?.[0];
  if (!parsed || !hasText(id) || !hasText(reason)) {
    output.stderr += "Usage: agent evidence defer --id <id> --reason <reason> [--file <path>]\n";
    return 1;
  }

  const path = evidencePath(config, parsed.file);
  const evidence = readEvidence(path);
  const claim = evidence.claims.find((candidate) => candidate.id === id);
  if (!claim) {
    output.stderr += `Unknown evidence id: ${id}\n`;
    return 1;
  }

  claim.kind = "deferral";
  claim.deferredReason = reason;
  writeEvidence(path, evidence);
  output.stdout += `Deferred ${id} in ${displayPath(config, path)}\n`;
  return 0;
}

function runCheck(args: string[], output: OperationOutput, config: AgentEvidenceConfig): number {
  const parsed = parseCheckOptions(args);
  if (!parsed) {
    output.stderr += "Usage: agent evidence check [--contract agent-hardening|--require-required-claims] [--file <path>]\n";
    return 1;
  }
  const contract = parsed.contract ? findEvidenceContract(parsed.contract) : undefined;
  if (parsed.contract && !contract) {
    output.stderr += `Unknown evidence contract: ${parsed.contract}\n`;
    return 1;
  }

  const path = evidencePath(config, parsed.file);
  const result = checkEvidenceFile(readEvidence(path), {
    requireRequiredClaims: parsed.requireRequiredClaims || contract !== undefined,
    requiredClaims: contract?.claims.map((claim) => claim.id)
  });
  if (result.ok) {
    output.stdout += "agent evidence check: ok\n";
    return 0;
  }

  for (const message of result.messages) {
    output.stderr += `${message}\n`;
  }
  return 1;
}

function parseClaimSpec(spec: string): { id: string; claim: string } | undefined {
  const separator = spec.indexOf(":");
  if (separator === -1) return undefined;
  const id = spec.slice(0, separator).trim();
  const claim = spec.slice(separator + 1).trim();
  if (!id || !claim) return undefined;
  return { id, claim };
}

function parseOptions(args: string[], allowed: Set<string>): { file?: string; values: Record<string, string[]> } | undefined {
  const values: Record<string, string[]> = {};
  let file: string | undefined;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (!allowed.has(arg)) {
      return undefined;
    }
    const value = args[index + 1];
    if (!value) {
      return undefined;
    }
    index += 1;

    if (arg === "--file") {
      file = value;
      continue;
    }

    const key = arg.slice(2);
    values[key] ??= [];
    values[key].push(value);
  }

  return { file, values };
}

function parseCheckOptions(args: string[]): { file?: string; requireRequiredClaims: boolean; contract?: string } | undefined {
  let file: string | undefined;
  let requireRequiredClaims = false;
  let contract: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--require-required-claims") {
      requireRequiredClaims = true;
      continue;
    }
    if (arg === "--file") {
      const value = args[index + 1];
      if (!value) return undefined;
      file = value;
      index += 1;
      continue;
    }
    if (arg === "--contract") {
      const value = args[index + 1];
      if (!value) return undefined;
      contract = value;
      index += 1;
      continue;
    }
    return undefined;
  }
  return { file, requireRequiredClaims, contract };
}

function evidencePath(config: AgentEvidenceConfig, overridePath: string | undefined): string {
  if (!overridePath) {
    return join(config.localProjectPath, defaultEvidencePath);
  }
  return resolve(config.localProjectPath, overridePath);
}

function displayPath(config: AgentEvidenceConfig, path: string): string {
  const defaultPath = join(config.localProjectPath, defaultEvidencePath);
  return path === defaultPath ? defaultEvidencePath : path;
}

function readEvidence(path: string): EvidenceFile {
  if (!existsSync(path)) {
    throw new Error(`Evidence file not found: ${path}`);
  }

  const parsed = JSON.parse(readFileSync(path, "utf8")) as EvidenceFile;
  if (!parsed || !Array.isArray(parsed.claims)) {
    throw new Error(`Invalid evidence file: ${path}`);
  }
  return parsed;
}

function writeEvidence(path: string, evidence: EvidenceFile): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(evidence, null, 2)}\n`);
}

function findEvidenceContract(name: string): EvidenceContract | undefined {
  return evidenceContracts.get(name);
}

function evidenceFromContract(contract: EvidenceContract): EvidenceFile {
  return {
    contract: contract.name,
    requiredClaims: contract.claims.map((claim) => claim.id),
    claims: contract.claims.map((claim) => ({ ...claim }))
  };
}

function hasText(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function normalizeRequiredClaims(
  file: EvidenceFile,
  options: { requiredClaims?: string[] }
): string[] {
  const required = options.requiredClaims ?? file.requiredClaims ?? [];
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const value of required) {
    if (typeof value !== "string") continue;
    const id = value.trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    ids.push(id);
  }
  return ids;
}

function evidenceUsage(): string {
  return [
    "Usage: agent evidence <init|add|defer|check>",
    "  agent evidence init [--contract agent-hardening | --claim id:text ...] [--file <path>]",
    "  agent evidence add --id <id> --green <evidence> [--artifact <path>] [--command <command>] [--file <path>]",
    "  agent evidence defer --id <id> --reason <reason> [--file <path>]",
    "  agent evidence check [--contract agent-hardening|--require-required-claims] [--file <path>]",
    ""
  ].join("\n");
}
