import { writeFileSync } from "node:fs";
import { appendArtifactPath, createRunArtifactPaths, writeRunSummary } from "./run-artifacts.js";
import { writeCommand } from "./run-artifacts.js";
import { ManagedCommandRunner } from "./managed-runner.js";
import { applyTotalOutputBudget } from "./managed-runner.js";
import type { ManagedRunPostprocessor, OutputBudget, PreviewSelection } from "./managed-runner.js";
import { runMutagenPreflight } from "./mutagen-preflight.js";
import type { StreamingCommandRunner } from "./commands.js";
import type { OperationConfig, OperationOutput } from "./operations.js";
import type { ResolvedOutputConfig, ResolvedSyncConfig } from "./project-config.js";
import type { RunArtifactPaths } from "./run-artifacts.js";

export type ManagedOperationConfig = OperationConfig & {
  localProjectPath: string;
  dockerContainer?: string;
  sync: ResolvedSyncConfig;
  output: ResolvedOutputConfig;
};

export type ManagedOperationRunner = StreamingCommandRunner;

export type ManagedOperationResult = {
  childExitCode: number;
  workflowExitCode: number;
  managedRun: Awaited<ReturnType<ManagedCommandRunner["runManaged"]>>;
  artifact: RunArtifactPaths;
};

export type ManagedOperationPostprocessor = ManagedRunPostprocessor;

export async function runManagedRemoteShell(input: {
  runner: ManagedOperationRunner;
  output: OperationOutput;
  config: ManagedOperationConfig;
  commandName: string;
  shellCommand: string;
  requireMutagenProof: boolean;
  postprocess?: ManagedOperationPostprocessor;
  fullOutput?: boolean;
  preview?: PreviewSelection;
  metadata?: Record<string, unknown>;
  artifactPaths?: string[];
}): Promise<ManagedOperationResult> {
  const artifact = createRunArtifactPaths({
    root: input.config.localProjectPath,
    commandName: input.commandName
  });
  input.output.stdout += `pgx-cli: starting ${input.commandName}\n`;
  input.output.stdout += `run id: ${artifact.runId}\n`;
  const startedAt = new Date();
  const artifactPaths = input.artifactPaths ?? [];
  for (const path of artifactPaths) {
    appendArtifactPath(artifact, path);
  }
  writeFileSync(artifact.artifactsPath, "", { flag: "a" });

  const command = input.config.runningOnRemote ? "bash" : "ssh";
  const remoteShell = `export PATH=$HOME/.local/bin:$PATH && cd ${quoteShell(input.config.remoteProjectPath)} && ${input.shellCommand}`;
  const args = input.config.runningOnRemote
    ? ["-c", remoteShell]
    : [input.config.sshHost, "bash", "-c", quoteShell(remoteShell)];
  const target = {
    kind: input.config.runningOnRemote ? "thor-local" : "ssh",
    ...(input.config.runningOnRemote ? {} : { sshHost: input.config.sshHost }),
    remoteProjectPath: input.config.remoteProjectPath,
    ...(input.config.dockerContainer ? { dockerContainer: input.config.dockerContainer } : {})
  };
  const project = {
    localProjectPath: input.config.localProjectPath,
    remoteProjectPath: input.config.remoteProjectPath,
    mutagenSession: input.config.mutagenSession
  };
  writeCommand(artifact, [command, ...args]);

  const preflight = await runMutagenPreflight({
    runner: input.runner,
    artifact,
    sessionName: input.config.mutagenSession,
    localProjectPath: input.config.localProjectPath,
    remoteProjectPath: input.config.remoteProjectPath,
    sshHost: input.config.sshHost,
    runId: artifact.runId,
    flushTimeoutSeconds: input.config.sync.flush_timeout_seconds,
    runningOnRemote: input.config.runningOnRemote,
    proofPath: input.config.sync.proof.path,
    requireProof: input.requireMutagenProof && input.config.sync.proof.enabled
  });

  if (!preflight.ok) {
    const finishedAt = new Date();
    const blockedText = [
      `pgx-cli: blocked before running ${input.commandName}`,
      `reason: ${preflight.message}`,
      "next:",
      ...preflight.next.map((next) => `- ${next}`),
      `transcript: ${artifact.combinedPath}`,
      `sync preflight: ${preflight.transcriptPath ?? artifact.syncPreflightPath}`,
      ""
    ].join("\n");
    writeFileSync(artifact.stdoutPath, "");
    writeFileSync(artifact.stderrPath, blockedText);
    writeFileSync(artifact.combinedPath, blockedText);
    const failureSummary = {
      kind: "mutagen",
      lines: [
        `mutagen: ${preflight.message}`,
        `transcript: ${artifact.combinedPath}`,
        `sync preflight: ${preflight.transcriptPath ?? artifact.syncPreflightPath}`
      ]
    };
    writeRunSummary(artifact, {
      runId: artifact.runId,
      commandName: input.commandName,
      command: [command, ...args],
      cwd: process.cwd(),
      startedAt: startedAt.toISOString(),
      finishedAt: finishedAt.toISOString(),
      durationMs: finishedAt.getTime() - startedAt.getTime(),
      childExitCode: 1,
      workflowExitCode: 1,
      timedOut: false,
      truncated: false,
      transcripts: {
        stdoutPath: artifact.stdoutPath,
        stderrPath: artifact.stderrPath,
        combinedPath: artifact.combinedPath
      },
      artifacts: {
        registryPath: artifact.artifactsPath,
        commandPath: artifact.commandPath,
        runDir: artifact.runDir
      },
      failureSummary,
      ...(input.metadata ?? {}),
      mutagenPreflight: preflight,
      target,
      project,
      artifactPaths
    });
    input.output.stderr += blockedText;
    const managedRun = {
      childExitCode: 1,
      workflowExitCode: 1,
      stdoutPreview: "",
      stderrPreview: blockedText,
      combinedPreview: blockedText,
      truncated: false,
      timedOut: false,
      artifact,
      failureSummary
    };
    return { childExitCode: 1, workflowExitCode: 1, managedRun, artifact };
  }

  const managedRun = await new ManagedCommandRunner(input.runner).runManaged({
    command,
    args,
    root: input.config.localProjectPath,
    commandName: input.commandName,
    artifact,
    budget: outputBudget(input.config.output, input.fullOutput),
    preview: input.fullOutput ? undefined : input.preview,
    postprocess: input.postprocess,
    summary: {
      ...(input.metadata ?? {}),
      mutagenPreflight: preflight,
      target,
      project,
      artifactPaths
    }
  });

  const rendered = applyTotalOutputBudget(
    [
      managedRun.combinedPreview,
      `exit: ${managedRun.workflowExitCode} (child: ${managedRun.childExitCode})\n`,
      `transcript: ${managedRun.artifact.combinedPath}\n`
    ],
    input.fullOutput ? Number.MAX_SAFE_INTEGER : input.config.output.max_lines_total
  );
  input.output.stdout += rendered.text;
  return {
    childExitCode: managedRun.childExitCode,
    workflowExitCode: managedRun.workflowExitCode,
    managedRun,
    artifact
  };
}

function outputBudget(output: ResolvedOutputConfig, fullOutput?: boolean): OutputBudget {
  if (fullOutput) {
    return {
      maxLinesPerStream: Number.MAX_SAFE_INTEGER,
      maxLinesTotal: Number.MAX_SAFE_INTEGER,
      successTailLines: Number.MAX_SAFE_INTEGER,
      failureTailLines: Number.MAX_SAFE_INTEGER
    };
  }
  return {
    maxLinesPerStream: output.max_lines_per_step,
    maxLinesTotal: output.max_lines_total,
    successTailLines: output.success_tail_lines,
    failureTailLines: output.failure_tail_lines
  };
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
