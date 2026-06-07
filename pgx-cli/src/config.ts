import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, relative, resolve } from "node:path";
import { loadProjectConfig } from "./project-config.js";
import type { OutputConfig, ProjectConfig, ResolvedOutputConfig, ResolvedSyncConfig, SyncConfig } from "./project-config.js";

export const DEFAULT_URL = "http://127.0.0.1:64343/stream";
export const DEFAULT_REMOTE_URL = "http://127.0.0.1:64342/stream";
export const DEFAULT_PROJECT_PATH = "/home/zel/repos/pgx-lower";
export const DEFAULT_SSH_HOST = "comfy";
export const DEFAULT_LOCAL_PROJECT_PATH = "/Users/nickvandermerwe/repos/pgx-lower";
export const DEFAULT_REMOTE_PROJECT_PATH = "/home/zel/repos/pgx-lower";
export const DEFAULT_MUTAGEN_SESSION = "pgx-lower";
export const DEFAULT_DOCKER_CONTAINER = "pgx-lower-dev";
export const DEFAULT_BUILD_QUEUE = "pgx-build";
export const DEFAULT_CHECK_QUEUE = "pgx-check";
export const DEFAULT_CONFIG_PATH = join(homedir(), ".config", "pgx-cli", "config.json");
export const DEFAULT_SYNC_CONFIG: ResolvedSyncConfig = {
  required_for_remote: true,
  flush_timeout_seconds: 45,
  proof: {
    enabled: true,
    path: ".pgx-cli/sync-probes",
    required_for: ["build", "test", "lint", "psql", "pg_regress", "bench", "profile", "run"]
  }
};
export const DEFAULT_OUTPUT_CONFIG: ResolvedOutputConfig = {
  mode: "agent",
  transcript_dir: ".pgx-cli/runs",
  max_lines_per_step: 80,
  max_lines_total: 180,
  failure_tail_lines: 60,
  success_tail_lines: 20,
  progress: "final-summary",
  full_output_requires_flag: true
};

export type Config = {
  url: string;
  projectPath: string;
  sshHost: string;
  remoteUrl: string;
  localProjectPath: string;
  remoteProjectPath: string;
  mutagenSession: string;
  dockerContainer: string;
  buildQueue: string;
  checkQueue: string;
  sync: ResolvedSyncConfig;
  output: ResolvedOutputConfig;
  runningOnRemote?: boolean;
};

type LoadConfigInput = {
  env: NodeJS.ProcessEnv;
  argvUrl?: string;
  readConfigFile?: () => Partial<Config> | undefined;
  readProjectConfig?: () => ProjectConfig | undefined;
  cwd?: string;
};

export function readDefaultConfigFile(): Partial<Config> | undefined {
  if (!existsSync(DEFAULT_CONFIG_PATH)) {
    return undefined;
  }

  return JSON.parse(readFileSync(DEFAULT_CONFIG_PATH, "utf8")) as Partial<Config>;
}

export function loadConfig(input: LoadConfigInput): Config {
  const fileConfig = input.readConfigFile?.() ?? readDefaultConfigFile();
  const projectConfig = input.readProjectConfig?.() ?? loadProjectConfig();
  const projectRemote = projectConfig?.remote;
  const projectQueues = projectConfig?.queues;
  const remoteProjectPath =
    fileConfig?.remoteProjectPath ?? input.env.PGX_REMOTE_PROJECT_PATH ?? projectRemote?.path ?? DEFAULT_REMOTE_PROJECT_PATH;
  const runningOnRemote = isPathWithin(remoteProjectPath, input.cwd ?? process.cwd());

  return {
    url: input.argvUrl ?? fileConfig?.url ?? input.env.CLION_MCP_URL ?? DEFAULT_URL,
    projectPath: fileConfig?.projectPath ?? input.env.CLION_PROJECT_PATH ?? projectRemote?.path ?? DEFAULT_PROJECT_PATH,
    sshHost: fileConfig?.sshHost ?? input.env.CLION_MCP_SSH_HOST ?? projectRemote?.host ?? DEFAULT_SSH_HOST,
    remoteUrl: fileConfig?.remoteUrl ?? input.env.CLION_MCP_REMOTE_URL ?? DEFAULT_REMOTE_URL,
    localProjectPath:
      fileConfig?.localProjectPath ??
      input.env.PGX_LOCAL_PROJECT_PATH ??
      (runningOnRemote ? remoteProjectPath : DEFAULT_LOCAL_PROJECT_PATH),
    remoteProjectPath,
    mutagenSession:
      fileConfig?.mutagenSession ?? input.env.PGX_MUTAGEN_SESSION ?? projectRemote?.mutagen_session ?? DEFAULT_MUTAGEN_SESSION,
    dockerContainer:
      fileConfig?.dockerContainer ?? input.env.PGX_DOCKER_CONTAINER ?? projectRemote?.docker_container ?? DEFAULT_DOCKER_CONTAINER,
    buildQueue: fileConfig?.buildQueue ?? input.env.PGX_BUILD_QUEUE ?? projectQueues?.build ?? DEFAULT_BUILD_QUEUE,
    checkQueue: fileConfig?.checkQueue ?? input.env.PGX_CHECK_QUEUE ?? projectQueues?.check ?? DEFAULT_CHECK_QUEUE,
    sync: resolveSyncConfig(projectConfig?.sync, fileConfig?.sync),
    output: resolveOutputConfig(projectConfig?.output, fileConfig?.output),
    runningOnRemote
  };
}

export function writeConfig(path: string, config: Config): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, `${JSON.stringify(config, null, 2)}\n`);
}

function isPathWithin(parent: string, child: string): boolean {
  const rel = relative(resolve(parent), resolve(child));
  return rel === "" || (!!rel && !rel.startsWith("..") && !isAbsolute(rel));
}

function resolveSyncConfig(project?: SyncConfig, personal?: SyncConfig): ResolvedSyncConfig {
  return {
    required_for_remote: personal?.required_for_remote ?? project?.required_for_remote ?? DEFAULT_SYNC_CONFIG.required_for_remote,
    flush_timeout_seconds: personal?.flush_timeout_seconds ?? project?.flush_timeout_seconds ?? DEFAULT_SYNC_CONFIG.flush_timeout_seconds,
    proof: {
      enabled: personal?.proof?.enabled ?? project?.proof?.enabled ?? DEFAULT_SYNC_CONFIG.proof.enabled,
      path: personal?.proof?.path ?? project?.proof?.path ?? DEFAULT_SYNC_CONFIG.proof.path,
      required_for: personal?.proof?.required_for ?? project?.proof?.required_for ?? DEFAULT_SYNC_CONFIG.proof.required_for
    }
  };
}

function resolveOutputConfig(project?: OutputConfig, personal?: OutputConfig): ResolvedOutputConfig {
  return {
    mode: personal?.mode ?? project?.mode ?? DEFAULT_OUTPUT_CONFIG.mode,
    transcript_dir: personal?.transcript_dir ?? project?.transcript_dir ?? DEFAULT_OUTPUT_CONFIG.transcript_dir,
    max_lines_per_step: personal?.max_lines_per_step ?? project?.max_lines_per_step ?? DEFAULT_OUTPUT_CONFIG.max_lines_per_step,
    max_lines_total: personal?.max_lines_total ?? project?.max_lines_total ?? DEFAULT_OUTPUT_CONFIG.max_lines_total,
    failure_tail_lines: personal?.failure_tail_lines ?? project?.failure_tail_lines ?? DEFAULT_OUTPUT_CONFIG.failure_tail_lines,
    success_tail_lines: personal?.success_tail_lines ?? project?.success_tail_lines ?? DEFAULT_OUTPUT_CONFIG.success_tail_lines,
    progress: personal?.progress ?? project?.progress ?? DEFAULT_OUTPUT_CONFIG.progress,
    full_output_requires_flag:
      personal?.full_output_requires_flag ?? project?.full_output_requires_flag ?? DEFAULT_OUTPUT_CONFIG.full_output_requires_flag
  };
}
