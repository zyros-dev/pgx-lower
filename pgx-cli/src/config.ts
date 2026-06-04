import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { loadProjectConfig } from "./project-config.js";
import type { ProjectConfig } from "./project-config.js";

export const DEFAULT_URL = "http://127.0.0.1:64343/stream";
export const DEFAULT_REMOTE_URL = "http://127.0.0.1:64342/stream";
export const DEFAULT_PROJECT_PATH = "/home/zel/repos/pgx-lower";
export const DEFAULT_SSH_HOST = "comfy";
export const DEFAULT_LOCAL_PROJECT_PATH = "/Users/nickvandermerwe/repos/pgx-lower";
export const DEFAULT_REMOTE_PROJECT_PATH = "/home/zel/repos/pgx-lower";
export const DEFAULT_MUTAGEN_SESSION = "pgx-lower";
export const DEFAULT_DOCKER_CONTAINER = "pgx-lower-dev";
export const DEFAULT_CONFIG_PATH = join(homedir(), ".config", "pgx-cli", "config.json");

export type Config = {
  url: string;
  projectPath: string;
  sshHost: string;
  remoteUrl: string;
  localProjectPath: string;
  remoteProjectPath: string;
  mutagenSession: string;
  dockerContainer: string;
};

type LoadConfigInput = {
  env: NodeJS.ProcessEnv;
  argvUrl?: string;
  readConfigFile?: () => Partial<Config> | undefined;
  readProjectConfig?: () => ProjectConfig | undefined;
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

  return {
    url: input.argvUrl ?? fileConfig?.url ?? input.env.CLION_MCP_URL ?? DEFAULT_URL,
    projectPath: fileConfig?.projectPath ?? input.env.CLION_PROJECT_PATH ?? projectRemote?.path ?? DEFAULT_PROJECT_PATH,
    sshHost: fileConfig?.sshHost ?? input.env.CLION_MCP_SSH_HOST ?? projectRemote?.host ?? DEFAULT_SSH_HOST,
    remoteUrl: fileConfig?.remoteUrl ?? input.env.CLION_MCP_REMOTE_URL ?? DEFAULT_REMOTE_URL,
    localProjectPath:
      fileConfig?.localProjectPath ?? input.env.PGX_LOCAL_PROJECT_PATH ?? DEFAULT_LOCAL_PROJECT_PATH,
    remoteProjectPath:
      fileConfig?.remoteProjectPath ?? input.env.PGX_REMOTE_PROJECT_PATH ?? projectRemote?.path ?? DEFAULT_REMOTE_PROJECT_PATH,
    mutagenSession:
      fileConfig?.mutagenSession ?? input.env.PGX_MUTAGEN_SESSION ?? projectRemote?.mutagen_session ?? DEFAULT_MUTAGEN_SESSION,
    dockerContainer:
      fileConfig?.dockerContainer ?? input.env.PGX_DOCKER_CONTAINER ?? projectRemote?.docker_container ?? DEFAULT_DOCKER_CONTAINER
  };
}

export function writeConfig(path: string, config: Config): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, `${JSON.stringify(config, null, 2)}\n`);
}
