import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { parse } from "yaml";

export type RuntimeConfig = {
  analyzer?: string;
  fallback?: string;
  logging?: string;
  latency_logging?: boolean;
  routing_ledger?: boolean;
};

export type CmakeConfig = {
  generator?: string;
  args?: Record<string, string | boolean | number>;
};

export type BuildConfig = {
  build_dir?: string;
  cmake?: CmakeConfig;
};

export type ProfileConfig = {
  inherits?: string;
  build?: BuildConfig;
  runtime?: RuntimeConfig;
};

export type ResolvedProfileConfig = {
  build: Required<BuildConfig> & { cmake: Required<CmakeConfig> };
  runtime: RuntimeConfig;
};

export type ProjectConfig = {
  project: string;
  remote: {
    host: string;
    path: string;
    mutagen_session: string;
    docker_container: string;
  };
  queues: {
    build: string;
    check: string;
  };
  profiles: Record<string, ProfileConfig>;
  configPath?: string;
  localConfigPath?: string;
};

const allowedTopLevelKeys = new Set(["project", "remote", "queues", "profiles"]);
const allowedRemoteKeys = new Set(["host", "path", "mutagen_session", "docker_container"]);
const allowedQueueKeys = new Set(["build", "check"]);
const allowedProfileKeys = new Set(["inherits", "build", "runtime"]);
const allowedBuildKeys = new Set(["build_dir", "cmake"]);
const allowedCmakeKeys = new Set(["generator", "args"]);
const allowedRuntimeKeys = new Set(["analyzer", "fallback", "logging", "latency_logging", "routing_ledger"]);

export function findProjectConfigPath(startDir: string): string | undefined {
  let current = startDir;
  while (true) {
    const candidate = join(current, "pgx-cli.yaml");
    if (existsSync(candidate)) {
      return candidate;
    }
    const parent = dirname(current);
    if (parent === current) {
      return undefined;
    }
    current = parent;
  }
}

export function loadProjectConfig(startDir: string = process.cwd()): ProjectConfig | undefined {
  const configPath = findProjectConfigPath(startDir);
  if (!configPath) {
    return undefined;
  }

  const root = dirname(configPath);
  const committed = readYamlObject(configPath);
  validateConfigObject(committed);

  const localConfigPath = join(root, "pgx-cli.local.yaml");
  const local = existsSync(localConfigPath) ? readYamlObject(localConfigPath) : {};
  validateConfigObject(local);

  const merged = deepMerge(committed, local) as ProjectConfig;
  return {
    ...merged,
    configPath,
    localConfigPath: existsSync(localConfigPath) ? localConfigPath : undefined
  };
}

export function resolveProfile(config: ProjectConfig, name: string): ResolvedProfileConfig {
  const seen = new Set<string>();

  function resolve(currentName: string): ProfileConfig {
    if (seen.has(currentName)) {
      throw new Error(`Profile inheritance cycle: ${[...seen, currentName].join(" -> ")}`);
    }
    const profile = config.profiles[currentName];
    if (!profile) {
      throw new Error(`Unknown profile: ${currentName}`);
    }
    seen.add(currentName);
    const inherited = profile.inherits ? resolve(profile.inherits) : {};
    seen.delete(currentName);
    return deepMerge(inherited, profile) as ProfileConfig;
  }

  const profile = resolve(name);
  return {
    build: {
      build_dir: profile.build?.build_dir ?? "",
      cmake: {
        generator: profile.build?.cmake?.generator ?? "",
        args: profile.build?.cmake?.args ?? {}
      }
    },
    runtime: profile.runtime ?? {}
  };
}

function readYamlObject(path: string): Record<string, unknown> {
  const parsed = parse(readFileSync(path, "utf8")) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${path} must contain a YAML object`);
  }
  return parsed as Record<string, unknown>;
}

function validateConfigObject(value: Record<string, unknown>): void {
  validateKeys(value, allowedTopLevelKeys, "project config");
  if (isObject(value.remote)) {
    validateKeys(value.remote, allowedRemoteKeys, "remote");
  }
  if (isObject(value.queues)) {
    validateKeys(value.queues, allowedQueueKeys, "queues");
  }
  if (isObject(value.profiles)) {
    for (const [profileName, profile] of Object.entries(value.profiles)) {
      if (!isObject(profile)) {
        throw new Error(`Profile ${profileName} must be an object`);
      }
      validateKeys(profile, allowedProfileKeys, `profile ${profileName}`);
      if (isObject(profile.build)) {
        validateKeys(profile.build, allowedBuildKeys, `profile ${profileName}.build`);
        if (isObject(profile.build.cmake)) {
          validateKeys(profile.build.cmake, allowedCmakeKeys, `profile ${profileName}.build.cmake`);
        }
      }
      if (isObject(profile.runtime)) {
        validateKeys(profile.runtime, allowedRuntimeKeys, `profile ${profileName}.runtime`);
      }
    }
  }
}

function validateKeys(value: Record<string, unknown>, allowed: Set<string>, context: string): void {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`Unknown ${context} key: ${key}`);
    }
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function deepMerge(left: unknown, right: unknown): unknown {
  if (!isObject(left) || !isObject(right)) {
    return right ?? left;
  }
  const merged: Record<string, unknown> = { ...left };
  for (const [key, value] of Object.entries(right)) {
    merged[key] = key in merged ? deepMerge(merged[key], value) : value;
  }
  return merged;
}
