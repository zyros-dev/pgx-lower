import { describe, expect, test } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_CONFIG_PATH, loadConfig, writeConfig } from "../src/config.js";

describe("loadConfig", () => {
  test("uses thor MCP tunnel settings by default", () => {
    const config = loadConfig({
      env: {},
      argvUrl: undefined,
      readConfigFile: () => undefined,
      cwd: "/Users/nickvandermerwe/repos/pgx-lower"
    });

    expect(config.url).toBe("http://127.0.0.1:64343/stream");
    expect(config.projectPath).toBe("/home/zel/repos/pgx-lower");
    expect(config.sshHost).toBe("comfy");
    expect(config.remoteUrl).toBe("http://127.0.0.1:64342/stream");
    expect(config.localProjectPath).toBe("/Users/nickvandermerwe/repos/pgx-lower");
    expect(config.remoteProjectPath).toBe("/home/zel/repos/pgx-lower");
    expect(config.mutagenSession).toBe("pgx-lower");
    expect(config.dockerContainer).toBe("pgx-lower-dev");
    expect(config.buildQueue).toBe("pgx-build");
    expect(config.checkQueue).toBe("pgx-check");
  });

  test("uses pgx-cli config directory", () => {
    expect(DEFAULT_CONFIG_PATH).toContain(".config/pgx-cli/config.json");
  });

  test("prefers explicit URL over environment and config file", () => {
    const config = loadConfig({
      env: { CLION_MCP_URL: "http://env.example/stream" },
      argvUrl: "http://argv.example/stream",
      readConfigFile: () => ({ url: "http://file.example/stream" })
    });

    expect(config.url).toBe("http://argv.example/stream");
  });

  test("uses config file before environment fallback", () => {
    const config = loadConfig({
      env: { CLION_MCP_URL: "http://env.example/stream" },
      argvUrl: undefined,
      readConfigFile: () => ({ url: "http://file.example/stream" })
    });

    expect(config.url).toBe("http://file.example/stream");
    expect(config.projectPath).toBe("/home/zel/repos/pgx-lower");
  });

  test("writes full config as JSON", () => {
    const dir = mkdtempSync(join(tmpdir(), "pgx-cli-"));
    const path = join(dir, "nested", "config.json");

    writeConfig(path, {
      url: "http://127.0.0.1:64343/stream",
      projectPath: "/home/zel/repos/pgx-lower",
      sshHost: "comfy",
      remoteUrl: "http://127.0.0.1:64342/stream",
      localProjectPath: "/Users/nickvandermerwe/repos/pgx-lower",
      remoteProjectPath: "/home/zel/repos/pgx-lower",
      mutagenSession: "pgx-lower",
      dockerContainer: "pgx-lower-dev",
      buildQueue: "pgx-build",
      checkQueue: "pgx-check"
    });

    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({
      url: "http://127.0.0.1:64343/stream",
      projectPath: "/home/zel/repos/pgx-lower",
      sshHost: "comfy",
      remoteUrl: "http://127.0.0.1:64342/stream",
      localProjectPath: "/Users/nickvandermerwe/repos/pgx-lower",
      remoteProjectPath: "/home/zel/repos/pgx-lower",
      mutagenSession: "pgx-lower",
      dockerContainer: "pgx-lower-dev",
      buildQueue: "pgx-build",
      checkQueue: "pgx-check"
    });

    rmSync(dir, { recursive: true, force: true });
  });

  test("uses project config for pgx-lower operation defaults", () => {
    const config = loadConfig({
      env: {},
      argvUrl: undefined,
      readConfigFile: () => undefined,
      readProjectConfig: () => ({
        project: "pgx-lower",
        remote: {
          host: "project-thor",
          path: "/project/remote",
          mutagen_session: "project-session",
          docker_container: "pgx-lower-dev"
        },
        queues: {
          build: "project-build",
          check: "project-check"
        },
        profiles: {}
      })
    });

    expect(config.sshHost).toBe("project-thor");
    expect(config.remoteProjectPath).toBe("/project/remote");
    expect(config.projectPath).toBe("/project/remote");
    expect(config.mutagenSession).toBe("project-session");
    expect(config.dockerContainer).toBe("pgx-lower-dev");
    expect(config.buildQueue).toBe("project-build");
    expect(config.checkQueue).toBe("project-check");
  });

  test("detects when pgx-cli is already running from the remote checkout", () => {
    const config = loadConfig({
      env: {},
      argvUrl: undefined,
      readConfigFile: () => undefined,
      cwd: "/project/remote/pgx-cli",
      readProjectConfig: () => ({
        project: "pgx-lower",
        remote: {
          host: "project-thor",
          path: "/project/remote",
          mutagen_session: "project-session",
          docker_container: "pgx-lower-dev"
        },
        queues: {
          build: "project-build",
          check: "project-check"
        },
        profiles: {}
      })
    });

    expect(config.runningOnRemote).toBe(true);
    expect(config.localProjectPath).toBe("/project/remote");
  });

  test("keeps personal JSON and environment overrides above project config", () => {
    const config = loadConfig({
      env: {
        PGX_REMOTE_PROJECT_PATH: "/env/remote",
        PGX_MUTAGEN_SESSION: "env-session"
      },
      argvUrl: undefined,
      readConfigFile: () => ({ sshHost: "personal-thor" }),
      readProjectConfig: () => ({
        project: "pgx-lower",
        remote: {
          host: "project-thor",
          path: "/project/remote",
          mutagen_session: "project-session",
          docker_container: "pgx-lower-dev"
        },
        queues: {
          build: "project-build",
          check: "project-check"
        },
        profiles: {}
      })
    });

    expect(config.sshHost).toBe("personal-thor");
    expect(config.remoteProjectPath).toBe("/env/remote");
    expect(config.mutagenSession).toBe("env-session");
  });
});
