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
      readConfigFile: () => undefined
    });

    expect(config.url).toBe("http://127.0.0.1:64343/stream");
    expect(config.projectPath).toBe("/home/zel/repos/pgx-lower");
    expect(config.sshHost).toBe("comfy");
    expect(config.remoteUrl).toBe("http://127.0.0.1:64342/stream");
    expect(config.localProjectPath).toBe("/Users/nickvandermerwe/repos/pgx-lower");
    expect(config.remoteProjectPath).toBe("/home/zel/repos/pgx-lower");
    expect(config.mutagenSession).toBe("pgx-lower");
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
      mutagenSession: "pgx-lower"
    });

    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({
      url: "http://127.0.0.1:64343/stream",
      projectPath: "/home/zel/repos/pgx-lower",
      sshHost: "comfy",
      remoteUrl: "http://127.0.0.1:64342/stream",
      localProjectPath: "/Users/nickvandermerwe/repos/pgx-lower",
      remoteProjectPath: "/home/zel/repos/pgx-lower",
      mutagenSession: "pgx-lower"
    });

    rmSync(dir, { recursive: true, force: true });
  });
});
