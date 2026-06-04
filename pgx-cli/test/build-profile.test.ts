import { describe, expect, test } from "vitest";
import { cmakeArgs, renderBuildExplain, renderConfigureCommand } from "../src/build-profile.js";
import type { ResolvedProfileConfig } from "../src/project-config.js";

const profile: ResolvedProfileConfig = {
  build: {
    build_dir: "build-artifacts/ptest",
    cmake: {
      generator: "Ninja",
      args: {
        CMAKE_BUILD_TYPE: "Debug",
        BUILD_ONLY_EXTENSION: true,
        CMAKE_EXPORT_COMPILE_COMMANDS: true
      }
    }
  },
  runtime: {
    analyzer: "auto",
    fallback: "auto",
    logging: "info",
    latency_logging: true,
    routing_ledger: true
  }
};

describe("build profile rendering", () => {
  test("renders cmake -D args deterministically", () => {
    expect(cmakeArgs(profile)).toEqual([
      "-DBUILD_ONLY_EXTENSION=ON",
      "-DCMAKE_BUILD_TYPE=Debug",
      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ]);
  });

  test("renders configure command", () => {
    expect(renderConfigureCommand(profile, "/workspace")).toEqual([
      "cmake",
      "-S",
      "/workspace",
      "-B",
      "build-artifacts/ptest",
      "-G",
      "Ninja",
      "-DBUILD_ONLY_EXTENSION=ON",
      "-DCMAKE_BUILD_TYPE=Debug",
      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ]);
  });

  test("renders explain output", () => {
    expect(renderBuildExplain("debug", profile)).toContain("profile: debug");
    expect(renderBuildExplain("debug", profile)).toContain("build_dir: build-artifacts/ptest");
    expect(renderBuildExplain("debug", profile)).toContain("runtime.logging: info");
  });
});
