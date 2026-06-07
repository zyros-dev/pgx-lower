import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { findProjectConfigPath, loadProjectConfig, resolveProfile } from "../src/project-config.js";

function withTempRepo(testFn: (root: string) => void): void {
  const root = mkdtempSync(join(tmpdir(), "pgx-cli-project-"));
  try {
    testFn(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

describe("project config", () => {
  test("finds pgx-cli.yaml by walking upward", () => {
    withTempRepo((root) => {
      mkdirSync(join(root, "src", "pgx-lower"), { recursive: true });
      writeFileSync(join(root, "pgx-cli.yaml"), "project: pgx-lower\n");

      expect(findProjectConfigPath(join(root, "src", "pgx-lower"))).toBe(join(root, "pgx-cli.yaml"));
    });
  });

  test("loads committed config and local override with strict keys", () => {
    withTempRepo((root) => {
      writeFileSync(
        join(root, "pgx-cli.yaml"),
        `
project: pgx-lower
remote:
  host: comfy
  path: /home/zel/repos/pgx-lower
  mutagen_session: pgx-lower
  docker_container: pgx-lower-dev
queues:
  build: pgx-build
  check: pgx-check
sync:
  required_for_remote: true
  flush_timeout_seconds: 45
  proof:
    enabled: true
    path: .pgx-cli/sync-probes
    required_for:
      - build
      - test
      - lint
      - psql
      - pg_regress
      - bench
      - profile
      - run
output:
  mode: agent
  transcript_dir: .pgx-cli/runs
  max_lines_per_step: 80
  max_lines_total: 180
  failure_tail_lines: 60
  success_tail_lines: 20
  progress: final-summary
  full_output_requires_flag: true
profiles:
  debug:
    build:
      build_dir: build-artifacts/ptest
      cmake:
        generator: Ninja
        args:
          CMAKE_BUILD_TYPE: Debug
          BUILD_ONLY_EXTENSION: true
    runtime:
      analyzer: auto
      fallback: auto
      logging: info
      latency_logging: true
      routing_ledger: true
  latency:
    inherits: debug
    build:
      build_dir: build-artifacts/latency
      cmake:
        args:
          CMAKE_BUILD_TYPE: Release
          CMAKE_CXX_FLAGS_RELEASE: "-O3"
    runtime:
      logging: error
      latency_logging: false
      routing_ledger: false
`
      );
      writeFileSync(
        join(root, "pgx-cli.local.yaml"),
        `
remote:
  host: local-thor
output:
  max_lines_total: 40
profiles:
  debug:
    runtime:
      logging: debug
`
      );

      const config = loadProjectConfig(root);
      expect(config?.remote.host).toBe("local-thor");
      expect(config?.remote.path).toBe("/home/zel/repos/pgx-lower");
      expect(config?.sync?.flush_timeout_seconds).toBe(45);
      expect(config?.sync?.proof?.path).toBe(".pgx-cli/sync-probes");
      expect(config?.output?.max_lines_per_step).toBe(80);
      expect(config?.output?.max_lines_total).toBe(40);
      expect(config?.output?.progress).toBe("final-summary");
      expect(config?.profiles.debug.runtime?.logging).toBe("debug");
      expect(config?.profiles.latency.runtime?.logging).toBe("error");
    });
  });

  test("rejects unknown keys", () => {
    withTempRepo((root) => {
      writeFileSync(
        join(root, "pgx-cli.yaml"),
        `
project: pgx-lower
mispelled_remote:
  host: comfy
`
      );

      expect(() => loadProjectConfig(root)).toThrow("Unknown project config key: mispelled_remote");
    });
  });

  test("rejects unknown nested sync and output keys", () => {
    withTempRepo((root) => {
      writeFileSync(
        join(root, "pgx-cli.yaml"),
        `
project: pgx-lower
sync:
  proof:
    bad_key: true
`
      );

      expect(() => loadProjectConfig(root)).toThrow("Unknown sync.proof key: bad_key");
    });
  });

  test("resolves inherited profiles", () => {
    withTempRepo((root) => {
      writeFileSync(join(root, "pgx-cli.yaml"), readFileSync(join(process.cwd(), "..", "pgx-cli.yaml"), "utf8"));

      const config = loadProjectConfig(root);
      expect(config).toBeDefined();
      const latency = resolveProfile(config!, "latency");
      expect(latency.build.build_dir).toBe("build-artifacts/latency");
      expect(latency.build.cmake.generator).toBe("Ninja");
      expect(latency.runtime.analyzer).toBe("auto");
      expect(latency.runtime.logging).toBe("error");
    });
  });
});
