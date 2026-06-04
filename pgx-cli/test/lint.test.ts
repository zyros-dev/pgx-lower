import { describe, expect, test } from "vitest";
import { fullLintCommand, targetedLintCommand } from "../src/lint.js";

describe("lint command rendering", () => {
  test("renders full clang-tidy command without scripts/run_lint.sh", () => {
    const command = fullLintCommand("/workspace", "pgx-lower-dev");

    expect(command).toContain("docker exec pgx-lower-dev bash -lc");
    expect(command).toContain("build-docker-lint");
    expect(command).toContain("clang-tidy-20");
    expect(command).toContain("src/pgx-lower");
    expect(command).not.toContain("scripts/run_lint.sh");
  });

  test("renders targeted clang-tidy command for selected files", () => {
    const command = targetedLintCommand("/workspace", "pgx-lower-dev", [
      "src/pgx-lower/runtime/tuple_access.cpp",
      "src/pgx-lower/runtime/PostgreSQLRuntime.cpp"
    ]);

    expect(command).toContain("LINT_SKIP_BUILD=1");
    expect(command).toContain("clang-tidy-20");
    expect(command).toContain("tuple_access.cpp");
    expect(command).toContain("PostgreSQLRuntime.cpp");
    expect(command).not.toContain("scripts/run_lint.sh");
  });
});
