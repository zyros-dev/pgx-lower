export function fullLintCommand(workspace: string, container: string): string {
  return dockerLintCommand(workspace, container, fullLintShellCommand(workspace));
}

export function targetedLintCommand(workspace: string, container: string, files: string[]): string {
  return dockerLintCommand(workspace, container, targetedLintShellCommand(workspace, files));
}

export function fullLintShellCommand(workspace: string): string {
  return lintShellCommand(workspace, { files: [] });
}

export function targetedLintShellCommand(workspace: string, files: string[]): string {
  return lintShellCommand(workspace, { files, skipBuild: true });
}

type LintOptions = {
  files: string[];
  skipBuild?: boolean;
};

function lintShellCommand(workspace: string, options: LintOptions): string {
  const lintDir = `${workspace}/build-docker-lint`;
  const fileSetup =
    options.files.length > 0
      ? `files=(${options.files.map(quoteShell).join(" ")})`
      : "mapfile -t files < <(find src/pgx-lower -name '*.cpp' | sort)";
  const buildBlock = options.skipBuild
    ? [
        "export LINT_SKIP_BUILD=1",
        `[ -f ${quoteShell(`${lintDir}/compile_commands.json`)} ] || { echo ${quoteShell(
          `LINT: ${lintDir}/compile_commands.json missing; run pgx-cli dev gate review once first`
        )}; exit 2; }`
      ].join(" && ")
    : [
        `mkdir -p ${quoteShell(lintDir)}`,
        `cd ${quoteShell(lintDir)}`,
        `[ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ${quoteShell(
          workspace
        )}`,
        "cmake --build ."
      ].join(" && ");
  const tidyArgs = [
    `-p ${quoteShell(lintDir)}`,
    "--quiet",
    "--allow-no-checks",
    "-warnings-as-errors='*'"
  ].join(" ");

  return [
    "set -o pipefail",
    `cd ${quoteShell(workspace)}`,
    buildBlock,
    `cd ${quoteShell(workspace)}`,
    fileSetup,
    "out=$(mktemp)",
    "rc=0",
    `printf '%s\\n' "\${files[@]}" | xargs -P$(nproc) -I{} clang-tidy-20 ${tidyArgs} {} >"$out" 2>&1 || rc=$?`,
    "sed '/^[0-9][0-9]* warnings generated\\.$/d' \"$out\"",
    "if [ \"$rc\" -eq 0 ]; then echo \"LINT CLEAN\"; else echo \"LINT FAILED - violations above (exit $rc).\"; fi",
    "rm -f \"$out\"",
    "exit \"$rc\""
  ].join(" && ");
}

function dockerLintCommand(workspace: string, container: string, shellCommand: string): string {
  void workspace;
  return `docker exec ${quoteShell(container)} bash -lc ${quoteShell(shellCommand)}`;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
