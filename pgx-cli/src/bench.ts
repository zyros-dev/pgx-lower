import { runManagedRemoteShell } from "./managed-operations.js";
import type { ManagedOperationConfig, ManagedOperationRunner } from "./managed-operations.js";
import type { OperationOutput } from "./operations.js";

export type BenchConfig = ManagedOperationConfig & {
  dockerContainer: string;
};

export async function runBenchCommand(
  args: string[],
  runner: ManagedOperationRunner,
  output: OperationOutput,
  config: BenchConfig
): Promise<number> {
  const [command, ...rest] = args;
  if (command !== "tpch") {
    output.stderr += "Usage: bench tpch -- <benchmark/tpch/run.py args...>\n";
    return 1;
  }

  const separator = rest.indexOf("--");
  if (separator === -1) {
    output.stderr += "Usage: bench tpch -- <benchmark/tpch/run.py args...>\n";
    return 1;
  }

  const payload = rest.slice(separator + 1);
  const result = await runManagedRemoteShell({
    runner,
    output,
    config,
    commandName: "bench-tpch",
    shellCommand: ["python3", "benchmark/tpch/run.py", ...payload].map(quoteShell).join(" "),
    requireMutagenProof: true
  });
  return result.workflowExitCode;
}

function quoteShell(value: string): string {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replaceAll("'", "'\"'\"'")}'`;
}
