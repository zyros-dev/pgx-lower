import { spawn } from "node:child_process";

export type RunResult = {
  exitCode: number;
  stdout: string;
  stderr: string;
};

export type CommandRunner = {
  run(command: string, args: string[]): Promise<RunResult>;
};

export class NodeCommandRunner implements CommandRunner {
  async run(command: string, args: string[]): Promise<RunResult> {
    return new Promise((resolve) => {
      const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk) => {
        const text = chunk.toString();
        stdout += text;
      });
      child.stderr.on("data", (chunk) => {
        const text = chunk.toString();
        stderr += text;
      });
      child.on("error", (error) => {
        resolve({ exitCode: 1, stdout, stderr: stderr + `${error.message}\n` });
      });
      child.on("close", (code) => {
        resolve({ exitCode: code ?? 1, stdout, stderr });
      });
    });
  }
}
