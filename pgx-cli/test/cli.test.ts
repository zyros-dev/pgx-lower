import { describe, expect, test } from "vitest";
import { runCli } from "../src/cli.js";
import type { McpClient } from "../src/mcp.js";

function client(): McpClient & { calls: Array<{ name: string; args: unknown }> } {
  const calls: Array<{ name: string; args: unknown }> = [];

  return {
    calls,
    async listTools() {
      return [
        {
          name: "get_file_text_by_path",
          description: "Read file text",
          inputSchema: {
            type: "object",
            properties: { path: { type: "string" } },
            required: ["path"]
          }
        },
        {
          name: "get_project_problems",
          description: "Inspect diagnostics",
          inputSchema: { type: "object", properties: {} }
        }
      ];
    },
    async callTool(name: string, args: unknown) {
      calls.push({ name, args });
      return { content: [{ type: "text", text: "ok" }] };
    },
    async close() {}
  };
}

describe("runCli", () => {
  test("lists server tools as readable commands", async () => {
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["tools"], client(), io);

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("get_file_text_by_path");
    expect(io.stdout).toContain("Read file text");
  });

  test("calls any MCP tool by exact name with JSON arguments", async () => {
    const fake = client();
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(
      ["call", "get_file_text_by_path", '{"path":"src/main.cpp"}'],
      fake,
      io,
      { projectPath: "/home/zel/repos/pgx-lower" }
    );

    expect(exitCode).toBe(0);
    expect(fake.calls).toEqual([
      {
        name: "get_file_text_by_path",
        args: { path: "src/main.cpp", projectPath: "/home/zel/repos/pgx-lower" }
      }
    ]);
    expect(io.stdout).toContain("ok");
  });

  test("does not overwrite an explicit projectPath", async () => {
    const fake = client();
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(
      ["call", "get_file_text_by_path", '{"path":"src/main.cpp","projectPath":"/custom"}'],
      fake,
      io,
      { projectPath: "/home/zel/repos/pgx-lower" }
    );

    expect(exitCode).toBe(0);
    expect(fake.calls).toEqual([
      { name: "get_file_text_by_path", args: { path: "src/main.cpp", projectPath: "/custom" } }
    ]);
  });

  test("returns non-zero when an MCP tool result is marked as an error", async () => {
    const fake = client();
    fake.callTool = async () => ({
      isError: true,
      content: [{ type: "text", text: "project is not open" }]
    });
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["call", "get_project_modules", "{}"], fake, io);

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("project is not open");
  });

  test("run convenience command resolves the actual CLion execute tool", async () => {
    const fake = client();
    fake.listTools = async () => [
      {
        name: "execute_run_configuration",
        inputSchema: { type: "object", properties: {} }
      }
    ];
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["run", "demo"], fake, io, {
      projectPath: "/home/zel/repos/pgx-lower"
    });

    expect(exitCode).toBe(0);
    expect(fake.calls).toEqual([
      {
        name: "execute_run_configuration",
        args: {
          name: "demo",
          configurationName: "demo",
          projectPath: "/home/zel/repos/pgx-lower"
        }
      }
    ]);
  });

  test("resolves diagnostics convenience command from discovered tools", async () => {
    const fake = client();
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["problems"], fake, io);

    expect(exitCode).toBe(0);
    expect(fake.calls).toEqual([{ name: "get_project_problems", args: {} }]);
  });

  test("prints a useful error when an alias cannot be resolved", async () => {
    const fake = client();
    fake.listTools = async () => [];
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["problems"], fake, io);

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("No CLion MCP tool found");
  });

  test("doctor reports configured thor MCP context and repository status", async () => {
    const fake = client();
    fake.callTool = async (name: string, args: unknown) => {
      fake.calls.push({ name, args });
      return { roots: [{ pathRelativeToProject: "", vcsName: "Git" }] };
    };
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["doctor"], fake, io, {
      url: "http://127.0.0.1:64343/stream",
      remoteUrl: "http://127.0.0.1:64342/stream",
      sshHost: "comfy",
      projectPath: "/home/zel/repos/pgx-lower"
    });

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("MCP URL: http://127.0.0.1:64343/stream");
    expect(io.stdout).toContain("Thor MCP: comfy http://127.0.0.1:64342/stream");
    expect(io.stdout).toContain("Project: /home/zel/repos/pgx-lower");
    expect(io.stdout).toContain("MCP tools: 2");
    expect(io.stdout).toContain("Repositories: ok");
    expect(io.stdout).toContain("Git repository roots: 1");
    expect(fake.calls).toEqual([
      {
        name: "get_repositories",
        args: { projectPath: "/home/zel/repos/pgx-lower" }
      }
    ]);
  });

  test("clion doctor is an alias for the CLion MCP doctor", async () => {
    const fake = client();
    fake.callTool = async (name: string, args: unknown) => {
      fake.calls.push({ name, args });
      return { roots: [] };
    };
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["clion", "doctor"], fake, io, {
      projectPath: "/home/zel/repos/pgx-lower"
    });

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("CLion MCP");
    expect(fake.calls[0]).toEqual({
      name: "get_repositories",
      args: { projectPath: "/home/zel/repos/pgx-lower" }
    });
  });

  test("help documents config commands", async () => {
    const io = { stdout: "", stderr: "" };
    const exitCode = await runCli(["help"], client(), io);

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("Usage: pgx-cli");
    expect(io.stdout).toContain("config set-project <project-path>");
    expect(io.stdout).toContain("setup install");
    expect(io.stdout).toContain("setup doctor");
    expect(io.stdout).toContain("pgx-cli dev status");
    expect(io.stdout).toContain("pgx-cli dev lint diff");
    expect(io.stdout).toContain("pgx-cli dev gate batch");
    expect(io.stdout).toContain("pgx-cli dev gate review");
    expect(io.stdout).toContain("tunnel");
    expect(io.stdout).toContain("doctor");
    expect(io.stdout).toContain("clion doctor");
    expect(io.stdout).toContain("sync status");
    expect(io.stdout).toContain("sync flush");
    expect(io.stdout).toContain("build compile");
    expect(io.stdout).toContain("check diff");
    expect(io.stdout).toContain("queue status");
    expect(io.stdout).toContain("thor just <recipe>");
    expect(io.stdout).toContain("thor shell --dangerous -- <cmd...>");
    expect(io.stdout).toContain("request feature <message...>");
    expect(io.stdout).toContain("request complaint <message...>");
  });
});
