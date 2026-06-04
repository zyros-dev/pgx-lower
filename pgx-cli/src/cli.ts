import type { McpClient, ToolInfo } from "./mcp.js";

type Io = {
  stdout: string;
  stderr: string;
};

type RunOptions = {
  url?: string;
  remoteUrl?: string;
  sshHost?: string;
  projectPath?: string;
};

type Alias = {
  command: string;
  candidates: string[];
  buildArgs(args: string[]): unknown;
  usage: string;
};

const aliases: Alias[] = [
  {
    command: "problems",
    candidates: [
      "get_project_problems",
      "get_problems",
      "get_current_file_errors",
      "get_current_file_problems"
    ],
    buildArgs: () => ({}),
    usage: "problems"
  },
  {
    command: "file",
    candidates: ["get_file_text_by_path", "get_file_text", "read_file"],
    buildArgs: ([path]) => requiredObject({ pathInProject: path, path, file_path: path }, "file <path>"),
    usage: "file <path>"
  },
  {
    command: "open",
    candidates: ["open_file_in_editor", "open_file"],
    buildArgs: ([path]) => requiredObject({ path }, "open <path>"),
    usage: "open <path>"
  },
  {
    command: "search",
    candidates: ["find_files_by_name_keyword", "find_files_by_name_substring", "search_files", "find_files"],
    buildArgs: ([query]) => requiredObject({ nameKeyword: query, nameSubstring: query, query }, "search <query>"),
    usage: "search <query>"
  },
  {
    command: "run",
    candidates: ["execute_run_configuration", "run_configuration", "run_run_configuration"],
    buildArgs: ([name]) => requiredObject({ name, configurationName: name }, "run <configuration-name>"),
    usage: "run <configuration-name>"
  }
];

export async function runCli(
  argv: string[],
  client: McpClient,
  io: Io,
  options: RunOptions = {}
): Promise<number> {
  const [command, ...rest] = argv;

  try {
    if (!command || command === "help" || command === "--help" || command === "-h") {
      io.stdout += helpText();
      return 0;
    }

    if (command === "tools") {
      return await listTools(client, rest, io);
    }

    if (command === "schema") {
      return await printSchema(client, rest, io);
    }

    if (command === "call") {
      return await callExact(client, rest, io, options);
    }

    if (command === "doctor") {
      return await doctor(client, io, options);
    }

    if (command === "clion") {
      return await clion(rest, client, io, options);
    }

    const alias = aliases.find((candidate) => candidate.command === command);
    if (alias) {
      return await callAlias(client, alias, rest, io, options);
    }

    io.stderr += `Unknown command: ${command}\n\n${helpText()}`;
    return 1;
  } catch (error) {
    io.stderr += `${error instanceof Error ? error.message : String(error)}\n`;
    return 1;
  } finally {
    await client.close();
  }
}

async function clion(
  args: string[],
  client: McpClient,
  io: Io,
  options: RunOptions
): Promise<number> {
  const [command, ...rest] = args;
  if (command === "doctor") {
    return doctor(client, io, options);
  }
  if (command === "tools") {
    return listTools(client, rest, io);
  }
  if (command === "schema") {
    return printSchema(client, rest, io);
  }
  if (command === "call") {
    return callExact(client, rest, io, options);
  }

  throw new Error("Usage: clion <doctor|tools|schema|call>");
}

async function doctor(client: McpClient, io: Io, options: RunOptions): Promise<number> {
  const tools = await client.listTools();
  const repositories = await client.callTool(
    "get_repositories",
    withProjectPath({}, options)
  );

  io.stdout += "CLion MCP\n";
  io.stdout += `MCP URL: ${options.url ?? "(unknown)"}\n`;
  io.stdout += `Thor MCP: ${options.sshHost ?? "(unknown)"} ${options.remoteUrl ?? "(unknown)"}\n`;
  io.stdout += `Project: ${options.projectPath ?? "(not configured)"}\n`;
  io.stdout += `MCP tools: ${tools.length}\n`;

  if (isErrorResult(repositories)) {
    const text = textContent(repositories) ?? JSON.stringify(repositories);
    io.stderr += text.endsWith("\n") ? text : `${text}\n`;
    return 1;
  }

  io.stdout += "Repositories: ok\n";
  io.stdout += repositorySummary(repositories);
  return 0;
}

function repositorySummary(result: unknown): string {
  const content = structuredContent(result);
  const roots = Array.isArray(content?.roots) ? content.roots : undefined;
  if (!roots) {
    return `${JSON.stringify(result)}\n`;
  }

  const gitRoots = roots.filter((root) => {
    return root && typeof root === "object" && (root as { vcsName?: unknown }).vcsName === "Git";
  });
  const lines = [`Git repository roots: ${gitRoots.length}`];
  for (const root of gitRoots) {
    const path = (root as { pathRelativeToProject?: unknown }).pathRelativeToProject;
    lines.push(`- ${typeof path === "string" && path ? path : "."}`);
  }
  return `${lines.join("\n")}\n`;
}

function structuredContent(result: unknown): Record<string, unknown> | undefined {
  if (!result || typeof result !== "object") {
    return undefined;
  }

  if ("structuredContent" in result) {
    const content = (result as { structuredContent?: unknown }).structuredContent;
    return content && typeof content === "object" && !Array.isArray(content)
      ? (content as Record<string, unknown>)
      : undefined;
  }

  return result as Record<string, unknown>;
}

async function listTools(client: McpClient, args: string[], io: Io): Promise<number> {
  const tools = await client.listTools();
  if (args.includes("--json")) {
    io.stdout += `${JSON.stringify(tools, null, 2)}\n`;
    return 0;
  }

  for (const tool of tools) {
    const description = tool.description ? ` - ${tool.description}` : "";
    io.stdout += `${tool.name}${description}\n`;
  }
  return 0;
}

async function printSchema(client: McpClient, args: string[], io: Io): Promise<number> {
  const [toolName] = args;
  if (!toolName) {
    throw new Error("Usage: schema <tool-name>");
  }

  const tool = (await client.listTools()).find((candidate) => candidate.name === toolName);
  if (!tool) {
    throw new Error(`Tool not found: ${toolName}`);
  }

  io.stdout += `${JSON.stringify(tool.inputSchema ?? {}, null, 2)}\n`;
  return 0;
}

async function callExact(
  client: McpClient,
  args: string[],
  io: Io,
  options: RunOptions
): Promise<number> {
  const [toolName, jsonArgs = "{}"] = args;
  if (!toolName) {
    throw new Error("Usage: call <tool-name> [json-args]");
  }

  const result = await client.callTool(toolName, withProjectPath(parseJsonObject(jsonArgs), options));
  return printResult(result, io);
}

async function callAlias(
  client: McpClient,
  alias: Alias,
  args: string[],
  io: Io,
  options: RunOptions
): Promise<number> {
  const tools = await client.listTools();
  const tool = resolveTool(tools, alias.candidates);
  if (!tool) {
    throw new Error(
      `No CLion MCP tool found for '${alias.command}'. Tried: ${alias.candidates.join(", ")}`
    );
  }

  const result = await client.callTool(tool.name, withProjectPath(alias.buildArgs(args), options));
  return printResult(result, io);
}

function withProjectPath(args: unknown, options: RunOptions): unknown {
  if (
    !options.projectPath ||
    !args ||
    typeof args !== "object" ||
    Array.isArray(args) ||
    "projectPath" in args
  ) {
    return args;
  }

  return { ...args, projectPath: options.projectPath };
}

function resolveTool(tools: ToolInfo[], candidates: string[]): ToolInfo | undefined {
  return candidates
    .map((name) => tools.find((tool) => tool.name === name))
    .find((tool): tool is ToolInfo => Boolean(tool));
}

function parseJsonObject(input: string): Record<string, unknown> {
  const parsed = JSON.parse(input) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Tool arguments must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

function requiredObject(values: Record<string, string | undefined>, usage: string): Record<string, string> {
  const missing = Object.values(values).every((value) => !value);
  if (missing) {
    throw new Error(`Usage: ${usage}`);
  }

  return Object.fromEntries(
    Object.entries(values).filter((entry): entry is [string, string] => Boolean(entry[1]))
  );
}

function printResult(result: unknown, io: Io): number {
  const text = textContent(result);
  if (isErrorResult(result)) {
    io.stderr += text ?? `${JSON.stringify(result, null, 2)}\n`;
    return 1;
  }

  io.stdout += text ?? `${JSON.stringify(result, null, 2)}\n`;
  return 0;
}

function isErrorResult(result: unknown): boolean {
  return Boolean(result && typeof result === "object" && (result as { isError?: unknown }).isError);
}

function textContent(result: unknown): string | undefined {
  if (!result || typeof result !== "object" || !("content" in result)) {
    return undefined;
  }

  const content = (result as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return undefined;
  }

  const text = content
    .map((item) => {
      if (!item || typeof item !== "object") {
        return undefined;
      }
      const maybeText = item as { type?: unknown; text?: unknown };
      return maybeText.type === "text" && typeof maybeText.text === "string"
        ? maybeText.text
        : undefined;
    })
    .filter((item): item is string => Boolean(item))
    .join("\n");

  return text ? `${text}\n` : undefined;
}

export function helpText(): string {
  const aliasHelp = aliases.map((alias) => `  ${alias.usage}`).join("\n");
  return [
    "Usage: pgx-cli [--url <mcp-url>] <command>",
    "",
    "Commands:",
    "  tools [--json]",
    "  schema <tool-name>",
    "  call <tool-name> [json-args]",
    "  config path",
    "  config show",
    "  config set-url <mcp-url>",
    "  config set-project <project-path>",
    "  config set-ssh-host <ssh-host>",
    "  setup install",
    "  setup doctor",
    "  tunnel",
    "  sync status",
    "  sync flush",
    "  build compile",
    "  build test",
    "  build utest-pg",
    "  build bench",
    "  check diff",
    "  check all",
    "  queue status",
    "  queue flush",
    "  queue tail <id>",
    "  queue cancel <id>",
    "  thor just <recipe> [args...]",
    "  thor shell --dangerous -- <cmd...>",
    "  request feature <message...>",
    "  request complaint <message...>",
    "  doctor",
    "  clion doctor",
    "  clion tools [--json]",
    "  clion schema <tool-name>",
    "  clion call <tool-name> [json-args]",
    aliasHelp,
    "",
    "Default workflow: SSH tunnel to thor MCP via comfy, then use /home/zel/repos/pgx-lower.",
    "Config precedence: --url, ~/.config/pgx-cli/config.json, CLION_MCP_URL, defaults.",
    ""
  ].join("\n");
}
