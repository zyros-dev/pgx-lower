import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

export type ToolInfo = {
  name: string;
  description?: string;
  inputSchema?: unknown;
};

export type McpClient = {
  listTools(): Promise<ToolInfo[]>;
  callTool(name: string, args: unknown): Promise<unknown>;
  close(): Promise<void>;
};

export async function connectMcp(url: string): Promise<McpClient> {
  const client = new Client(
    { name: "pgx-cli", version: "1.0.0" },
    { capabilities: {} }
  );
  const transport = new StreamableHTTPClientTransport(new URL(url));

  await client.connect(transport);

  return {
    async listTools() {
      const result = await client.listTools();
      return result.tools;
    },
    async callTool(name: string, args: unknown) {
      return client.callTool({ name, arguments: objectArgs(args) });
    },
    async close() {
      await transport.close();
    }
  };
}

function objectArgs(args: unknown): Record<string, unknown> {
  if (args === undefined || args === null) {
    return {};
  }
  if (typeof args !== "object" || Array.isArray(args)) {
    throw new Error("Tool arguments must be a JSON object");
  }
  return args as Record<string, unknown>;
}
