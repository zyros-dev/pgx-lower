# pgx-cli

Small TypeScript CLI for using the CLion MCP server without hand-writing MCP calls.

Default workflow: CLion runs through JetBrains Gateway on thor, so the useful MCP server is on thor's loopback interface. `pgx-cli` defaults to a local SSH tunnel:

```sh
ssh -L 127.0.0.1:64343:127.0.0.1:64342 comfy
```

Default local MCP URL:

```sh
http://127.0.0.1:64343/stream
```

Default project path:

```sh
/home/zel/repos/pgx-lower
```

Config precedence is:

```text
--url
~/.config/pgx-cli/config.json
CLION_MCP_URL
default URL
```

## Commands

```sh
pgx-cli tools
pgx-cli tools --json
pgx-cli schema <tool-name>
pgx-cli call <tool-name> '{"arg":"value"}'
pgx-cli tunnel
pgx-cli sync status
pgx-cli sync flush
pgx-cli build compile
pgx-cli build test
pgx-cli build utest-pg
pgx-cli build bench
pgx-cli check diff
pgx-cli check all
pgx-cli queue status
pgx-cli queue tail <id>
pgx-cli queue cancel <id>
pgx-cli queue flush
pgx-cli thor just --list
pgx-cli thor shell --dangerous -- git status --short
pgx-cli request feature make the compile gate easier to inspect
pgx-cli request complaint thor command output is too noisy
pgx-cli doctor
pgx-cli clion doctor
pgx-cli clion tools
pgx-cli clion schema <tool-name>
pgx-cli clion call <tool-name> '{"arg":"value"}'
pgx-cli problems
pgx-cli file <path>
pgx-cli open <path>
pgx-cli search <query>
pgx-cli run <configuration-name>
```

The `call` command supports every tool the CLion MCP server advertises. The convenience commands resolve against the discovered tool list, so they fail loudly if the installed CLion MCP exposes a different tool surface.

`pgx-cli thor just ...` always flushes the `pgx-lower` Mutagen session first, then runs `just` on `comfy` in `/home/zel/repos/pgx-lower`. `pgx-cli thor shell ...` is reserved for explicit remote shell work and requires `--dangerous`.

`pgx-cli build ...` and `pgx-cli check ...` are the normal pgx-lower development commands. They flush Mutagen and then run the existing queued thor `just` recipes, so agents do not need to know task-spooler details. `pgx-cli queue ...` is for diagnostics and recovery only.

`pgx-cli request feature ...` and `pgx-cli request complaint ...` write timestamped Markdown notes to `~/.config/pgx-cli/requests/`. They are local inbox commands for agents to lodge friction quickly and continue with the current task.

Planned command groups:

```text
pgx-cli clion ...
pgx-cli build ...
pgx-cli check ...
pgx-cli thor ...
pgx-cli queue ...
pgx-cli lsp ...
pgx-cli sync ...
```

The current top-level CLion commands are the first slice. Future work should move them under `pgx-cli clion` while keeping compatibility aliases.

## Roadmap

Conversation mining found these recurring workflows worth turning into commands:

1. `pgx-cli doctor` - read-only health check for thor, mutagen, branch, Docker image, and local-build hazards.
2. `pgx-cli thor <cmd>` / `pgx-cli just <recipe>` - flush mutagen, then run on `comfy` in `/home/zel/repos/pgx-lower`.
3. `pgx-cli gate` - standard diff/check/compile/unit/test gate with logs and summary.
4. `pgx-cli clion doctor` - Gateway backend, MCP tunnel, project-path, and local-vs-thor endpoint diagnosis.
5. `pgx-cli clion db` - compile database / CLion indexing validation for PG, LLVM 20, and generated includes.
6. `pgx-cli lsp ...` - code-intelligence helpers such as diagnostics, symbols, definition, references, and compile-command checks.
7. `pgx-cli sync doctor` - mutagen status, ignored artifacts, and session drift.
8. `pgx-cli bench` - expensive benchmark/report wrapper with explicit subset controls.
9. `pgx-cli test` - correctness test summary focused on TPC-H result interpretation.
10. `pgx-cli expected` - guarded expected-output update flow.
11. `pgx-cli crash-backtrace` - collect real thor-side backtraces and PG logs.
12. `pgx-cli plan` - list ready wiki plans and enforce the single-checkout branch workflow.

## Config

```sh
pgx-cli config path
pgx-cli config show
pgx-cli config set-url http://127.0.0.1:64343/stream
pgx-cli config set-project /home/zel/repos/pgx-lower
pgx-cli config set-ssh-host comfy
```

`projectPath` is added automatically to tool calls unless the JSON arguments already include it.

## Development

```sh
npm install
npm test
npm run build
npm link
```
