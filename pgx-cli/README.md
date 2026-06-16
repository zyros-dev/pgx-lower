# pgx-cli

TypeScript CLI for pgx-lower development workflows. It wraps CLion MCP, thor,
Mutagen, queued workflow commands, and setup diagnostics from the pgx-lower
checkout.

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
pgx-cli sync doctor
pgx-cli run thor -- true
pgx-cli run docker -- bash -lc 'echo ok'
pgx-cli run psql --query 'SELECT 1'
pgx-cli run psql --file tests/debug/q17.sql
pgx-cli logs show <run-id>
pgx-cli logs latest
pgx-cli dev status
pgx-cli dev lint diff
pgx-cli dev test focused
pgx-cli dev test tpch
pgx-cli dev gate batch
pgx-cli dev gate review
pgx-cli dev build explain --profile debug
pgx-cli dev build compile --profile debug
pgx-cli test route-check --help
pgx-cli test compare-postgres --workload tpch-correctness
pgx-cli test unit-sql --root .
pgx-cli test psql-regression-burndown --help
pgx-cli docker status
pgx-cli docker build ptest
pgx-cli docker build release
pgx-cli repo audit-tools
pgx-cli queue status
pgx-cli queue tail <id>
pgx-cli queue cancel <id>
pgx-cli queue flush
pgx-cli thor shell --dangerous -- git status --short
pgx-cli codex-policy check -- ssh comfy true
pgx-cli codex-policy rules
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

`pgx-cli dev ...` commands are the normal pgx-lower development workflow. They
run the needed thor/Docker/task-spooler steps through managed output and Mutagen
preflight.
`pgx-cli test route-check` validates pg_regress route notices against SQL
directives and writes a route summary without contacting CLion MCP.
`pgx-cli test compare-postgres --workload tpch-correctness` compares stock
PostgreSQL rows against pgx-lower rows through the managed thor/dev path and
writes Markdown/JSON artifacts under `build-artifacts/test-runs/...`.
`pgx-cli test unit-sql` generates pg_regress SQL wrappers for PGX_TEST_FN C++
unit tests.
`pgx-cli test psql-regression-burndown` runs the opt-in upstream PostgreSQL
regression burn-down ledger and compares failures against a reviewed baseline.
`pgx-cli docker ...` commands wrap explicit thor-side Docker maintenance flows.
`pgx-cli run thor -- ...`, `pgx-cli run docker -- ...`, and
`pgx-cli run psql ...` are the bounded escape hatch when no typed command exists
yet.
`pgx-cli repo audit-tools` enforces that workflow entrypoints stay in pgx-cli
instead of drifting back into loose shell or Python helper scripts.
`pgx-cli thor shell ...` is reserved for explicit remote shell work and requires
`--dangerous`. `pgx-cli queue ...` is for diagnostics and recovery.

Managed commands print a run id and transcript path, and write
`.pgx-cli/runs/<run-id>/summary.json`, `stdout.log`, `stderr.log`,
`combined.log`, and `sync-preflight.log`. Use `pgx-cli logs show <run-id>` or
`pgx-cli logs latest` to inspect bounded transcript excerpts without rerunning a
workflow. Use `--head N`, `--tail N`, or `--full` only when the default preview
is not the view you need.

Mutagen-dependent commands fail closed before remote execution when the session
is missing, paused, disconnected, conflicted, stale, or the sync proof fails.
Use `pgx-cli sync status` and `pgx-cli sync doctor` to diagnose those failures.
Project-local Codex rules live in `.codex/rules/default.rules`; Codex CLI must
trust the project `.codex/` layer for raw workflow command prefix blocks to
load. `pgx-cli codex-policy check` also inspects common
`bash -lc`/`zsh -c`/`sh -c` wrappers for raw workflow commands, large log/IR
reads, and unsafe `gh pr comment --body` backticks. Automatic external Codex
hook wiring beyond the project rules and classifier is still outside this
repo-local workflow.

`pgx-cli request feature ...` and `pgx-cli request complaint ...` write timestamped Markdown notes to `~/.config/pgx-cli/requests/`. They are local inbox commands for agents to lodge friction quickly and continue with the current task.

Planned command groups:

```text
pgx-cli clion ...
pgx-cli dev ...
pgx-cli thor ...
pgx-cli queue ...
pgx-cli lsp ...
pgx-cli sync ...
```

The current top-level CLion commands are the first slice. Future work should move them under `pgx-cli clion` while keeping compatibility aliases.

## Roadmap

Conversation mining found these recurring workflows worth turning into commands:

1. `pgx-cli doctor` - read-only health check for thor, mutagen, branch, Docker image, and local-build hazards.
2. `pgx-cli dev ...` - flush mutagen, then run the direct workflow on local/thor/Docker as needed.
3. `pgx-cli dev gate ...` - standard diff/check/compile/unit/test gate with logs and summary.
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

## Dev workflow

Use `pgx-cli dev ...` for normal pgx-lower agent workflows:

```sh
pgx-cli dev status
pgx-cli dev lint diff
pgx-cli dev lint file src/pgx-lower/runtime/tuple_access.cpp
pgx-cli dev test unit type_mapping
pgx-cli dev gate batch
pgx-cli dev gate review
pgx-cli dev logs latest
pgx-cli docker status
pgx-cli docker build ptest
pgx-cli docker build release
pgx-cli run thor -- true
pgx-cli run docker -- bash -lc 'echo ok'
pgx-cli run psql --query 'SELECT 1'
pgx-cli run psql --file tests/debug/q17.sql
pgx-cli logs show <run-id>
pgx-cli logs latest
pgx-cli sync doctor
pgx-cli repo audit-tools
```

`dev gate batch` is the fast handoff and pre-push gate. `dev gate review` is
the explicit final PR review gate and runs the full lint/compile/unit/regression
sequence.

## CMake profiles

Build profiles live in the pgx-lower root `pgx-cli.yaml`.

```sh
pgx-cli config validate
pgx-cli config show --profile debug
pgx-cli config show --sources
pgx-cli dev build explain --profile latency
pgx-cli dev build configure --profile debug
pgx-cli dev build compile --profile debug
pgx-cli dev build install --profile debug
```

Use `dev build explain` before expensive work when changing profiles. The command
prints the resolved inherited profile without running CMake.

## Route Checks

```sh
pgx-cli test route-check \
  --run-name pgx-regression-correctness \
  --profile debug \
  --execution-mode extension-auto \
  --sql-dir tests/pgx-regression/sql \
  --output-dir build-artifacts/make/ptest/extension/pgx-regression/results \
  --summary build-artifacts/test-runs/pgx-regression-correctness/summary.md \
  --default-auto-should-route-to lower \
  --require-route-directives
```

Use `--pg-regress -- <pg_regress command...>` to run pg_regress before checking
route notices.

## Compare-Postgres

```sh
pgx-cli test compare-postgres --workload tpch-correctness
```

Use compare-postgres for row-value truth checks against stock PostgreSQL.
Existing pg_regress `.out` files still catch output shape, route notices, and
harness regressions. A compare-postgres mismatch is a correctness failure; a
route-check mismatch is a routing failure; a `.out` mismatch is fixture/log
evidence until compare-postgres confirms row divergence. File-level workload
exclusions require reasons in `tests/workloads.yaml`; do not add per-query
compare flags.

## PostgreSQL Regression Burndown

```sh
pgx-cli test psql-regression-burndown \
  --source tests/psql-regression \
  --baseline tests/psql-regression/baselines/current.txt \
  --output-dir tests/psql-regression/results \
  --summary build-artifacts/test-runs/psql-regression-burndown/summary.md \
  --route-summary build-artifacts/test-runs/psql-regression-burndown/route-summary.md \
  --pg-regress /usr/local/pgsql/lib/pgxs/src/test/regress/pg_regress \
  --bindir /usr/local/pgsql/bin \
  --dlpath /usr/local/pgsql/lib \
  --schedule parallel_schedule \
  --load-extension pgx_lower
```

Use `--record` only after reviewing the run; it replaces the baseline with the
current failing upstream PostgreSQL test names.
The default `--pg-regress` path matches the pgx-lower dev container; pass an
explicit path when using a different PostgreSQL installation.

## Development

```sh
npm install
npm test
npm run build
npm link
```
