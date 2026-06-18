# pgx-lower

pgx-lower (PostgreSQL Extension, Lower e.g. MLIR lowerings) is an extension that rewrites PostgreSQL's execution
engine with a compiler. Read more at https://pgx.zyros.dev/

## Development

Builds run in the dev container on thor through `pgx-cli` (the mac is an edit host;
edits sync via mutagen). All generated build and benchmark scratch output lives
under the gitignored `build-artifacts/` directory; `bench-results/` holds the
committed per-PR benchmark reports.

## pgx-cli workflow

`pgx-cli` is the command surface for pgx-lower development workflows. The CLI
lives in `pgx-cli/` and owns thor, Mutagen, task-spooler, CLion, build, check,
Docker, Postgres, logs, and queue operations.

Install or relink it from this checkout:

```bash
pgx-cli setup install
```

Diagnose the local/thor workflow:

```bash
pgx-cli setup doctor
pgx-cli sync doctor
pgx-cli sync status
pgx-cli queue status
```

Use `pgx-cli dev ...`, `pgx-cli queue ...`, and `pgx-cli run ...` as the normal
workflow commands. Raw `ssh comfy`, `docker`, `psql`, `pg_regress`, `ctest`,
`cmake`, `ninja`, `tsp`, `mutagen`, `just`, and migrated helper scripts are not
normal agent workflow. The old recipe layer has been retired.

When no typed command exists yet, use the bounded gateway:

```bash
pgx-cli run thor -- true
pgx-cli run docker -- bash -lc 'echo ok'
pgx-cli run psql --query 'SELECT 1'
pgx-cli run psql --file tests/debug/q17.sql
```

Mutagen-dependent commands fail before remote execution when session health,
flush, or sync proof fails. Managed commands print a compact summary and store
full transcripts under `.pgx-cli/runs/<run-id>/`:

```bash
pgx-cli logs show <run-id>
pgx-cli logs latest
pgx-cli sync doctor
```

Use `--head N`, `--tail N`, or `--full` when you need a different transcript
preview.

Codex CLI command rules live in `.codex/rules/default.rules`; the project
`.codex/` layer must be trusted for those rules to load.

Workflow entrypoints live in `pgx-cli`. New shell or Python helper scripts need
a plan-level exception and must pass:

```bash
pgx-cli repo audit-tools
```

Cheap PR hygiene checks:

```bash
git diff --check
pgx-cli dev lint diff
pgx-cli dev gate batch
```

The pre-push hook runs `pgx-cli dev gate batch` so normal pushes stay quick.
Run `pgx-cli dev gate review` explicitly before handing a PR to a human.

For row-value correctness, use the stock PostgreSQL oracle gate:

```bash
pgx-cli test compare-postgres --workload tpch-correctness
```

This managed command runs through thor/dev infrastructure from a local checkout
and compares pgx-lower result rows against stock PostgreSQL. Existing `.out`
files remain fixture, route-notice, and harness checks; inspect the
compare-postgres Markdown summary and JSON diff rather than pasting raw result
sets into chat.

## Temporal Type Support

pgx-lower supports DATE, TIMESTAMP WITHOUT TIME ZONE, and INTERVAL only where
the analyzer accepts the operation. TIME, TIMETZ, and TIMESTAMPTZ are
fallback-only until explicit support specs exist. Intervals must preserve
PostgreSQL time/day/month fields; average-month flattening is forbidden.

## String Type Support

PostgreSQL text, varchar, and bpchar values lower as `!db.pg_text`,
`!db.pg_varchar`, and `!db.pg_bpchar` semantic types, preserving OID, typmod,
collation, and nullability metadata. Do not infer PostgreSQL string identity from
`VarLen32`; unsupported string-like PostgreSQL types route fallback, and
supported string operations use PostgreSQL-equivalent runtime semantics or are
rejected by the analyzer.

### IDE setup (compile_commands.json)

The build runs in a Docker container on thor (LLVM 20 / MLIR 20 / PG 17.6 from
source), so the IDE can't drive CMake itself. Instead, every build exports a
`compile_commands.json` and a remote IDE reads it.

- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is set on the debug build profile, so the
  raw DB is re-emitted as a side effect of normal build and test commands.
- The raw Docker DB lands at `build-artifacts/ptest/compile_commands.json` on
  thor and contains `/workspace/...` paths.
- The raw Docker DB can be copied or rewritten for the thor host checkout when
  the IDE needs a refreshed `compile_commands.json`.

First-time bootstrap: run `pgx-cli dev build compile --profile debug` so
the host-path DB exists, then in the IDE:

- CLion: open the project in Compilation Database mode and select
  `compile_commands.json`. If diagnostics look stale after a build or branch
  switch, reload the compilation database project (`Ctrl+Shift+O` or
  Tools | Compilation Database | Reload Compilation Database Project).
- VS Code / clangd: point `clangd` at the same path
  (`--compile-commands-dir=.`).

After that, normal builds keep the file current. CLion may still need a project
reload depending on its per-project auto-reload setting.
