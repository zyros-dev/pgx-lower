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
and queue operations.

Install or relink it from this checkout:

```bash
pgx-cli setup install
```

Diagnose the local/thor workflow:

```bash
pgx-cli setup doctor
pgx-cli sync status
pgx-cli queue status
```

Use `pgx-cli dev ...` and `pgx-cli queue ...` as the normal workflow commands;
treat raw `ssh comfy`, raw `mutagen`, and raw `tsp` as debugging escape hatches.
The old recipe layer has been retired.

The inherited `tools/` tree is tracked in `docs/tools-ledger.md`; do not delete
entries without updating that ledger.

Cheap PR hygiene checks:

```bash
git diff --check
pgx-cli dev lint diff
pgx-cli dev gate batch
```

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
