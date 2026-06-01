# pgx-lower

pgx-lower (PostgreSQL Extension, Lower e.g. MLIR lowerings) is an extension that rewrites PostgreSQL's execution
engine with a compiler. Read more at https://pgx.zyros.dev/

## Development

Builds run in the dev container on thor, driven by `just` (the mac is an edit host;
edits sync via mutagen). All generated build and benchmark scratch output lives
under the gitignored `build-artifacts/` directory; `bench-results/` holds the
committed per-PR benchmark reports.

The inherited `tools/` tree is tracked in `docs/tools-ledger.md`; do not delete
entries without updating that ledger.

Cheap PR hygiene checks:

```bash
git diff --check
just check-diff
just ffix-diff  # apply clang-format to changed C/C++ hunks
```

### IDE setup (compile_commands.json)

The build runs in a Docker container on thor (LLVM 20 / MLIR 20 / PG 17.6 from
source), so the IDE can't drive CMake itself. Instead, every build exports a
`compile_commands.json` and a remote IDE reads it.

- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is set on the configure in `just compile`,
  `just test`, and `just utest-pg`, so the raw DB is re-emitted as a side effect
  of any normal build.
- The raw Docker DB lands at `build-artifacts/ptest/compile_commands.json` on
  thor and contains `/workspace/...` paths.
- Those normal build recipes automatically rewrite the DB for the thor host
  checkout and write `compile_commands.json` at the repo root. `just clion-db`
  exists only as a manual repair/bootstrap command.

First-time bootstrap: run any normal build command (`just compile` is enough) so
the host-path DB exists, then in the IDE:

- CLion: open the project in Compilation Database mode and select
  `compile_commands.json`. If diagnostics look stale after a build or branch
  switch, reload the compilation database project (`Ctrl+Shift+O` or
  Tools | Compilation Database | Reload Compilation Database Project).
- VS Code / clangd: point `clangd` at the same path
  (`--compile-commands-dir=.`).

After that, normal builds keep the file current. CLion may still need a project
reload depending on its per-project auto-reload setting.
