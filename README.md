# pgx-lower

pgx-lower (PostgreSQL Extension, Lower e.g. MLIR lowerings) is an extension that rewrites PostgreSQL's execution
engine with a compiler. Read more at https://pgx.zyros.dev/

## Development

Builds run in the dev container on thor, driven by `just` (the mac is an edit host;
edits sync via mutagen). All generated build trees live under the gitignored
`build-artifacts/` directory; `bench-results/` holds the committed per-PR benchmark
baseline history.

### IDE code-resolution (CLion / VS Code)

Every build (`just compile` / `just test` / `just utest-pg`) configures CMake with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, so `build-artifacts/ptest/compile_commands.json`
is re-emitted as a side effect — there is no separate "generate the JSON" step.

Point a remote IDE (CLion via Gateway, or VS Code Remote-SSH on thor) at that file
once:

- CLion: open the project as a Compilation Database, or set the compile-commands path
  to `build-artifacts/ptest/compile_commands.json`.
- VS Code / clangd: set `compileCommandsDir` to `build-artifacts/ptest`.

After that, a normal `just compile` keeps the database current and the IDE auto-reloads
it — no manual refresh. The DB lives in the mutagen-ignored `build-artifacts/`, so the
supported path is a remote IDE on thor (a local mac IDE can't resolve the containerized
toolchain).

## IDE setup (compile_commands.json)

The build runs in a Docker container on thor (LLVM 20 / MLIR 20 / PG 17.6 from
source), so the IDE can't drive CMake itself. Instead, every build exports a
`compile_commands.json` and a remote IDE reads it.

- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` is set on the configure in `just compile`,
  `just test`, and `just utest-pg`, so the DB is re-emitted as a side effect of
  any normal build. There is no separate "generate the JSON" step.
- The DB lands at `build-artifacts/ptest/compile_commands.json` on thor.
  `build-artifacts/` is mutagen-ignored, so it does not sync to the mac — point a
  *remote* IDE (CLion Gateway or VS Code Remote-SSH on thor) at it.

First-time bootstrap: run `just compile` once so the DB exists, then in the IDE:

- CLion: open the project in Compilation Database mode and select
  `build-artifacts/ptest/compile_commands.json`.
- VS Code / clangd: point `clangd` at the same path
  (`--compile-commands-dir=build-artifacts/ptest`).

After that it auto-reloads when the file changes — no manual refresh.
