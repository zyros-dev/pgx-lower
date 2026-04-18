---
name: pgx-lower-overview
description: High-level architecture map of pgx-lower. The shape of the codebase, what each top-level directory does, and how a query flows from PostgreSQL through MLIR to JITed native code. Use when orienting in this codebase, when asked broad architectural questions, or as the starting point before diving into a specific area.
---

# pgx-lower architecture overview

pgx-lower is a PostgreSQL extension that intercepts query execution, translates
the plan tree into MLIR, lowers it through the LingoDB dialect chain, and JIT
compiles native code via LLVM. The aim is to beat PG's interpreted executor on
analytical queries.

## Top-level layout

```
pgx-lower/
├── extension/             PG extension scaffolding (control file, SQL install script)
│   ├── CMakeLists.txt    Builds pgx_lower.so (the loaded module)
│   ├── control/          pgx_lower.control template
│   └── sql/              pgx_lower--1.0.sql — runs on CREATE EXTENSION
│
├── src/pgx-lower/         Our own code
│   ├── execution/        executor hook, MLIR runner, JIT engine, MLIR setup
│   ├── frontend/SQL/     PG plan tree → MLIR RelAlg translator
│   ├── runtime/          C runtime called from JITed code (FFI symbols)
│   └── utility/          logging, error handling
│
├── src/lingodb/           Vendored LingoDB snapshot (pre-SubOp)
│   ├── mlir/Dialect/     RelAlg, DB, DSA, util dialects
│   ├── mlir/Conversion/  Lowering passes between dialects
│   ├── mlir-support/     parser/eval helpers
│   └── runtime/          Hashtable, Vector, Sort, Hash — generic runtime
│
├── include/               Headers mirroring src/ layout
├── benchmark/             TPC-H harness, Python orchestration, SQLite results
├── tests/                 Unit (GTest) + SQL regression (pg_regress)
├── docker/                Dev/benchmark/profile containers, Makefile targets
├── cmake/                 FindPostgreSQL, PostgreSQLExtension macros
├── tools/                 build-tools (TableGen helper), profiling utilities
├── resources/sql/tpch/    The 22 TPC-H queries
├── specs/                 Performance roadmap specs (planning docs)
└── lingo-db/              Empty — placeholder for future submodule
```

## Query flow (one paragraph)

PostgreSQL parses+plans a query, then calls the registered `ExecutorRun_hook`.
pgx-lower's `custom_executor` (`src/pgx-lower/execution/postgres/executor_c.c`)
catches it, dispatches into `MyCppExecutor::execute`, which calls
`QueryAnalyzer::analyzePlan` to decide if pgx-lower can handle the query. If
yes, control flows into `mlir_runner::run_mlir_with_dest_receiver`, which (1)
translates the `PlannedStmt` into MLIR RelAlg via `PostgreSQLASTTranslator`,
(2) runs the lowering pipeline RelAlg→DB+DSA+Util→Standard→LLVM through six
PassManagers, and (3) hands the LLVM module to `JITEngine` which compiles and
executes it. The JITed function calls back into our C runtime
(`src/pgx-lower/runtime/`, `src/lingodb/runtime/`) for tuple I/O, hashtables,
sort, and type conversions. Results stream into a `DestReceiver`. If anything
fails or the query isn't supported, control falls through to PG's
`standard_ExecutorRun`.

## The four key abstractions

**1. The executor hook** — `executor_c.c:69-90` installs `custom_executor` as
`ExecutorRun_hook` in `_PG_init`. This is the only PG integration point.
Everything else is downstream of this hook firing.

**2. The MLIR pipeline** — three phases of lowering (RelAlg → DB+DSA+Util →
Standard → LLVM), implemented in `mlir_runner_phases.cpp`. Each phase builds
fresh PassManagers per query (no caching today — see specs/03).

**3. The JIT engine** — `jit_execution_engine.cpp` wraps `mlir::ExecutionEngine`
with LLVM-level optimization passes. Calls `main` and `rt_set_execution_context`
in the JITed module to drive execution.

**4. The runtime FFI surface** — JITed code calls back into C via globally
exported symbols (`-Wl,--export-dynamic`). Hot paths: `get_*_field_mlir`,
`extract_field<T>`, `numeric_to_i128`, `Hashtable::*`, `PgSortState::*`,
`add_tuple_to_result`.

## Where to go next

| Question | Skill |
|----------|-------|
| How does the executor hook → JIT chain actually work? | pgx-lower-execution-path |
| How does PG's plan tree become MLIR RelAlg? | pgx-lower-ast-translation |
| What are RelAlg / DB / DSA / util? How do they lower? | pgx-lower-mlir-dialects |
| What runtime functions does the JIT call? Where do conversions happen? | pgx-lower-runtime-ffi |
| What does JITEngine actually do with the LLVM module? | pgx-lower-jit-compilation |
| How do I build / test this? | pgx-lower-build-and-test |
| How do I benchmark it / A/B test a change? | pgx-lower-benchmarks |
| Why Clang? Why these LLVM/PG versions? What lessons from history? | pgx-lower-versions-and-history |

## High-level facts worth knowing up front

- **Build host**: macOS edits, thor (Linux Ryzen) compiles. Mutagen syncs the
  tree; everything builds inside `pgx-lower-dev` Docker container.
- **Compiler**: Clang only. GCC 14.2.0's `-O3` infinite-loops in
  `mlir::reconcileUnrealizedCasts`. CMakeLists.txt enforces this hard.
- **Versions**: LLVM 20, MLIR 20, PG 17.6, C++20, CMake 3.22+. All pinned in
  `docker/dev/Dockerfile`.
- **What's missing vs LingoDB upstream**: SubOp dialect. Modern LingoDB's
  vectorisation lives there; we don't have it. See `pgx-lower-mlir-dialects`.
- **What's missing vs LLVM defaults**: Auto-vectorization is off in
  `jit_execution_engine.cpp:306-308`. See specs/02.
- **What's missing vs ideal**: No compile cache (every query re-MLIRs and
  re-LLVMs). See specs/03.

## Things this codebase intentionally doesn't do

- **No parallel query support.** `T_Gather` and `T_GatherMerge` are
  pass-through in the translator; the JIT runs single-threaded per backend.
- **No write path.** SELECT only. INSERT/UPDATE/DELETE fall through to PG.
- **No columnar storage.** PG heap is the source of truth; we transpose on
  read if needed (spec 10).
- **No FULL OUTER JOIN, no T_Result, no T_Group, no T_SetOp.** Translator
  rejects these; PG handles them.

## Related context files

- `CLAUDE.md` (if present) — project-specific instructions for Claude.
- `specs/00-dag.md` — the performance-roadmap DAG. Read this when asked
  "what should we work on next?"
- `specs/ab-test-template.md` — A/B procedure for any benchmark-driving change.
