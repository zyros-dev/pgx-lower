# Spec 01 — Hoist MLIR/JIT state out of the per-query path

**Tier:** 1 (free wins)
**Stack on:** `main`
**Blocks:** 03 (compile cache), simplifies 04
**Estimated effort:** 1–2 days

## Goal

Stop allocating a fresh `MLIRContext`, dialect registry, target machine, and
LLVM target initialisation on every query. These are expensive setup objects
that are functionally identical across queries and can live for the lifetime
of the backend.

This is also a precondition for spec 03 (the compile cache stores
`mlir::ExecutionEngine` instances which hold dialect-typed Operations and
*must* share a context with the cache key's IR).

## Background

Today, every query hits this allocation pattern (`src/pgx-lower/execution/mlir_runner.cpp:60-148`):

```cpp
::mlir::MLIRContext context;          // line 70 — fresh per-query
setupMLIRContextForJIT(context);      // line 71 — re-loads all dialects
auto translator = create_postgresql_ast_translator(context);
// ...
runCompleteLoweringPipeline(*module); // creates 6 fresh PassManagers per query
                                       // (mlir_runner_phases.cpp lines 45, 62, 98, 117, 136, 166)
JITEngine engine(llvm::CodeGenOptLevel::Default);  // mlir_runner.cpp:152
                                                    // → setup_llvm_target() at jit_execution_engine.cpp:184
```

Inside `JITEngine::create_llvm_optimizer()` (`jit_execution_engine.cpp:286-303`)
the `TargetMachine` is also rebuilt every compile.

`setup_llvm_target()` already has an `initialized` static guard
(`jit_execution_engine.cpp:185-191`), so target init is once-only — but the
*MLIR* target init (`mlir::registerAllToLLVMIRTranslations` at line 199) and
dialect registration are not.

## What to change

### 1. Long-lived MLIR runtime singleton

Create a new file `src/pgx-lower/execution/mlir_setup/mlir_runtime.cpp`
(and matching header in `include/pgx-lower/execution/`) exposing:

```cpp
namespace pgx_lower::execution {

// Process-lifetime owners. Constructed on first use, destroyed at _PG_fini.
struct MLIRRuntime {
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    std::unique_ptr<llvm::TargetMachine> target_machine;
    // future: cache map (spec 03)
};

MLIRRuntime& get_mlir_runtime();   // lazy-init singleton
void shutdown_mlir_runtime();      // called from _PG_fini
}
```

The constructor must:
- Run `llvm::InitializeNativeTarget*` once (move from `JITEngine::setup_llvm_target`).
- Build the `DialectRegistry` once (move logic from `JITEngine::register_dialects`,
  `mlir_runner_passes.cpp:setupMLIRContextForJIT`, and
  `mlir_runner_passes.cpp:initialize_mlir_context`).
- `appendDialectRegistry(registry)` and load all dialects on the context once.
- Build the `TargetMachine` once (move logic from `create_llvm_optimizer` lines 286–303).

### 2. Replace per-query construction

| File | Line | Change |
|------|------|--------|
| `src/pgx-lower/execution/mlir_runner.cpp` | 70 | Replace `MLIRContext context;` with `auto& rt = get_mlir_runtime(); auto& context = rt.context;` |
| `src/pgx-lower/execution/mlir_runner.cpp` | 71 | Delete `setupMLIRContextForJIT(context)` call (now in singleton init) |
| `src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp` | 52–94 | Either delete `setupMLIRContextForJIT` or have it become a no-op asserting the runtime is initialised |
| `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` | 184–195 | Delete `setup_llvm_target` body — singleton owns it |
| `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` | 286–303 | `TargetMachine` lookup → `rt.target_machine.get()` instead of constructing locally |
| `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` | 197–207 | Delete `register_dialects` call from `JITEngine::compile` (now context-wide) |

### 3. PassManagers stay per-query for now

Do **not** try to reuse `PassManager` instances. They're cheap relative to
LLVM codegen, and the LingoDB pipeline builders (`createRelAlgToDBPipeline`
etc.) don't expose a clean reset path. Leave the six `PassManager` constructions
in `mlir_runner_phases.cpp` alone. Spec 03 will move past them via caching.

### 4. Per-query state that must still reset

The per-query setup uses several mutable globals that survive across queries.
Document them at the top of the new singleton header so spec 03 inherits the
list:

- `g_execution_context` (`PostgreSQLRuntime.cpp:48-50`) — set per-query via
  `rt_set_execution_context(estate)`. Already called from JITEngine::execute.
- `g_tuple_streamer` (`tuple_access.cpp:47`) — initialized/shutdown per-query
  by `my_executor.cpp` (lines 240–253). Leave behaviour unchanged.
- `g_current_tuple_passthrough`, `g_computed_results`, `g_jit_table_oid`
  (`tuple_access.cpp`) — per-query state, owned by the executor not the JIT.

### 5. Init/shutdown wiring

`_PG_init` (`src/pgx-lower/execution/postgres/executor_c.c:69-90`) currently
calls `initialize_mlir_passes()`. Add immediately after it:

```c
initialize_mlir_runtime();   // wraps get_mlir_runtime() to force first-use init
```

Add `shutdown_mlir_runtime()` to `_PG_fini` (`executor_c.c:92-95`).

### 6. Thread safety

The PostgreSQL backend is single-threaded per process. Don't add locks. Add a
comment to the header stating "single-threaded backend; do not call from a
parallel worker". Parallel workers don't go through the executor hook in the
current code path (verified — see `query_analyzer.cpp:151-233` analyzeNode
which treats `T_Gather` parents specially), but state this explicitly.

## Acceptance criteria

- Build clean on thor (`make build-release` in `docker/`).
- Existing tests pass (`docker exec pgx-lower-dev bash -c "cd /workspace/build-docker-ptest && ctest"`).
- A/B numbers per template show neutral-to-better warm performance — this spec
  alone won't move the needle dramatically; it pays off via spec 03. Acceptance:
  **no warm regression > 2%**, cold runs neutral or improved.
- New singleton has no static-destruction-order issues at backend exit
  (the destructor must not log via `PGX_LOG` — by the time `_PG_fini` runs
  the logging GUCs may already be torn down).

## Risks

- MLIRContext shared across queries means dialect mutations (registerInterface
  etc.) persist. Audit `mlir_runner_passes.cpp:84-91` and the LingoDB dialect
  init paths for any per-query dialect registration. There shouldn't be any —
  if there is, it's a bug surfaced by this change.
- LLVM `TargetMachine` ownership: `mlir::ExecutionEngine` doesn't take ownership
  of an externally-supplied TargetMachine. We pass a raw pointer in
  `create_llvm_optimizer`; that stays valid because the singleton outlives
  every JITEngine. Document this in the header.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `01-hoist`.

Expected impact for this spec alone:
- Cold queries: 5–20 ms shaved per query (dialect/target init time).
- Warm queries: minimal direct impact (these costs were small relative to LLVM
  codegen). The win compounds with spec 03.

## Rollback

Revert the singleton — all per-query allocations go back to where they were.
No data migration. Branch is fully self-contained.
