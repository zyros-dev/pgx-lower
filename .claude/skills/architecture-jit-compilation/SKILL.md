---
name: architecture-jit-compilation
description: The LLVM JIT engine, optimization pipeline, ExecutionEngine lifecycle, MLIR→LLVM IR translation, and the per-query compile flow. Use when working on jit_execution_engine.cpp, LLVM passes, JIT compilation issues, ExecutionEngine ownership, function lookup, or symbol resolution.
---

# JIT compilation

After phase 3c, the MLIR module is pure LLVM dialect. `JITEngine` translates
it to LLVM IR, runs optimization passes, hands it to `mlir::ExecutionEngine`
for ORC-based JIT compile, looks up the entry symbols, and runs them.

## File layout

- `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` — the engine.
- `include/pgx-lower/execution/jit_execution_engine.h` — header with class
  members.
- `src/pgx-lower/execution/mlir_runner.cpp:151-165` — the caller
  (`executeJITWithDestReceiver`).

## Public API

```cpp
class JITEngine {
public:
    explicit JITEngine(llvm::CodeGenOptLevel opt_level);
    bool compile(mlir::ModuleOp module);            // 69-114
    bool execute(void* estate, void* dest) const;   // 116-182
    bool link_static();                              // 393-413 — alternative path

private:
    std::unique_ptr<mlir::ExecutionEngine> engine_;
    void* main_fn_ = nullptr;
    void* set_context_fn_ = nullptr;
    bool compiled_ = false;
    llvm::CodeGenOptLevel opt_level_;
};
```

`opt_level` is one of `None / Less / Default (=O2) / Aggressive (=O3)`.
Today: hardcoded `Default` at `mlir_runner.cpp:152`.

## Compile flow (`compile()` — line 69-114)

1. Verify the input module (`pgx_lower::log::verify_module_or_throw`).
2. `register_dialects(module)` (line 197-207) — appends LLVM IR translation
   interfaces to the context. Done per-call today; spec 01 hoists it.
3. Create `MLIR→LLVM translator` lambda (line 209-263) — captures
   `mlir::translateModuleToLLVMIR`. Verifies the LLVM module after translation
   (in non-Release builds). Sets `main` and `_mlir_ciface_main` to
   `ExternalLinkage` so they're addressable.
4. Create `LLVM optimizer` lambda (line 265-365) — see "Optimization" below.
5. `mlir::ExecutionEngine::create(module, options)` (line 93) — invokes the
   translator + optimizer, then ORC-JITs. Owns the resulting native code.
6. `lookup_functions()` (line 367-391) — resolves `main` (or fallback
   `_mlir_ciface_main`) and `rt_set_execution_context` to raw void pointers.

## Execute flow (`execute()` — line 116-182)

1. Save `CurrentMemoryContext`.
2. PG_TRY block:
   - Cast `set_context_fn_` to `void(*)(void*)`, call with `estate`.
   - Cast `main_fn_` to `void(*)()`, call. **The JITed module has no
     parameters.** All input is via globals (estate via
     `rt_set_execution_context`, output via `g_tuple_streamer`).
   - `CHECK_FOR_INTERRUPTS()` for SIGINT handling.
3. PG_CATCH: copy ereport data, log it, restore memory context, re-throw via
   `PG_RE_THROW`.
4. Restore `CurrentMemoryContext`.

## LLVM optimization (line 265-365)

The optimizer runs inside the `ExecutionEngine::create` call. Today's pipeline
is **manually assembled, not the standard PassBuilder pipeline**.

Pipeline tuning options (line 305-308):
```cpp
PTO.LoopUnrolling      = false;   // ⚠ DISABLED
PTO.LoopVectorization  = false;   // ⚠ DISABLED
PTO.SLPVectorization   = false;   // ⚠ DISABLED
```

`spec 02` flips these.

Hand-rolled `FunctionPassManager` (line 324-337):
- SROA (with ModifyCFG)
- InstCombine
- Promote
- LICM (with MemorySSA)
- Reassociate
- GVN
- SimplifyCFG

Run per-function with skip for declarations and `optnone` functions.

This is roughly an O1-ish subset, missing the vectorization/loop-opt passes
the standard `buildPerModuleDefaultPipeline(O2)` would include.

`TargetMachine` (line 286-303): created per-compile from
`llvm::sys::getDefaultTargetTriple()` and `getHostCPUName()`. Spec 01 hoists
this to a singleton.

`install_fatal_error_handler` (line 280-283): catches LLVM aborts and routes
them to PGX_ERROR before the abort actually fires.

## Symbol resolution

`lookup_functions()` (line 367-391):
- Looks up `main`, falls back to `_mlir_ciface_main`.
- Looks up `rt_set_execution_context`.

The MLIR-emitted entry function is named `main`. The `_mlir_ciface_*` prefix
is MLIR's C-interface convention for functions that take MLIR aggregate types
in their signature; for our parameterless `main()` it's a no-op alias.

Other runtime symbols (Hashtable::create, get_int64_field_mlir, etc.) are
resolved by `mlir::ExecutionEngine` against the host process's symbol table —
which is why everything is built with `-Wl,--export-dynamic`.

## Static linking path (`link_static()` — line 393-413)

An alternative to ORC JIT: dump the compiled object file via
`engine_->dumpToObjectFile`, compile with `g++ -shared`, `dlopen` the result.

Slower than JIT but produces a real `.so` for inspection. Used for debugging,
not the production path.

## What's per-query and what's per-process

| Object | Lifetime today | Should be |
|--------|----------------|-----------|
| `MLIRContext` | per-query (mlir_runner.cpp:70) | per-process (spec 01) |
| `DialectRegistry` | per-query (registered each call) | per-process (spec 01) |
| `llvm::TargetMachine` | per-compile (line 286-303) | per-process (spec 01) |
| `JITEngine` | per-query (mlir_runner.cpp:152) | per-cache-entry (spec 03) |
| `mlir::ExecutionEngine` | owned by JITEngine, per-query | per-cache-entry (spec 03) |
| LLVM target init | per-process (`initialized` static guard at line 185-191) | already correct |

## Compile cost (typical, from thesis Q20 numbers)

- MLIR pipeline (phases 3a-c): ~236 ms
- LLVM codegen + JIT: ~15 ms
- Total compile: ~250 ms
- Execution: ~929 ms

So compile is ~21% of cold-query wall time. Spec 03 caches it away. Spec 02
adds ~30-80ms more compile time but produces faster code.

## Verifying generated code

To dump LLVM IR after optimization, set `pgx_lower.log_ir = on`. The dump
runs at `jit_execution_engine.cpp:352` via `mlir_runner::dumpLLVMIR`.

To dump MLIR IR between phases, set `pgx_lower.log_debug = on`. The phase
files (`mlir_runner_phases.cpp`) call `dumpModuleWithStats` between every
pass group.

## Common gotchas

- **`engine_->lookup` returns `Expected<>`** — uninspected `Expected` aborts.
  Use `if (auto x = engine_->lookup(...))` and handle the error case.
- **MLIR aggregate args** — if you ever change `main`'s signature, the
  `_mlir_ciface_main` wrapper auto-generated by MLIR may be the right entry
  point instead. Both are looked up.
- **TargetMachine outliving the optimizer** — today the lambda owns the TM
  via local stack capture. The lambda is consumed by `ExecutionEngine::create`
  before the local goes out of scope, so it works. After spec 01 (hoisted TM),
  the singleton owns the TM and the lambda holds a raw pointer; that's still
  correct because the singleton outlives every JITEngine.
- **`enableObjectDump = true`** (line 91): keeps the compiled object in
  memory for `dumpToObjectFile`. Costs RSS. Acceptable today; revisit if
  caching many engines.
- **PG_TRY across the JIT call**: PG's setjmp/longjmp can long-jump *out of*
  the JIT call. Cleanup of MLIR-owned state is responsibility of the catch
  block. Don't add C++ destructors that run only on normal return.
- **The JITed `main` has no params, no return**. Don't try to pass arguments;
  globals are the only input/output channel.

## Adding LLVM-level optimizations

To add a new pass to the manual pipeline (line 324-337):
- Add `FPM.addPass(YourPass());`
- Make sure the analysis it depends on is registered (e.g.,
  `MemorySSAAnalysis` for LICM).

To switch to the standard pipeline (spec 02):
```cpp
auto MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
MPM.run(*module, MAM);
```
This is one of the cheapest perf wins available — flip the three PTO flags
to true and use `buildPerModuleDefaultPipeline`.

## Related skills

- `architecture-execution-path` — what calls `JITEngine::compile` / `execute`.
- `architecture-mlir-dialects` — what the input LLVM-dialect module looks like.
- `architecture-runtime-ffi` — what symbols the JITed code resolves at runtime.
- `architecture-versions-and-history` — LLVM 20 specifics, pass manager
  migration history (commits 7816cd2, f8ff190, 5add478).
