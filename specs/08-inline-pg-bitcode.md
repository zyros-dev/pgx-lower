# Spec 08 — Inline PostgreSQL bitcode

**Tier:** 2 (type-gap track polish)
**Stack on:** 05, 06, 07 (FFI surface must be stable)
**Blocks:** —
**Estimated effort:** 1–2 weeks

## Goal

PG ships its own LLVM bitcode for hot internal functions (date arithmetic,
numeric ops, type input/output) under `$pkglibdir/bitcode/postgres/`. PG's
own JIT (`llvmjit_inline.cpp`) loads this bitcode and inlines it into the
generated code so the optimizer can fold across the FFI boundary.

pgx-lower currently calls those same functions as opaque externs. After
inlining, LLVM can DCE conversion paths, hoist constant computations out of
loops, and combine multi-call expressions into a single fused operation.

## Background

PG's JIT layer's bitcode-inlining file:
`postgres/src/backend/jit/llvm/llvmjit_inline.cpp` — reference implementation.

The bitcode is built during PG's own compile and installed at
`$(pg_config --pkglibdir)/bitcode/postgres/`. It has a `.index.bc` summary
file plus per-source-file `.bc` files. Loading it requires:
1. Find the bitcode dir.
2. Parse the index to map symbol → bitcode file.
3. On encountering an extern call to a known PG function, lazily load the
   bitcode module, link it into our LLVM module, and let the optimizer inline.

After specs 05–07, the FFI surface from JITed code into PG is small and
stable:
- `j2date` (date extraction — used by `extractYear` etc. after spec 06)
- `numeric_in`, `numeric_out`, `numeric_add`, `numeric_mul` etc. (after spec
  06's date work touched only date funcs, numeric stays whichever way spec
  05 leaves it)
- `text_substring`, `cstring_to_text_with_len` (after spec 07)
- `heap_deform_tuple` (called from spec 05's bulk decoder — but this is too
  large to be a good inlining target; skip it)

Inlining `j2date` and the small numeric helpers is the win.

## Design

### 1. Build-time discovery

`pg_config --pkglibdir` gives the directory. Cache it in the
`MLIRRuntime` singleton (spec 01) — it doesn't change at runtime.

### 2. Bitcode loader

New file `src/pgx-lower/execution/jit_engine/pg_bitcode_inliner.cpp`:

```cpp
class PgBitcodeInliner {
public:
    PgBitcodeInliner(llvm::LLVMContext& ctx);
    // Runs as an LLVM module pass: for each declared-but-not-defined
    // function in the module that matches a PG bitcode symbol, link the
    // bitcode module in.
    llvm::Error inline_pg_calls(llvm::Module& m);
private:
    void load_bitcode_index();
    std::unordered_map<std::string, std::string> symbol_to_bc_file_;
    llvm::LLVMContext& ctx_;
};
```

The implementation can crib heavily from `llvmjit_inline.cpp` — it's BSD
licensed.

### 3. Pipeline integration

The inliner runs *before* `buildPerModuleDefaultPipeline` (spec 02). Order
in `JITEngine::create_llvm_optimizer()` (`jit_execution_engine.cpp:265-365`):

```cpp
PgBitcodeInliner inliner(*ctx);
if (auto err = inliner.inline_pg_calls(*module); err) return err;

auto MPM = PB.buildPerModuleDefaultPipeline(O2);
MPM.run(*module, MAM);
```

The default pipeline's `InlinerPass` then does the actual inlining, since the
PG bitcode functions are now defined in our module.

### 4. Allowlist

Don't inline everything — bitcode adds compile time and many PG functions are
massive. Maintain a deliberately small allowlist:

```cpp
static constexpr const char* PG_INLINE_ALLOWLIST[] = {
    "j2date", "date2j",
    "DirectFunctionCall1", "DirectFunctionCall2",
    "numeric_short_to_int128",   // hypothetical helpers from spec 06
    "VARSIZE_4B", "VARSIZE_1B",  // inline length probing — wins big in spec 07's loops
    // add as profiling reveals more wins
};
```

Make it a sorted array, binary search per declared function. Cache results.

### 5. Caching

The bitcode-loaded `llvm::Module`s should be cached in the singleton — parsing
bitcode is not free. After parsing, `llvm::Linker::linkModules` consumes the
module, so cache the *raw bitcode buffer* and re-parse per JIT compilation.
(Or: clone the module each use; benchmark which is faster.)

### 6. Compile-cost ceiling

Inlining adds work at compile time. The win comes when the cached entry
(spec 03) reuses the inlined+optimized engine. **Only enable inlining when
the cache is enabled** — controlled via a new GUC:

```c
DefineCustomBoolVariable("pgx_lower.inline_pg_bitcode",
    "Inline PostgreSQL bitcode into JITed code (requires cache_enabled)",
    NULL, &g_inline_pg_bitcode,
    true,    // default on after the spec lands
    PGC_USERSET, 0, NULL, NULL, NULL);
```

## Files to touch

| File | Change |
|------|--------|
| `src/pgx-lower/execution/jit_engine/pg_bitcode_inliner.cpp` (new) | Loader + linker |
| `include/pgx-lower/execution/pg_bitcode_inliner.h` (new) | Public API |
| `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp:265-365` | Wire inliner before module pipeline |
| `src/pgx-lower/execution/postgres/executor_c.c:79-85` | Add `pgx_lower.inline_pg_bitcode` GUC |
| `include/pgx-lower/execution/mlir_runtime.h` (from spec 01) | Cache bitcode dir + parsed modules |
| `cmake/` | Detect `pg_config --pkglibdir`, sanity check bitcode dir exists |

## Don't change

- The PG bitcode itself (it's in the PG install).
- The runtime FFI surface — this spec is purely a codegen optimisation.

## Acceptance criteria

- Build clean.
- Existing tests pass.
- Inspection: post-optimization IR for q01 should show *no* call to `j2date`
  if `extractYear` is used — the call should be inlined and folded into the
  surrounding loop. Verify via `pgx_lower.log_ir = on`.
- A/B (warm, SF=0.01, 4-query subset):
  - **q01 (date-heavy, after spec 06): ≥5% faster.**
  - q06 (no date functions): unchanged or marginally better (VARSIZE inlining
    if any string columns).
  - q03, q12: any direction; report.
- Compile-cost test: cold q01 should be ≤30% slower than without inlining
  (acceptable given the cache amortizes it).

## Risks

- PG bitcode is built against a specific PG version. If the runtime PG's
  bitcode disagrees with the headers we built against, inlining produces
  wrong code. Verify PG version match at startup; refuse to inline if mismatch.
- Bitcode may reference symbols not present in our module. The linker
  handles unresolved externals (they remain externs); but if a transitive
  dependency goes unresolved, JIT lookup fails at execution. Test with the
  full TPC-H sweep.
- Some PG functions use `PG_TRY` / `PG_CATCH` macros that expand to setjmp.
  Inlined setjmp into a JIT function is dangerous — the JIT fn isn't on PG's
  exception stack. Allowlist only functions known not to use these macros.
  `j2date` is safe; many others aren't.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `08-bitcode`.

Required runs:
1. Standard 4-query sweep.
2. Full 22-query validation.
3. With `pgx_lower.inline_pg_bitcode = off` — must match the spec-07-completed
   baseline.
4. IR inspection on q01 — paste a snippet to PR showing inlined `j2date`.

## Rollback

`SET pgx_lower.inline_pg_bitcode = false` is the runtime kill-switch.
