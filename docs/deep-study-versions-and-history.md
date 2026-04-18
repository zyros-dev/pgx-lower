# Deep study: pgx-lower versions and lessons from history

Reference doc for the `pgx-lower-versions-and-history` skill. Long-form
content; the skill is the short summary.

Written 2026-04-19. Sources cited inline.

---

## Part 1: Version pinning

### What's pinned and where

`docker/dev/Dockerfile`:

```dockerfile
ENV LLVM_VERSION=20
ENV PG_VERSION=17.6
```

Tied together via `apt.llvm.org/llvm.sh 20` and a from-source PG 17.6 build.
CMake 3.31.6 is downloaded explicitly. Ubuntu 24.04 base.

`CMakeLists.txt`:
- `cmake_minimum_required(VERSION 3.22)`
- `set(CMAKE_CXX_STANDARD 20)` — C++20 required
- Hard requirement on Clang (lines 19-35)
- LLVM/MLIR found via `find_package(LLVM REQUIRED)` and the hardcoded
  `MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`

The Postgres path is `/usr/local/pgsql` (from `cmake/PostgreSQLConfig.cmake`
line 1, hardcoded). PG is built with `--with-openssl --with-icu`.

### Why Clang only, not GCC

`CMakeLists.txt:19-35`:

```cmake
# Enforce Clang compiler due to GCC 14.2.0 -O3 bug
# (infinite loop in mlir::reconcileUnrealizedCasts)
```

GCC 14.2.0's `-O3` infinite-loops on certain MLIR template patterns. The
project enforces Clang via a hard `FATAL_ERROR` if `CMAKE_CXX_COMPILER_ID`
isn't `Clang`. Don't try to override.

The Dockerfile installs both `gcc-14` and `clang-20` and uses
`update-alternatives` to make `clang` the default `cc/c++`.

---

## Part 2: LLVM 20 (20.1.0, released 2025-03-04)

Sources:
- https://releases.llvm.org/20.1.0/docs/ReleaseNotes.html
- https://releases.llvm.org/20.1.0/tools/clang/docs/ReleaseNotes.html
- https://llvm.org/docs/NewPassManager.html
- https://github.com/llvm/llvm-project/blob/release/20.x/llvm/include/llvm/Passes/PassBuilder.h

### PassBuilder / PassManager — what changed

**No source-breaking PassBuilder API changes** vs LLVM 18/19 for our usage in
`jit_execution_engine.cpp`. Constructor signature unchanged:

```cpp
explicit PassBuilder(TargetMachine *TM = nullptr,
                     PipelineTuningOptions PTO = PipelineTuningOptions(),
                     std::optional<PGOOptions> PGOOpt = std::nullopt,
                     PassInstrumentationCallbacks *PIC = nullptr);
```

The `register*Analyses + crossRegisterProxies` pattern at
`jit_execution_engine.cpp:310-321` is forward-compatible.

**Extension-point callbacks available in 20** (full list in `PassBuilder.h`):

- `registerPipelineStartEPCallback` — runs before `buildPerModuleDefaultPipeline`.
  This is the right hook for spec 08 (PG bitcode inlining).
- `registerOptimizerEarlyEPCallback`, `registerOptimizerLastEPCallback`
- `registerPeepholeEPCallback`, `registerLateLoopOptimizationsEPCallback`
- `registerVectorizerStartEPCallback` — relevant for spec 09 if we want to
  inject custom vector lowerings before the standard vectorizer.

**Pass set we currently use is unchanged in 20**:
- `SROAPass(SROAOptions::ModifyCFG)`
- `InstCombinePass`, `PromotePass`, `LICMPass(LICMOptions())`
- `createFunctionToLoopPassAdaptor(..., /*UseMemorySSA=*/true)`
- `ReassociatePass`, `GVNPass`, `SimplifyCFGPass`

**Notable new pass**: `IRNormalizerPass` (`-passes=normalize`) — reorders
instructions while preserving semantics. Useful as a *debug tool* for
diffing two pipeline outputs, not for the production pipeline.

### PipelineTuningOptions

Full struct (release/20.x):

```cpp
bool LoopInterleaving;
bool LoopVectorization;
bool SLPVectorization;
bool LoopUnrolling;
bool ForgetAllSCEVInLoopUnroll;
unsigned LicmMssaOptCap;
unsigned LicmMssaNoAccForPromotionCap;
bool CallGraphProfile;
bool UnifiedLTO;
bool MergeFunctions;
int InlinerThreshold;          // -1 = use opt-level default
bool EagerlyInvalidateAnalyses;
```

We currently disable `LoopUnrolling`, `LoopVectorization`, `SLPVectorization`
at `jit_execution_engine.cpp:306-308`. **No defaults changed for these between
18 and 20.** When spec 02 enables them, also leave `LoopInterleaving` on (its
default tracks `LoopVectorization`).

### ORC JIT / ExecutionEngine

`mlir::ExecutionEngine` is **still built on LLJIT/RuntimeDyld** in LLVM 20.
JITLink migration is in flight but hasn't reached MLIR's wrapper.

API unchanged for our usage:
- `mlir::ExecutionEngine::create(module, options)` — same signature.
- `ExecutionEngineOptions{llvmModuleBuilder, transformer, jitCodeGenOptLevel,
   enableObjectDump}` — same fields.
- `engine_->lookup("symbol")` returns `Expected<void*>` — handle the `Expected`.

**Object cache** (`ExecutionEngineOptions::cache`,
`std::unique_ptr<ObjectCache>`) is unchanged. **This is the integration point
for spec 03's compile cache**. Implement an `ObjectCache` subclass; LLVM
serializes compiled object files transparently.

**Symbol resolution**: nothing changed. Our `-Wl,--export-dynamic` linking
strategy continues to make runtime symbols resolvable via dlsym from the
host process.

**Lazy compilation**: ORC's `LazyCallThroughManager` exists but MLIR's
`ExecutionEngine` does eager compilation; LLVM 20 doesn't change this.

### Vectorization improvements

If/when spec 02 flips the flags on:

**Loop vectorizer in 20**:
- VPlan now fully owns def-use chains — better cost decisions, fewer
  pathological vectorized loops.
- More cases of partial-load/strided-access vectorize correctly.
- EVL-based VPlans (RVV/SVE; not us on x86) compute AVL correctly.

**SLP vectorizer in 20**:
- Improved load-instruction ordering and clustering — Arm's writeup cites
  ~3% on SPEC `525.x264_r`.
- Better handling of reorderable trees; fewer bailouts on FMA chains.

For pgx-lower's emitted IR (small straight-line code with predicate masks
for nulls), **SLP is the cheaper-and-more-relevant first try**. Loop
vectorization carries higher compile-time cost and rarely helps unless the
loop body is large; pair with `LoopUnrolling=true` if enabled.

### Bitcode loading / inlining (spec 08)

For inlining PG's bitcode (`$pkglibdir/bitcode/postgres/*.bc`):

- **Opaque pointers** stable since LLVM 16. No `--force-opaque-pointers`
  upgrade dance needed. Our PG is built against LLVM 20 → all opaque.
- **`x86_mmx` removed in LLVM 20** but auto-upgraded to `<1 x i64>`. PG
  bitcode doesn't use `x86_mmx`. No impact.
- **Recursive types removed in LLVM 20** — verifier rejects them. PG bitcode
  doesn't have any.
- **`captures(none)` attribute** landed experimentally in 20; auto-upgrade
  from `nocapture` is in 21. On 20, both spellings accepted. PG bitcode
  emits `nocapture`. No action.
- **`Linker::linkModules`** with `LinkOnlyNeeded` — unchanged API. The
  `llvmjit_inline.cpp` pattern works as-is.
- **`getLazyBitcodeModule`** — unchanged. Lazy materialization works for
  per-file `.bc` modules.

### Clang 20 — what affects our build

**On by default that may bite C++20 code**:
- `-Wdeprecated-literal-operator` is **on**. `operator"" _foo` (with space)
  warns; under `-Werror` it's an error. Fix: `operator""_foo`.
- `-Wenum-constexpr-conversion` is **no longer suppressible**. Casting an
  int outside enum range to an enum in a constexpr context is now a hard
  error (was suppressible warning in 16-19).
- Extraneous `template <>` heads are now ill-formed by default.
- `[[clang::lifetimebound]]` on void-returning functions is now an error
  (was silently ignored).

**Pointer-overflow / strict-aliasing tightening (relevant for FFI)**:
- Clang 20 emits **distinct TBAA tags for incompatible pointer types** by
  default. May silently change behaviour of strict-aliasing-violating code.
  Workaround: `-fno-pointer-tbaa`. PG itself compiles with
  `-fno-strict-aliasing`; consider matching if FFI miscompiles surface.
- `-fwrapv` no longer implies `-fwrapv-pointer`. `-fno-strict-overflow`
  now means `-fwrapv -fwrapv-pointer` (matches GCC).
- Pointer-add overflow used more aggressively for optimization.
  `ptr + unsigned_offset < ptr` optimizes to `false`. Use
  `(uintptr_t)ptr + offset < (uintptr_t)ptr` for overflow checks.

**ABI**:
- Itanium mangling fix for construction vtable name. Incompatibility with
  Clang 19-built objects unless `-fclang-abi-compat=19`. Only matters for
  prebuilt `.o` files; our build is from source.

**Removed flag warnings**:
- `-Wenum-constexpr-conversion` (and `-Wno-`/`-Wno-error-` forms) gone.
  Audit our build flags.

### Known LLVM 20 bugs

- **`ExecutionEngine/OrcLazy` test failures in 20.1.0-rc1** (#125393) —
  cosmetic; we don't use OrcLazy.
- **mlir-runner static destructors not run** (#100414) — resource leaks.
  We use in-process API, less impact.
- **mlir JitRunner memory leak** (#114018) — open. Per-compile leak.
  Mitigation: rebuild MLIR context per N queries (spec 01 already hoists
  context; spec 03's cache eviction can serve as the N-trigger).
- **CPU feature mis-detection in ORC** (#130509). Affects users of
  `JITTargetMachineBuilder::detectHost()`. We use
  `llvm::sys::getHostCPUName()` directly (`jit_execution_engine.cpp:297`).
  **We're safe.**
- **MLIR ExecutionEngine on RuntimeDyld** (still). JITLink migration coming
  in 21+; pin tests when it lands.

---

## Part 3: MLIR 20

Sources:
- https://mlir.llvm.org/docs/ReleaseNotes/
- https://mlir.llvm.org/docs/DialectConversion/
- https://mlir.llvm.org/docs/PassManagement/
- https://mlir.llvm.org/docs/Dialects/LLVM/
- PRs #116470, #121389, #119975, #121440, #123776

### The biggest single thing: 1:N dialect conversion

The 1:1 and 1:N dialect-conversion drivers were **merged in LLVM 20**. Every
`ConversionPattern`/`OpConversionPattern` now has a 1:N overload.

If a `TypeConverter` declares any 1:N rule (one source type → several target
types), or if a pattern calls `rewriter.replaceOpWithMultiple(...)`, every
pattern that touches such a value must implement the 1:N overload. Otherwise:

```
fatal: pattern '<name>' does not support 1:N conversion
```

For pgx-lower this is the most likely upgrade-time breakage. Our `RelAlgToDB`,
`DBToStd`, `DSAToStd` patterns are 1:1 today. If we ever pull in a newer
upstream pattern (e.g. via `populateFuncToLLVMConversionPatterns` for variadic
results, or vector lowerings) that introduces a 1:N rule, every connected
pattern in the same converter call needs the new overload.

Old (still works for 1:1):
```cpp
LogicalResult matchAndRewrite(Operation *op,
                              ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) const override;
```

New 1:N overload (preferred when the converter might split values):
```cpp
LogicalResult matchAndRewrite(Operation *op,
                              ArrayRef<ValueRange> operands,
                              ConversionPatternRewriter &rewriter) const override;
```

Default 1:N implementation calls `getOneToOneAdaptorOperands(operands, oneToOne)`
and falls through to the 1:1 version. **Fails fatally if any operand has
cardinality > 1**.

### `TypeConverter::addArgumentMaterialization` deprecated

Argument materializations were a workaround for the lack of 1:N support.
Replace with `addSourceMaterialization`. Target stays `addTargetMaterialization`.

For pgx-lower: scan our `LLVMTypeConverter` subclasses and per-dialect type
converters in `RelAlgToDB`, `DBToStd`, `DSAToStd`. If `addArgumentMaterialization`
appears, swap it.

### Standalone 1:N driver removed

`mlir/include/mlir/Transforms/OneToNTypeConversion.h` is gone. Removed APIs:
`OneToNTypeMapping`, `OneToNPatternRewriter`, `OneToNConversionPattern`,
`OneToNOpConversionPattern`, `applyPartialOneToNConversion`,
`addArgumentMaterialization`, `materializeArgumentConversion`,
`populateSCFStructuralOneToNTypeConversions` (replaced by
`populateSCFStructuralTypeConversions`).

We don't use any of these directly, so no migration needed — but if a future
agent looks at upstream LingoDB and copies a pattern, it might.

### Vector dialect changes

If we ever load `populateVectorToLLVMConversionPatterns` (spec 09):

It's now **strictly conversion patterns**. Rewrite patterns it used to bundle
(FMA rank-reduction, insert/extract strided-slice transforms, vector step
lowering, vector transfer lowering) moved out. Must drive a greedy-rewrite
phase **before** dialect conversion:

```cpp
RewritePatternSet rewrites(&ctx);
populateVectorRankReducingFMAPattern(rewrites);
populateVectorInsertExtractStridedSliceTransforms(rewrites);
populateVectorTransferLoweringPatterns(rewrites);
(void)applyPatternsAndFoldGreedily(module, std::move(rewrites));

RewritePatternSet conv(&ctx);
populateVectorToLLVMConversionPatterns(typeConverter, conv);
// then applyFullConversion(...)
```

Driven by PR #119975. Without the split, vector ops stay in the IR after
conversion → "failed to legalize" errors.

New: `populateVectorRankReducingFMAPattern` public API. New:
`VectorToLLVMDialectInterface` (PR #121440) — lets `--convert-to-llvm`
discover vector lowering through a promised interface.

### Pass management

**Stable in 20.** No breaking changes. Constraints to remember:

- Passes must be copy-constructible and stateless across runs.
- `nest<OpT>()`, `nestAny()` work as before.
- Mark preserved analyses with `markAnalysesPreserved<>()` or
  `markAllAnalysesPreserved()`.
- Crash reproduction in two modes — global pipeline and local pass.
- Textual pipelines (`builtin.module(func.func(cse,canonicalize))`) and
  `PassPipelineCLParser` unchanged.

### LLVM dialect / `translateModuleToLLVMIR`

- Signature unchanged. `translateModuleToLLVMIR(op, ctx, name)` returns
  `unique_ptr<llvm::Module>`.
- Per-process registration unchanged: `registerLLVMDialectTranslation(ctx)`
  + `registerBuiltinDialectTranslation(ctx)`.
- **Pointer-element type fully gone — opaque pointers throughout.** If
  `LLVM::LLVMPointerType::get(ctx, elemTy)` calls survive in the codebase,
  replace with `LLVM::LLVMPointerType::get(ctx)`.
- `llvm.experimental.stepvector` renamed to `llvm.stepvector`.
- New: `llvm.atomicrmw usub_cond/usub_sat`, `noalias.addrspace` metadata.

### TableGen / ODS

No breaking `.td` syntax in 20. Notable:
- `Pure` trait is the canonical replacement for `NoSideEffect` (deprecated
  for several releases).
- Properties are the default for attributes (since 18; opt-out
  `usePropertiesForAttributes=0` *removed in 19*). Inherent attrs now live
  in `op.getProperties().foo`, **not** in `op->getAttr("foo")`. If custom
  printers/parsers pull `op->getAttr("foo")` for inherent attrs, the value
  is null — use the generated `op.getFoo()` accessor.

### Known MLIR 20 issues

1. **1:N pattern fatal errors at runtime** (see above). The error is
   `report_fatal_error`, not `LogicalResult::failure`. Aborts the host
   process. From PG's perspective: backend crashes, not just query failure.
2. **Translator + `ConversionPatternRewriter` + dropped block args**: stricter
   in 20. Leftover `unrealized_conversion_cast` after `applyFullConversion`
   almost always means a missing `addSourceMaterialization` for that type pair.
3. **mlir-runner consolidation**: `mlir-cpu-runner` and friends merged into
   `mlir-runner`. Doesn't affect us (we use in-process API), but lit tests
   may need updating.
4. **`TargetMachine` `RelocModel` defaults to `PIC_`**. Large-model addressing
   for global runtime functions may now route through GOT. Usually fine; check
   that runtime symbols are registered before first lookup. (We're fine
   today.)

---

## Part 4: PostgreSQL 17.6 (released 2025-08-14)

Sources:
- https://www.postgresql.org/docs/17/release-17.html
- https://www.postgresql.org/docs/release/17.6/
- https://www.postgresql.org/about/news/postgresql-176-1610-1514-1419-1322-and-18-beta-3-released-3118/
- https://www.postgresql.org/docs/17/jit-extensibility.html

### What's new in PG 17 generally (relevant to us)

**Planner / executor**:
- **CTE planning improvements** — planner considers stats and sort order of
  earlier CTEs. Tree shape may differ — same node types, possibly more/fewer
  `Material` and `Sort` nodes.
- **Partition pruning expanded** — boolean column pruning on
  `IS [NOT] UNKNOWN`, better range pruning. `PartitionPruneInfo` entries
  may appear in plans that didn't have them in 16.
- **`enable_group_by_reordering` GUC** (default on) — `GROUP BY` columns
  may reorder. If we compare against SQL text ordering, switch to comparing
  against `Agg`/`Group` node order.
- **MERGE improvements** including `RETURNING`. New plan-node behaviour.
- **B-tree IN-list lookups more efficient** — same semantics, different perf
  profile in baseline.
- **VACUUM rework**: `pg_stat_progress_vacuum` columns renamed.
  (We don't read those.)

**JIT**:
- Minimum LLVM raised to 10. PG 17 builds against LLVM 17/18/19/20 fine.
- `jit_provider` GUC unchanged. `llvmjit.so` lazy-loaded based on
  `jit_above_cost`.
- **No JIT API changes in 17** that affect a custom JIT extension.

**Source-level for extensions**:
- **User-defined `recv` functions**: input data is **no longer null-terminated**.
  Doesn't affect us — we consume types, don't define wire-format ones.
- `--disable-thread-safety` configure removed.
- AIX support removed; VS-specific Windows build removed (Meson only); OpenSSL
  1.0.1 dropped.
- **Injection points** (compile-time `--enable-injection-points`) — useful
  for testing extensions that need to provoke specific server states.
- **Custom wait events** — extensions can register named wait events. Useful
  if pgx-lower ever blocks (e.g. waiting on a JIT compile thread).

### PG 17.6 specific fixes

**Security**:
- **CVE-2025-8713** — tightened security checks in planner estimation. Effect:
  **planner-time permission checks on views may now fire earlier than before.**
  If our `ExecutorRun_hook` previously assumed planning was permission-free
  for view-mediated access, we may now see `ERROR: permission denied for
  relation …` arrive *before* the hook runs. Should be tolerated by our
  PG_TRY/PG_CATCH wrapping.
- Two `pg_dump`-related security fixes — irrelevant to us.

**Correctness fixes that may affect us**:
- **BRIN `numeric_minmax_multi_ops` distance miscalculation** — wrong results
  on 64-bit, *wildly* wrong on 32-bit. After 17.6 upgrade, `REINDEX` BRIN
  multi-ops indexes. If any TPC-H benchmark uses BRIN, re-baseline.
- **MERGE on inheritance parents** — was crashing or wrong; now correct.
  Tree shape unchanged.
- **MERGE with `BEFORE ROW` triggers + concurrent updates** — was crashing;
  now correct.
- **Out-of-memory crash with "bump" allocator** — only if we explicitly use
  `BumpContext` (we don't).
- **Per-relation memory leak in autovacuum** — fixed.

**APIs we use — no changes in 17.6**:
- `ExecutorRun_hook` signature unchanged.
- `PlannedStmt` struct field layout unchanged.
- `heap_getattr` / `heap_deform_tuple` semantics unchanged.
- `MemoryContext` API unchanged.
- `j2date`, `numeric_in`, `numeric_out`, `datumCopy` unchanged.
- `tuplesort_begin_heap` unchanged.
- `PG_TRY` / `PG_CATCH` machinery unchanged.

The headers we link against are ABI/API-stable across 17.0-17.6 by policy.
**No recompile required for a 17.x→17.6 bump on the consumer side**, but
typical practice is to rebuild against headers from the same minor.

### PG's `llvmjit` and our LLVM — collision surface

Two LLVMs in the same process is the genuine risk. PG's `llvmjit.so` is
lazy-loaded when `jit_above_cost` is exceeded *and* `jit = on`.

**Mitigation we use today**: build PG 17.6 against the **same LLVM 20** our
MLIR uses. The Dockerfile is set up that way. Verify with:

```bash
pg_config --configure | tr ' ' '\n' | grep -i llvm
llvm-config --version
```

Both should reference LLVM 20.

**Backup mitigation**: turn off PG's JIT entirely for sessions that load
pgx-lower:

```c
SetConfigOption("jit", "off", PGC_USERSET, PGC_S_SESSION);
```

Heavy-handed; sidesteps the collision completely.

### Known PG 17.6 gotchas (likelihood-ordered)

1. Planner permission errors arrive earlier (CVE-2025-8713 fix). Tolerate
   in hook wrapper.
2. MERGE on inheritance parents now correct — update test goldens if any.
3. BRIN multi-ops needs REINDEX after upgrade.
4. CTE-heavy and `GROUP BY`-reordered plans have different shapes — reset
   plan-shape assertions if any.
5. Two LLVMs in one process — confirmed mitigated by Docker pin.

---

## Part 5: Lessons from git history

Mined from ~961 commits (project bootstrap to 2025-11). Themes that recur.

### A. "This works but I don't like it" — deferred-decision debt

- **8e2fa71** (2025-09-07): "stuck on aggregation due to our null architecture.
  need to step back and change null architecture" — entire null architecture
  was wrong for aggregates; rewrite affected 30+ downstream commits.
- **b930a27** (2025-10-05): "clean out todo. I think its overly complicated"
  — correlation parameter handling flagged as over-complicated.
- **59db350**, **22ee030** (2025-09-28): refactoring-marked aggregation and
  initplan handling.

**Lesson**: When you write "REVERT ME LATER" or "I don't like it", file a
ticket with a deadline. One deferred decision (nullable aggregates) cost
30+ downstream commits.

### B. "REVERT ME LATER" — debug prints in MLIR lowerings

- **559d80e** (2025-08-16): Revert "REVERT ME LATER!" — surgically removed
  PrintOps from 7 files (RelAlgToDB translators + DB RuntimeFunctions +
  PrintRuntime).
- **092f2f1** (2025-09-06): "haha my own logs were causing the crash" —
  diagnostic logging in Vector.cpp dereferenced invalid memory and itself
  caused the crash being investigated.

**Lesson**: Never commit PrintOps in MLIR lowering passes — they introduce
real LLVM IR calls in hot loops. Use the safe logging module (PGX_LOG)
introduced by `d0831d3` ("Unify logging architecture across entire codebase").

### C. The decimal/numeric saga — 27 days, 14 commits

Chronologically:
1. 573a025 (08-16): "add decimal support... not working?"
2. fbefb69 (09-06): "fix numeric issue"
3. 64b68b2 (09-09): "column naming + decimal precision. decimals still seem broken"
4. 24d056c (09-20): "fixing support for decimal types"
5. bcef4a8 (09-20): "string formatting with decimals; suffix was overflowing"
   — 48-byte buffer assumption overflowed.
6. 2a18a01 (09-20): "strip trailing zeros from decimals"
7. ca39005 (09-21): "fixing decimal scaling"
8. 9baf612 (09-27): "lower the decimal limit to below 10^38" — i128 overflow.
9. cca0764 (10-07): "Enforce i128 precision for decimals"
10. 33b2655 (10-10): "37 - cap numeric values"
11. 081c1c4 (10-13): "fix decimal conversion"
12. 1d46d60 (10-17): "numeric -> i128" — extracted to `NumericConversion.cpp`.
13. 5295091 (10-21): "okie dokie, numerics are isolated"

**Lesson**: For numeric types, establish bit-width constraints and serialization
buffer sizes upfront. Decimal/numeric is a category where the type system must
validate at creation time, not be debug-patched. Once chosen i128, every
downstream code path should validate "will this fit?" Isolating to a single
conversion module (`NumericConversion.cpp`) was the unlock — should have done
that in week 1, not week 4.

### D. The null-handling saga — required rearchitect at 60% progress

1. 3ee652a (08-02): "Fix NullableType auto-wrapping causing GetColumnOp crashes"
2. 5b920d4 (08-02): "Fix NULL table name issue in SubOp lowering"
3. 8ea2f83 (08-10): "nullable type stuff"
4. 4311185 (08-24): "nullability and coalesce operator"
5. 195953a (08-31): "null handling appears to work" — premature
6. 8e2fa71 (09-07): **"stuck on aggregation due to our null architecture.
   need to step back and change null architecture"** — pivot.
7. ee64bde (09-09): "fix nullability in projection! this one was difficult to debug"
8. 7e78318 (09-12): "fix for non-null strings"
9. 638146e (09-20): "fix count operator's treatment of nulls" — classic SQL
   COUNT/NULL gotcha.
10. 1df388e (10-08): "null flag fix"
11. edd5589 (10-13): "fixing join nullability"

**Lesson**: Null semantics demand upfront type-system design. Auto-wrapping
NullableType broke at the aggregation step; explicit null-flag columns at
schema level was the right choice. Test aggregates and joins early — they
expose null handling gaps faster than projections.

### E. The hash join saga

1. 8b613d3 (09-15): "implement hash join" — initial.
2. 6d80e17 (10-14): "fix for hash join thign" — one-line scan fix.
3. c1e33c5 (10-17): "add impl hash join to enforce using hash joins" —
   planner wasn't picking hash joins by default; force via `impl="hash"`
   attribute.
4. cb7b5cc (10-21): "great pain with supporting hashes" — DeriveTruth
   handling for nullable bool predicates; null initializer for empty hash
   accumulation.
5. edd5589 (10-13): "fixing join nullability" — null propagation through joins.

**Lesson**: Hash joins are a higher-correctness bar than nested loop joins.
Edge cases that bite: empty hash table, NULL predicates, duplicate keys,
outer joins. Add explicit hash-join lowering tests. If the planner doesn't
pick hash joins by default, force them rather than relying on cost.

### F. The benchmark thrash — 10 reverts in 11 hours (2025-11-16)

```
70811cd  08:37:06  drop to 1 for dry run
cff922e  09:20:27  Revert "drop to 1 for dry run"
8233581  09:43:17  test just sf = 1
7a7bffa  09:54:21  Revert "test just sf = 1"
ce7b542  18:14:20  Reapply "test just sf = 1"
8ad55f3  18:15:14  Revert "Reapply..."
1e69c37  18:15:26  Reapply "drop to 1 for dry run"
5ad181a  18:33:48  1 -> 2
9ad38fa  19:29:28  Revert "Reapply..."
d44bc15  19:29:26  Revert "1 -> 2"
```

**Lesson**: Never iterate on benchmark configuration via commit/revert on
master. Use parameterized config (the YAML profiles in `benchmark-config.yaml`
came out of this lesson) or a feature branch. The thrash indicated no stable
measurement methodology.

### G. The LLVM modernization — pass manager migration

Three commits, one morning (2025-10-16):

- **7816cd2** (05:31): "llvm optimization passes" — added timing/statistics/
  crash-reproduction/verification (these later turned out to break PG memory
  contexts).
- **f8ff190** (06:05): "upgrade to modern LLVM pass manager" — 88-line
  rewrite of pass manager code.
- **5add478** (06:34): "llvm modern pass manager" — refinement.

What broke: SCF IfOp builder API (lambdas → manual insertion guards in LLVM
20 — see commit f1c0318). Also, PassManager debugging features bypassed PG
memory contexts.

**Lesson**: LLVM upgrades are breaking changes. Builder APIs shift between
majors. Test each pass independently. **Never enable PassManager debugging
features in a PG extension** — they bypass memory-context safety.

### H. Build-time wins

- **96baf79** (10-06): "add a precompiled header to massively reduce
  compilation time!" — `translation_pch.h` added. 2-3x speedup.
- **f1c0318** (10-08): SCF IfOp builder API fix for LLVM 20 compatibility.
- **65420fb** (08-23): DSA pattern registration linker fix.
- **033995b** (08-21): "ensure qbuild dies if there's a compilation failure"
  — prevent silent partial builds.

**Lesson**: PCH for template-heavy code (MLIR) is free outsized win. Aggressive
linker checks prevent insidious partial-build bugs.

### I. Crash fixes — memory and type system

- **c869bd5** (08-15): "Major fix: Resolve MLIR PassManager segfault in
  PostgreSQL extensions" — PassManager debugging features bypass PG memory
  contexts.
- **244532f** (08-15): "Major server crash fix: Eliminate PostgreSQL crash
  during MLIR→LLVM lowering" — `CodeGenOptLevel::Default` triggered
  aggressive inlining that broke PG calling conventions. Set to `None`.
- **a997aa1** (08-15): "CRITICAL: Fix MLIR Type ID collision causing Phase 5
  server crashes" — PG and MLIR type-system clash; fixed via pragma
  push/pop and macro isolation in CMakeLists.txt + logging.h.
- **0768a75** (08-15): "Fix critical segfault: Implement MLIR context
  isolation and MapCreationHelper system" — MLIR context shared memory
  with PG contexts; isolated via MapCreationHelper.
- **3ee652a** (08-02): "Fix NullableType auto-wrapping causing GetColumnOp
  crashes" — nested nullable types created invalid IR.
- **cc51bc4** (08-14): "Fix multi-column JIT execution crash: implement
  proper RecordBatchInfo structure" — uninitialized struct corrupted offset
  calculations.
- **4049a9c** (10-14): "fix for pfree called on invalid pointer" — classic
  double-free; PG owns `originalTuple`, our destructor was freeing it again.
  Fixed by disabling `heap_freetuple` in `POSTGRESQL_EXTENSION` mode.

**Lesson**: PG memory isolation is non-negotiable. Every MLIR/LLVM
integration point must isolate memory contexts. Never use default
optimization levels in a PG extension. Type-system conflicts require
compiler-level pragma isolation. Double-free bugs come from unclear
ownership — document who allocated what.

### J. The translator simplification — 4ef299c

**4ef299c** (10-16): "refactor the jit engine to be wayyyy simpler"

- 7 files changed. 921 → 387 insertions (-472 lines).
- Deleted: `jit_execution_wrapper.cpp` (186 lines), `jit_execution_interface.h`
  (21 lines).
- Inverse of earlier modular-architecture commits (f96a403, 1fc1090, ae66ecc).

**Lesson**: Premature abstraction (wrapper layers, interface indirection)
causes complexity debt. Periodically delete code; if tests pass, it was dead
code or over-engineered. Simplification beats modularity when still exploring
the problem space.

### K. Other patterns

- **The aggregation rearchitecture** (8e2fa71 → 30+ downstream commits) —
  aggregation semantics in MLIR+SQL are fundamentally hard. Test with
  multi-row tables early.
- **The PG memory-context isolation arc** (c869bd5, 244532f, a997aa1,
  0768a75) — multiple commits on the same root cause across one week. The
  realization that MLIR's memory and type systems fight PG's at the language
  level was a design pivot, not a bug fix.
- **The decimal precision hard limit** (9baf612, 2025-09-27) — chose i128;
  hit 10^38 ceiling; capped. PG decimal can go to 10^131k. Architectural
  limitation worth documenting.
- **Logging standardization** (d0831d3, 08-15) — followed the
  log-corruption-causes-the-crash bug (092f2f1). Unified to a safe logging
  module to prevent future instrumentation disasters.
- **Docker stabilization** (432d7b5, 6cf9176, 08-09 to 10-09) — 3 commits to
  get reproducible benchmarks in container. Docker was added to enable
  reproducible builds, not for deployment.

---

## Part 6: Synthesis — 10 lessons for future work

1. **Establish type-system constraints upfront.** Numeric bit-width, null
   representation, and aggregate type mapping are not "refine later."
2. **Isolate MLIR memory contexts from PG.** Non-negotiable. Every MLIR
   integration point assumes conflicting memory models.
3. **Never commit debug PrintOps in MLIR lowerings.** Use `PGX_LOG`.
4. **Null semantics require redesign, not patching.** Explicit null flags at
   schema level beat auto-wrapping types.
5. **Hash joins need explicit planner enforcement.** Don't trust the cost
   model to pick.
6. **PCH pays for itself immediately.** MLIR template bloat demands it.
7. **Modularity can be premature.** Simplify before abstracting.
8. **LLVM/MLIR upgrades are breaking changes.** Builder APIs shift.
   Test passes independently.
9. **Never enable PassManager debugging in production.** Bypasses PG
   memory-context safety.
10. **Benchmark infrastructure requires isolation.** Use parameterized
    config (YAML profiles), not commit-per-experiment.

## Part 7: TL;DR action items

**MLIR 20 audit**:
- Audit `RelAlgToDB`, `DBToStd`, `DSAToStd` patterns for 1:N readiness.
- Replace `addArgumentMaterialization` with `addSourceMaterialization`.
- If we add vector lowering (spec 09): split greedy-rewrite from
  dialect-conversion phases.
- Verify no `LLVMPointerType::get(ctx, elemTy)` calls remain (opaque
  pointers).

**LLVM 20 audit**:
- No PassBuilder code changes needed.
- Spec 02: try `SLPVectorization=true` first; LoopVectorization needs paired
  unrolling.
- Spec 03 cache: `ExecutionEngineOptions::cache` is the integration point
  (unchanged in 20).
- Spec 08 bitcode: `registerPipelineStartEPCallback`.

**Clang 20 build flags**:
- Audit for `-Wno-enum-constexpr-conversion` (now unknown).
- Consider `-fno-pointer-tbaa` if FFI miscompiles surface.
- Consider `-fwrapv-pointer` to match GCC for any pointer-arith overflow
  checks.

**PG 17.6 verification**:
- Confirm `pg_config --configure` shows LLVM 20 path, matching MLIR's LLVM.
- Tolerate planner-time permission errors in `ExecutorRun_hook` wrapper.
- Re-baseline goldens for BRIN-multi-ops queries and MERGE-on-inheritance
  queries.
- Update plan-shape assertions for CTE-heavy / GROUP BY-reordered queries.
