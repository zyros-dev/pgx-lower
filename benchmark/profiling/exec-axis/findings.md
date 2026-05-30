# Execution-Axis Profiling Findings (Task 5)

**Date:** 2026-05-17
**Query:** Q01 @ SF=1
**Builds profiled:** Debug (no PGX_RELEASE_MODE) + **RelWithDebInfo (PGX_RELEASE_MODE, optimized)** — see below
**Profiler:** perf 6.8.1, frame-pointer call-graph (`--call-graph fp`), attached to postgres backend PID as postgres user

---

## 1. No-Logging Mechanism

Two complementary mechanisms disable logging overhead:

**Runtime (GUC default):** `pgx_lower.log_enable` defaults to `false`.
When false, `should_log()` short-circuits before the `std::set<Category>` RB-tree lookup.

**CRITICAL:** `postgresql.auto.conf` may override the default via a prior phase-timing run
(`ALTER SYSTEM SET pgx_lower.log_enable = 'on'`). This persists across sessions and
contaminates profiles. Always verify and reset before production profiles:
```sql
ALTER SYSTEM RESET pgx_lower.log_enable;
ALTER SYSTEM RESET pgx_lower.log_io;
ALTER SYSTEM RESET pgx_lower.enabled_categories;
SELECT pg_reload_conf();
```

**Compile-time (`PGX_RELEASE_MODE`):** Set automatically for `Release`, `RelWithDebInfo`,
`Profile` build types (CMakeLists.txt:135-137). Eliminates `PGX_HOT_LOG` but NOT `PGX_IO`.
The default `just compile` uses `Debug` — no `PGX_RELEASE_MODE`.

---

## 2. A vs B Verdict on `should_log()`

**VERDICT: (A) — `!log_enable` short-circuits BEFORE the `std::set::find()` call.**

```cpp
// src/pgx-lower/utility/logging.cpp:108-140
bool should_log(const Category cat, const Level level) {
    initialize_if_needed();
    if (cat == Category::PROBLEM) return true;   // always log errors

    if (!log_enable)
        return false;                            // <- early exit; NO set lookup

    if (!enabled_categories.contains(cat)) {     // <- std::set RB-tree is HERE
        return false;
    }
    // ...
    return true;
}
```

The prior profile (55-65% in `std::_Rb_tree::find`) was GUC-ON contamination:
`postgresql.auto.conf` had `pgx_lower.log_enable='on'` from a previous phase-timing run,
causing the RB-tree lookup on every `should_log()` call (4.4M tuples/query).

---

## 3. Contamination Confirmed — Hardware Counter Comparison

| Metric | Logging ON (contaminated) | Logging OFF (clean) |
|--------|---------------------------|---------------------|
| Cycles | 91.8B | 34.2B |
| Instructions | 126.2B | 54.5B |
| IPC | 1.38 | 1.60 |
| Frontend stalls | 47.05% | 36.15% |
| Top symbol | `std::_Rb_tree::find` 7.56% | `DataSourceIteration::access` 8.49% |

**2.67x fewer cycles with logging off.** The contamination was severe.

---

## 4. Clean Profile Top Symbols (Logging OFF, GUCs Reset)

Source: `benchmark/profiling/perf-exec/q01-sf1-nolog/perf-report.txt`
34,276 samples; 34.2B cycles; IPC=1.60; 36.15% frontend stalls

| Symbol | Self % | Group |
|--------|--------|-------|
| `runtime::DataSourceIteration::access(RecordBatchInfo*)` | 8.49% | per-tuple decode |
| `numeric_to_i128` | 7.67% | numeric conversion |
| `process_tuple_into_batch` | 5.73% | tuple ingestion |
| `pgx_lower::log::should_log()` | 4.88% + 1.22%@plt | PGX_IO overhead |
| `heap_deform_tuple` (postgres) | 4.94% | PG tuple decode |
| `pgx_lower::log::log()` | 4.36% + 0.65%@plt | logging (see note) |
| `AllocSetAlloc` (postgres) | 2.30% | PG allocator |
| `pgx_lower::log::initialize_if_needed()` | 2.10% | PGX_IO overhead |
| `detoast_attr` (postgres) | 1.61% | PG detoast |
| `read_next_tuple_from_table` | 1.42% | tuple scan |
| `std::optional<ScopeLogger>` ctor/dtor | ~5-6% | PGX_IO overhead |
| `__divti3` (libgcc) | 1.22% | 128-bit division |
| `xxh::endian_align_sub_ending` | 1.14% | hash computation |
| `StringRuntime::compareEq` | 1.05% | string compare |

**Note on `log::log` at 4.36% with logging off:** Two explanations:
1. Frame-pointer misattribution from JITed code — the JIT frame (`0x705010a021b6`)
   breaks the FP chain; orphaned samples land on the last traced C++ function.
2. Real `PGX_IO` overhead — `std::optional<ScopeLogger>` ctor/dtor executes per
   `PGX_IO` invocation even when `should_log()` returns false.

**Logging symbols with logging OFF: ~19-22% of samples** (vs 55-65% with logging on).
Real `PGX_IO` guard cost (should_log + optional overhead): ~11-13%.

---

## 5. FFI-Wall Hypothesis Assessment: NOT CONFIRMED

`extract_field` and `get_*_field_mlir` do NOT appear in the profile.

The JIT→C++ FFI boundary is visible in call chains (address `0x705010a021b6`)
but the FFI entry points themselves are not hot as independent functions.

**True bottlenecks (in order):**
1. `DataSourceIteration::access` — per-column batch fill (8.49%)
2. `numeric_to_i128` — PostgreSQL NUMERIC → int128 conversion (7.67%)
3. `process_tuple_into_batch` — tuple ingestion (5.73%)
4. `PGX_IO` guard overhead — per-tuple `should_log()` + optional (11-13%)
5. PostgreSQL internals — `heap_deform_tuple` + `AllocSetAlloc` (7-8%)

---

## 6. IPC and Stall Rate

- **IPC: 1.60** (debug build; acceptable)
- **Frontend stalls: 36.15%** — elevated; likely causes:
  - Debug build code size pressure on L1-I cache
  - JIT→C++ indirect call overhead at `DataSourceIteration` boundary
  - `PGX_IO` machinery adding instruction pressure per tuple
- **Backend stalls: not supported** on this AMD host

---

## 7. RelWithDebInfo Profile — Before/After vs Debug Clean

Source: `benchmark/profiling/perf-exec/q01-sf1-relwithdebinfo/`
15,701 samples; 15.5B cycles; IPC=2.22; 18.82% frontend stalls

| Metric | Debug+GUC-OFF (clean) | RelWithDebInfo+GUC-OFF (decisive) |
|--------|----------------------|-----------------------------------|
| PGX_RELEASE_MODE | No | **Yes** |
| Query wall time | ~23 s | **~4 s (5.75× faster)** |
| Cycles | 34.2B | **15.5B** |
| IPC | 1.60 | **2.22** |
| Frontend stalls | 36.15% | **18.82%** |
| `heap_deform_tuple` | 4.94% | **11.12%** |
| `should_log` (PGX_IO) | ~6.1% + 1.22%@plt | **10.23% + 1.26%@plt** |
| `numeric_to_i128` | 7.67% | **9.21%** |
| `DataSourceIteration::access` | 8.49% | **8.02%** |
| `log::log` (PGX_IO) | ~5.0% | **7.28%** |
| `process_tuple_into_batch` | 5.73% | **5.15%** |
| JIT `main` | (raw address, ~0.09%) | **4.28% (symbolized!)** |
| `__divti3` | 1.22% | **2.42%** |
| FFI symbols (`extract_field` etc.) | NOT FOUND | **NOT FOUND** |

Key observations:
- **`heap_deform_tuple` rose from 4.94% to 11.12%** because PGX_IO overhead shrank
  proportionally with optimizer eliminating redundant code — heap decode is now more
  visible as a fraction.
- **PGX_IO overhead did NOT disappear in RelWithDebInfo** — `PGX_RELEASE_MODE` only
  eliminates `PGX_HOT_LOG`, not `PGX_IO`. The guard overhead is real and still ~19%
  of total samples.
- **JIT frames ARE now symbolized** — `main` in `jitted-<pid>-1.so` at 4.28%.
- **FFI-wall still not present** — `extract_field`/`get_*_field_mlir` absent.

---

## 8. FFI-Wall Hypothesis: DEFINITIVELY REFUTED

**VERDICT: FFI-WALL REFUTED (decisive, production-representative build)**

With JIT symbolization working and code fully optimized:
- The JIT function (`main` in `jitted-<pid>-1.so`) runs 4.28% of cycles
- Its subtree calls: `should_log` 1.12%, `DataSourceIteration::access` 0.67%, self 0.84%
- `extract_field`/`get_*_field_mlir` do NOT appear at any percentage
- The JIT→C++ boundary is NOT the bottleneck

True production bottlenecks (in order of impact):
1. **PGX_IO overhead** (~19.6%): `should_log` + `log::log` not compiled out by PGX_RELEASE_MODE
2. **PG tuple decode** (~16%): `heap_deform_tuple` 11.12% + `AllocSetAlloc` 4.52%
3. **Numeric conversion** (~11.6%): `numeric_to_i128` 9.21% + `__divti3` 2.42%
4. **Per-column decode** (~8%): `DataSourceIteration::access` 8.02%
5. **JIT loop overhead** (~4.28%): the optimized query loop itself (arithmetic + FFI calls)

---

## 9. Notes for Downstream Work

- **Spec 06 (data types):** `numeric_to_i128` 9.21% + `__divti3` 2.42% = 11.6% on
  NUMERIC conversion for Q01. This is the single largest addressable per-query cost
  after PGX_IO. Priority target for spec 06.

- **PGX_IO overhead (19.6% in RelWithDebInfo):** `PGX_RELEASE_MODE` only eliminates
  `PGX_HOT_LOG`, NOT `PGX_IO`. The `PGX_IO` macro must also be gated on
  `#ifndef PGX_RELEASE_MODE` (or equivalent) to eliminate per-tuple `should_log()`
  calls in production builds. This is the largest single addressable bottleneck overall.

- **PG tuple decode (11.12% `heap_deform_tuple` + 4.52% `AllocSetAlloc`):** Structural
  PostgreSQL cost. Spec 05 (decode-at-scan) addresses part of this. Remaining cost
  is inherent in the PG heap access model.

- **FFI-wall is not the bottleneck.** Do not optimize `extract_field`/`get_*_field_mlir`
  dispatch — they are not independently hot in either Debug or RelWithDebInfo profiles.
  The JIT-to-C++ boundary cost is real but modest (~1-2% visible in JIT call tree).

- **RelWithDebInfo build:** `just compile-rwdi` triggers a RelWithDebInfo build to
  `build-docker-rwdi/`, installs to PG extension dir, with `PGX_RELEASE_MODE` active.
  Run `just compile` to restore the Debug build. Build takes ~3 minutes on thor.

- **JIT symbolization:** Use `perf record -k 1` + `perf inject --jit` for all future
  profiles. MLIR ExecutionEngine already has `enablePerfNotificationListener=true` by
  default — no code changes needed. The jitdump approach is confirmed working.

- **Call-graph note:** `--call-graph fp` (1.2MB perf.data, ~60s report) is correct for
  this codebase. Prior `--call-graph dwarf` produced 289MB data, multi-hour perf report.
  Avoid dwarf call-graph for routine profiling.
