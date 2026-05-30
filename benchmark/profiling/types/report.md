# Data-Type Misalignment: Cost and Risk Classification

**Date:** 2026-05-17
**Query scope:** Q01 @ SF=1 (NUMERIC-heavy; no hot string or date extraction path)
**Decisive profile:** `benchmark/profiling/perf-exec/q01-sf1-relwithdebinfo/perf-report.txt`
  (RelWithDebInfo, PGX_RELEASE_MODE, jitdump, spec-reviewed; 15K samples, 15.5B cycles, IPC=2.22)
**Cross-check profile:** `benchmark/profiling/perf-exec/q01-sf1-nolog/perf-report.txt`
  (Debug, GUCs reset, 34K samples)

---

## 1. Measured Data-Type Conversion Execution Fraction

From `q01-sf1-relwithdebinfo/perf-report.txt` (decisive, optimized build):

| Symbol | Self % | Source |
|--------|--------|--------|
| `numeric_to_i128` | 9.21% | `pgx_lower.so` |
| `__divti3` (128-bit division) | 2.42% | `libgcc_s.so.1` |
| **Total numeric conversion** | **~11.6%** | |
| `detoast_attr` | 2.89% | `postgres` (NUMERIC varlena detoast) |

Cross-check (`q01-sf1-nolog`): `numeric_to_i128` 7.67%, `__divti3` 1.22% = ~8.9%.
Both profiles confirm NUMERIC conversion as the #3 addressable bottleneck after PGX_IO
overhead (~19.6%) and PG tuple decode (`heap_deform_tuple` + `AllocSetAlloc`, ~16%).

`detoast_attr` (2.89%) is attributable to NUMERIC being a varlena type: every
NUMERIC column read calls `DatumGetNumeric(datum)` which may detoast the datum.
This is part of the NUMERIC->i128 conversion pipeline.

---

## 2. Mismatch Enumeration with File:Function Citations

### M1 — NUMERIC/DECIMAL: PG varlena -> `__int128` with per-tuple conversion

**Class: (a) Significant execution contributor — 9.21% + 2.42% = ~11.6% measured**

| Site | File | Function | What happens |
|------|------|----------|--------------|
| Ingestion | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:739-743` | `process_tuple_into_batch` (NUMERIC case) | Calls `numeric_to_i128(value, meta.numeric_scale)` for every NUMERIC cell per tuple |
| Conversion | `src/pgx-lower/runtime/NumericConversion.cpp:99-146` | `numeric_to_i128` | Decodes PG `NumericData` struct (short/long format), reconstructs value digit-by-digit (base-10000 loop), rescales with `value /= 10` or `value *= 10` loops — the `/= 10` path drives `__divti3` |
| Detoast | `DatumGetNumeric(datum)` inside `NumericConversion.cpp:102` | (PG macro -> `detoast_attr`) | NUMERIC is varlena; every call may invoke detoast |
| Output | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:316-337` | `TableBuilder::addDecimal` | Calls `i128_to_numeric(value, scale)` to produce a PG `Numeric` datum for result streaming — additional `__divti3` source |
| Precision cap | `src/pgx-lower/frontend/SQL/pgx_lower_constants.h:91` | (constant) | `MAX_NUMERIC_PRECISION = 32` — LingoDB `__int128` supports ~38-39 decimal digits; PG NUMERIC supports up to 131072 digits before decimal; values beyond `__int128` range silently overflow with no detection |

The conversion loop in `numeric_to_i128` does `value = value * NBASE + digits[i]` for
each base-10000 digit, then adjusts scale with repeated multiply-or-divide-by-10.
The scale-adjustment loops (`value /= 10`) trigger `__divti3` (software 128-bit division),
which is the direct cause of the 2.42% `__divti3` cost.

---

### M2 — NULL-flag inversion: LingoDB `1=valid` vs PostgreSQL `1=null`

**Class: (b) Correctness/robustness debt — not a measured perf cost, but gates all type correctness**

| Site | File | Lines | What happens |
|------|------|-------|--------------|
| Batch store | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:774` | `process_tuple_into_batch` | `iter->batch->column_nulls[col][row_idx] = !is_null;` — PG `is_null=true` stored as `false`; LingoDB reads buffer as a "valid" flag (1=valid, 0=null) |
| Output | `src/pgx-lower/runtime/tuple_access.cpp:141-147` | `table_builder_add<T>` | Receives LingoDB `is_valid` (true=non-null), negates to `is_null` for PG storage |
| INTERVAL special | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:748` | `process_tuple_into_batch` (INTERVAL case) | Sets `column_nulls[col][row_idx] = false` for null interval (inside `if (is_null)` branch) — LingoDB valid=true convention, consistent |

The inversion is handled correctly in both directions today. The risk is structural:
any new consumer of `column_nulls` that treats the buffer as a PG-style null flag
(1=null) rather than LingoDB valid flag (1=valid) will silently invert null semantics.
The convention is implicit — no type alias, no comment at `BatchStorage::column_nulls`.

---

### M3 — BPCHAR -> `db.string`: fixed-width character semantics lost

**Class: (b) Correctness/robustness debt — not exercised by Q01**

| Site | File | Lines | What happens |
|------|------|-------|--------------|
| Type mapping | `src/pgx-lower/frontend/SQL/translation/translation_core.cpp:49-55` | `PostgreSQLTypeMapper::map_postgre_sqltype` | `BPCHAROID` maps to `mlir::db::StringType` with a TODO comment acknowledging it should be `!db.char<X>` |
| Ingestion | `src/pgx-lower/runtime/tuple_access.cpp:820-824` | `get_string_field` (BPCHAR case) | `DatumGetBpCharPP(value)` returns raw varlena without space-padding — trailing spaces that BPCHAR guarantees are discarded |
| String decoding | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:725-737` | `process_tuple_into_batch` (STRING case) | `VARSIZE_ANY_EXHDR` gives byte length without padding for BPCHAR |

BPCHAR comparison semantics differ: PG pads to declared length for equality tests;
`db.string` comparison is byte-exact with no padding. A `WHERE bpchar_col = 'foo'`
can give wrong results if the stored value has trailing spaces the query literal lacks.

Q01 does not query any BPCHAR column — zero measured cost. In workloads with BPCHAR
equality/inequality filters (e.g. TPC-H `l_returnflag`, `l_linestatus` which are CHAR(1)),
the correctness risk is live.

---

### M4 — INTERVAL: month fields approximated with 30-day average; addMonths crashes

**Class: (b) Correctness/robustness debt — not exercised by Q01**

| Site | File | Lines | What happens |
|------|------|-------|--------------|
| Ingestion | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:745-758` | `process_tuple_into_batch` (INTERVAL case) | `interval->month * 30 * USECS_PER_DAY` — hard-coded 30-day month approximation |
| Constant translation | `src/pgx-lower/frontend/SQL/translation/translation_core.cpp:263-275` | `translate_const` (INTERVALOID) | Same 30-day approximation via `AVERAGE_DAYS_PER_MONTH` (30.4167 in constants.h); TODO comment: "Our datetime representation needs to be smarter" |
| Type restriction | `src/pgx-lower/frontend/SQL/translation/translation_core.cpp:70` | `map_postgre_sqltype` | `IntervalType::get(..., IntervalUnitAttr::daytime)` — only daytime subtype supported |
| Runtime guard | `src/pgx-lower/runtime/DateRuntime.cpp:26-39` | `DateRuntime::subtractMonths`, `addMonths` | Both unconditionally `ereport(ERROR)` — month arithmetic is unimplemented |

Any query involving `INTERVAL '1 month'` arithmetic either returns an approximated
(wrong) result or crashes with an error. Q01 uses `INTERVAL '90 day'` (day-only)
which is exact; the month mismatch is dormant.

---

### M5 — DATE: PG days-since-epoch stored, LingoDB runtime expects nanoseconds

**Class: (b) Correctness debt + minor per-access inefficiency — not measurably hot in Q01**

| Site | File | Lines | What happens |
|------|------|-------|--------------|
| Batch ingestion | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:664` | `process_tuple_into_batch` (DATE -> DATUM_BYVAL) | DateADT (int32 days since epoch) passes through unchanged in the Datum |
| MLIR lowering | `src/lingodb/mlir/Conversion/DBToStd/LowerToStd.cpp:120-147` | AtOp lowering for DateType::day | Sign-extends to i64 then multiplies by `86400000000000` (nanos/day) before passing value to runtime functions |
| DateRuntime interface | `src/pgx-lower/runtime/DateRuntime.cpp:14-17` | `nanosToPostgresDate` | Expects nanos, divides by `86400000000000` to recover days |

The MLIR lowering correctly inserts the days->nanos multiply. The net result is
numerically correct but structurally wasteful: two inverse conversions cancel out
on every date-column access. Not measurably hot in Q01 because date extraction is
not in the batch-inner-loop hot path.

---

### M6 — TIMESTAMP: PG microseconds vs LingoDB nanoseconds (x1000 multiply per access)

**Class: (b) Minor correctness/perf debt — not exercised by Q01**

| Site | File | Lines | What happens |
|------|------|-------|--------------|
| MLIR lowering | `src/lingodb/mlir/Conversion/DBToStd/LowerToStd.cpp:141-143` | AtOp lowering for TimestampType | Multiplies PG microseconds by `1000` to get nanoseconds for LingoDB runtime |
| Constant translation | `src/pgx-lower/frontend/SQL/translation/translation_core.cpp:243-252` | `translate_const` TIMESTAMPOID | Converts PG timestamp via `timestamp_out` string, passes string to LingoDB |

Precision is preserved (microseconds * 1000 -> nanoseconds -> microseconds in output
is lossless). The x1000 multiply per TIMESTAMP access is a minor per-tuple cost when
TIMESTAMP columns are in the query projection.

---

### M7 — STRING/BYTEA: per-cell `datumTransfer` allocation for varlena strings

**Class: (b) Allocation pressure debt — not exercised in Q01's hot path**

| Site | File | Lines | What happens |
|------|------|-------|--------------|
| String ingestion | `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:730-735` | `process_tuple_into_batch` (STRING case) | `datumTransfer(value, meta.attbyval, meta.attlen)` allocates a copy of each varlena string into the batch memory context per cell |

For text-heavy queries (e.g. TPC-H queries on `l_comment`, `p_name`), this per-cell
allocation drives `AllocSetAlloc` cost. In Q01, string columns are not projected; the
`AllocSetAlloc` 4.52% is attributed to NUMERIC detoasting and batch setup, not string copies.

---

## 3. Classification Summary

| ID | Mismatch | Class | Measured % (Q01 RWDi) | Q01 exercises? |
|----|----------|-------|-----------------------|----------------|
| M1 | NUMERIC->\_\_int128 per-tuple conversion + \_\_divti3 | **(a) Significant exec cost** | 9.21% + 2.42% = **~11.6%** | Yes — hot |
| M2 | NULL-flag inversion (LingoDB 1=valid vs PG 1=null) | **(b) Correctness debt** | Not measured | Implicitly (all columns) |
| M3 | BPCHAR->db.string, trailing-space semantics lost | **(b) Correctness debt** | Not measured | No |
| M4 | INTERVAL month: 30-day approximation + addMonths crashes | **(b) Correctness debt** | Not measured | Partially (day-interval only) |
| M5 | DATE: days->nanos->days double-conversion via MLIR lowering | **(b) Minor inefficiency + correctness debt** | Not measured as hot | Indirectly |
| M6 | TIMESTAMP: PG microseconds -> LingoDB nanoseconds (x1000) | **(b) Minor perf debt** | Not measured | No |
| M7 | STRING/BYTEA: per-cell datumTransfer allocation | **(b) Allocation pressure debt** | Not measured | No (Q01 strings not projected) |

---

## 4. Scope Caveat

**This profile covers Q01 at SF=1 only.** Q01 is NUMERIC-heavy (TPC-H `lineitem.l_extendedprice`,
`l_discount`, `l_tax` are all NUMERIC columns) with no hot string-filter or date-extraction
path in its inner loop. Consequently:

- **M1 is the only mismatch with a measured execution cost** in this workload.
  The 11.6% figure is specific to Q01's NUMERIC projection and aggregation structure.
- **M3, M4, M5, M6, M7 are NOT exercised in Q01's hot path.** Classifying them as
  (c) negligible based on this profile would be misleading — they are (b) because they
  are real correctness and performance risks in other TPC-H queries and real workloads.
  Q01 simply does not stress these paths. They are cheap here, not cheap in general.
- `detoast_attr` at 2.89% is a component of M1, not a separate mismatch: NUMERIC
  varlena detoast is a direct consequence of `DatumGetNumeric` inside `numeric_to_i128`.

---

## 5. Fix-Ordering Recommendation (for Task 8 Final Synthesis)

1. **M1 (NUMERIC->i128) — fix first.** The only data-type mismatch with confirmed,
   measured execution cost (~11.6% in Q01). Options in order of impact:
   - Replace the `value /= 10` scale-adjustment loop with a precomputed power-of-10
     divisor and a single 128-bit divide, or defer scale adjustment to output time.
   - Cache the detoasted pointer to avoid redundant `detoast_attr` on already-detoasted
     datums (addresses the 2.89% `detoast_attr` cost).
   - Long-term: native PG int128 accumulation without per-tuple detoasting (aligns
     with spec 06 intent).

2. **M4 (INTERVAL month arithmetic) — fix before supporting any month-interval query.**
   The `ereport(ERROR)` guard is a safe stop, but the 30-day approximation for month
   fields in the daytime path is silent data corruption.

3. **M3 (BPCHAR) — fix before any workload with BPCHAR equality filters.** TPC-H
   `l_returnflag CHAR(1)` and `l_linestatus CHAR(1)` are exactly this case.

4. **M2 (null-flag inversion) — document and add type alias, then leave.** The inversion
   is handled consistently today. A `using ValidBit = bool` typedef and a comment at
   `BatchStorage::column_nulls` would reduce future confusion risk.

5. **M5/M6/M7 — revisit when the relevant query types become a priority.** The
   days->nanos double-conversion (M5) is a structural inefficiency but not hot in
   current workloads; string allocation (M7) matters for text-heavy queries.
