# Spec 06 — Switch internal date repr to PG-native

**Tier:** 2 (type-gap track)
**Stack on:** 05 (decode-at-scan)
**Blocks:** 08 (paired with 07)
**Estimated effort:** 1 week

## Goal

Replace LingoDB's nanosecond-since-2000 internal date/timestamp representation
with PostgreSQL's native:
- `DateADT` (int32, days since 2000-01-01)
- `Timestamp` (int64, microseconds since 2000-01-01)

This kills the multiply/divide pair on every scan/output, makes the conversion
purely a no-op pointer cast, and unblocks the unimplemented `addMonths` /
`subtractMonths` (which currently `ereport(ERROR)`).

## Background

Today, `db.date` and `db.timestamp` are int64 nanoseconds. The conversions
sit at every boundary:

- `DateRuntime.cpp:14` — `nanosToPostgresDate`: divides by 86,400,000,000,000.
- `DateRuntime.cpp:20` — `postgresTimestampToNanos`: adds offset, multiplies
  by 1000.
- `LowerToStd.cpp:140` — at scan: multiplies day-DateADT by 86,400,000,000,000.
- `LowerToStd.cpp:142` — at scan: multiplies microsecond-Timestamp by 1000.
- `LowerToStd.cpp:194` — at output: divides nanos by 86,400,000,000,000.
- `LowerToStd.cpp:206` — at output: divides nanos by 1000.
- `LowerToStd.cpp:735, 740, 966` — constant lowering.
- `parsing.cpp:143` — literal parsing converts to nanos.

`addMonths` / `subtractMonths` (`DateRuntime.cpp:26, 34`) just throw
"feature not supported", because in nanosecond representation, "add 1 month"
isn't well-defined (28/29/30/31 days). PostgreSQL's `date_pli` /
`timestamp_pl_interval` handle this via the `Interval` struct — switching to
PG-native types lets us call those directly.

## Design

### 1. New representation

| DB type | New repr | Old repr |
|---------|----------|----------|
| `db.date<day>` | int32 (DateADT) | int64 nanoseconds |
| `db.timestamp<microsecond>` | int64 (Timestamp) | int64 nanoseconds |
| `db.timestamp<nanosecond>` | (drop) | int64 nanoseconds |
| `db.timestamp<second>` / `<millisecond>` | (drop) | int64 nanoseconds |
| `db.interval<daytime>` | int64 microseconds | int64 nanoseconds |
| `db.interval<months>` | int32 months | int64 (always 0 in current code) |

Drop sub-microsecond and second/millisecond timestamp variants from the dialect.
PG only supports microsecond timestamps; the variants existed to support
LingoDB's broader scope. Removing them removes dead code paths in lowerings.

### 2. Lowering changes

`src/lingodb/mlir/Conversion/DBToStd/LowerToStd.cpp`:
- Line 140 (date scan): replace `multiply by 86400e12` with `extsi i32 → i64`
  (or just remove if downstream uses i32 directly — preferred).
- Line 142 (timestamp scan): remove the multiply-by-1000.
- Line 152 (interval daytime): remove the multiply-by-1000.
- Line 194 (date output): remove the divide-by-86400e12.
- Line 206 (interval output): remove the divide-by-1000.
- Lines 735, 740, 966 (constants): emit DateADT/Timestamp values directly.

`src/lingodb/mlir/parsing.cpp:143`: parse literals into PG-native units.

### 3. Runtime changes

`src/pgx-lower/runtime/DateRuntime.cpp`:

- Delete `nanosToPostgresDate` and `postgresTimestampToNanos`.
- Rewrite `extractYear`/`extractMonth`/`extractDay` to take `DateADT`
  directly:
  ```cpp
  int64_t extractYear(DateADT d) {
      int year, month, day;
      j2date(d + POSTGRES_EPOCH_JDATE, &year, &month, &day);
      return year;
  }
  ```
- Implement `addMonths`/`subtractMonths` properly using PG's
  `date_pli`/`timestamp_pl_interval` or hand-rolled Julian calendar logic.
  PG's `DirectFunctionCall2(date_pli, DateADTGetDatum(d), Int32GetDatum(days))`
  is the path of least resistance; for month arithmetic, build an `Interval`
  with `interval->month = months` and call `date_pl_interval`.

### 4. Bulk decode/encode (interaction with spec 05)

The bulk decode added in spec 05 stops doing the days→nanos conversion. The
struct field type for date columns becomes `int32_t` (DateADT) instead of
`int64_t` (nanoseconds).

The bulk encode similarly stops doing the inverse conversion.

### 5. PG bitcode opportunity (preview of spec 08)

After this spec, `extractYear`/`extractMonth`/`extractDay` are thin wrappers
around PG's `j2date`. Spec 08 will inline `j2date`'s LLVM bitcode so the
optimizer can fold extract patterns across multiple rows. Don't do that
optimisation here — keep the FFI call.

## Files to touch

| File | Change |
|------|--------|
| `include/lingodb/mlir/Dialect/DB/IR/DBOps.td` | Drop sub-microsecond timestamp variants; document new representation |
| `src/lingodb/mlir/Conversion/DBToStd/LowerToStd.cpp:140-206, 735-966` | Remove conversion arithmetic |
| `src/lingodb/mlir/parsing.cpp:143` | Parse to PG units |
| `src/pgx-lower/runtime/DateRuntime.cpp` | Rewrite with PG-native types; implement month arithmetic |
| `include/lingodb/runtime/DateRuntime.h` | Update signatures |
| `src/pgx-lower/runtime/tuple_access.cpp` (if `extract_field<DateADT>` exists) | Adjust decoding |

## Don't change

- The PG `Interval` struct itself — pgx-lower returns it unchanged via
  `copy_datum_to_postgresql_memory:543-553`. After this spec the conversion
  there becomes an identity copy (drop it).
- Spec 05's bulk-decode infrastructure — only the per-type field type for
  dates/timestamps changes.

## Acceptance criteria

- Build clean.
- All existing tests pass.
- New tests:
  - `SELECT date '2024-01-01' + interval '1 month'` returns `2024-02-01`
    (was `ereport(ERROR)`).
  - `SELECT extract(year from o_orderdate) FROM orders` matches PG output.
  - Timestamp arithmetic round-trips with microsecond precision.
- A/B (warm, SF=0.01, 4-query subset):
  - **q01 (touches l_shipdate): ≥5% faster** (one less conversion per tuple
    on a date-heavy query).
  - q03, q06, q12: any direction; report.
- Validation: full 22-query TPC-H sweep, bit-identical results vs baseline
  (date-handling queries q01, q03, q04, q05, q06, q07, q10, q12, q14, q15
  are the ones to watch).

## Risks

- LingoDB internal arithmetic over dates/intervals may assume nanosecond units
  in places the lowering map missed. Search for `86400` and `1000` in
  `src/lingodb/` after the change — anything left is suspect.
- Date overflow: DateADT is int32, nanoseconds was int64. Year 5874897 wraps
  DateADT — irrelevant for any real query but may surface in unit tests with
  edge-case values.
- The interval representation change (months stay separate from microseconds)
  matches PG's `Interval` struct exactly; if any LingoDB code packed both
  into one int64, fix at the same time.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `06-date`.

Required runs:
1. Standard 4-query sweep.
2. Validation: full 22-query sweep with result hashing.

## Rollback

Type-system change — not gated by a GUC. Revert is the branch revert. No data
migration since types only matter inside the JITed code; PG storage is untouched.
