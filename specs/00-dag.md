# pgx-lower performance roadmap — task DAG

Goal: get pgx-lower clearly ahead of stock PostgreSQL on warm analytical TPC-H queries
at small scale factors. Each spec below is independently executable by a single agent
on its own branch. The DAG defines what must land first.

## How to use this directory

1. Pick a task with no unresolved blockers (see DAG below).
2. Branch from `main` (or stack on a parent branch if listed under "stack on").
3. Hand the spec file to an implementation agent. Each spec is self-contained.
4. Run the A/B harness from `specs/ab-test-template.md` before opening the PR.
5. Capture the A/B numbers in the PR description.
6. Land to `main` (or to the parent branch it stacks on).

Tasks marked **decision** are not implementation work — produce a short writeup, not code.

## DAG

```
                    ┌──────────────────────────┐
                    │ 01 hoist MLIR/JIT state  │
                    └────┬─────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌────────────────┐        ┌──────────────────┐
   │ 02 enable LLVM │        │ 03 plan-shape    │
   │ vectorization  │        │    compile cache │
   └────┬───────────┘        └────┬─────────────┘
        │                         │
        │                         ▼
        │                ┌─────────────────────┐
        │                │ 04 cost-gate small  │
        │                │    queries to PG    │
        │                └─────────────────────┘
        │
        │  (informs scope of 09)
        ▼
   ┌────────────────────┐
   │ 09 MLIR vector     │
   │    dialect lower   │ ◀──── 10 row-to-column transpose
   └────┬───────────────┘       (paired with 09)
        │
        ▼
   ┌────────────────────┐
   │ 11 LingoDB rebase  │   (decision spec, gated on 09/10 result)
   │    decision        │
   └────────────────────┘

Independent track (parallel agent):

   ┌────────────────────┐
   │ 05 decode-at-scan  │
   │  / encode-at-out   │
   └────┬───────────────┘
        │
        ├────────────► 06 PG-native date repr
        │
        ├────────────► 07 PG varlena strings
        │
        └────────────► 08 inline PG bitcode
                       (after 05–07 stabilise FFI boundaries)

Deferred (research):
   12 copy-and-patch backend       (after Phase 1–3 land)
   13 two-tier adaptive execution  (after 09 lands)
```

## Tier ordering

| Spec | Tier | Stack on | Expected impact |
|------|------|----------|-----------------|
| 01 hoist MLIR/JIT state | 1 | main | Cleans seams for 03; small per-query setup-cost win |
| 02 enable LLVM vectorization | 1 | main | One-line change; measure before committing to 09 |
| 03 plan-shape compile cache | 1 | 01 | **Largest single warm-query win** (~21% on Q20 per thesis) |
| 04 cost-gate small queries | 1 | 03 | Cuts wasted compile budget on tiny queries |
| 05 decode-at-scan | 2 | main | Foundation of type track; lifts conversions out of hot loops |
| 06 PG-native date repr | 2 | 05 | Kills nanosecond round-trip; unblocks month arithmetic |
| 07 PG varlena strings | 2 | 05 | Removes string fudges |
| 08 inline PG bitcode | 2 | 05+06+07 | Lets LLVM DCE through conversion paths |
| 09 MLIR vector dialect | 3 | 02 result | DuckDB-range numbers on analytical queries |
| 10 row-to-column transpose | 3 | 09 | Pairs with 09 — vector ops want columnar in-register |
| 11 LingoDB rebase | decision | 09+10 result | Defer until you have a concrete reason |
| 12 copy-and-patch | 4 | 01–04 landed | Cold-small-query wins |
| 13 adaptive execution | 4 | 09 landed | Umbra-style two-tier |

## Rules of engagement for parallel agents

- The two tracks (1→2→3→4 and 5→6→7→8) are independent. Run them concurrently.
- 05–08 must not modify any file that 01/03 modify. The conversion runtime
  (`src/pgx-lower/runtime/`) is owned by the type track. The execution path
  (`src/pgx-lower/execution/`) is owned by the compile-cache track.
- If both tracks need to touch `tuple_access.cpp`, the type track owns it —
  the compile-cache track only references it.
- Final integration benchmark required after each branch merges. Two changes
  that each won 5% in isolation can cancel or compound. Budget time for it.

## A/B test policy (binding)

- All A/B tests use **SF=0.01**. Never SF=1 in this loop.
- Use the 4-query subset (q01, q03, q06, q12) for fast iteration unless the
  spec calls for the full TPC-H sweep.
- Pin the run to thor (Linux) — perf stat and magic-trace are unavailable on macOS.
- Record both warm (iteration 2+) and cold (iteration 1) numbers. Several specs
  affect one but not the other.
- See `specs/ab-test-template.md` for the exact commands.
