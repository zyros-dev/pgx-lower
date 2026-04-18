---
name: pgx-lower-build-and-test
description: How to build pgx-lower (Docker on thor, mutagen-synced from mac), run unit tests (GTest) and SQL regression tests (pg_regress), where artifacts land. Use when building, debugging build failures, adding new tests, running CI checks, or understanding the CMake/Docker layout.
---

# Build and test

pgx-lower builds in Docker on thor (Linux Ryzen). Mac edits sync via mutagen.
**Do not try to build on macOS** — LLVM 20 + MLIR 20 + PG 17.6 from source is
hours of yak.

## The remote build flow

Per `~/.claude/projects/-Users-nickvandermerwe-repos-pgx-lower/memory/project_thor_remote_build.md`:

- SSH alias: `ssh comfy` (user `zel`)
- Repo on thor: `/home/zel/repos/pgx-lower`
- Mutagen session: `pgx-lower` (`mutagen sync list pgx-lower`)
- Mounted into container at `/workspace`

## docker/Makefile targets

The most-used targets:

| Target | What it does |
|--------|--------------|
| `build-debug` | Debug CMake (`-O0 -g`), installs `.so` to `/usr/local/pgsql/lib` |
| `build-release` | Release CMake (`-O2 -g`), same install |
| `build-profile` | `-pg -fno-omit-frame-pointer` for callgrind/perf |
| `build-postgres-debug` | Builds PG 17.6 from source with `-g -O2 -pg` |
| `ptest-debug` | Runs SQL regression tests (`ctest`) on debug build |
| `ptest-release` | Same on release build |
| `ptest-profile` | Same on profile build |
| `bench-dockers` | Runs the full benchmark suite via `benchmark/run_benchmark_config.py` |
| `release-container` | Copies built `.so` + control to benchmark container |
| `clean-dockers` | Stops/removes containers, build dirs |
| `rebuild` | Clean rebuild with fresh image |
| `all` | Full setup chain |

All targets exec inside the `pgx-lower-dev` container via
`docker exec pgx-lower-dev bash -c "..."`.

## Manual build commands (when Make is too coarse)

Build the extension:

```bash
ssh comfy
cd ~/repos/pgx-lower/docker
docker compose up -d dev
docker exec pgx-lower-dev bash -c "
  cd /workspace &&
  mkdir -p build-docker-ptest &&
  cd build-docker-ptest &&
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON .. &&
  cmake --build .
"
```

Artifact: `build-docker-ptest/extension/pgx_lower.so`.

For Release: replace `Debug` with `Release`. For perf measurements, **always
use Release** — Debug skews vectorisation, inlining, IPC numbers.

## CMake graph (key options)

`CMakeLists.txt` (top-level):
- **Compiler enforcement (lines 19-35)**: Clang only. GCC 14.2.0's `-O3`
  infinite-loops in `mlir::reconcileUnrealizedCasts`. CMake aborts if not Clang.
- **C++20** (line 28).
- **Position-independent code globally** (line 38).
- **Build type defaults to Debug** (lines 42-46).
- **`BUILD_ONLY_EXTENSION`** option (lines 170-211): when ON, skips `tests/`
  and `tools/` subdirs. CI uses this.
- **`PGX_RELEASE_MODE`** (lines 135-138): disables MLIR verifier and
  DEBUG/TRACE logging in Release.
- **`ENABLE_COVERAGE`** option (lines 57-70): adds `-fprofile-instr-generate
  -fcoverage-mapping` (Clang) for coverage runs.

Subdirectories added (line 149-174):
- `tools/build-tools` → `runtime-header-tool` executable
- `include/lingodb/mlir/Dialect/{RelAlg,DB,DSA}/IR` → TableGen outputs
- `src/lingodb/mlir`, `src/lingodb/mlir-support`, `src/lingodb/runtime`
- `extension` → produces `pgx_lower.so`
- `tests`, `tools` (only if not BUILD_ONLY_EXTENSION)

## Extension linking strategy

`extension/CMakeLists.txt:87-112`:

- `-Wl,--whole-archive` around dialect/pass/runtime libraries (lines 89-102).
  Ensures every symbol from `MLIRRelAlgDialect`, `MLIRDBDialect`, `MLIRDSA`,
  `MLIRUtilDialect`, `MLIRRelAlgToDB`, `MLIRDBToArrowStd`, `MLIRDSAToStd`,
  `MLIRUtilToLLVM`, `StandardToLLVM`, `MLIRPasses`, `MLIRCustomTransforms`,
  `mlir-support`, `runtime` is linked into the `.so`.
- Standard LLVM/MLIR dialects linked normally (lines 104-108).
- `-Wl,--export-dynamic` (line 112): JIT can resolve symbols from the
  loaded module via dlsym.
- **Precompiled headers**: `translation_pch.h` (line 126). MLIR headers are
  the build-time bottleneck.

## Object libraries

The runtime is split into object libraries linked into the `.so`:

- `src/lingodb/runtime/CMakeLists.txt`: `runtime` object lib —
  `helpers.cpp MetaData.cpp Hash.cpp Hashtable.cpp Vector.cpp PgSortRuntime.cpp
  LazyJoinHashtable.cpp PrintRuntime.cpp` + the cross-included
  `../../pgx-lower/runtime/StringRuntime.cpp`.
- `src/pgx-lower/runtime/CMakeLists.txt`: `pgx_runtime` object lib —
  `PostgreSQLRuntime.cpp NumericConversion.cpp tuple_access.cpp
  PostgreSQLDataSource.cpp StringRuntime.cpp`.

Both pass `-Wl,--export-dynamic` to children.

## TableGen flow

For each dialect (RelAlg, DB, DSA), `include/lingodb/mlir/Dialect/<X>/IR/CMakeLists.txt`
runs `mlir_tablegen()`:
- `<X>OpsDialect.h.inc/.cpp.inc`
- `<X>Ops.h.inc/.cpp.inc`
- `<X>OpsEnums.h.inc/.cpp.inc`
- `<X>OpsTypes.h.inc/.cpp.inc`
- `<X>OpsAttributes.h.inc/.cpp.inc`
- `<X>OpsInterfaces.h.inc/.cpp.inc`
- Auto-generated docs.

Generated under `${CMAKE_BINARY_DIR}/include/`. Public target name:
`PGXLower<X>OpsIncGen`.

`tools/build-tools/runtime-header-tool` (`gen_rt_def()` macro, line 1-36
of its CMakeLists.txt) extracts C++ runtime function signatures into
`runtime-defs/*.h` so MLIR conversion passes can declare them.

## Tests

### Unit tests (GoogleTest)

`tests/unit/test_lowerings/CMakeLists.txt`:
- Targets: `test_mlir_lowerings`, `test_boolean_lowering`.
- Sources: `standalone_mlir_runner.cpp`, `test_pipeline_phases.cpp`,
  `test_boolean_lowering.cpp`.
- Registered via `add_test(NAME MLIRLoweringTests COMMAND test_mlir_lowerings)`.
- Timeouts: 600s (MLIR), 60s (boolean).
- Labels: `unit;mlir;lowering`, `unit;mlir;lowering;boolean`.

GoogleTest 1.14.0 is fetched via CMake `FetchContent` (top CMakeLists.txt
line 162-168).

Run via:
```bash
docker exec pgx-lower-dev bash -c "cd /workspace/build-docker-ptest && ctest --output-on-failure"
```

### SQL regression tests (pg_regress)

`tests/sql/` — 47 `.sql` files, named `NN_description.sql`.
`tests/expected/` — corresponding `.out` files.

43 tests are registered in `extension/CMakeLists.txt:39-85`. Topics:
- `1_one_tuple` … `3_lots_of_tuples` — basics
- `4`–`8_*` — type coverage
- `9`–`13_*` — arithmetic, comparison, logical, NULL, text
- `14`–`15_*` — aggregates, special ops
- `17`–`29_*` — WHERE, ORDER BY, GROUP BY
- `31_distinct`, `32_decimal_maths`, `33_basic_joins`, `34_advanced_joins`
- `35`–`42_*`, `tpch*` — TPC-H queries

The pg_regress harness:
- Driver: `pg_regress` binary (found by `cmake/FindPostgreSQL.cmake:150`).
- Inputs: `tests/sql/`, expected: `tests/expected/`.
- Diffs land in `build-docker-ptest/extension/regression.diffs` on failure.
- DB: container's PG on port 54320.
- Loads extension via `--load-extension=pgx_lower`.

### Adding a new test

**Unit test**:
1. Create `tests/unit/test_lowerings/test_<feature>.cpp` with `TEST(...)` /
   `EXPECT_*` macros.
2. Add to `tests/unit/test_lowerings/CMakeLists.txt`.
3. `add_test(...)` for ctest registration.

**Regression test**:
1. Create `tests/sql/NN_description.sql` and matching
   `tests/expected/NN_description.out`.
2. Add `NN_description` to the REGRESS list in `extension/CMakeLists.txt`
   (lines 39-85).

## Build artifact directories

```
build-docker-ptest/         Debug build
├── extension/
│   ├── pgx_lower.so        Loaded by PG
│   ├── pgx_lower.control   Generated control file
│   ├── regression.out      Latest regression output
│   └── regression.diffs    Diff on failure
└── include/                TableGen .h.inc/.cpp.inc outputs

build-docker-ptest-release/ Release build (-O2 -g)
build-docker-ptest-profile/ Profile build (-pg)

postgres-debug/             PG 17.6 from source (in dev container)
└── lib/compat/             ICU, MLIR libs for compatibility
```

Three build dirs because Debug/Release/Profile cohabit cleanly.

## Container layout (docker-compose.yml)

| Service | Port | Purpose |
|---------|------|---------|
| `dev` | 54320→5432 | Build + ptest. Volumes: workspace, postgres-data, build caches |
| `benchmark` | 54322→5432 | A/B benchmark target. Volume: workspace, benchmark-data |
| `profile` | 54325 | Host-native PG (no container) for profiling without Docker overhead |

The `profile` service uses files under `~/.pgx-profile/` extracted from a
container build, so perf can attach without Docker isolation issues.

## Common build failures

- **"This project requires Clang"**: you set `CC=gcc` somewhere. CMakeLists
  enforces Clang via `CMAKE_CXX_COMPILER_ID` check.
- **"GCC infinite loop in reconcileUnrealizedCasts"**: see above. Don't even
  try GCC.
- **TableGen `*.inc` not found**: check the dialect's `CMakeLists.txt` is
  included from the parent. Build dir's `include/` should contain the .inc.
- **"unresolved external symbol rt_..." at JIT time**: the runtime function
  isn't exported. Check the object lib's CMakeLists has `-Wl,--export-dynamic`.
- **Out-of-memory on `cmake --build`**: MLIR translation units are RAM-heavy
  during compile. Drop `-j` count. Spec commit `943a78d` notes "optimize debug
  flags to reduce compilation RAM usage."
- **PG version mismatch at extension load**: extension built against
  different PG headers than runtime PG. Rebuild after `build-postgres-debug`.

## Resources

`resources/sql/tpch/` — the 22 standard TPC-H SQL queries (1.sql … 22.sql),
used by `benchmark/`. See `pgx-lower-benchmarks` skill.

## Related skills

- `pgx-lower-benchmarks` — running A/B tests post-build.
- `pgx-lower-versions-and-history` — version pinning rationale and historical
  build issues.
- `pgx-lower-overview` — top-level layout.
