# Profiling Tooling Decision Record

## Thor CPU vendor

`AuthenticAMD` — confirmed via `/proc/cpuinfo` on thor (kernel 6.14.0-36-generic).

---

## Tool selection

### Magic Trace — REJECTED

Magic Trace requires Intel Processor Trace (Intel PT), a hardware feature only
present on Intel CPUs.  Thor is AMD; Intel PT is not available.

### perf — CHOSEN (primary)

`perf` is AMD-compatible.  It provides:

- Sampled call-graph profiling (`perf record -g --call-graph dwarf`) using
  AMD hardware performance-monitoring units (PMU) visible at
  `/sys/bus/event_source/devices/cpu`, `ibs_fetch`, `ibs_op`.
- Hardware counter snapshot per invocation (`perf stat -e cycles,instructions,
  branches,branch-misses,cache-references,cache-misses,
  stalled-cycles-frontend,stalled-cycles-backend`).

**In-container perf version**: `perf version 6.8.1` (Linux 6.8 kernel tools
package, extracted from `linux-tools-6.8.0-31` and installed manually into the
container because the Ubuntu HWE 6.14 kernel tools package for this host does
not ship the `perf` binary — `linux-tools-6.14.0-36-generic` and
`linux-hwe-6.14-tools-6.14.0-36` only contain `cpupower`, `turbostat`, `rtla`,
etc., not `perf` itself).

### AMD uProf — FALLBACK

AMD uProf provides deeper µarch counters (IBS instruction-based sampling,
L3 cache details, memory-controller bandwidth) useful for diagnosing specific
bottlenecks.  It is not installed on thor.  If perf hardware counters prove
insufficient for Task 5, install `AMDuProf` from
<https://developer.amd.com/amd-uprof/>.

---

## Blocker: perf_event_paranoid=4 on thor

**Status: RESOLVED as of 2026-05-17.**

**Resolution:** A human with sudo access to thor ran the one-time fix documented
below.  `kernel.perf_event_paranoid=1` is now set and persisted in
`/etc/sysctl.d/99-perf.conf`.  perf 6.8.1 is functional in-container.
The dry-run completed successfully (see "Dry-run status" below).

### Historical context (why it was blocked)

Thor's kernel had `kernel.perf_event_paranoid = 4`.  Ubuntu's 6.14 kernel
extends the standard paranoid range (-1 … 2) with a value of 4 that denies
`perf_event_open(2)` to ALL processes, including those holding
`CAP_PERFMON`, `CAP_SYS_ADMIN`, and `CAP_SYS_PTRACE`.  The error observed
on every invocation (host and inside container):

```
Error:
Access to performance monitoring and observability operations is limited.
Consider adjusting /proc/sys/kernel/perf_event_paranoid setting to open
access to performance monitoring and observability operations for processes
without CAP_PERFMON, CAP_SYS_PTRACE or CAP_SYS_ADMIN Linux capability.
perf_event_paranoid setting is 4
```

### What was tried before the fix

| Attempt | Result |
|---------|--------|
| `perf stat` on host as user `zel` (in `sudo` group) | EPERM — paranoid=4 |
| Docker `--privileged` + `--security-opt seccomp=unconfined` + `--security-opt apparmor=unconfined` | EPERM — rootless Docker cannot modify host sysctl |
| `docker-compose` `cap_add: [SYS_ADMIN, PERFMON]` + `security_opt: [seccomp:unconfined]` | EPERM — same rootless barrier |
| `docker run --pid=host --cgroupns=host --privileged sysctl -w kernel.perf_event_paranoid=1` | "sysctl: permission denied" |
| Python `ctypes` direct `perf_event_open(2)` syscall from inside container (with both caps) | `errno=13 EPERM` |

Root cause: Thor runs **rootless Docker** (context `rootless`, daemon socket
`unix:///run/user/1000/docker.sock`).  In rootless mode, even `--privileged`
containers do not hold real `CAP_SYS_ADMIN` at the host level and cannot
write to host-namespace `/proc/sys` entries.  `perf_event_paranoid=4` is a
host-kernel parameter that cannot be changed without an interactive root
session or a passwordless sudoers rule.

### Fix applied (one-time, by a human with sudo access to thor)

```bash
# Lower paranoid to 1: allows kernel profiling by processes with CAP_PERFMON.
sudo sysctl -w kernel.perf_event_paranoid=1

# Made permanent across reboots:
echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl -p /etc/sysctl.d/99-perf.conf
```

`/etc/sysctl.d/99-perf.conf` now contains `kernel.perf_event_paranoid=1`.

### Rootless Docker attach constraint (paranoid=1 residual)

With `paranoid=1`, `perf record -p <pid>` is blocked for processes owned by a
**different user** — even from the container's "root" (which is not real host
root in rootless Docker).  The postgres backend runs as user `postgres`
(uid=1001); perf must be launched **as the postgres user** to attach to it.
The `run_perf_profile.py` helper was updated to use `su postgres` for the
attach step.  `perf record -- <child>` (tracing own child) works unrestricted.

---

## RelWithDebInfo Build (PGX_RELEASE_MODE)

**Status: CONFIRMED working as of 2026-05-17.**

### Build invocation

```bash
# In worktree or main repo:
just compile-rwdi

# Under the hood (in Docker container on thor):
mkdir -p build-docker-rwdi
cd build-docker-rwdi
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_ONLY_EXTENSION=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      /workspace/.worktrees/<slug>
cmake --build .
cmake --install .
```

Build dir: `build-docker-rwdi/` (separate from `build-docker-ptest/` for Debug).
Install: overwrites `/usr/local/pgsql/lib/pgx_lower.so`.
Run `just compile` to restore the Debug build.

### What PGX_RELEASE_MODE eliminates (and what it does NOT)

CMakeLists.txt:135-137 defines `PGX_RELEASE_MODE=1` for Release/RelWithDebInfo/Profile.

| Macro | Debug | RelWithDebInfo (PGX_RELEASE_MODE) |
|-------|-------|----------------------------------|
| `PGX_HOT_LOG` | active (per-tuple LOG call) | **compiled out** → `((void)0)` |
| `PGX_IO` (per-tuple `should_log` + `ScopeLogger`) | active | **NOT eliminated** — still fires |
| LLVM module verification | active | **compiled out** |

**Critical finding:** `PGX_IO` overhead (10.23% `should_log` + 7.28% `log::log`) persists
in the RelWithDebInfo profile. `PGX_RELEASE_MODE` only gates `PGX_HOT_LOG`. To eliminate
per-tuple PGX_IO overhead in production builds, `PGX_IO` itself must also be gated on
`#ifndef PGX_RELEASE_MODE`.

### Performance impact of RelWithDebInfo vs Debug

At SF=1, Q01:

| Metric | Debug | RelWithDebInfo |
|--------|-------|---------------|
| Query wall time | ~23 s | ~4 s (5.75× faster) |
| Cycles | 34.2B | 15.5B |
| IPC | 1.60 | 2.22 |
| Frontend stalls | 36.15% | 18.82% |

---

## Why SF=1 makes compile-phase gating unnecessary

At SF=1, TPC-H Q01 execution dominates compile by ~70–175×:
- Execution: ~23,409 ms
- Total compile (setup + translate + lowering + jit): ~134 ms

A full-query `perf record` at SF=1 captures ~99% execution samples.
The compile "mountain" is negligible noise (< 1% of samples).  No elaborate
gating, two-stage profiling, or signal/synchronization tricks are needed to
isolate execution from compile — just profile the whole query and read the
result.

---

## FFI symbol resolution

The hot code is JITed into the PG backend process.  The extension exports
FFI entry-points (e.g. `extract_field`, `get_int32_field_mlir`,
`get_int64_field_mlir`, `get_date_field_mlir`, `get_decimal_field_mlir`)
via `-Wl,--export-dynamic` on the `pgx_lower.so` link.

**Named-symbol (`extract_field` / `get_*_field_mlir`) resolution**:
Confirmed functional as of 2026-05-17.  `perf record -g --call-graph dwarf`
on the postgres backend resolves symbols from `pgx_lower.so` ELF table
directly.  The per-tuple decode path symbols that DID appear (with sample %):

| Symbol | % samples |
|--------|-----------|
| `runtime::DataSourceIteration::access(...)` | 4.21% |
| `process_tuple_into_batch` | 2.18% |
| `read_next_tuple_from_table` | 0.83% |
| `runtime::DataSourceIteration::next()` | 0.32% |
| `read_and_fill_batch` | 0.29% |
| `check_batch_validity` | 0.23% |
| `runtime::DataSourceIteration::isValid()` | 0.25% |

`extract_field` and `get_*_field_mlir` did **NOT** appear in the report,
meaning they are not independently sampled hot functions.  The FFI-wall
hypothesis (those functions dominating per-tuple decode) is **not confirmed
by this profile**.  The actual dominant cost is the logging subsystem
(see Dry-run status below for details).

**JITed-frame symbolization (perf jitdump / `perf inject --jit`)**: CONFIRMED working.

Method:
1. `perf record -k 1` (CLOCK_MONOTONIC clockid) — required for jitdump timestamp correlation
2. MLIR ExecutionEngine has `enablePerfNotificationListener=true` by default, which registers
   `createPerfJITEventListener()`. This writes `/tmp/jit-<pid>.dump` during execution.
3. `perf inject --jit -i perf.data -o perf-jit.data` merges the jitdump into the output.
4. `perf report -i perf-jit.data` resolves JIT frames by name.

**No code changes were needed** — the LLVM perf JIT listener is already wired in MLIR's
ExecutionEngine. Only `-k 1` in `perf record` and the `perf inject --jit` step were missing.

Result from the RelWithDebInfo profile:
- JIT function `main` appears as `4.28%` in `jitted-6238-1.so`
- Prior Debug profile: JIT frames were raw addresses `0x705010a021b6` (~0.09% of samples)
- The 4.28% in RelWithDebInfo is the actual JIT-compiled query loop cost (much larger
  than Debug's 0.09% because JIT optimization is ON in RelWithDebInfo)

`run_perf_profile.py` now includes `-k 1` in `perf record` and the `perf inject --jit`
step automatically (results written to `perf-jit.data`; used for `perf report`).

---

## Dry-run status

**COMPLETED** — 2026-05-17, paranoid=1, perf 6.8.1.

`just profile-exec QUERY=q01 SF=1` ran successfully.  Artifacts at
`benchmark/profiling/perf-exec/q01-sf1/`:

| File | Size | Notes |
|------|------|-------|
| `perf-backend.data` | 789 MB | 94,033 samples from postgres backend (not committed — see .gitignore) |
| `perf-report.txt` | 127 KB | committed |
| `perf-stat.txt` | 1.4 KB | committed; backend process counters |

### Key findings from the profile

The dominant hot path is the **logging subsystem**, not FFI field decode:

| Symbol group | Approx. % |
|---|---|
| `pgx_lower::log::should_log` + `std::_Rb_tree::find` (log category set) | ~38% |
| `std::_Rb_tree` helpers (lower_bound, _S_key, end, begin, etc.) | ~17% |
| `std::optional<ScopeLogger>` ctor/dtor | ~6% |
| `runtime::DataSourceIteration::access` | ~4% |
| `numeric_to_i128` | ~3.5% |
| `process_tuple_into_batch` | ~2.2% |
| postgres binary (`heap_deform_tuple`, `AllocSetAlloc`, etc.) | ~8% |

**Total logging-related symbols: ~55–65% of execution samples.**

The 47% frontend stall rate in `perf stat` (43B stalled cycles out of 91B
total) is consistent with pointer-chasing in the RB-tree that backs the log
category filter set: each `should_log()` call traverses a `std::set<Category>`
whose nodes are heap-allocated and not cache-friendly at 4.4M tuples/query.

**THIS PROFILE WAS CONTAMINATED — see section below.**

---

## Profile Contamination (Task 5 finding)

**Root cause:** `postgresql.auto.conf` had `pgx_lower.log_enable='on'` and
`pgx_lower.enabled_categories='general'` set by a prior phase-timing run
(`ALTER SYSTEM SET`).  These persist across sessions.  The dry-run profile
above was run with logging ON, making the RB-tree set lookup fire on every
`should_log()` call — 4.4M tuples × multiple calls per tuple.

**`should_log()` verdict:** The `!log_enable` bool DOES short-circuit before
the `std::set::find()` call.  With `log_enable=false`, the RB-tree is NOT reached.
The 55–65% logging dominance in the dry-run was 100% GUC-ON contamination.

**Before any production profile, always reset system GUCs:**
```sql
ALTER SYSTEM RESET pgx_lower.log_enable;
ALTER SYSTEM RESET pgx_lower.log_io;
ALTER SYSTEM RESET pgx_lower.enabled_categories;
SELECT pg_reload_conf();
```

---

## Clean profile (logging OFF) — Task 5

**Artifacts:** `benchmark/profiling/perf-exec/q01-sf1-nolog/`

| File | Size | Notes |
|------|------|-------|
| `perf.data` | 3.6 MB | fp call-graph, 34,276 samples (gitignored) |
| `perf-report.txt` | 306 KB | committed |
| `perf-stat.txt` | 1.4 KB | committed |

**Call-graph method:** `--call-graph fp` (frame pointer).
- Prior: `--call-graph dwarf` → 289MB perf.data, hours of `perf report` processing on Debug build (123MB DWARF)
- Now: `--call-graph fp` → 3.6MB, ~60s report processing
- Use `dwarf` only for specific call-chain investigations; `fp` is the default.

**Key hardware counters (logging OFF):**

| Metric | Logging ON | Logging OFF |
|--------|-----------|-------------|
| Cycles | 91.8B | 34.2B |
| IPC | 1.38 | 1.60 |
| Frontend stalls | 47.05% | 36.15% |

**2.67× cycle reduction** by turning logging off.

**True hot path (logging OFF):**

| Symbol | % |
|--------|---|
| `DataSourceIteration::access` | 8.49% |
| `numeric_to_i128` | 7.67% |
| `process_tuple_into_batch` | 5.73% |
| `pgx_lower::log::should_log` (PGX_IO per-tuple cost) | ~6.1% |
| `heap_deform_tuple` (postgres) | 4.94% |
| `pgx_lower::log::log` (fp misattrib + PGX_IO) | ~5.0% |
| ScopeLogger optional ctor/dtor | ~5-6% |
| `AllocSetAlloc` (postgres) | 2.30% |
| `__divti3` (128-bit division) | 1.22% |

**FFI-wall hypothesis: NOT confirmed.**  `extract_field`/`get_*_field_mlir` not in profile.
True bottleneck: numeric conversion + per-column batch fill + PGX_IO overhead.

Full analysis: `benchmark/profiling/exec-axis/findings.md`.

---

## Decisive profile (RelWithDebInfo, logging OFF, JIT symbolized) — Task 5 Round 2

**Artifacts:** `benchmark/profiling/perf-exec/q01-sf1-relwithdebinfo/`

| File | Size | Notes |
|------|------|-------|
| `perf.data` | 1.3 MB | fp call-graph, -k 1, 15,701 samples (gitignored) |
| `perf-jit.data` | 1.3 MB | jit-injected (perf inject --jit; gitignored) |
| `perf-report.txt` | 77 KB | committed; generated from perf-jit.data |
| `perf-stat.txt` | 1.5 KB | committed |

**Build:** RelWithDebInfo (`just compile-rwdi`), `PGX_RELEASE_MODE=1`, Clang 20 O2+.
**GUC contamination:** `ALTER SYSTEM RESET` baked into `run_perf_profile.py` — runs automatically.
**JIT symbolization:** `-k 1` + `perf inject --jit` — JIT `main` visible at 4.28%.

**Key hardware counters:**

| Metric | Debug+GUC-OFF | RelWithDebInfo+GUC-OFF |
|--------|---------------|------------------------|
| Cycles | 34.2B | 15.5B |
| IPC | 1.60 | 2.22 |
| Frontend stalls | 36.15% | 18.82% |
| Query wall time | ~23 s | ~4 s |

**Top symbols (RelWithDebInfo):**

| Symbol | % | Group |
|--------|---|-------|
| `heap_deform_tuple` | 11.12% | PG decode |
| `should_log` | 10.23% + 1.26%@plt | PGX_IO (NOT eliminated by PGX_RELEASE_MODE) |
| `numeric_to_i128` | 9.21% | numeric conversion |
| `DataSourceIteration::access` | 8.02% | per-column decode |
| `log::log` | 7.28% + 0.86%@plt | PGX_IO |
| `process_tuple_into_batch` | 5.15% | batch fill |
| `AllocSetAlloc` | 4.52% | PG allocator |
| `main` (JIT, `jitted-<pid>-1.so`) | 4.28% | JIT query loop |
| `detoast_attr` | 2.89% | PG detoast |
| `__divti3` | 2.42% | 128-bit division |

**FFI-wall hypothesis: DEFINITIVELY REFUTED.**
`extract_field`/`get_*_field_mlir` not in profile in either Debug or RelWithDebInfo.
JIT is now symbolized and shows only 4.28% self — not dominated by FFI overhead.

Full analysis: `benchmark/profiling/exec-axis/findings.md`.
