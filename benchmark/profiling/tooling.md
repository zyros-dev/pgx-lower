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

**JITed-frame symbolization (perf jitdump / `perf inject --jit`)**: LLVM's
MCJIT can write a `jit-<pid>.dump` file that `perf inject --jit` uses to
annotate JITed frames with source-line information.  This is nice-to-have.
Status: JITed frames appear as `[JIT] tid 865 [.] 0x0000705010a021b6` (two
raw-address JIT entries totalling ~0.09% of samples).  JIT execution time
is negligible in this profile — the hot path is entirely in `pgx_lower.so`
C++ code, not in the JITed MLIR-generated LLVM IR.  `perf inject --jit`
was not attempted; not needed given the JIT share is < 0.1%.

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

**Implication for Task 5:** The FFI-wall hypothesis (execution dominated by
`extract_field`/`get_*_field_mlir` per-tuple FFI overhead) is NOT confirmed.
The real bottleneck is the logging guard `pgx_lower::log::should_log()` called
on every per-tuple operation.  Disabling or short-circuiting the logging check
in production builds is the high-priority fix, not FFI restructuring.
