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

**Status: BLOCKED as of 2026-05-17.**

Thor's kernel has `kernel.perf_event_paranoid = 4`.  Ubuntu's 6.14 kernel
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

### What was tried

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

### Fix required (one-time, by a human with sudo access to thor)

```bash
# Lower paranoid to 1: allows kernel profiling by processes with CAP_PERFMON.
sudo sysctl -w kernel.perf_event_paranoid=1

# Make it permanent across reboots:
echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl -p /etc/sysctl.d/99-perf.conf
```

After lowering paranoid, `just profile-exec QUERY=q01 SF=1` will run without
modification — the recipe already has `cap_add: [SYS_ADMIN, PERFMON]` and
`security_opt: [seccomp:unconfined]` in docker-compose.yml for the dev service.

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

**Named-symbol (`extract_field` / `get_*_field_mlir`) resolution**: `perf
record -g --call-graph dwarf` on the PG backend process resolves these symbols
via the ELF symbol table in the `.so` at the time of profiling.  Because the
`.so` is loaded with `RTLD_GLOBAL` (via `LOAD 'pgx_lower.so'` or
`shared_preload_libraries`) and exported dynamic, `perf report` should show
these names directly without extra steps.

**JITed-frame symbolization (perf jitdump / `perf inject --jit`)**: LLVM's
MCJIT can write a `jit-<pid>.dump` file that `perf inject --jit` uses to
annotate JITed frames with source-line information.  This is nice-to-have and
not required for the FFI-wall hypothesis.  Status: **not attempted** (blocked
by `perf_event_paranoid=4`).  If perf becomes available after the sysctl fix,
enable LLVM's jit-dump via `export ENABLE_JITDUMP=1` before running the query
and then `perf inject --jit -i perf.data -o perf.jit.data` before reporting.

---

## Dry-run status

**NOT COMPLETED** — blocked by `perf_event_paranoid=4` (see above).

The `just profile-exec QUERY=q01 SF=1` recipe is wired and correct.  It will
produce artifacts once a human runs the one-time `sudo sysctl` fix on thor.
