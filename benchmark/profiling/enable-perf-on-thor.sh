#!/usr/bin/env bash
# Enable `perf` on thor for the pgx-lower execution-axis profiling (Task 5).
#
# Why this is needed: thor runs rootless Docker and the host kernel has
# kernel.perf_event_paranoid=4 (Ubuntu hardened default), which makes
# perf_event_open() return EPERM inside the container regardless of
# capabilities / --privileged. Only a host sysctl (needs sudo on thor)
# fixes it. See benchmark/profiling/tooling.md.
#
# Run this from the Mac:   ./benchmark/profiling/enable-perf-on-thor.sh
# It will prompt once for your sudo password on thor.
set -euo pipefail

THOR="${THOR_SSH_ALIAS:-comfy}"   # thor SSH alias (user zel); override via THOR_SSH_ALIAS

echo ">> Enabling perf on thor ($THOR): setting kernel.perf_event_paranoid=1 (runtime + persistent)"

# -t allocates a TTY so the interactive sudo password prompt works over SSH.
# One sudo shell does all three: set runtime value, persist, read back.
ssh -t "$THOR" "sudo bash -c '
  set -e
  sysctl -w kernel.perf_event_paranoid=1
  echo kernel.perf_event_paranoid=1 > /etc/sysctl.d/99-perf.conf
  echo -n \"now: \"; sysctl -n kernel.perf_event_paranoid
'"

echo
echo ">> Verifying..."
VAL="$(ssh "$THOR" 'sysctl -n kernel.perf_event_paranoid' | tr -d '[:space:]')"
if [ "$VAL" = "1" ] || [ "$VAL" = "0" ] || [ "$VAL" = "-1" ]; then
  echo ">> OK: kernel.perf_event_paranoid=$VAL on thor — perf is now usable."
  echo "   Task 5 (FFI execution-axis profiling) is unblocked:"
  echo "     just profile-exec QUERY=q01 SF=1"
else
  echo "!! Unexpected value: kernel.perf_event_paranoid=$VAL — perf may still be blocked." >&2
  exit 1
fi
