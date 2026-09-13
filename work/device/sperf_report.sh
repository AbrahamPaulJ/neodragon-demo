#!/usr/bin/env bash
# Accelerator time for the streaming decoder, min-of-N per round.
# The DSP throttles (trap #10), so compare MIN within one thermal session,
# never averages across sessions.
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
D=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/sperf
for f in "$D"/*.log; do
  n=$(basename "$f" .log)
  out=$(qnn-profile-viewer --input_log "$f" 2>/dev/null)
  for S in Min Average; do
    v=$(echo "$out" | grep -A18 "Execute Stats ($S)" \
        | grep -m1 'Accelerator (execute) time' | grep -oE '[0-9]+ us' | grep -oE '[0-9]+')
    printf "%-10s %-8s %12s us\n" "$n" "$S" "${v:-?}"
  done
done
