#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
D=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/ab
for f in "$D"/ab_*.log; do
  n=$(basename "$f" .log)
  # min is the least thermally-contaminated inference
  out=$(qnn-profile-viewer --input_log "$f" 2>/dev/null)
  avg_c=$(echo "$out" | grep -A18 'Execute Stats (Average)' | grep -m1 'Accelerator (execute) time (cycles)' | grep -oE '[0-9]+ cycles' | grep -oE '[0-9]+')
  avg_u=$(echo "$out" | grep -A18 'Execute Stats (Average)' | grep -m1 'Accelerator (execute) time):' | grep -oE '[0-9]+ us' | grep -oE '[0-9]+')
  min_c=$(echo "$out" | grep -A18 'Execute Stats (Min)' | grep -m1 'Accelerator (execute) time (cycles)' | grep -oE '[0-9]+ cycles' | grep -oE '[0-9]+')
  min_u=$(echo "$out" | grep -A18 'Execute Stats (Min)' | grep -m1 'Accelerator (execute) time):' | grep -oE '[0-9]+ us' | grep -oE '[0-9]+')
  printf "%-4s  avg %10s cyc / %7s us   min %10s cyc / %7s us\n" \
     "$n" "${avg_c:-?}" "${avg_u:-?}" "${min_c:-?}" "${min_u:-?}"
done
