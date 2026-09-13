#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
D=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/vperf
for f in "$D"/*.log; do
  n=$(basename "$f" .log)
  out=$(qnn-profile-viewer --input_log "$f" 2>/dev/null)
  for S in Average Min; do
    v=$(echo "$out" | grep -A18 "Execute Stats ($S)" \
        | grep -m1 'Accelerator (execute) time):' | grep -oE '[0-9]+ us' | grep -oE '[0-9]+')
    printf "%-3s %-8s %10s us\n" "$n" "$S" "${v:-?}"
  done
done
