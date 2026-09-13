#!/usr/bin/env bash
# Accelerator time for the 2-D VAE encoder, min-of-N per round.
# The DSP throttles (trap #10), so compare MIN within one thermal session,
# never averages across sessions.
#
# Reference points: paper Table 7 puts VAE Enc at 1206.5 ms on 8 Elite Gen4.
# Our collapsed graph is 363.33 GMAC against ~1071.2 GMAC for a naive T=1
# 5-D export, and the decoder sustained 205.07 GMAC in 100.18 ms.
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
D=${1:-${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/eperf}
for f in "$D"/*.log; do
  n=$(basename "$f" .log)
  out=$(qnn-profile-viewer --input_log "$f" 2>/dev/null)
  for S in Min Average; do
    v=$(echo "$out" | grep -A18 "Execute Stats ($S)" \
        | grep -m1 'Accelerator (execute) time' | grep -oE '[0-9]+ us' | grep -oE '[0-9]+')
    printf "%-12s %-8s %12s us\n" "$n" "$S" "${v:-?}"
  done
done
