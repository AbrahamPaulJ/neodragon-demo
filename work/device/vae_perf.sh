#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
L=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/prof_v.log
out=$(qnn-profile-viewer --input_log "$L" 2>/dev/null)
for S in Average Min Max; do
  echo "--- $S ---"
  echo "$out" | grep -A16 "Execute Stats ($S)" \
    | grep -E 'NetRun:|Accelerator \(execute\) time' | head -4
done
