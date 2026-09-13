#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
LOG=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/prof.log
CSV=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/prof.csv

echo "=== profile-viewer options ==="
qnn-profile-viewer --help 2>&1 | head -30

echo
echo "=== dumping to csv ==="
qnn-profile-viewer --input_log "$LOG" --output "$CSV" 2>&1 | tail -5
ls -la "$CSV" 2>/dev/null && wc -l "$CSV"
