#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
qnn-profile-viewer --input_log ${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/work/device/sperf/sperf_1.log 2>/dev/null \
  | grep -E "Execute Stats|Accelerator \(execute\)|NetRun|Num Inferences|inferences|Backend \(execute\)" | head -30
