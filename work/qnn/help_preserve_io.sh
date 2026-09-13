#!/usr/bin/env bash
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
qnn-onnx-converter --help 2>&1 | grep -n "Use this option to preserve IO" -A 18
