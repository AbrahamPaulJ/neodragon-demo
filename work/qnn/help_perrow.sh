#!/usr/bin/env bash
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
qnn-onnx-converter --help > /tmp/h.txt 2>&1
sed -n '340,356p' /tmp/h.txt
