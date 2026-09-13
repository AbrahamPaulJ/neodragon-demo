#!/usr/bin/env bash
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
qnn-onnx-converter --help > /tmp/h.txt 2>&1
grep -inE "float_bitwidth|float_fallback|keep_weights|pack" /tmp/h.txt | head -12
echo "---- context of float_bitwidth ----"
awk '/--float_bitwidth/,/^  --[a-z]/' /tmp/h.txt | head -8
