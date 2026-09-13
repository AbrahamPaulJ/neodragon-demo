#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
qnn-onnx-converter --help 2>&1 | grep -B2 -A6 -iE '^\s*-n\b|--input_list|--act_bitwidth|--bias_bitwidth|--use_per_channel|--algorithms|--quantization_overrides|--float_bitwidth' | head -90
