#!/usr/bin/env bash
p=$(pgrep -f qnn-onnx-converter | head -1)
[ -z "$p" ] && { echo "converter not running"; exit 1; }
echo "pid=$p"
tr '\0' ' ' < /proc/$p/cmdline | grep -oE '\-\-use_per_row_quantization|\-\-enable_per_row_quantized_bias|\-\-use_per_channel_quantization' | sort -u
