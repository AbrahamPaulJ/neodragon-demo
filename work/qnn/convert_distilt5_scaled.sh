#!/usr/bin/env bash
# Phase 1b v2 -- residual-scaled DistilT5 -> QNN -> HTP V79 context binary.
set -uo pipefail
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ONNX=$ND/work/onnx/distilt5_scaled_qnn.onnx
OUT=$HOME/neodragon-build/scaled
mkdir -p "$OUT"

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

step "1/3  qnn-onnx-converter"
qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/distilt5s.cpp" \
    --input_dim input_ids 1,128 \
    --input_dim attention_mask 1,128 \
    2>&1 | tail -12
echo "converter exit: ${PIPESTATUS[0]}"

step "2/3  qnn-model-lib-generator"
qnn-model-lib-generator \
    -c "$OUT/distilt5s.cpp" -b "$OUT/distilt5s.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -8
echo "libgen exit: ${PIPESTATUS[0]}"

step "3/3  qnn-context-binary-generator (HTP V79)"
qnn-context-binary-generator \
    --model "$OUT/lib/x86_64-linux-clang/libdistilt5s.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" \
    --binary_file distilt5s_v79 \
    --config_file "$ND/work/qnn/htp_backend_v79_scaled.json" 2>&1 | grep -viE '^\s*\[#* *\]|^\s*\[#+ *\] *[0-9]+%' | tail -25
echo "ctxgen exit: ${PIPESTATUS[0]}"
ls -la "$OUT/ctx" 2>/dev/null

step "copy to windows"
cp "$OUT/ctx/distilt5s_v79.bin" "$ND/work/device/" && echo "copied to work/device/"
step "done"
