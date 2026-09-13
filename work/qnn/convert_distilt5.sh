#!/usr/bin/env bash
# Phase 1b -- DistilT5 ONNX -> QNN model -> HTP V79 context binary.
#
# Float conversion first: the point of this pass is to answer "does the graph
# convert and compile for V79 at all", not to hit an accuracy target. W8A16
# quantisation comes after, once calibration data exists.
set -uo pipefail
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ONNX=$ND/work/onnx/distilt5_qnn.onnx
OUT=$HOME/neodragon-build
mkdir -p "$OUT"

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

step "0/3  inputs"
ls -la "$ONNX"
python - <<PY
import onnx
m = onnx.load("$ONNX", load_external_data=False)
for i in m.graph.input:
    d = [x.dim_value for x in i.type.tensor_type.shape.dim]
    print(f"  input  {i.name:<16} {d}  elem_type={i.type.tensor_type.elem_type}")
for o in m.graph.output:
    d = [x.dim_value for x in o.type.tensor_type.shape.dim]
    print(f"  output {o.name:<16} {d}  elem_type={o.type.tensor_type.elem_type}")
PY

step "1/3  qnn-onnx-converter (float)"
time qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/distilt5.cpp" \
    --input_dim input_ids 1,128 \
    --input_dim attention_mask 1,128 \
    2>&1 | tail -40
echo "converter exit: ${PIPESTATUS[0]}"
ls -la "$OUT"/distilt5.* 2>/dev/null

step "2/3  qnn-model-lib-generator"
time qnn-model-lib-generator \
    -c "$OUT/distilt5.cpp" -b "$OUT/distilt5.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -20
echo "libgen exit: ${PIPESTATUS[0]}"
find "$OUT/lib" -name '*.so' -exec ls -la {} \; 2>/dev/null

step "3/3  qnn-context-binary-generator (HTP V79)"
time qnn-context-binary-generator \
    --model "$OUT/lib/x86_64-linux-clang/libdistilt5.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" \
    --binary_file distilt5_v79 \
    --config_file "$ND/work/qnn/htp_backend_v79.json" 2>&1 | tail -30
echo "ctxgen exit: ${PIPESTATUS[0]}"
ls -la "$OUT/ctx" 2>/dev/null

step "done"
