#!/usr/bin/env bash
# Generic: ONNX -> QNN -> HTP V79 context binary.
#   usage: convert.sh <onnx-basename-without-ext> <graph-name>
# e.g.   convert.sh distilt5_folded_qnn distilt5f
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SRC=$1
NAME=$2
ONNX=$ND/work/onnx/${SRC}.onnx
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

# per-graph HTP config (graph_names must match the .cpp graph name)
CFG=$ND/work/qnn/htp_config_${NAME}.json
BE=$ND/work/qnn/htp_backend_${NAME}.json
cat > "$CFG" <<EOF
{
  "graphs": [ { "vtcm_mb": 8, "graph_names": ["$NAME"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$HTP_ARCH", "soc_model": $SOC_MODEL,
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ]
}
EOF
cat > "$BE" <<EOF
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "$CFG"
  }
}
EOF

step() { echo; echo "########## $* ##########"; }

step "1/3 converter"
qnn-onnx-converter --input_network "$ONNX" --output_path "$OUT/$NAME.cpp" \
    --input_dim input_ids 1,128 --input_dim attention_mask 1,128 2>&1 | tail -6
echo "exit ${PIPESTATUS[0]}"

step "2/3 lib-generator"
qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -4
echo "exit ${PIPESTATUS[0]}"

step "3/3 context-binary-generator"
qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$NAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -18
echo "exit ${PIPESTATUS[0]}"

ls -la "$OUT/ctx"
cp "$OUT/ctx/${NAME}_${HTP_ARCH}.bin" "$ND/work/device/" && echo "-> work/device/${NAME}_${HTP_ARCH}.bin"
