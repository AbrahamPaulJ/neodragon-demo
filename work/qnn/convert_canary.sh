#!/usr/bin/env bash
# The FP16 canary: a few-KB float graph that ships inside the APK and is run before any
# 8 GB download is offered. See work/export/export_canary.py for what it proves.
#
#   ./convert_canary.sh
#
# Built as a FLOAT graph deliberately -- no --input_list, no quantisation flags -- so the
# HTP runs it in fp16 exactly as it runs ctxadaptfp16 and distilt5f.
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=canary
MODE=${1:-full}
ONNX=$ND/work/onnx/canary/canary_raw.onnx
CALIB=$ND/work/calib/canary
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"
[ -f "$ONNX" ] || { echo "ERROR: no $ONNX -- run: py -3.10 work/export/export_canary.py"; exit 1; }

DIMS=$(tr -d '\\\n' < "$CALIB/dims.txt" | tr -s ' ')
# Always a float build: the canary exists to test the fp16 path, so there is no
# quantisation branch at all.
QFLAGS=""
n=0
echo "mode=$MODE  calibration samples: $n"

CFG=$ND/work/qnn/htp_config_${NAME}.json
BE=$ND/work/qnn/htp_backend_${NAME}.json
cat > "$CFG" <<EOF
{ "graphs": [ { "vtcm_mb": 8, "graph_names": ["$NAME"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$HTP_ARCH", "soc_model": $SOC_MODEL,
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
EOF
cat > "$BE" <<EOF
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "$CFG" } }
EOF

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

step "1/3 converter ($MODE, $NAME, $n samples)"
/usr/bin/time -v qnn-onnx-converter --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    --preserve_io layout x y \
    $DIMS $QFLAGS \
    2>&1 | tee "$OUT/converter_raw.log" | grep -viE '^\s*\[[# ]*\]' | tail -12
echo "exit ${PIPESTATUS[0]}"
cp "$OUT/${NAME}_net.json" "$ND/work/device/" 2>/dev/null \
  && echo "-> work/device/${NAME}_net.json"
[ "$MODE" = "--layout" ] && { echo; echo "LAYOUT screen done -- run analyze_net.py"; exit 0; }

step "2/3 lib-generator"
qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -3

step "3/3 context-binary-generator"
qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$NAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -6

cp "$OUT/ctx/${NAME}_${HTP_ARCH}.bin" "$ND/work/device/" && echo "-> work/device/${NAME}_${HTP_ARCH}.bin"
ls -la "$ND/work/device/${NAME}_${HTP_ARCH}.bin"
