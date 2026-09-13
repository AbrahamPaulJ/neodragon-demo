#!/usr/bin/env bash
# Convert one tiny concat reproducer (work/audit/concat_repro.py) end to end.
#
# The point of the reproducer is a fast loop: the real stage-1 graph costs ~55 minutes
# per attempt, this costs about one.
#
#   wsl -d Ubuntu -e bash work/qnn/convert_repro.sh cat3
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
V=${1:-cat3}
NAME=crepro_$V
ONNX=$ND/work/onnx/$NAME/$NAME.onnx
CALIB=$ND/work/calib/$NAME
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

[ -f "$ONNX" ] || { echo "ERROR: no $ONNX"; exit 1; }
DIMS=$(tr -d '\\\n' < "$CALIB/dims.txt" | tr -s ' ')
LAYOUTS=""
for nm in $(grep -oP '(?<=--input_dim )\S+' "$CALIB/dims.txt"); do
  LAYOUTS="$LAYOUTS --input_layout $nm NONTRIVIAL"
done

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

echo "== converting $NAME =="
qnn-onnx-converter --input_network "$ONNX" --output_path "$OUT/$NAME.cpp" \
    --preserve_io layout out \
    $LAYOUTS $DIMS \
    --input_list "$CALIB/calib_list_host.txt" \
    --use_per_channel_quantization --act_bitwidth 16 \
    --weights_bitwidth 8 --bias_bitwidth 32 2>&1 | tail -6

qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -2

qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$NAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | tail -3

cp "$OUT/ctx/${NAME}_${HTP_ARCH}.bin" "$ND/work/device/" && echo "-> work/device/${NAME}_${HTP_ARCH}.bin"
cp "$OUT/${NAME}_net.json" "$ND/work/device/" 2>/dev/null
ls -la "$ND/work/device/${NAME}_${HTP_ARCH}.bin"
