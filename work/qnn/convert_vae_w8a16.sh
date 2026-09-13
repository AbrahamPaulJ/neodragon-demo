#!/usr/bin/env bash
# Phase 4d -- W8A16 PTQ of the TAEHV decoder.
#
# Not an optimisation: the float build measured 10.88 s on device vs the paper's
# 248.9 ms, because HMX only accelerates quantised convolution. Flags mirror the
# proven LocalDream recipe (convert_inpaint_unet.sh).
#
#   usage: convert_vae_w8a16.sh <onnx-basename> <graph-name> <T> <calib-dir>
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SRC=$1; NAME=$2; T=$3; CALIB=$4
ONNX=$ND/work/onnx/${SRC}.onnx
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

# calibration list: one absolute path per line, raw fp32 in graph input layout
LIST=$OUT/calib_list.txt
find "$CALIB" -name 'latent_*.raw' | sort | head -50 > "$LIST"
echo "calibration samples: $(wc -l < "$LIST")  (paper uses 50 for VAE Dec)"
head -2 "$LIST"

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
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "$CFG" } }
EOF

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

step "1/3 converter (W8A16 PTQ)"
qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    --preserve_io layout \
    --input_dim latent "1,16,$T,40,64" \
    --input_list "$LIST" \
    --use_per_channel_quantization \
    --act_bitwidth 16 \
    --weights_bitwidth 8 \
    --bias_bitwidth 32 \
    2>&1 | tail -20
echo "exit ${PIPESTATUS[0]}"

step "2/3 lib-generator"
qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -4
echo "exit ${PIPESTATUS[0]}"

step "3/3 context-binary-generator"
qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$NAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -20
echo "exit ${PIPESTATUS[0]}"

ls -la "$OUT/ctx" && cp "$OUT/ctx/${NAME}_${HTP_ARCH}.bin" "$ND/work/device/" \
  && echo "-> work/device/${NAME}_${HTP_ARCH}.bin"
