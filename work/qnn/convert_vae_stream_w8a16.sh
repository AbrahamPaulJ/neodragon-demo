#!/usr/bin/env bash
# Phase 4f -- W8A16 PTQ of the STREAMING TAEHV decoder (explicit MemBlock state).
#
# Replaces convert_vae_w8a16.sh for the new graph. Two differences that matter:
#
#   1. Ten inputs, not one. --input_dim is repeated per input, and the calibration
#      list carries "name:=path" pairs (produced by work/device/make_stream_calib.py).
#   2. The graph is rank-4 throughout -- the 1x1x1 post_quant_conv is folded to a
#      Conv2d -- so there is no 5-D tensor left. --preserve_io layout is still
#      passed (trap #7: mandatory for any conv model), it simply has less to pin.
#
# The point of the rewrite is the Transpose count. Baseline to beat, from
# analyze_net.py on the old vaedec1q_net.json: 47 Transposes moving 779.80 MB.
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${1:-vaedecs}
ONNX=$ND/work/onnx/vaedec_stream_qnn.onnx
CALIB=$ND/work/calib/vae_dec_stream
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

# paper Table 9 uses 50 calibration samples for VAE Dec; 50-55 stay held out.
# calib_list_host.txt, NOT input_list.txt: the latter carries /data/local/tmp
# device paths for qnn-net-run and the converter would not find any of them.
LIST=$OUT/calib_list.txt
head -50 "$CALIB/calib_list_host.txt" > "$LIST"
echo "calibration samples: $(wc -l < "$LIST")  (paper uses 50 for VAE Dec)"
missing=$(tr ' ' '\n' < "$LIST" | sed 's/^.*:=//' | sort -u | while read -r f; do
             [ -f "$f" ] || echo x; done | wc -l)
[ "$missing" -eq 0 ] || { echo "ERROR: $missing calibration files missing"; exit 1; }

CFG=$ND/work/qnn/htp_config_${NAME}.json
BE=$ND/work/qnn/htp_backend_${NAME}.json
# soc_model 69 verified correct 2026-08-21: QNN_SOC_MODEL_SM8750 = 69 (QnnTypes.h)
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

DIMS=(--input_dim latent "1,16,40,64")
for spec in "0 256,40,64" "1 256,40,64" "2 256,40,64" \
            "3 128,80,128" "4 128,80,128" "5 128,80,128" \
            "6 64,160,256" "7 64,160,256" "8 64,160,256"; do
  set -- $spec
  DIMS+=(--input_dim "state_$1" "1,$2")
done

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

step "1/3 converter (W8A16 PTQ, 10 inputs)"
qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    --preserve_io layout \
    "${DIMS[@]}" \
    --input_list "$LIST" \
    --use_per_channel_quantization \
    --act_bitwidth 16 \
    --weights_bitwidth 8 \
    --bias_bitwidth 32 \
    2>&1 | tail -25
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
cp "$OUT/${NAME}_net.json" "$ND/work/device/" 2>/dev/null \
  && echo "-> work/device/${NAME}_net.json  (run analyze_net.py on this)"
