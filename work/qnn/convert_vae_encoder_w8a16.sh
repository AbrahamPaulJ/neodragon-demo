#!/usr/bin/env bash
# Phase 4b -- W8A16 PTQ of the T=1 VAE ENCODER, collapsed to pure 2-D.
#
# Graph: image [1,3,320,512] in [-1,1]  ->  moments [1,32,40,64]
# One input, one output, rank-4 throughout, no 5-D op anywhere -- the T=1
# causal-pad collapse in work/export/export_vae_encoder_2d.py removed every
# Conv3d and every CausalGroupNorm time-into-batch fold before conversion.
#
# Two variants, selected by $1 (trap #7 vs the vaedecs->vaedecsn lesson):
#   vaeenc   --preserve_io layout image moments   both pinned NCHW (baseline)
#   vaeencn  --preserve_io layout moments         image left channel-last.
#            NHWC is the image's NATIVE host layout -- _pil_to_numpy returns
#            [1,H,W,3] and generation_utils transposes it to NCHW only to match
#            torch -- so feeding NHWC costs the host nothing and drops a
#            boundary transpose on the largest input in the graph.
#            Host raws for this build come from make_enc_io.py --nhwc.
#
# Targets (paper Table 7 / Table 9, 8 Elite Gen4): 1206.5 ms, 40 dB, calib 50.
# 363.33 GMAC collapsed vs 1071.2 GMAC for a naive T=1 5-D export -- the paper's
# figure is consistent with the uncollapsed cost, so expect well under 1206 ms.
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${1:-vaeenc}
ONNX=$ND/work/onnx/vaeenc2d_qnn.onnx
CALIB=$ND/work/calib/vae_enc
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

case "$NAME" in
  vaeenc)  PRESERVE=(--preserve_io layout image moments) ;;
  vaeencn) PRESERVE=(--preserve_io layout moments)
           CALIB=$CALIB/nhwc ;;
  *) echo "ERROR: unknown build name '$NAME' (want vaeenc or vaeencn)"; exit 1 ;;
esac

# Table 9 uses 50 calibration samples for VAE Enc. We captured 64 real SSD1B
# first frames from vbench prompts; 50 calibrate, the rest stay unused, and the
# deploy number is measured on 16 showcase-prompt frames that were never seen.
LIST=$OUT/calib_list.txt
ls "$CALIB"/image_*.raw 2>/dev/null | sort | head -50 > "$LIST"
n=$(wc -l < "$LIST")
echo "calibration samples: $n  (paper uses 50 for VAE Enc)"
[ "$n" -eq 50 ] || { echo "ERROR: expected 50 calibration files in $CALIB, got $n"; exit 1; }

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

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

step "1/3 converter (W8A16 PTQ, $NAME)"
qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    "${PRESERVE[@]}" \
    --input_dim image "1,3,320,512" \
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
