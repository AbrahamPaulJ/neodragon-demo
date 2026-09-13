#!/usr/bin/env bash
# Phase 3 -- W8A16 PTQ of the SSD1B UNet (first-frame path, 640x1024, latent 80x128).
#
#   ./convert_ssd1bunet_w8a16.sh --layout     # NO calibration: layout screen only, fast
#   ./convert_ssd1bunet_w8a16.sh --probe      # 1 sample, reports peak RSS
#   ./convert_ssd1bunet_w8a16.sh              # full run, 500 samples (Table 9)
#
# --layout exists because this is the most conv-heavy module in the pipeline and therefore
# the one most likely to inherit trap #11 -- the reshape-into-batch that cost the VAE
# decoder a 53.5x latency penalty. Axis tracking runs in IrOptimizer passes BEFORE the
# quantiser, so a calibration-free build makes identical layout decisions (trap #24) and
# `analyze_net.py` reads the transpose/compute ratio off it for free. 1.16 was
# catastrophic, 0.28 livable, 0.09 clean. Screen BEFORE spending GPU time on capture.
#
# --use_per_row_quantization is included from the start: --use_per_channel_quantization is
# documented CONVOLUTION-ONLY, and this graph has MatMul x496. Omitting it left the
# ContextAdapter's weights per-tensor 8-bit and cost that module ~25 dB.
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=ssd1bunet
MODE=${1:-full}
ONNX=$ND/work/onnx/$NAME/${NAME}_raw.onnx
CALIB=$ND/work/calib/$NAME
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

[ -f "$ONNX" ] || { echo "ERROR: no $ONNX -- run export_ssd1b_unet.py --export"; exit 1; }

DIMS="--input_dim sample 1,4,80,128 --input_dim t_emb 1,320 --input_dim aug_emb 1,1280 --input_dim encoder_hidden_states 1,77,2048"

# Only `sample` has image semantics and feeds conv, so it keeps its layout (trap #7).
# The rest are rank 2/3 conditioning with no spatial meaning -> NONTRIVIAL (trap #23).
PRESERVE="sample noise_pred"
LAYOUTS="--input_layout t_emb NONTRIVIAL --input_layout aug_emb NONTRIVIAL --input_layout encoder_hidden_states NONTRIVIAL"

case "$MODE" in
  --layout) QFLAGS="" ; n=0 ;;
  --probe)  head -1 "$CALIB/calib_list_host.txt" > "$OUT/calib_list.txt"
            QFLAGS="--input_list $OUT/calib_list.txt"; n=1 ;;
  *)        head -500 "$CALIB/calib_list_host.txt" > "$OUT/calib_list.txt"
            QFLAGS="--input_list $OUT/calib_list.txt"; n=$(wc -l < "$OUT/calib_list.txt") ;;
esac
if [ -n "$QFLAGS" ]; then
  QFLAGS="$QFLAGS --use_per_channel_quantization --use_per_row_quantization"
  QFLAGS="$QFLAGS --enable_per_row_quantized_bias"
  QFLAGS="$QFLAGS --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32"
  missing=$(tr ' ' '\n' < "$OUT/calib_list.txt" | sed 's/^.*:=//' | sort -u |
            while read -r f; do [ -f "$f" ] || echo x; done | wc -l)
  [ "$missing" -eq 0 ] || { echo "ERROR: $missing calibration files missing"; exit 1; }
fi
echo "mode=$MODE  calibration samples: $n"

# Stage on ext4: /mnt/c is a 9p mount and this is 4.94 GB streamed more than once.
LOCAL=$HOME/neodragon-build/onnx/$NAME
if [ ! -f "$LOCAL/$(basename "$ONNX")" ] || \
   [ "$(stat -c %s "$ONNX")" != "$(stat -c %s "$LOCAL/$(basename "$ONNX")" 2>/dev/null)" ]; then
  echo "staging ONNX to ext4 ($LOCAL) ..."
  mkdir -p "$LOCAL" && cp "$(dirname "$ONNX")"/* "$LOCAL/" || exit 1
fi
ONNX=$LOCAL/$(basename "$ONNX")

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
/usr/bin/time -v qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    --preserve_io layout $PRESERVE \
    $LAYOUTS $DIMS $QFLAGS \
    2>&1 | tee "$OUT/converter_raw.log" | grep -viE '^\s*\[[# ]*\]' | tail -14
echo "exit ${PIPESTATUS[0]}"

cp "$OUT/${NAME}_net.json" "$ND/work/device/" 2>/dev/null \
  && echo "-> work/device/${NAME}_net.json"

if [ "$MODE" = "--layout" ]; then
  echo
  echo "LAYOUT screen done. Now:  py -3.10 work/device/analyze_net.py work/device/${NAME}_net.json"
  echo "  ratio 1.16 was catastrophic (VAE decoder), 0.28 livable (DistilT5), 0.09 clean."
  exit 0
fi
[ "$MODE" = "--probe" ] && { echo; echo "PROBE done -- read Maximum resident set size against the 11 GB WSL cap."; exit 0; }

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
