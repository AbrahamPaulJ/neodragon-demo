#!/usr/bin/env bash
# W8A16 PTQ of the SSD1B VAE decoder (Table 9 row "SSD1B Dec", 31 dB).
#
#   ./convert_ssd1bvaedec_w8a16.sh --layout   # layout screen only
#   ./convert_ssd1bvaedec_w8a16.sh            # full run
#
# This is a conv decoder that upsamples 80x128 -> 640x1024x3, so it is squarely in trap
# #11 territory (the video VAE decoder's reshape-into-batch cost 53.5x) and trap #17
# (GroupNorm arrives fused natively -- verified there, all 22, ratio 0.05). Screen the
# ratio with analyze_net.py before spending device time.
#
# `latent` genuinely feeds convolution, so it keeps its image layout (trap #7 -- omitting
# --preserve_io layout on a conv model measured -0.75 dB with nothing looking broken).
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=ssd1bvaedec
MODE=${1:-full}
ONNX=$ND/work/onnx/$NAME/${NAME}_raw.onnx
CALIB=$ND/work/calib/$NAME
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"
[ -f "$ONNX" ] || { echo "ERROR: no $ONNX -- run export_ssd1b_vaedec.py --export"; exit 1; }

DIMS=$(tr -d '\\\n' < "$CALIB/dims.txt" | tr -s ' ')
case "$MODE" in
  --layout) QFLAGS=""; n=0 ;;
  *) cp "$CALIB/calib_list_host.txt" "$OUT/calib_list.txt"
     n=$(wc -l < "$OUT/calib_list.txt")
     QFLAGS="--input_list $OUT/calib_list.txt --use_per_channel_quantization"
     QFLAGS="$QFLAGS --use_per_row_quantization --enable_per_row_quantized_bias"
     QFLAGS="$QFLAGS --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32"
     missing=$(sed 's/^.*:=//' "$OUT/calib_list.txt" | while read -r f; do
                  [ -f "$f" ] || echo x; done | wc -l)
     [ "$missing" -eq 0 ] || { echo "ERROR: $missing calibration files missing"; exit 1; } ;;
esac
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
    --preserve_io layout latent image \
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
