#!/usr/bin/env bash
# W8A16 PTQ of the ContextAdapter -- the module the phased plan never listed.
#
# 130 M params, a SkipMLP: 4096 -> 4096 x4 with concat skips -> 1536. No convolution and
# no attention, so traps #7 (preserve_io on conv inputs) and #11 (layout) do not bite.
# Both tensors are rank 3 with no image semantics, hence NONTRIVIAL (trap #23).
#
# It runs once per generation rather than per AR unit, so latency is not the point --
# this exists so the video path is entirely on the NPU.
#
#   wsl -d Ubuntu -e bash work/qnn/convert_ctxadapt_w8a16.sh
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=ctxadapt
# WBITS overrides the weight bitwidth and renames the artefact, so W8 and W16 builds
# coexist. MEASURED on the host: even a faithful per-row W8 simulation of this module
# only reaches 16.84 dB, because it is a 5-layer MLP with weight outliers and NO
# normalisation layers, so weight error compounds multiplicatively with depth.
# Activations are innocent (A16 alone: 72.00 dB). It is 130 M params and runs once per
# generation, so spending 16-bit weights costs 260 MB and no measurable time.
ONNX=$ND/work/onnx/$NAME/${NAME}_raw.onnx
CALIB=$ND/work/calib/$NAME
# only the OUTPUT is renamed by WBITS -- the ONNX and calibration set are shared
OUTNAME=${NAME}${WBITS:+w$WBITS}
OUT=$HOME/neodragon-build/$OUTNAME
mkdir -p "$OUT"

[ -f "$ONNX" ] || { echo "ERROR: no $ONNX -- run export_context_adapter.py --export"; exit 1; }
[ -f "$CALIB/dims.txt" ] || { echo "ERROR: no $CALIB/dims.txt"; exit 1; }

n=$(wc -l < "$CALIB/calib_list_host.txt")
echo "calibration samples: $n"
missing=$(sed 's/^.*:=//' "$CALIB/calib_list_host.txt" | while read -r f; do
             [ -f "$f" ] || echo x; done | wc -l)
[ "$missing" -eq 0 ] || { echo "ERROR: $missing calibration files missing"; exit 1; }

DIMS=$(tr -d '\\\n' < "$CALIB/dims.txt" | tr -s ' ')
CFG=$ND/work/qnn/htp_config_${OUTNAME}.json
BE=$ND/work/qnn/htp_backend_${OUTNAME}.json
cat > "$CFG" <<EOF
{ "graphs": [ { "vtcm_mb": 8, "graph_names": ["$OUTNAME"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$HTP_ARCH", "soc_model": $SOC_MODEL,
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
EOF
cat > "$BE" <<EOF
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "$CFG" } }
EOF

step() { echo; echo "########## $* ##########"; date '+%H:%M:%S'; }

# FLOAT=1 converts with NO --input_list at all: the converter then emits a float graph
# and HTP runs float graphs in fp16 (trap #3). That is exactly how DistilT5 ships at
# 49.04 dB, and Table 9 deploys the whole text path (CLIP L, CLIP G, DistilT5) in FP16
# with no ContextAdapter row at all -- so FP16 here matches the reference deployment.
# Safe for this module: it has no LayerNorm (trap #3 NaN risk) and its activations peak
# near 80, far inside fp16 range.
if [ "${FLOAT:-0}" = "1" ]; then
  QFLAGS=""
  OUTNAME=${NAME}fp16
  OUT=$HOME/neodragon-build/$OUTNAME
  CFG=$ND/work/qnn/htp_config_${OUTNAME}.json
  BE=$ND/work/qnn/htp_backend_${OUTNAME}.json
  cat > "$CFG" <<EOF
{ "graphs": [ { "vtcm_mb": 8, "graph_names": ["$OUTNAME"], "O": 3.0 } ],
  "devices": [ { "dsp_arch": "$HTP_ARCH", "soc_model": $SOC_MODEL,
                 "cores": [ { "perf_profile": "burst", "rpc_control_latency": 100 } ] } ] }
EOF
  cat > "$BE" <<EOF
{ "backend_extensions": { "shared_library_path": "libQnnHtpNetRunExtensions.so",
                          "config_file_path": "$CFG" } }
EOF
  mkdir -p "$OUT"
else
  QFLAGS="--input_list $CALIB/calib_list_host.txt --use_per_channel_quantization --use_per_row_quantization --enable_per_row_quantized_bias --act_bitwidth 16 --weights_bitwidth ${WBITS:-8} --bias_bitwidth 32"
fi

step "1/3 converter (W8A16 PTQ, $NAME, $n samples)"
/usr/bin/time -v qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/$OUTNAME.cpp" \
    --preserve_io layout context \
    --input_layout prompt_embeds NONTRIVIAL \
    $DIMS \
    ${QFLAGS} \
    2>&1 | tee "$OUT/converter_raw.log" | grep -viE '^\s*\[[# ]*\]' | tail -12
echo "exit ${PIPESTATUS[0]}"

step "2/3 lib-generator"
qnn-model-lib-generator -c "$OUT/$OUTNAME.cpp" -b "$OUT/$OUTNAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -3

step "3/3 context-binary-generator"
qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$OUTNAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${OUTNAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -6

cp "$OUT/ctx/${OUTNAME}_${HTP_ARCH}.bin" "$ND/work/device/" && echo "-> work/device/${OUTNAME}_${HTP_ARCH}.bin"
cp "$OUT/${OUTNAME}_net.json" "$ND/work/device/" 2>/dev/null \
  && echo "-> work/device/${OUTNAME}_net.json"
ls -la "$ND/work/device/${OUTNAME}_${HTP_ARCH}.bin"
