#!/usr/bin/env bash
# CLIP L / CLIP G -> FP16 graphs, exactly as Table 9 deploys them.
#
# No --input_list at all. The converter then emits a FLOAT graph and HTP runs float
# graphs in fp16 (trap #3) -- this is the same recipe DistilT5 ships with at 49.04 dB,
# and Table 9 lists CLIP L, CLIP G and DistilT5 all as "FP16, calib NA".
#
# `input_ids` is INTEGER. On a float graph that means qnn-net-run needs
# --use_native_input_files at run time; on a QUANTISED graph it must be omitted. Getting
# it wrong yields identical, plausible-looking outputs for every input (trap #8), so the
# compare script asserts two different inputs give two different outputs.
#
#   wsl -d Ubuntu -e bash work/qnn/convert_clip_fp16.sh clipl
#   wsl -d Ubuntu -e bash work/qnn/convert_clip_fp16.sh clipg
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${1:?usage: convert_clip_fp16.sh <clipl|clipg>}
# QUANT=1 builds W8A16 instead of FP16 and names the artefact <name>q. Used for clipg:
# at fp32 it cannot link (trap #37) and at --float_bitwidth 16 it measured 14.71 dB on
# the penultimate hidden state against CLIP L's 60.75. An 8-bit blob is ~0.69 GB, links
# comfortably, and A16 activations carry more through 32 layers than fp16 does.
QUANT=${QUANT:-0}
ONNX=$ND/work/onnx/$NAME/${NAME}_raw.onnx
OUTNAME=$NAME$([ "$QUANT" = "1" ] && echo q)
OUT=$HOME/neodragon-build/$OUTNAME
mkdir -p "$OUT"
[ -f "$ONNX" ] || { echo "ERROR: no $ONNX -- run export_clip.py --export"; exit 1; }

# clipg is 2.59 GB in external-data format; stage it on ext4 (trap: /mnt/c is 9p).
LOCAL=$HOME/neodragon-build/onnx/$NAME
if [ ! -f "$LOCAL/$(basename "$ONNX")" ] || \
   [ "$(stat -c %s "$ONNX")" != "$(stat -c %s "$LOCAL/$(basename "$ONNX")" 2>/dev/null)" ]; then
  echo "staging ONNX to ext4 ..."
  mkdir -p "$LOCAL" && cp "$(dirname "$ONNX")"/* "$LOCAL/" || exit 1
fi
ONNX=$LOCAL/$(basename "$ONNX")

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

if [ "$QUANT" = "1" ]; then
  QFLAGS="--input_list $ND/work/calib/$NAME/calib_list_host.txt --use_per_channel_quantization --use_per_row_quantization --enable_per_row_quantized_bias --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32"
else
  QFLAGS="--float_bitwidth 16"
fi
step "1/3 converter ($OUTNAME)"
# --float_bitwidth 16 is REQUIRED for clipg, not merely nice. At fp32 its 694.7 M params
# make a 2.78 GB weight blob inside the model .so and the x86-64 link fails with
#     relocation truncated to fit: R_X86_64_PC32
# because PC32 relocations cannot address beyond 2 GB. fp16 halves it to 1.39 GB and
# links. It is also what HTP executes anyway (trap #3), so nothing is lost.
qnn-onnx-converter --input_network "$ONNX" --output_path "$OUT/$OUTNAME.cpp" \
    --input_dim input_ids 1,77 $QFLAGS \
    2>&1 | tee "$OUT/converter_raw.log" | grep -viE '^\s*\[[# ]*\]' | tail -10
echo "exit ${PIPESTATUS[0]}"

step "2/3 lib-generator"
cd "$OUT" || exit 1        # trap #30: lib-gen scratch tree lands under the CWD
qnn-model-lib-generator -c "$OUT/$OUTNAME.cpp" -b "$OUT/$OUTNAME.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -3

step "3/3 context-binary-generator"
qnn-context-binary-generator --model "$OUT/lib/x86_64-linux-clang/lib$OUTNAME.so" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${OUTNAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -6

cp "$OUT/ctx/${OUTNAME}_${HTP_ARCH}.bin" "$ND/work/device/" && echo "-> work/device/${OUTNAME}_${HTP_ARCH}.bin"
cp "$OUT/${OUTNAME}_net.json" "$ND/work/device/" 2>/dev/null
ls -la "$ND/work/device/${OUTNAME}_${HTP_ARCH}.bin"
