#!/usr/bin/env bash
# Phase 5d -- W8A16 PTQ of one MMDiT pyramid stage (padded envelope graph).
#
# This is a different weight class from everything before it. DistilT5 is 130 M params
# and the VAE encoder 37 M; the MMDiT is **1512 M**, i.e. a 5.64 GiB fp32 ONNX in
# external-data format. Host RAM is 15.6 GB and .wslconfig gives WSL 11 GB + 48 GB swap,
# so the converter's working set is the practical constraint on this phase, not time.
# Run `--probe` first: it converts with a SINGLE calibration sample under
# /usr/bin/time -v and reports peak RSS, which tells you whether the full run is a
# 40-minute job or a swap-thrashing disaster.
#
# Inputs come from work/device/make_mmdit_io.py, which also writes dims.txt so this
# script never hardcodes a stage's shapes.
#
#   ./convert_mmdit_w8a16.sh 0 --layout            # NO calibration: layout only, ~2 min
#   ./convert_mmdit_w8a16.sh 0 --layout pinned     # ... with the old all-pinned flags
#   ./convert_mmdit_w8a16.sh 0 --probe             # 1 sample, measure peak RSS
#   ./convert_mmdit_w8a16.sh 0 11                  # 11 samples, measure the slope
#   ./convert_mmdit_w8a16.sh 0                     # full run, 300 samples (Table 9)
#
# --layout is the loop to iterate transposes in. Quantisation costs ~15 s per
# calibration sample and ~6 min of fixed cost, and it does not change the axis-tracking
# decisions -- those are IrOptimizer passes that run before the quantiser. Dropping
# --input_list gives the same net.json layout structure in a fraction of the time and
# memory. Byte counts in analyze_net.py double (fp32 vs A16) but the RATIO is unchanged.
set -uo pipefail
# Target SoC. Defaults are the S25 Ultra (SM8750, HTP v79). See docs/porting-other-socs.md
HTP_ARCH=${HTP_ARCH:-v79}
SOC_MODEL=${SOC_MODEL:-69}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAGE=${1:-0}
MODE=${2:-full}
# NAME lets several variants of one stage coexist: the session-6 fix build is
# mmdit_s0f, its ablations mmdit_s0fa / mmdit_s0fb, and the session-5 baseline stays
# mmdit_s0. It selects the ONNX directory, the calibration directory and every output.
#   NAME=mmdit_s0f ./convert_mmdit_w8a16.sh 0
NAME=${NAME:-mmdit_s${STAGE}}
ONNX=$ND/work/onnx/${NAME}_env/${NAME}_env_raw.onnx
CALIB=$ND/work/calib/${NAME}
OUT=$HOME/neodragon-build/$NAME
mkdir -p "$OUT"

[ -f "$ONNX" ] || { echo "ERROR: no ONNX at $ONNX"; echo "run: py -3.10 work/export/export_mmdit_stage.py --stage $STAGE --envelope --export"; exit 1; }
[ -f "$CALIB/dims.txt" ] || { echo "ERROR: no dims.txt in $CALIB -- run make_mmdit_io.py"; exit 1; }

# $2 may be --probe (1 sample), a plain integer (that many samples, still probe
# mode), or absent (the full 300 Table 9 calibration).
LIST=$OUT/calib_list.txt
case "$MODE" in
  --layout) : ;;
  --probe) head -1   "$CALIB/calib_list_host.txt" > "$LIST" ;;
  *[!0-9]*|"") head -300 "$CALIB/calib_list_host.txt" > "$LIST" ;;
  *)       head -"$MODE" "$CALIB/calib_list_host.txt" > "$LIST"; MODE=--probe ;;
esac
n=$([ "$MODE" = "--layout" ] && echo 0 || wc -l < "$LIST")
echo "calibration samples: $n  (paper Table 9 uses 300 for each MMDiT row)"
if [ "$MODE" != "--layout" ]; then
  missing=$(tr ' ' '\n' < "$LIST" | sed 's/^.*:=//' | sort -u | while read -r f; do
               [ -f "$f" ] || echo x; done | wc -l)
  [ "$missing" -eq 0 ] || { echo "ERROR: $missing calibration files missing"; exit 1; }
fi

echo "ONNX: $(du -sh "$(dirname "$ONNX")" | cut -f1) in $(ls "$(dirname "$ONNX")" | wc -l) files"
echo "host: $(free -g | awk '/^Mem:/{print $2}') GiB RAM, $(free -g | awk '/^Swap:/{print $2}') GiB swap"

# Stage the model on WSL's ext4. /mnt/c is a 9p mount: fine for the 149 MB VAE encoder,
# but this is 5.6 GiB and the converter streams it more than once.
LOCAL=$HOME/neodragon-build/onnx/$(basename "$(dirname "$ONNX")")
if [ ! -f "$LOCAL/$(basename "$ONNX")" ] || \
   [ "$(stat -c %s "$ONNX")" != "$(stat -c %s "$LOCAL/$(basename "$ONNX")" 2>/dev/null)" ]; then
  echo "staging ONNX to ext4 ($LOCAL) ..."
  mkdir -p "$LOCAL" && cp "$(dirname "$ONNX")"/* "$LOCAL/" || exit 1
fi
ONNX=$LOCAL/$(basename "$ONNX")
echo "using: $ONNX"

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

# dims.txt is a pre-formatted, line-continued --input_dim block
DIMS=$(tr -d '\\\n' < "$CALIB/dims.txt" | tr -s ' ')

# Layout control. Measured on the first stage-0 build: 70% of all transpose traffic
# (287.6 MB of 407.2 MB) was TWO permutations of the [24,408,408] attention score
# matrix per block -- MatMul and Softmax both run NCF, but the mask-add Eltwise_Binary
# runs NFC, so every block paid a round trip. The tensor names in the net.json say it
# outright: _MatMul_output_0_nfc then _Add_5_output_0_ncf.
#
# QNN's MaskedSoftmax op would fuse the three, but htp.json does not list it -- only
# Softmax and LogSoftmax -- so that route is closed on this backend.
#
# The fix is to tell the converter these tensors have no layout semantics at all.
# NONTRIVIAL means "as authored", so unlike NHWC it needs no host-side permutation and
# cannot silently reinterpret bytes (trap #8). Only the latents keep image semantics:
# they are 5-D and feed pos_embed.proj, a real Conv2d, where trap #7 applies.
FLAVOUR=${3:-nontrivial}
LAYOUTS=""
PRESERVE=""
if [ "$FLAVOUR" = "pinned" ]; then
  PRESERVE=""                      # bare --preserve_io layout pins everything
else
  for nm in $(grep -oP '(?<=--input_dim )\S+' "$CALIB/dims.txt"); do
    case "$nm" in
      latent_*) PRESERVE="$PRESERVE $nm" ;;
      *)        LAYOUTS="$LAYOUTS --input_layout $nm NONTRIVIAL" ;;
    esac
  done
  PRESERVE="$PRESERVE noise_pred"
fi
echo "layout flavour: $FLAVOUR"
echo "preserve_io layout:$PRESERVE"
echo "NONTRIVIAL inputs:$(echo "$LAYOUTS" | grep -oP '(?<=--input_layout )\S+' | tr '\n' ' ')"

if [ "$MODE" = "--layout" ]; then
  QFLAGS=()
  step "1/3 converter (LAYOUT ONLY -- no quantisation, $NAME)"
else
  # 16-bit DYNAMIC weights. QNN treats a MatMul's second operand as a "weight", and
  # 16-bit dynamic weights are off by default -- so K and V, which are pure activations,
  # were quantised to 8 bits in all 18 attention blocks (measured on the first build:
  # 36 MatMuls at u16 x u8 with the second operand UNSTARRED, i.e. not static).
  # The converter says so itself during conversion:
  #   "mixedPrecisionForWeights: The override provided for the following weight:
  #    /Transpose_71_output_0 would not be honoured. Since 16 bit dynamic weights are
  #    not supported by default. Kindly use the flag --use_dynamic_16_bit weights"
  # htp.json lists u16 x u16 -> u16 as a supported MatMul kernel, so the 8-bit operand
  # is a converter default, NOT a hardware limit. --restrict_quantization_steps is the
  # documented companion ("required for 16-bit Matmul operations") and is only honoured
  # because we pass --use_per_channel_quantization (gate at qnn_quantizer.py:399).
  #
  # ---- MEASURED 2026-08-21: THIS DOES NOT FIX THE ACCURACY GAP. ----------------
  # The flags work exactly as advertised -- matmul_bits.py confirms all 36 MatMuls
  # move u16 x u8 -> u16 x u16 and the 255 FullyConnected keep their static u8
  # weights -- but on device it measured:
  #      accuracy  19.47 -> 19.48 dB   (+0.01, i.e. nothing; target is 29)
  #      latency   248.3 -> 294.5 ms   (+18.6%, min-of-3-rounds)
  # so it is a strict regression and is now OFF by default. It is kept because it
  # does not rule out mattering LATER: if the dominant error source is found and
  # fixed, 8-bit K/V could become the next binding constraint. Re-enable with
  # DYN16=1 and re-measure BOTH axes. Conversion also costs 42:26 instead of 35:04.
  # See docs/phase5-mmdit-accuracy.md.
  # -----------------------------------------------------------------------------
  # Verify with: py -3.10 work/device/matmul_bits.py work/device/mmdit_s0_net.json
  # --use_per_row_quantization: --use_per_channel_quantization is documented
  # CONVOLUTION-ONLY, and the blocks are 428 FullyConnected with 0 convolutions, so
  # without this EVERY weight in the model was per-TENSOR 8-bit (verified in net.json:
  # 0 per-row, 428 per-tensor). The host ablation that predicted 43.14 dB assumed
  # per-channel weights the graph never got, so the real W8 ceiling was lower all along.
  QFLAGS=(--input_list "$LIST" --use_per_channel_quantization
          --use_per_row_quantization --enable_per_row_quantized_bias
          --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32)
  # OFF BY DEFAULT -- MEASURED DEAD END. Opt in with DYN16=1.
  if [ "${DYN16:-0}" = "1" ]; then
    QFLAGS+=(--use_dynamic_16_bit_weights
             --restrict_quantization_steps "-0x8000 0x7F7F")
  fi
  step "1/3 converter (W8A16 PTQ, $NAME, $n samples)"
fi
/usr/bin/time -v qnn-onnx-converter \
    --input_network "$ONNX" \
    --output_path "$OUT/$NAME.cpp" \
    --preserve_io layout $PRESERVE \
    $LAYOUTS \
    $DIMS \
    "${QFLAGS[@]}" \
    2>&1 | tee "$OUT/converter_raw.log" | grep -viE '^\s*\[[# ]*\]' | tail -40
echo "exit ${PIPESTATUS[0]}"

cp "$OUT/${NAME}_net.json" "$ND/work/device/" 2>/dev/null \
  && echo "-> work/device/${NAME}_net.json"

if [ "$MODE" = "--layout" ]; then
  echo
  echo "LAYOUT run done ($FLAVOUR). Now: py -3.10 work/device/analyze_net.py \\"
  echo "     work/device/${NAME}_net.json   and trace_transposes.py on the same file."
  exit 0
fi
if [ "$MODE" = "--probe" ]; then
  echo
  echo "PROBE done. Read 'Maximum resident set size' above against the 11 GB WSL limit."
  echo "If it fits, rerun without --probe for the full 300-sample calibration."
  exit 0
fi

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
