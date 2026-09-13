#!/usr/bin/env bash
# Re-target an ALREADY-CONVERTED graph to another Hexagon revision / SoC.
#
# A conversion is three steps: qnn-onnx-converter (quantisation, the slow part -- 3 h for
# MMDiT stage 2), qnn-model-lib-generator, and qnn-context-binary-generator. Only the
# last one is SoC-specific. The model library from step 2 is a plain x86_64 .so that
# carries the quantised graph and its encodings, so porting to a new chip reuses it and
# re-runs only graph preparation -- minutes, not hours.
#
#   HTP_ARCH=v75 SOC_MODEL=57 ./retarget_ctx.sh mmdit_s2fs      # 8 Gen 3 / SM8650
#   HTP_ARCH=v73 SOC_MODEL=43 ./retarget_ctx.sh vaedecsn        # 8 Gen 2 / SM8550
#
# It needs $HOME/neodragon-build/<name>/lib/x86_64-linux-clang/lib<name>.so, which every
# convert_*.sh leaves behind. If you do not have it, run the module's convert_*.sh with
# HTP_ARCH/SOC_MODEL set instead -- that does all three steps for the new target.
#
# SOC_MODEL values are the QNN_SOC_MODEL_* enum in $QNN_SDK_ROOT/include/QNN/QnnTypes.h.
# See docs/porting-other-socs.md.
set -uo pipefail
HTP_ARCH=${HTP_ARCH:?set HTP_ARCH, e.g. v75}
SOC_MODEL=${SOC_MODEL:?set SOC_MODEL, e.g. 57 for SM8650 -- see QnnTypes.h}
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
NAME=${1:?usage: HTP_ARCH=v75 SOC_MODEL=57 $0 <graph name>}
OUT=${OUT:-$HOME/neodragon-build/$NAME}
SO=$OUT/lib/x86_64-linux-clang/lib$NAME.so
[ -f "$SO" ] || { echo "ERROR: no $SO -- run the module's convert_*.sh with HTP_ARCH set"; exit 1; }

# Same graph-prep options as every shipping v79 build, only the device changes.
CFG=$OUT/htp_config_${NAME}_${HTP_ARCH}.json
BE=$OUT/htp_backend_${NAME}_${HTP_ARCH}.json
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

BIN=$OUT/ctx/${NAME}_${HTP_ARCH}.bin
rm -f "$BIN"   # a partial binary from a killed run must not pass for a good one
echo "########## context-binary-generator: $NAME -> $HTP_ARCH / soc_model $SOC_MODEL ##########"
date '+%H:%M:%S'
qnn-context-binary-generator --model "$SO" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -20
rc=${PIPESTATUS[0]}
echo "exit $rc"; date '+%H:%M:%S'
[ "$rc" -eq 0 ] || exit "$rc"

cp "$BIN" "$ND/work/device/" && echo "-> work/device/${NAME}_${HTP_ARCH}.bin ($(stat -c %s "$BIN") bytes)"
