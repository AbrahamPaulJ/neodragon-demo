#!/usr/bin/env bash
# Layerwise intermediate dump WITHOUT the device.
#
# Why not on device: qnn-net-run's --debug and --set_output_tensors both refuse to work
# with --retrieve_context ("can only be used with --model option or --dlc_path"). The
# shipping artefact IS a context binary, so neither is available on the phone without
# first building an aarch64-android model lib (needs the NDK + a 1.5 GB push).
#
# But step 2/3 of the conversion already built the model as an x86 shared library, and
# the quantisation encodings live in the model, not the backend. So the same graph can
# be run on the host CPU backend with --debug and every intermediate dumped. That
# isolates QUANTISATION error, which is the question -- HTP-specific numerics are a
# separate, second-order concern.
#
#   ./dump_blocks_cpu.sh 0
set -uo pipefail
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAGE=${1:-0}
NAME=mmdit_s${STAGE}
OUT=$HOME/neodragon-build/$NAME
SO=$OUT/lib/x86_64-linux-clang/lib$NAME.so
DUMP=$OUT/dbg_s${STAGE}

[ -f "$SO" ] || { echo "ERROR: no $SO"; exit 1; }

# one input case only -- --debug writes ~4 GB per inference
LIST=$OUT/one_input.txt
head -1 "$ND/work/device/mio_s${STAGE}/input_list.txt" \
  | sed "s#/data/local/tmp/nd/mio_s${STAGE}#$ND/work/device/mio_s${STAGE}#g" > "$LIST"
echo "input list:"; tr ' ' '\n' < "$LIST" | head -3; echo "  ..."

echo; echo "running on CPU backend with --debug (slow: 1.5 GB graph, one inference)"
date '+%H:%M:%S'
qnn-net-run --model "$SO" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so" \
    --input_list "$LIST" --output_dir "$DUMP" --debug 2>&1 | tail -8
echo "exit ${PIPESTATUS[0]}"; date '+%H:%M:%S'

echo; echo "dumped: $(find "$DUMP" -name '*.raw' 2>/dev/null | wc -l) tensors, $(du -sh "$DUMP" 2>/dev/null | cut -f1)"
