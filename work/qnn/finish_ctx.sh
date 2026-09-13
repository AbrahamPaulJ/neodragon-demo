#!/usr/bin/env bash
# Re-run ONLY step 3/3 (context-binary-generator) for an already-converted stage.
#
# Steps 1/3 (35 min) and 2/3 (6.5 min) write $OUT/mmdit_sN.cpp/.bin and
# lib/x86_64-linux-clang/libmmdit_sN.so. Step 3 consumes only the .so, so if the
# pipeline dies during it -- as it did on 2026-08-21 -- there is no reason to redo
# the conversion. Verify the .so is intact, then finish.
#
#   ./finish_ctx.sh 0
set -uo pipefail
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

ND=${ND:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
HTP_ARCH=${HTP_ARCH:-v79}
STAGE=${1:-0}
NAME=mmdit_s${STAGE}
OUT=$HOME/neodragon-build/$NAME
SO=$OUT/lib/x86_64-linux-clang/lib$NAME.so
BE=$ND/work/qnn/htp_backend_${NAME}.json

[ -f "$SO" ] || { echo "ERROR: no $SO -- steps 1/2 did not complete"; exit 1; }
[ -f "$BE" ] || { echo "ERROR: no backend config $BE"; exit 1; }
echo "model:   $SO ($(stat -c %s "$SO") bytes, $(date -r "$SO" '+%H:%M:%S'))"
echo "backend: $BE"
cat "$BE"

# a partially-written binary from a killed run must not be mistaken for a good one
if [ -f "$OUT/ctx/${NAME}_${HTP_ARCH}.bin" ]; then
  echo "removing stale/partial $OUT/ctx/${NAME}_${HTP_ARCH}.bin ($(stat -c %s "$OUT/ctx/${NAME}_${HTP_ARCH}.bin") bytes)"
  rm -f "$OUT/ctx/${NAME}_${HTP_ARCH}.bin"
fi

echo; echo "########## 3/3 context-binary-generator ##########"; date '+%H:%M:%S'
qnn-context-binary-generator --model "$SO" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --output_dir "$OUT/ctx" --binary_file "${NAME}_${HTP_ARCH}" \
    --config_file "$BE" 2>&1 | grep -viE '^\s*\[[# ]*\]' | tail -20
rc=${PIPESTATUS[0]}
echo "exit $rc"; date '+%H:%M:%S'
[ "$rc" -eq 0 ] || exit "$rc"

ls -la "$OUT/ctx"
cp "$OUT/ctx/${NAME}_${HTP_ARCH}.bin" "$ND/work/device/" && echo "-> work/device/${NAME}_${HTP_ARCH}.bin"
cp "$OUT/${NAME}_net.json" "$ND/work/device/" 2>/dev/null && echo "-> work/device/${NAME}_net.json"
