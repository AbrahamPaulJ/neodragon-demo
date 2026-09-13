#!/usr/bin/env bash
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
OUT=$HOME/neodragon-build/clipg
cd "$OUT" || exit 1          # trap #30: scratch tree lands under CWD; keep it off 9p
echo "cwd=$(pwd)  free=$(df -h . | tail -1 | awk '{print $4}')"
qnn-model-lib-generator -c "$OUT/clipg.cpp" -b "$OUT/clipg.bin" \
    -t x86_64-linux-clang -o "$OUT/lib" 2>&1 | tail -40
echo "exit ${PIPESTATUS[0]}"
ls -la "$OUT/lib/x86_64-linux-clang/" 2>/dev/null
