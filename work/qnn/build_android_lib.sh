#!/usr/bin/env bash
# Build an aarch64-android model .so for one MMDiT stage, so the phone can run the
# graph with ONLINE prep and therefore with --debug / --set_output_tensors.
#
# Why this exists: qnn-net-run refuses --debug and --set_output_tensors against a
# context binary ("can only be used with graph prepared online using --model or
# --dlc_path"), and the host CPU backend rejects the rank-5 latent Transpose. The only
# route to intermediate values is an aarch64-android model library.
#
# Needs ndk-build on PATH -> $HOME/ndk/android-ndk-r26d.
#
#   ./build_android_lib.sh 0
set -uo pipefail
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

STAGE=${1:-0}
# NAME selects the variant, matching convert_mmdit_w8a16.sh:
#   NAME=mmdit_s0f ./build_android_lib.sh 0
NAME=${NAME:-mmdit_s${STAGE}}
OUT=$HOME/neodragon-build/$NAME
NDK=${ANDROID_NDK_ROOT:-$HOME/ndk/android-ndk-r26d}
export ANDROID_NDK_ROOT=$NDK
export PATH="$NDK:$PATH"

command -v ndk-build >/dev/null || { echo "ERROR: ndk-build not on PATH ($NDK)"; exit 1; }
[ -f "$OUT/$NAME.cpp" ] || { echo "ERROR: no $OUT/$NAME.cpp"; exit 1; }
[ -f "$OUT/$NAME.bin" ] || { echo "ERROR: no $OUT/$NAME.bin"; exit 1; }

# trap #30: lib-gen builds its scratch tree under the CWD and explodes the weight
# blob into ~900 .raw files. Peak scratch was 4.2 GB; keep it off the 9p /mnt/c mount.
cd "$OUT" || exit 1

echo "ndk : $(ndk-build --version | head -1) at $NDK"
echo "cpp : $(stat -c %s "$OUT/$NAME.cpp") bytes"
echo "bin : $(stat -c %s "$OUT/$NAME.bin") bytes"
date '+%H:%M:%S  start'

/usr/bin/time -v qnn-model-lib-generator \
    -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t aarch64-android -o "$OUT/lib" 2>&1 | tail -30
echo "exit ${PIPESTATUS[0]}"
date '+%H:%M:%S  done'

ls -la "$OUT/lib/aarch64-android/" 2>/dev/null
