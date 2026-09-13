#!/usr/bin/env bash
# aarch64-android model library for ANY converted module, so its intermediates can be
# dumped. `build_android_lib.sh` is MMDiT-specific; this takes the build name.
#
# qnn-net-run refuses --debug and --set_output_tensors against a context binary, so an
# online-prepared model library is the only route to intermediate values.
#
#   wsl -d Ubuntu -e bash work/qnn/build_android_lib_any.sh ctxadaptw16
set -uo pipefail
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"

NAME=${1:?usage: build_android_lib_any.sh <build-name>}
OUT=$HOME/neodragon-build/$NAME
NDK=${ANDROID_NDK_ROOT:-$HOME/ndk/android-ndk-r26d}
export ANDROID_NDK_ROOT=$NDK
export PATH="$NDK:$PATH"

command -v ndk-build >/dev/null || { echo "ERROR: ndk-build not on PATH ($NDK)"; exit 1; }
[ -f "$OUT/$NAME.cpp" ] || { echo "ERROR: no $OUT/$NAME.cpp"; exit 1; }

# trap #30: lib-gen builds its scratch tree under the CWD. Keep it off the 9p mount.
cd "$OUT" || exit 1
qnn-model-lib-generator -c "$OUT/$NAME.cpp" -b "$OUT/$NAME.bin" \
    -t aarch64-android -o "$OUT/lib" 2>&1 | tail -3
ls -la "$OUT/lib/aarch64-android/"
