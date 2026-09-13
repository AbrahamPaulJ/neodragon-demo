# Copy to work/qnn/qnn_env.sh (gitignored) and edit, or point $QNN_ENV at your own copy.
# Every conversion script sources it. Runs under WSL.

# QAIRT SDK 2.49.0.260730, e.g. unpacked on the Windows side and reached through /mnt/c
export QNN_SDK_ROOT=/mnt/c/path/to/qairt/2.49.0.260730

# A Python 3.10 venv with numpy 1.26.4 (the SDK's tested pin) and the packages
# `$QNN_SDK_ROOT/bin/check-python-dependency` asks for. work/qnn/setup_env.sh builds one.
export VENV="$HOME/neodragon-qnn"

# libPyIrGraph310 needs libpython3.10.so.1.0 on the loader path. With a uv / standalone
# CPython the venv's interpreter lives next to it; with a distro python3.10 this is harmless.
PYLIB="$(dirname "$(dirname "$(readlink -f "$VENV/bin/python")")")/lib"

export PYTHONPATH="$QNN_SDK_ROOT/lib/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$PYLIB:${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$QNN_SDK_ROOT/bin/x86_64-linux-clang:$HOME/.local/bin:$PATH"

# Android NDK (Linux build) -- only for qnn-model-lib-generator -t aarch64-android
export ANDROID_NDK_ROOT="$HOME/ndk/android-ndk-r26d"

# Target SoC. Defaults in the scripts are the S25 Ultra (SM8750, HTP v79).
# export HTP_ARCH=v75 SOC_MODEL=57     # Snapdragon 8 Gen 3 / SM8650
