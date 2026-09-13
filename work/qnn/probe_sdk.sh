#!/usr/bin/env bash
S=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT SDK root}
echo "=== bin/ ==="
ls "$S/bin"
echo
echo "=== bin/x86_64-linux-clang/ ==="
ls "$S/bin/x86_64-linux-clang" 2>/dev/null | head -40
echo
echo "=== lib/ targets ==="
ls "$S/lib"
echo
echo "=== python deps expected by converter ==="
ls "$S/lib/python" 2>/dev/null | head -20
echo
echo "=== check-python-dependency / env setup ==="
ls "$S/bin/check-python-dependency" "$S/bin/envsetup.sh" "$S/bin/aimet_env_setup.sh" 2>/dev/null
echo
echo "=== host python ==="
python3 --version
python3 -c "import onnx, numpy; print('onnx', onnx.__version__, 'numpy', numpy.__version__)" 2>&1 | tail -1
