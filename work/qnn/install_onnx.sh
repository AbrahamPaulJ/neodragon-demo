#!/usr/bin/env bash
set -u
S=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT SDK root}
VENV=$HOME/neodragon-qnn
source "$VENV/bin/activate"

echo "=== what the SDK pins ==="
python "$S/bin/check-python-dependency" 2>&1 | head -30

echo
echo "=== installing onnx / onnxruntime ==="
pip install --quiet onnx onnxruntime 2>&1 | tail -5

echo
echo "=== versions ==="
python -c "
import numpy, onnx, onnxruntime
print('numpy', numpy.__version__)
print('onnx', onnx.__version__)
print('onnxruntime', onnxruntime.__version__)
"
