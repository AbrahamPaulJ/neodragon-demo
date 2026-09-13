#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
echo "=== venv python ==="
python --version
python -c "
import importlib
for m in ('numpy','onnx','onnxruntime','torch'):
    try:
        mod=importlib.import_module(m); print(' ',m,getattr(mod,'__version__','?'))
    except Exception: print(' ',m,'MISSING')
"
echo
echo "=== converter reachable? ==="
which qnn-onnx-converter qnn-model-lib-generator qnn-context-binary-generator
echo
echo "=== hexagon v79 libs ==="
ls "$QNN_SDK_ROOT/lib/hexagon-v79/unsigned" 2>/dev/null | head
echo
echo "=== soc_model / dsp_arch reference in SDK ==="
grep -rhoE '"?soc_model"?[^0-9]{0,12}[0-9]+' "$QNN_SDK_ROOT/docs" 2>/dev/null | sort -u | head -20
echo "--- SM8750 mentions ---"
grep -rliE 'SM8750|8 Elite' "$QNN_SDK_ROOT/docs" 2>/dev/null | head -5
