#!/usr/bin/env bash
S=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT SDK root}
echo "=== python ==="
python3 --version
echo "=== pip? ==="
python3 -m pip --version 2>&1 | head -2
echo "=== ensurepip? ==="
python3 -m ensurepip --version 2>&1 | head -2
echo "=== venv module? ==="
python3 -c "import venv; print('venv module present')" 2>&1 | head -2
echo "=== already-installed relevant pkgs ==="
python3 -c "
import importlib
for m in ['numpy','onnx','onnxruntime','yaml','packaging','google.protobuf']:
    try:
        mod=importlib.import_module(m); print(' ',m,getattr(mod,'__version__','?'))
    except Exception: print(' ',m,'MISSING')
"
echo "=== SDK python lib ==="
ls "$S/lib/python" 2>/dev/null
echo "=== converter shebang ==="
head -3 "$S/bin/x86_64-linux-clang/qnn-onnx-converter"
echo "=== dep check (with python3) ==="
python3 "$S/bin/check-python-dependency" 2>&1 | tail -25
