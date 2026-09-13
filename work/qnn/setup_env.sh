#!/usr/bin/env bash
# Stand up the QAIRT converter environment in WSL without sudo.
#
# Ubuntu 24.04 ships python3.12 with neither pip nor ensurepip, and the SDK's
# check-python-dependency refuses to run outside a virtualenv. So: create a
# venv --without-pip, bootstrap pip into it from get-pip.py, then let the SDK
# installer populate it. Everything lives in $VENV and can be deleted whole.
set -u
S=${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your QAIRT SDK root}
VENV=$HOME/neodragon-qnn

if [ ! -x "$VENV/bin/pip" ]; then
  echo "=== creating venv (--without-pip) ==="
  rm -rf "$VENV"
  python3 -m venv --without-pip "$VENV" || exit 1

  echo "=== bootstrapping pip ==="
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py || exit 1
  "$VENV/bin/python" /tmp/get-pip.py --quiet || exit 1
fi

echo "=== pip ==="
"$VENV/bin/pip" --version

echo
echo "=== SDK dependency installer ==="
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python "$S/bin/check-python-dependency" 2>&1 | tail -25

echo
echo "=== versions ==="
python - <<'PY'
import importlib, sys
print("python", sys.version.split()[0])
for m in ("numpy", "onnx", "onnxruntime", "yaml", "packaging"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m} MISSING ({type(e).__name__})")
PY
