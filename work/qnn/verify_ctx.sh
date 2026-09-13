#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
OUT=$HOME/neodragon-build
BIN=$OUT/ctx/distilt5_v79.bin

echo "=== artefacts ==="
ls -la "$OUT"/distilt5.cpp "$OUT"/distilt5.bin "$BIN"
find "$OUT/lib" -name '*.so' -exec ls -la {} \;

echo
echo "=== context binary metadata ==="
qnn-context-binary-utility --context_binary "$BIN" --json_file /tmp/ctx.json >/dev/null 2>&1
python - <<'PY'
import json
d = json.load(open("/tmp/ctx.json"))
def walk(o, depth=0):
    pass
info = d.get("info", d)
print("top-level keys:", list(d.keys()))
graphs = json.dumps(d)
# pull the interesting bits without guessing the schema
for key in ("backendBinaryVersion", "contextBlobVersion", "coreApiVersion"):
    if key in str(d):
        pass
g = d.get("info", {}).get("graphs") or d.get("graphs")
if g:
    for gr in g:
        gi = gr.get("info", gr)
        print(f"\ngraph: {gi.get('graphName')}")
        for io in ("graphInputs", "graphOutputs"):
            for t in gi.get(io, []):
                ti = t.get("info", t)
                print(f"  {io[5:-1]:<7} {ti.get('name'):<20} "
                      f"dims={ti.get('dimensions')} "
                      f"dtype={ti.get('dataType')}")
PY

echo
echo "=== htp arch actually targeted ==="
grep -aoE 'v(6[0-9]|7[0-9]|8[0-9])' "$BIN" | sort | uniq -c | sort -rn | head -5
