#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
B=$HOME/neodragon-build/vaedec1/ctx/vaedec1_v79.bin
qnn-context-binary-utility --context_binary "$B" --json_file /tmp/vctx.json >/dev/null 2>&1
python - <<'PY'
import json
d = json.load(open("/tmp/vctx.json"))
g = d.get("info", {}).get("graphs") or d.get("graphs")
for gr in g:
    gi = gr.get("info", gr)
    print("graph:", gi.get("graphName"))
    for io in ("graphInputs", "graphOutputs"):
        for t in gi.get(io, []):
            ti = t.get("info", t)
            print(f"  {io[5:-1]:<7} {ti.get('name'):<10} dims={ti.get('dimensions')} "
                  f"dtype={ti.get('dataType')}")
PY
