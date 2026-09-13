#!/usr/bin/env bash
set -u
source "${QNN_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/work/qnn/qnn_env.sh}"
for N in vaedec1 vaedec1q; do
  B=$HOME/neodragon-build/$N/ctx/${N}_v79.bin
  [ -f "$B" ] || continue
  qnn-context-binary-utility --context_binary "$B" --json_file /tmp/${N}.json >/dev/null 2>&1
  echo "=== $N ==="
  python - "$N" <<'PY'
import json, sys
n = sys.argv[1]
d = json.load(open(f"/tmp/{n}.json"))
g = d.get("info", {}).get("graphs") or d.get("graphs")
for gr in g:
    gi = gr.get("info", gr)
    for io in ("graphInputs", "graphOutputs"):
        for t in gi.get(io, []):
            ti = t.get("info", t)
            enc = ti.get("quantizeParams", {})
            se = enc.get("scaleOffsetEncoding", {}) if isinstance(enc, dict) else {}
            print(f"  {io[5:-1]:<7} {ti.get('name'):<8} dims={ti.get('dimensions')} "
                  f"dtype={ti.get('dataType')}")
            if se:
                print(f"           scale={se.get('scale')} offset={se.get('offset')}")
PY
done
