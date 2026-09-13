#!/usr/bin/env python3
"""Write a converter `dims.txt` straight from an ONNX graph's own inputs.

`convert_mmdit_w8a16.sh` needs `$CALIB/dims.txt` -- a pre-formatted `--input_dim` block
-- even in `--layout` mode, where no calibration data is used at all. Normally
`make_mmdit_io.py` emits it as a side effect of building the calibration set, so a
layout-only run of a NEW graph variant is blocked on GPU work it does not need.

The input shapes are a property of the exported graph, so read them from the graph.
That is also safer than copying a sibling stage's file: if a rewrite ever does change an
input, this notices and a hand-copied dims.txt would silently lie.

Reads only the ONNX *proto* (`load_external_data=False`), so it does not touch the
multi-GB weights blob and runs in well under a second.

  usage: py -3.10 work/device/dims_from_onnx.py <model.onnx> <out_dir>
"""
import sys
from pathlib import Path

import onnx


def main():
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    m = onnx.load(str(src), load_external_data=False)

    # Initializers also appear in graph.input on some exporters; keep only real inputs.
    init = {t.name for t in m.graph.initializer}

    dims = []
    for vi in m.graph.input:
        if vi.name in init:
            continue
        shape = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(str(d.dim_value))
            else:
                raise SystemExit(
                    "input {!r} has a dynamic axis ({!r}); these graphs are static "
                    "by construction, so this means the export is wrong".format(
                        vi.name, d.dim_param or "?"))
        dims.append("--input_dim {} {}".format(vi.name, ",".join(shape)))

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "dims.txt"
    # The converter consumes this with `tr -d '\\\n'`, so the exact line continuation
    # and indent are what the existing files use. LF only -- CRLF has broken the
    # toolchain twice (HANDOFF, "Environment reminders").
    dst.write_text(" \\\n    ".join(dims) + "\n", newline="\n")
    print("wrote {} ({} inputs)".format(dst, len(dims)))
    for d in dims:
        print("  " + d)


if __name__ == "__main__":
    main()
