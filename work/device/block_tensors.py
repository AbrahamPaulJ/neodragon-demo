"""Emit every tensor name inside one MMDiT block, for a targeted --set_output_tensors run.

`--debug` does not work on this graph: forcing all 2374 intermediates to be
client-readable makes HTP's finalize fail outright (`Finalize Graph for Idx = 0 failed
with error = 1002`). 35 tensors finalize fine. So intra-block localisation has to be
done a block at a time, and this picks the names.

A block is delimited by the two residual boundaries `find_block_io.py` derives; every
node output whose topological index falls between them belongs to that block. Names are
emitted in the converter's sanitised form, ready to paste into --set_output_tensors.

  usage: py -3.10 work/device/block_tensors.py --block 0 [--stage 0] [--max 120]
         py -3.10 work/device/block_tensors.py --block 0 --emit > names.txt
"""

import argparse
import re
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
ND = W.parent.parent
sys.path.insert(0, str(W))
from layer_snr import sanitize                       # noqa: E402
from find_block_io import boundaries                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--max", type=int, default=120,
                    help="cap the count; HTP finalize gets unhappy well before 2374")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--name", default="",
                    help="variant, e.g. mmdit_s0f -- picks its net.json and its ONNX")
    a = ap.parse_args()

    import onnx
    suf = (a.name[6:] if a.name.startswith("mmdit_") else a.name) or "s{}".format(a.stage)
    net = W / "mmdit_{}_net.json".format(suf)
    b, streams = boundaries(str(net))
    img = b[streams[0]]                               # image stream, sorted by index
    ends = img[1::2]                                  # post-FFN == block output

    def idx(name):
        m = re.search(r"_(\d+)_output", name)
        return int(m.group(1)) if m else -1

    onnx_path = ND / "work/onnx/mmdit_{0}_env/mmdit_{0}_env_raw.onnx".format(suf)
    model = onnx.load(str(onnx_path), load_external_data=False)
    pos = {}
    for i, node in enumerate(model.graph.node):
        for o in node.output:
            pos.setdefault(sanitize(o), i)

    lo_name = ends[a.block - 1] if a.block > 0 else None
    hi_name = ends[a.block]
    lo = pos[lo_name] if lo_name else -1
    hi = pos[hi_name]
    inside = sorted((s for s, i in pos.items() if lo < i <= hi), key=lambda s: pos[s])

    if not a.emit:
        print("block {}: nodes ({}, {}]  ->  {} tensors".format(
            a.block, lo_name or "<graph start>", hi_name, len(inside)))
        if len(inside) > a.max:
            print("  capped at --max {} (evenly spaced)".format(a.max))
    if len(inside) > a.max:
        step = len(inside) / a.max
        inside = [inside[int(k * step)] for k in range(a.max)]
        if hi_name not in inside:
            inside[-1] = hi_name
    if a.emit:
        print(",".join(inside))
    else:
        for s in inside:
            print("   {:>5}  {}".format(pos[s], s))


if __name__ == "__main__":
    main()
