"""Locate the 18 MMDiT block boundaries in a converted graph, by topology.

Why this is needed: `qnn-onnx-converter` flattens all scope information, so a block's
residual output is called `_Add_97_output_0`, not `blocks.8.residual`. Node numbering
also shifts whenever the exported graph changes, so the names cannot be hardcoded --
they have to be re-derived per build.

The derivation uses no names at all:

  * a block-boundary tensor is produced by an Eltwise_Binary (the residual add)
  * it is consumed by a LayerNorm (the next block's AdaLayerNorm) AND by another
    Eltwise_Binary (the next residual add)
  * it has the stream's full shape -- [1, n_img, C] or [1, n_text, C]

That yields exactly two per block (post-attention and post-FFN) and, in a correct
graph, a perfectly periodic index pattern. The period is measured, not assumed, and the
script FAILS LOUDLY if the pattern is irregular -- an irregular pattern means the
heuristic has mismatched the graph and every downstream number would be garbage.

  usage: py -3.10 work/device/find_block_io.py <net.json> [--emit]

`--emit` prints the comma-separated list for qnn-net-run --set_output_tensors.
"""

import json
import re
import sys
from collections import defaultdict


def idx(name):
    m = re.search(r"_(\d+)_output", name)
    return int(m.group(1)) if m else -1


def boundaries(net_path):
    g = json.load(open(net_path))["graph"]
    tensors, nodes = g["tensors"], g["nodes"]

    prod, cons = {}, defaultdict(list)
    for k, v in nodes.items():
        for o in v.get("output_names", []):
            prod[o] = v["type"]
        for i in v.get("input_names", []):
            cons[i].append(v["type"])

    # Find every candidate first, then let the shapes fall out of the result. Picking
    # the "two most common rank-3 shapes" up front does NOT work: [1,1,1536] (the
    # pooled timestep embedding, broadcast per block) is more common than the text
    # stream, and silently displaces it.
    by_shape = defaultdict(list)
    for name, t in tensors.items():
        d = tuple(t["dims"])
        if len(d) != 3 or d[0] != 1 or d[1] < 2:
            continue
        if prod.get(name) != "Eltwise_Binary":
            continue
        if "LayerNorm" not in set(cons.get(name, [])):
            continue
        by_shape[d].append(name)

    # a real residual stream has one boundary pair per block, so >= 2 per block over
    # 18 blocks. Anything with a handful of hits is incidental.
    streams = sorted((d for d, v in by_shape.items() if len(v) >= 10),
                     key=lambda d: -d[1])        # image (more tokens) first
    assert streams, "no residual streams found -- heuristic mismatched this graph"

    out = {d: sorted(by_shape[d], key=idx) for d in streams}
    return out, streams


def main():
    net = sys.argv[1]
    emit = "--emit" in sys.argv
    b, streams = boundaries(net)

    picked = []
    for shape in streams:
        hits = b[shape]
        ids = [idx(h) for h in hits]
        # two boundaries per block: post-attention then post-FFN. The block OUTPUT --
        # the tensor the next block consumes -- is the second of each pair.
        assert len(hits) % 2 == 0, (
            "odd boundary count {} for {} -- expected one post-attention and one "
            "post-FFN add per block".format(len(hits), list(shape)))
        pairs = list(zip(hits[0::2], hits[1::2]))
        deltas = {ids[i + 2] - ids[i] for i in range(len(ids) - 2)}
        if not emit:
            print("stream {}: {} boundary tensors -> {} blocks".format(
                list(shape), len(hits), len(pairs)))
            print("   period between blocks: {}".format(
                sorted(deltas) if deltas else "n/a"))
            print("   first block: post-attn {}  post-ffn {}".format(*pairs[0]))
            print("   last  block: post-attn {}  post-ffn {}".format(*pairs[-1]))
        assert len(deltas) == 1, (
            "IRREGULAR block pattern for {}: deltas {} -- the topology heuristic has "
            "mismatched this graph, do NOT trust the dump".format(list(shape), sorted(deltas)))
        picked += [p[1] for p in pairs]          # post-FFN == the block output

    if emit:
        print(",".join(picked))
    else:
        print("")
        print("block outputs to dump: {}".format(len(picked)))
        print("run with --emit to get the --set_output_tensors list")


if __name__ == "__main__":
    main()
