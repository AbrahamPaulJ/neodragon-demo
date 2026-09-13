"""Rank a QNN *_net.json's activation encodings by how much dynamic range they burn.

Trap #3's mechanism: with A16 min-max calibration, one rare outlier sets a tensor's
encoding range, and every ordinary value in that tensor then gets fewer effective bits.
In an 18-block residual stream the effect compounds.

This is the free static check for it. For every quantised activation it prints the
encoding range and the implied step, and flags the tensors whose range is wildly out of
line with the graph's median -- those are where percentile / MSE calibration would pay.

  usage: py -3.10 work/device/analyze_encodings.py <net.json> [top_n]
"""

import json
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    g = json.load(open(path))["graph"]
    tensors, nodes = g["tensors"], g["nodes"]

    producer = {}
    for name, n in nodes.items():
        for o in n.get("output_names", []):
            producer[o] = n["type"]

    rows = []
    for name, t in tensors.items():
        q = t.get("quant_params") or {}
        enc = q.get("scale_offset") or {}
        scale = enc.get("scale")
        if not scale:
            continue
        # activations only: type 3 is native/app-writeable in QNN net.json; use the
        # producer map instead -- anything a node emits is an activation.
        if name not in producer:
            continue
        bits = int("{:02x}".format(t["data_type"] & 0xFF))
        rng = scale * (2 ** bits)
        rows.append((rng, scale, name, producer[name], t["dims"], bits))

    rows.sort(reverse=True)
    med = rows[len(rows) // 2][0] if rows else 0.0

    print("=" * 92)
    print("Activation encodings in {}".format(path.split("/")[-1]))
    print("  {} quantised activations, median range {:.4g}".format(len(rows), med))
    print("=" * 92)
    print("")
    print("  {:>12} {:>11} {:>8}  {:<26} {}".format(
        "range", "step", "xmedian", "producer", "tensor"))
    print("  " + "-" * 88)
    for rng, scale, name, ptype, dims, bits in rows[:top]:
        print("  {:>12.4g} {:>11.3e} {:>8.0f}x  {:<26} {}".format(
            rng, scale, rng / med if med else 0, ptype, name[:44]))

    # per-producer-type summary: where does the range live?
    by = defaultdict(list)
    for rng, _, _, ptype, _, _ in rows:
        by[ptype].append(rng)
    print("")
    print("  by producer op type (median range, count):")
    for ptype, v in sorted(by.items(), key=lambda kv: -sorted(kv[1])[len(kv[1]) // 2]):
        v.sort()
        print("    {:<26} {:>12.4g}  n={}".format(ptype, v[len(v) // 2], len(v)))

    over = [r for r in rows if r[0] > 20 * med]
    print("")
    print("  tensors with >20x the median range: {} of {}".format(len(over), len(rows)))
    if over:
        print("  -> these are what percentile / MSE activation calibration targets")


if __name__ == "__main__":
    main()
