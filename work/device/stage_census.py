#!/usr/bin/env python3
"""Static per-stage census of the three MMDiT graphs -- no device needed.

Written 2026-08-23 (session 7) to answer ONE question before spending any device
time on stage 2:

    stage 2 costs 1851 ms against stage 0's 169.2 ms -- 10.94x the latency for
    4.24x the tokens. Is that DISPATCH (op count) or ARITHMETIC (MACs)?

The roadmap asserted dispatch, on the strength of stage 0's profile ("2378 ops at
~129K cycles each") plus a node count for stage 2. But op count and MAC count
scale completely differently with sequence length: the graph structure is
identical across stages -- same 18 blocks, same ops -- while attention is O(n^2).
So the node count CANNOT distinguish them and the question was still open.

This counts MACs exactly, from the operand shapes the converter recorded, and
attributes them per op type. If MACs scale like the measured latency, the cost is
arithmetic and op-count work is pointless.

  usage: py -3.10 work/device/stage_census.py \
             work/device/mmdit_s0g_net.json \
             work/device/mmdit_s1f_net.json \
             work/device/mmdit_s2f_net.json
"""
import collections
import json
import sys

CATEGORY = {0x00: "INT", 0x01: "UINT", 0x02: "FLOAT", 0x03: "SFIXED",
            0x04: "UFIXED", 0x05: "BOOL", 0x06: "STRING"}


def dtype_bits(code):
    # 0x0416 is UFIXED_16: the low byte spells the width in DECIMAL (trap #12).
    return int(f"{code & 0xFF:02x}")


def nelem(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def tbytes(tensors, name):
    t = tensors[name]
    return nelem(t["dims"]) * dtype_bits(t["data_type"]) // 8


def macs(node, tensors):
    """MACs for one node, from its recorded operand shapes.

    MatMul  [.., M, K] x [.., K, N] -> batch * M * N * K
    FullyConnected  in [.., K], weight [N, K] -> (elems_in / K) * N * K
    Conv2d  out [N,H,W,Co] with weight [kh,kw,Ci,Co] -> N*H*W*Co*Ci*kh*kw
    Everything else is not arithmetic in the MAC sense.
    """
    t = node["type"]
    ins = node["input_names"]
    outs = node["output_names"]
    if not ins or not outs or outs[0] not in tensors:
        return 0
    out = tensors[outs[0]]["dims"]

    if t == "MatMul" and len(ins) >= 2 and ins[0] in tensors and ins[1] in tensors:
        a = tensors[ins[0]]["dims"]
        if len(a) >= 1:
            k = a[-1]
            return nelem(out) * k

    if t == "FullyConnected" and len(ins) >= 2 and ins[1] in tensors:
        w = tensors[ins[1]]["dims"]
        if len(w) >= 2:
            # weight is [N, K]; every output element costs K MACs
            return nelem(out) * w[-1]

    if t in ("Conv2d", "DepthWiseConv2d") and len(ins) >= 2 and ins[1] in tensors:
        w = tensors[ins[1]]["dims"]          # [kh, kw, Ci, Co]
        if len(w) == 4:
            return nelem(out) * w[0] * w[1] * w[2]

    return 0


def census(path):
    graph = json.load(open(path))["graph"]
    nodes, tensors = graph["nodes"], graph["tensors"]

    by_type = collections.Counter()
    mac_by_type = collections.Counter()
    out_bytes = collections.Counter()
    for n in nodes.values():
        t = n["type"]
        by_type[t] += 1
        mac_by_type[t] += macs(n, tensors)
        if n["output_names"] and n["output_names"][0] in tensors:
            out_bytes[t] += tbytes(tensors, n["output_names"][0])

    # The attention score matrix is the tensor that is quadratic in sequence
    # length; find the largest one to report it explicitly.
    biggest = max(((nelem(t["dims"]), name, tuple(t["dims"]),
                    dtype_bits(t["data_type"]))
                   for name, t in tensors.items()), default=(0, "", (), 0))

    return dict(path=path, nodes=len(nodes), tensors=len(tensors),
                by_type=by_type, mac_by_type=mac_by_type,
                out_bytes=out_bytes, biggest=biggest,
                total_macs=sum(mac_by_type.values()),
                total_out_bytes=sum(out_bytes.values()))


def main(paths):
    cs = [census(p) for p in paths]

    print("\n=== totals ===")
    hdr = f"{'stage':<26}{'nodes':>8}{'GMAC':>12}{'out MB':>12}{'largest tensor':>34}"
    print(hdr)
    print("-" * len(hdr))
    for c in cs:
        big = f"{c['biggest'][2]} @{c['biggest'][3]}b"
        print(f"{c['path'].split('/')[-1]:<26}{c['nodes']:>8}"
              f"{c['total_macs']/1e9:>12.2f}{c['total_out_bytes']/1e6:>12.1f}"
              f"{big:>34}")

    print("\n=== scaling vs first stage ===")
    base = cs[0]
    print(f"{'stage':<26}{'nodes x':>10}{'GMAC x':>10}{'outMB x':>10}")
    for c in cs:
        print(f"{c['path'].split('/')[-1]:<26}"
              f"{c['nodes']/base['nodes']:>10.2f}"
              f"{c['total_macs']/base['total_macs']:>10.2f}"
              f"{c['total_out_bytes']/base['total_out_bytes']:>10.2f}")

    print("\n=== GMAC by op type ===")
    types = sorted({t for c in cs for t in c["mac_by_type"] if c["mac_by_type"][t]})
    print(f"{'op':<20}" + "".join(f"{c['path'].split('/')[-1][6:9]:>14}" for c in cs))
    for t in types:
        print(f"{t:<20}" + "".join(f"{c['mac_by_type'][t]/1e9:>14.2f}" for c in cs))

    print("\n=== op count by type ===")
    types = sorted({t for c in cs for t in c["by_type"]},
                   key=lambda t: -max(c["by_type"][t] for c in cs))
    print(f"{'op':<20}" + "".join(f"{c['path'].split('/')[-1][6:9]:>8}" for c in cs))
    for t in types:
        print(f"{t:<20}" + "".join(f"{c['by_type'][t]:>8}" for c in cs))

    print("\n=== output bytes by op type (MB) ===")
    types = sorted({t for c in cs for t in c["out_bytes"]},
                   key=lambda t: -max(c["out_bytes"][t] for c in cs))
    print(f"{'op':<20}" + "".join(f"{c['path'].split('/')[-1][6:9]:>10}" for c in cs))
    for t in types[:14]:
        print(f"{t:<20}" + "".join(f"{c['out_bytes'][t]/1e6:>10.1f}" for c in cs))


if __name__ == "__main__":
    main(sys.argv[1:])
