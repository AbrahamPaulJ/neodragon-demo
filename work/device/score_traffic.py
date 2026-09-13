#!/usr/bin/env python3
"""Count how many times a graph materialises the attention SCORE matrix.

This is the one number the session-7 fix is about (trap #40). Stage 2's score is
`[24,1728,1728]` -- 143.3 MB at 16 bits, one tensor -- and the shipping graph writes it
FIVE times per block:

    MatMul(q_even,k_even) -> MatMul(q_odd,k_odd) -> Add(the two) -> Add(mask) -> Softmax

The fix contracts once at full width, which should leave THREE:

    MatMul(q,k) -> Add(mask) -> Softmax

Comparing total output bytes between builds does not work directly here, because a
`--layout` build is fp32 and a shipping build is A16, so every activation differs by 2x
for reasons unrelated to the change. Counting score-shaped tensors is dtype-independent
and answers the question exactly.

  usage: py -3.10 work/device/score_traffic.py <net.json> [<net.json> ...]
"""
import collections
import json
import sys


def dtype_bits(code):
    return int(f"{code & 0xFF:02x}")          # trap #12


def nelem(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def report(path):
    g = json.load(open(path))["graph"]
    nodes, tensors = g["nodes"], g["tensors"]

    # The score matrix is the tensor whose last two dims are equal and large -- the only
    # square-in-the-last-two-axes activation in the graph. Find its shape by picking the
    # largest such tensor, then count every node that produces one of that shape.
    sq = [(nelem(t["dims"]), tuple(t["dims"]))
          for t in tensors.values()
          if len(t["dims"]) >= 2 and t["dims"][-1] == t["dims"][-2] and t["dims"][-1] > 16]
    if not sq:
        print(f"{path}: no square activation found")
        return
    _, shape = max(sq)

    per_type = collections.Counter()
    total_bytes = 0
    for n in nodes.values():
        for o in n.get("output_names", []):
            t = tensors.get(o)
            if t and tuple(t["dims"]) == shape:
                per_type[n["type"]] += 1
                total_bytes += nelem(t["dims"]) * dtype_bits(t["data_type"]) // 8

    n_score = sum(per_type.values())
    bits = max(dtype_bits(t["data_type"]) for t in tensors.values()
               if tuple(t["dims"]) == shape)
    one = nelem(shape) * bits // 8

    # A16-normalised bytes, so a fp32 --layout build can be compared against a shipping
    # A16 build. Without this the raw totals differ by 2x for reasons that have nothing
    # to do with the graph, and the comparison reads backwards: the session-7 rewrite
    # first showed as "1.200x score traffic" when it had in fact cut it by 40%.
    norm = n_score * (nelem(shape) * 16 // 8)

    print(f"\n===== {path} =====")
    print(f"score shape {shape} @ {bits}b = {one/1e6:.1f} MB per materialisation")
    print(f"materialisations: {n_score}  ({n_score/18:.2f} per block over 18 blocks)")
    for t, c in per_type.most_common():
        print(f"    {t:<18} {c:>4}   ({c/18:.2f}/block)")
    print(f"total score traffic: {total_bytes/1e6:.1f} MB "
          f"({norm/1e6:.1f} MB normalised to 16-bit)")
    return n_score, norm


def main(paths):
    totals = [(p, report(p)) for p in paths]
    if len(totals) == 2 and all(t for _, t in totals):
        (pa, (na, a)), (pb, (nb, b)) = totals
        print(f"\n{pb}\n  vs {pa}")
        print(f"  materialisations : {na} -> {nb}  ({na/18:.2f} -> {nb/18:.2f} per block)")
        print(f"  16-bit-normalised: {a/1e6:.1f} -> {b/1e6:.1f} MB "
              f"= {b/a:.3f}x  ({(b-a)/1e6:+.1f} MB)")
        if bits_differ(pa, pb):
            print("  (raw byte totals are NOT comparable here -- the builds use "
                  "different activation widths; the normalised line is the answer)")


def bits_differ(pa, pb):
    def w(p):
        g = json.load(open(p))["graph"]
        return max(dtype_bits(t["data_type"]) for t in g["tensors"].values())
    return w(pa) != w(pb)


if __name__ == "__main__":
    main(sys.argv[1:])
