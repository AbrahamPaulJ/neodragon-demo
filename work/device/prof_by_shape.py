#!/usr/bin/env python3
"""Break one op TYPE down by output SHAPE, from a detailed profile.

`prof_by_optype.py` says Eltwise_Binary is 29.9% of stage 2 and Softmax 27.2%. That is
where the time is, but not what to do about it: 543 Eltwise ops are not one problem, and
knowing which SHAPES carry the cycles is what turns a share into a target.

Also prints cycles-per-byte per shape, which is the number that actually transfers between
ops -- the session-7 mistake was pricing a graph change with one aggregate ms/MB when op
types differ by 20x on the identical tensor.

  usage: py -3.10 work/device/prof_by_shape.py <prof.csv> <net.json> [op_type ...]
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


def load_profile(csv_path):
    execs, cur = [], None
    for line in open(csv_path, encoding="utf-8", errors="replace"):
        p = line.split(",")
        if len(p) < 7:
            continue
        msg, val, unit, lvl, ident = p[1], p[2], p[3], p[5], ",".join(p[6:]).strip()
        if msg != "EXECUTE":
            continue
        try:
            v = int(val)
        except ValueError:
            continue
        if lvl == "ROOT" and "Accelerator (execute) time (cycles)" in ident:
            if cur:
                execs.append(cur)
            cur = {}
        elif lvl == "SUB-EVENT" and unit == "CYCLES" and cur is not None:
            k = ident[:-len(" (cycles)")] if ident.endswith(" (cycles)") else ident
            cur[k] = cur.get(k, 0) + v
    if cur:
        execs.append(cur)
    return execs[-1] if execs else {}


def main():
    csv_path, net_path = sys.argv[1], sys.argv[2]
    want = set(sys.argv[3:]) or None

    g = json.load(open(net_path))["graph"]
    nodes, tensors = g["nodes"], g["tensors"]
    prof = load_profile(csv_path)

    info = {}
    for name, n in nodes.items():
        shape, nbytes = (), 0
        if n.get("output_names") and n["output_names"][0] in tensors:
            t = tensors[n["output_names"][0]]
            shape = tuple(t["dims"])
            nbytes = nelem(shape) * dtype_bits(t["data_type"]) // 8
        rec = (n["type"], shape, nbytes)
        info[name] = rec
        for o in n.get("output_names", []):
            info.setdefault(o, rec)

    # (type, shape) -> [cycles, bytes, count]
    agg = collections.defaultdict(lambda: [0, 0, 0])
    total = 0
    for ident, cyc in prof.items():
        rec = info.get(ident.split(":OpId_")[0].strip())
        if rec is None:
            continue
        t, shape, nbytes = rec
        total += cyc
        if want and t not in want:
            continue
        a = agg[(t, shape)]
        a[0] += cyc
        a[1] += nbytes
        a[2] += 1

    print(f"total graph cycles: {total:,}\n")
    hdr = f"{'op type':<18}{'n':>5}{'cycles':>15}{'share':>8}{'MB':>9}{'cyc/byte':>10}  shape"
    print(hdr)
    print("-" * (len(hdr) + 18))
    for (t, shape), (c, b, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:24]:
        cpb = c / b if b else float("nan")
        print(f"{t:<18}{n:>5}{c:>15,}{100*c/total:>7.1f}%"
              f"{b/1e6:>9.1f}{cpb:>10.4f}  {shape}")


if __name__ == "__main__":
    main()
