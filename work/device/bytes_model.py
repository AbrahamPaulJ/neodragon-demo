#!/usr/bin/env python3
"""Test whether MMDiT latency is predicted by OUTPUT BYTES -- validation, not a guess.

Session 7 (2026-08-23). The static census showed the three MMDiT stages have
essentially the same node count (1719/1719/1730) but latencies of 169.2/375.8/1851 ms.
So the roadmap's "stage 2 is dispatch-bound" cannot be right -- op count is flat while
latency moves 10.94x.

Across the three stages, latency tracked total op OUTPUT BYTES (1.00/1.96/9.71x)
far better than MACs (1.00/1.63/4.86x). That is only 3 points from one graph family,
which is not enough to design on. This script tests the same claim inside a SINGLE
graph, where there are ~1700 independent points and a measured cycle count for each:

    if bytes-written predicts cost, then cycles/byte should be roughly constant
    across op types -- and where it is NOT, that op type is paying a granularity
    penalty (trap #26) rather than a bandwidth cost.

  usage: py -3.10 work/device/bytes_model.py <prof.csv> <net.json>
"""
import json
import sys
from collections import defaultdict


def dtype_bits(code):
    return int(f"{code & 0xFF:02x}")          # trap #12: decimal in the low byte


def nelem(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def load_profile(csv_path):
    """Return {identifier: cycles} for the last complete EXECUTE event."""
    # Columns: timestamp, Message, Time, Unit, Timing Source, Event Level, Identifier.
    # Per-op rows are Event Level "SUB-EVENT" with Unit "CYCLES", and the identifier
    # reads "<op name>:OpId_<n> (cycles)".
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
    g = json.load(open(net_path))["graph"]
    nodes, tensors = g["nodes"], g["tensors"]

    prof = load_profile(csv_path)
    print(f"profile rows in last EXECUTE: {len(prof)}   nodes in net.json: {len(nodes)}")

    # The profile identifies an op by its own name or by its output tensor name, and
    # suffixes ":OpId_N". Build both lookups.
    info = {}
    for name, n in nodes.items():
        ob = 0
        if n.get("output_names") and n["output_names"][0] in tensors:
            t = tensors[n["output_names"][0]]
            ob = nelem(t["dims"]) * dtype_bits(t["data_type"]) // 8
        rec = (n["type"], ob)
        info[name] = rec
        for o in n.get("output_names", []):
            info.setdefault(o, rec)

    agg = defaultdict(lambda: [0, 0, 0])       # type -> [cycles, bytes, n]
    matched = unmatched = 0
    for ident, cyc in prof.items():
        key = ident.split(":OpId_")[0].strip()
        rec = info.get(key)
        if rec is None:
            unmatched += 1
            continue
        matched += 1
        t, ob = rec
        a = agg[t]
        a[0] += cyc
        a[1] += ob
        a[2] += 1

    print(f"matched {matched}, unmatched {unmatched}\n")
    tot_c = sum(a[0] for a in agg.values())
    tot_b = sum(a[1] for a in agg.values())
    print(f"total cycles {tot_c:,}   total output bytes {tot_b/1e6:.1f} MB")
    print(f"overall {tot_c/max(tot_b,1):.2f} cycles/byte\n")

    hdr = (f"{'op type':<20}{'n':>6}{'cycles':>14}{'share':>8}"
           f"{'out MB':>10}{'cyc/byte':>10}{'vs bulk':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    bulk = tot_c / max(tot_b, 1)
    for t, (c, b, n) in rows:
        cpb = c / b if b else float("nan")
        print(f"{t:<20}{n:>6}{c:>14,}{100*c/tot_c:>7.1f}%"
              f"{b/1e6:>10.1f}{cpb:>10.2f}{cpb/bulk:>9.2f}x")


if __name__ == "__main__":
    main()
