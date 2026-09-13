"""Aggregate per-op cycles from the QNN profiling CSV to find the hotspots."""

import re
import sys
from collections import defaultdict
from pathlib import Path

W = Path(__file__).resolve().parent
CSV = W / (sys.argv[1] if len(sys.argv) > 1 else "prof.csv")


def norm_name(n):
    """Collapse per-block instances so the 12 encoder layers aggregate."""
    n = re.sub(r":OpId_\d+.*$", "", n)
    n = re.sub(r"^_enc_encoder_encoder_", "", n)
    n = re.sub(r"^rms_norm__enc_encoder_encoder_", "rms_norm:", n)
    n = re.sub(r"block_\d+_", "block_N_", n)
    n = re.sub(r"_\d+$", "", n)
    return n


def main():
    rows = CSV.read_text(encoding="utf-8", errors="replace").splitlines()

    execs = []           # list of dicts: per EXECUTE event
    cur = None
    tot_cycles = tot_us = None
    accel_us = []
    accel_cy = []

    for line in rows:
        p = line.split(",")
        if len(p) < 7:
            continue
        _, msg, val, unit, src, lvl, ident = p[0], p[1], p[2], p[3], p[4], p[5], ",".join(p[6:])
        if msg != "EXECUTE":
            continue
        try:
            v = int(val)
        except ValueError:
            continue
        if lvl == "ROOT" and "Accelerator (execute) time (cycles)" in ident:
            accel_cy.append(v)
            cur = defaultdict(int)
            execs.append(cur)
        elif lvl == "ROOT" and ident.strip() == "Accelerator (execute) time":
            accel_us.append(v)
        elif lvl == "SUB-EVENT" and unit == "CYCLES" and cur is not None:
            nm = ident.replace(" (cycles)", "").strip()
            cur[norm_name(nm)] += v

    print(f"EXECUTE events with per-op data: {len(execs)}")
    if accel_cy and accel_us:
        n = min(len(accel_cy), len(accel_us))
        hz = sum(accel_cy[:n]) / (sum(accel_us[:n]) / 1e6)
        print(f"derived accelerator clock: {hz/1e6:.0f} MHz")
    else:
        hz = 1e9

    # aggregate across all inferences, then report per-inference average
    agg = defaultdict(int)
    for e in execs:
        for k, v in e.items():
            agg[k] += v
    n_inf = max(len(execs), 1)

    total = sum(agg.values())
    print(f"\nsum of per-op cycles / inference: {total/n_inf:,.0f}")
    print(f"  == {total/n_inf/hz*1e3:.2f} ms at the derived clock\n")

    print(f"{'cycles/inf':>13} {'ms':>7} {'%':>6}  op")
    print("-" * 74)
    for k in sorted(agg, key=lambda x: -agg[x])[:28]:
        c = agg[k] / n_inf
        if c < 1:
            continue
        print(f"{c:>13,.0f} {c/hz*1e3:>7.3f} {100*agg[k]/total:>5.1f}%  {k[:52]}")

    # group by coarse kind
    print("\n--- by op kind ---")
    kind = defaultdict(int)
    for k, v in agg.items():
        if "rms_norm" in k:
            kind["rms_norm"] += v
        elif "MatMul" in k:
            kind["MatMul"] += v
        elif "Softmax" in k:
            kind["Softmax"] += v
        elif "Transpose" in k or "Reshape" in k:
            kind["Transpose/Reshape"] += v
        elif "Add" in k:
            kind["Add"] += v
        elif "Mul" in k:
            kind["Mul"] += v
        elif "Tanh" in k or "Gelu" in k or "Relu" in k or "Sigmoid" in k:
            kind["activation"] += v
        elif "Gather" in k:
            kind["Gather"] += v
        else:
            kind["other"] += v
    for k in sorted(kind, key=lambda x: -kind[x]):
        c = kind[k] / n_inf
        print(f"  {c:>13,.0f} cycles  {c/hz*1e3:>7.3f} ms  {100*kind[k]/total:>5.1f}%  {k}")


if __name__ == "__main__":
    main()
