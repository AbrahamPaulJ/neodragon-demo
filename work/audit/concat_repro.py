"""Minimal reproducer for the multi-input concat interleave, and a bench for workarounds.

Session 6 found QNN executing the MMDiT stage-1 patchify concat -- three tensors of
200/160/160 rows joined into 520 -- as EIGHT round-robin chunks: the consumer reads the
three producer buffers in the order 25,20,20, 25,20,20, ... The arithmetic is perfect
(applying that permutation scores 81.85 dB) but the ORDER is wrong, which is fatal
because the host supplies attn_mask and the RoPE tables indexed by sequential token
position. Stage 0's 280-row concat is untouched; the fault is invisible in net.json.

Iterating on that inside the real model costs ~55 minutes per attempt. This builds
tiny graphs with the same shapes so a variant can be converted and measured in minutes.

Variants:
  cat3      three-input concat, rank 2, axis 0      -- the shipping graph, expected BAD
  cat3r3    three-input concat, rank 3, axis 1      -- what the graph did before
  cat2      nested two-input concats                -- is it multi-input specific?
  padadd    zero-pad each part to full length and add -- no concat at all
  cat3big   three-input concat with 5x the rows     -- where is the threshold?

Each graph is: inputs -> (join) -> LayerNorm -> output. The LayerNorm is there because
in the real graph it is the concat's consumer, and the consumer is what reads the
buffers.

  py -3.10 work/audit/concat_repro.py --variant padadd --export
  py -3.10 work/audit/concat_repro.py --variant padadd --check work/device/crepro
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "onnx"
CAL = ROOT / "work" / "calib"
DEV = ROOT / "work" / "device"
DEVICE_ROOT = "/data/local/tmp/nd"

C = 1536
PARTS = {
    "cat3":   [200, 160, 160],
    "cat3r3": [200, 160, 160],
    "cat2":   [200, 160, 160],
    "padadd": [200, 160, 160],
    "cat3big": [1000, 800, 800],
    # the real shapes this project cares about
    "s0":     [200, 40, 40],      # stage 0 patchify -- known good on device
    "s2":     [200, 160, 160, 640],
    "attn1":  [128, 520],         # stage 1 attention: text ++ image, 648 rows
    "attn0":  [128, 280],         # stage 0 attention -- known good on device
    "two360": [200, 160],         # smallest 2-input case above stage 0's 280
    "two320": [200, 120],         # bisecting the threshold between 280 and 360
    "two300": [200, 100],
    # RANK 3, [n, 24, 64] -- the shape the real attention concat actually uses
    "qk1":    [128, 520],
    "qk0":    [128, 280],
}
# variants whose parts are [n, 24, 64] rather than [n, 1536]
RANK3 = {"qk0", "qk1"}


class Net(nn.Module):
    def __init__(self, variant, parts):
        super().__init__()
        self.variant = variant
        self.parts = parts
        self.r3 = variant in RANK3
        self.norm = nn.LayerNorm(64 if self.r3 else C,
                                 elementwise_affine=False, eps=1e-6)

    def forward(self, *xs):
        v = self.variant
        if v in ("cat3big", "cat3", "s0", "s2", "attn1", "attn0",
                 "two360", "two320", "two300", "qk0", "qk1"):
            h = torch.cat(xs, dim=0)
        elif v == "cat3r3":
            h = torch.cat([x[None] for x in xs], dim=1)[0]
        elif v == "cat2":
            h = torch.cat([torch.cat([xs[0], xs[1]], dim=0), xs[2]], dim=0)
        elif v == "padadd":
            # No concat at all: pad each part to the full length and sum. Padding is
            # zeros and the parts are disjoint, so the sum IS the concatenation.
            n = sum(self.parts)
            off, acc = 0, None
            for x, p in zip(xs, self.parts):
                y = F.pad(x, (0, 0, off, n - off - p))
                acc = y if acc is None else acc + y
                off += p
            h = acc
        else:
            raise ValueError(v)
        return self.norm(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(PARTS))
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--check", default="", help="device output dir to score")
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()

    parts = PARTS[a.variant]
    tag = "crepro_{}".format(a.variant)
    net = Net(a.variant, parts).eval()

    g = torch.Generator().manual_seed(0)
    shp = (lambda p: (p, 24, 64)) if a.variant in RANK3 else (lambda p: (p, C))
    xs = [torch.randn(*shp(p), generator=g) for p in parts]
    with torch.no_grad():
        ref = net(*xs)
    plain = net.norm(torch.cat(xs, dim=0))
    assert torch.allclose(ref, plain, atol=1e-5), "variant is not equivalent to a concat"
    print("[check] {} == concat+LayerNorm  (max|d| {:.2e})".format(
        a.variant, float((ref - plain).abs().max())))

    if a.check:
        d = Path(a.check)
        got = np.fromfile(d / "Result_0" / "out.raw", np.float32).reshape(ref.shape)
        r = ref.numpy()
        err = np.linalg.norm(r - got)
        snr = float("inf") if err == 0 else 20 * np.log10(np.linalg.norm(r) / err)
        print("")
        print("  device vs fp32 (identity order): {:.2f} dB".format(snr))
        r2 = r.reshape(r.shape[0], -1)
        g2 = got.reshape(got.shape[0], -1)
        rn = r2 / np.linalg.norm(r2, axis=1, keepdims=True)
        gn = g2 / np.linalg.norm(g2, axis=1, keepdims=True)
        j = (gn @ rn.T).argmax(1)
        ident = int((j == np.arange(len(j))).sum())
        print("  rows in the right place: {} of {}".format(ident, len(j)))
        if ident != len(j):
            runs, s = [], 0
            for i in range(1, len(j) + 1):
                if i == len(j) or j[i] != j[i - 1] + 1:
                    runs.append((s, i - 1, int(j[s]))); s = i
            print("  {} runs; first few:".format(len(runs)))
            for s0, e0, r0 in runs[:6]:
                print("     dev {:5d}:{:<5d} <- ref {:5d}  len {}".format(
                    s0, e0 + 1, r0, e0 - s0 + 1))
        print("")
        print("  VERDICT: {}".format("ORDER PRESERVED" if ident == len(j)
                                     else "INTERLEAVED -- workaround does not help"))
        return

    if not a.export:
        print("pass --export to write the ONNX and calibration set")
        return

    dst = OUT / tag
    dst.mkdir(parents=True, exist_ok=True)
    names = ["x{}".format(i) for i in range(len(parts))]
    torch.onnx.export(net, tuple(xs), str(dst / (tag + ".onnx")),
                      input_names=names, output_names=["out"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("[onnx]  {}".format(dst / (tag + ".onnx")))

    cal = CAL / tag
    cal.mkdir(parents=True, exist_ok=True)
    for f in cal.glob("*.raw"):
        f.unlink()
    lines, dev_lines = [], []
    for s in range(a.samples):
        gg = torch.Generator().manual_seed(100 + s)
        row, drow = [], []
        for nm, p in zip(names, parts):
            v = torch.randn(*shp(p), generator=gg).numpy().astype(np.float32)
            v.tofile(cal / "{}_{:04d}.raw".format(nm, s))
            row.append("{}:={}".format(nm, str(cal / "{}_{:04d}.raw".format(nm, s))
                                       .replace("C:", "/mnt/c").replace("\\", "/")))
            drow.append("{}:={}/{}/{}_{:04d}.raw".format(nm, DEVICE_ROOT, tag, nm, s))
        lines.append(" ".join(row))
        dev_lines.append(" ".join(drow))
    (cal / "calib_list_host.txt").write_text("\n".join(lines) + "\n", newline="\n")

    # sample 0 doubles as the device test input, so the fp32 reference above matches
    io = DEV / tag
    io.mkdir(parents=True, exist_ok=True)
    for f in io.glob("*.raw"):
        f.unlink()
    for nm, x in zip(names, xs):
        x.numpy().astype(np.float32).tofile(io / "{}_0000.raw".format(nm))
    (io / "input_list.txt").write_text(
        " ".join("{}:={}/{}/{}_0000.raw".format(nm, DEVICE_ROOT, tag, nm)
                 for nm in names) + "\n", newline="\n")
    dims = " ".join("--input_dim {} {}".format(
        nm, ",".join(str(x) for x in shp(p))) for nm, p in zip(names, parts))
    (cal / "dims.txt").write_text(dims + "\n", newline="\n")
    print("[calib] {} samples -> {}".format(a.samples, cal))
    print("[io]    device inputs -> {}".format(io))


if __name__ == "__main__":
    main()
