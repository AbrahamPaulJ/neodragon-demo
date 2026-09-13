"""MMDiT stage graph on device vs fp32 ONNX, held-out real DiT calls.

Paper Table 9's deploy SNR targets are the worst in the pipeline, and they differ per
stage -- this is the only module where the target is not a single number:

    MMDiT+CA [7x10x16]  (stage 0)  29 dB
    MMDiT+CA [7x20x32]  (stage 1)  22 dB
    MMDiT+CA [7x40x64]  (stage 2)  24 dB

Trap #3 is the thing to watch here. Unlike DistilT5 -- which ships FP16 and escapes it --
the MMDiT is W8A16 and carries a residual stream through 18 blocks, so the effective-bits
argument in trap-audit.md section 3 applies for real. If a stage falls short, the residual
is the first place to look, not the attention.

  usage: py -3.10 work/device/compare_mmdit.py --stage 0 [outdir]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
TARGET = {0: 29.0, 1: 22.0, 2: 24.0}
LAT_HW = {0: (10, 16), 1: (20, 32), 2: (40, 64)}


def snr(ref, got):
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[0, 1, 2])
    ap.add_argument("outdir", nargs="?", default=None)
    ap.add_argument("--name", default="",
                    help="variant, e.g. mmdit_s0f -- selects mio_s0f / dout_s0f")
    args = ap.parse_args()

    suf = (args.name[6:] if args.name.startswith("mmdit_")
           else args.name) or "s{}".format(args.stage)
    sub = args.outdir or "dout_{}".format(suf)
    d = W / sub
    if not d.is_dir():
        sys.exit("no such output dir: {}\nrun the device step first".format(d))

    io_dir = W / "mio_{}".format(suf)
    refs = sorted(io_dir.glob("nref_*.raw"))
    assert refs, "no references in {} -- run make_mmdit_io.py --split test".format(io_dir)
    n_res = len(list(d.glob("Result_*")))
    assert n_res >= len(refs), "{} results for {} references".format(n_res, len(refs))

    h, w = LAT_HW[args.stage]
    shape = (1, 16, 1, h, w)

    print("=" * 76)
    print("MMDiT stage {} on S25 Ultra HTP V79 -- held-out real DiT calls".format(args.stage))
    print("outputs: {}".format(d))
    print("=" * 76)

    rows, bad = [], 0
    print("")
    print("  {:>4} {:>10} {:>12} {:>12}".format("case", "SNR dB", "max|d|", "ref rms"))
    print("  " + "-" * 42)
    for c in range(len(refs)):
        ref = np.fromfile(io_dir / "nref_{:04d}.raw".format(c),
                          dtype=np.float32).reshape(shape)
        got = np.fromfile(d / "Result_{}".format(c) / "noise_pred.raw",
                          dtype=np.float32).reshape(shape)
        bad += int(np.count_nonzero(~np.isfinite(got)))
        s = snr(ref, got)
        rows.append((s, float(np.abs(ref - got).max()), float(np.sqrt((ref ** 2).mean()))))
        print("  {:>4} {:>10.2f} {:>12.4e} {:>12.4f}".format(c, *rows[-1]))

    a = np.array(rows)
    t = TARGET[args.stage]
    print("")
    print("  SNR: mean {:6.2f} dB   min {:6.2f} dB".format(a[:, 0].mean(), a[:, 0].min()))
    print("  nan+inf: {}".format(bad))
    print("  paper deploy SNR target for stage {}: {} dB  -> {}".format(
        args.stage, t, "MET" if a[:, 0].mean() >= t else "NOT met"))

    g0 = np.fromfile(d / "Result_0" / "noise_pred.raw", dtype=np.float32)
    g1 = np.fromfile(d / "Result_1" / "noise_pred.raw", dtype=np.float32)
    md = np.abs(g0 - g1).max()
    print("  distinct outputs across cases 0/1: max|d| = {:.4f}  {}".format(
        md, "OK" if md > 1e-3 else "FAILED -- inputs not reaching graph (trap #8)"))


if __name__ == "__main__":
    main()
