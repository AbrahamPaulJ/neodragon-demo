"""SSD1B VAE decoder on device vs fp32. Table 9 target: 31 dB.

This one is worth reporting in PIXELS as well as dB, because its output is an image in
[-1, 1] rather than an intermediate tensor -- the video VAE decoder's 34.27 dB works out
to a mean pixel error of 1.25/255, which is the only intuition this project has for what
a dB is worth perceptually.

  usage: py -3.10 work/device/compare_ssd1bvaedec.py [outdir]
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
TARGET = 31.0
SHAPE = (1, 3, 640, 1024)


def main():
    io = W / "io_ssd1bvaedec"
    d = W / (sys.argv[1] if len(sys.argv) > 1 else "dout_ssd1bvaedec")
    refs = sorted(io.glob("vref_*.raw"))
    if not refs:
        sys.exit("no references -- run export_ssd1b_vaedec.py --export")
    if not d.is_dir():
        sys.exit("no device outputs at {}".format(d))

    print("=" * 70)
    print("SSD1B VAE decoder on S25 Ultra HTP V79 -- held-out latents")
    print("=" * 70)
    print("")
    print("  {:>4} {:>10} {:>12} {:>14}".format("case", "SNR dB", "max|d|", "mean px /255"))
    print("  " + "-" * 44)
    rows, bad = [], 0
    for c in range(len(refs)):
        ref = np.fromfile(io / "vref_{:04d}.raw".format(c), np.float32).reshape(SHAPE)
        got = np.fromfile(d / "Result_{}".format(c) / "image.raw",
                          np.float32).reshape(SHAPE)
        bad += int(np.count_nonzero(~np.isfinite(got)))
        e = np.linalg.norm(ref - got)
        snr = float("inf") if e == 0 else 20 * np.log10(np.linalg.norm(ref) / e)
        # [-1,1] -> 8-bit: half the range is 127.5 levels
        px = float(np.abs(ref - got).mean()) * 127.5
        rows.append((snr, px))
        print("  {:>4} {:>10.2f} {:>12.4e} {:>14.2f}".format(
            c, snr, float(np.abs(ref - got).max()), px))

    a = np.array(rows)
    print("")
    print("  SNR: mean {:6.2f} dB   min {:6.2f} dB".format(a[:, 0].mean(), a[:, 0].min()))
    print("  mean pixel error: {:.2f}/255   (video VAE dec at 34.27 dB = 1.25/255)".format(
        a[:, 1].mean()))
    print("  nan+inf: {}".format(bad))
    print("  paper Table 9 target: {} dB  -> {}".format(
        TARGET, "MET" if a[:, 0].mean() >= TARGET else "NOT met"))
    g0 = np.fromfile(d / "Result_0" / "image.raw", np.float32)
    g1 = np.fromfile(d / "Result_1" / "image.raw", np.float32)
    md = float(np.abs(g0 - g1).max())
    print("  distinct outputs across cases 0/1: max|d| = {:.4f}  {}".format(
        md, "OK" if md > 1e-3 else "FAILED -- inputs not reaching graph (trap #8)"))


if __name__ == "__main__":
    main()
