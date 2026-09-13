"""SSD1B UNet on device vs fp32, on held-out prompts. Paper Table 9 target: 33 dB.

The first frame runs 4 denoising steps once per video, so unlike the MMDiT this module's
error does not compound across an AR loop -- but it does seed everything downstream: the
frame is LANCZOS-downscaled to 512x320 and encoded by the causal video VAE to become the
AR loop's starting latent.

  usage: py -3.10 work/device/compare_unet.py [outdir]
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
TARGET = 33.0
SHAPE = (1, 4, 80, 128)


def main():
    io = W / "io_ssd1bunet"
    d = W / (sys.argv[1] if len(sys.argv) > 1 else "dout_ssd1bunet")
    refs = sorted(io.glob("uref_*.raw"))
    if not refs:
        sys.exit("no references -- run make_unet_refs.py")
    if not d.is_dir():
        sys.exit("no device outputs at {}".format(d))

    print("=" * 66)
    print("SSD1B UNet on S25 Ultra HTP V79 -- held-out prompts")
    print("=" * 66)
    print("")
    print("  {:>4} {:>10} {:>12} {:>12}".format("case", "SNR dB", "max|d|", "ref rms"))
    print("  " + "-" * 42)
    rows, bad = [], 0
    for c in range(len(refs)):
        ref = np.fromfile(io / "uref_{:04d}.raw".format(c), np.float32).reshape(SHAPE)
        got = np.fromfile(d / "Result_{}".format(c) / "noise_pred.raw",
                          np.float32).reshape(SHAPE)
        bad += int(np.count_nonzero(~np.isfinite(got)))
        e = np.linalg.norm(ref - got)
        snr = float("inf") if e == 0 else 20 * np.log10(np.linalg.norm(ref) / e)
        rows.append(snr)
        print("  {:>4} {:>10.2f} {:>12.4e} {:>12.4f}".format(
            c, snr, float(np.abs(ref - got).max()), float(ref.std())))

    a = np.array(rows)
    print("")
    print("  SNR: mean {:6.2f} dB   min {:6.2f} dB".format(a.mean(), a.min()))
    print("  nan+inf: {}".format(bad))
    print("  paper Table 9 target: {} dB  -> {}".format(
        TARGET, "MET" if a.mean() >= TARGET else "NOT met"))
    g0 = np.fromfile(d / "Result_0" / "noise_pred.raw", np.float32)
    g1 = np.fromfile(d / "Result_1" / "noise_pred.raw", np.float32)
    md = float(np.abs(g0 - g1).max())
    print("  distinct outputs across cases 0/1: max|d| = {:.4f}  {}".format(
        md, "OK" if md > 1e-3 else "FAILED -- inputs not reaching graph (trap #8)"))


if __name__ == "__main__":
    main()
