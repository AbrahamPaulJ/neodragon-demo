"""2-D VAE encoder on device vs fp32 ONNX, held-out real SSD1B first frames.

Headline metric is SNR on the MEAN half of `moments`. See make_enc_io.py for why
the whole-tensor figure is misleading here: the logvar half is a near-constant
~-30 that dominates the norm and is trivially reproduced.

The third number, "latent SNR", is what the DiT actually receives:
    image_latent = (mean - VAE_SHIFT_FACTOR) * VAE_SCALE_FACTOR
An affine map, so its SNR equals the mean-half SNR only when the shift is small
relative to the signal; reporting it separately keeps that assumption visible.

  usage: py -3.10 work/device/compare_enc.py [outdir]
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
SHAPE = (1, 32, 40, 64)
TARGET = 40.0            # paper Table 9, VAE Enc deploy SNR


def snr(ref, got):
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "eout_Q"
    d = W / sub
    if not d.is_dir():
        sys.exit("no such output dir: {}\nrun the device step first".format(d))

    refs = sorted((W / "eio").glob("mref_*.raw"))
    assert refs, "no references -- run make_enc_io.py first"

    # generation_utils.py imports these two; hardcoded here to keep the compare
    # script free of the neodragon package.
    from_gen = {}
    gu = (W.parents[1] / "src" / "neodragon" / "neodragon" / "utils"
          / "generation_utils.py").read_text(encoding="utf-8")
    for line in gu.splitlines():
        for k in ("VAE_SHIFT_FACTOR", "VAE_SCALE_FACTOR"):
            if line.startswith(k + " ="):
                from_gen[k] = float(line.split("=", 1)[1].strip())
    shift = from_gen.get("VAE_SHIFT_FACTOR", 0.0)
    scale = from_gen.get("VAE_SCALE_FACTOR", 1.0)

    print("=" * 74)
    print("VAE encoder on S25 Ultra HTP V79 -- held-out showcase first frames")
    print("outputs: {}   shift={} scale={}".format(d, shift, scale))
    print("=" * 74)

    rows, bad = [], 0
    print("")
    print("  {:>4} {:>10} {:>10} {:>10} {:>11}".format(
        "case", "mean dB", "latent dB", "all dB", "max|d| mean"))
    print("  " + "-" * 50)
    for c in range(len(refs)):
        ref = np.fromfile(W / "eio" / "mref_{}.raw".format(c),
                          dtype=np.float32).reshape(SHAPE)
        got = np.fromfile(d / "Result_{}".format(c) / "moments.raw",
                          dtype=np.float32).reshape(SHAPE)
        bad += int(np.count_nonzero(~np.isfinite(got)))
        s_mean = snr(ref[:, :16], got[:, :16])
        s_all = snr(ref, got)
        s_lat = snr((ref[:, :16] - shift) * scale, (got[:, :16] - shift) * scale)
        e = float(np.abs(ref[:, :16] - got[:, :16]).max())
        rows.append((s_mean, s_lat, s_all, e))
        print("  {:>4} {:>10.2f} {:>10.2f} {:>10.2f} {:>11.4e}".format(
            c, s_mean, s_lat, s_all, e))

    a = np.array(rows)
    print("")
    print("  mean-half SNR:  mean {:6.2f} dB   min {:6.2f} dB".format(
        a[:, 0].mean(), a[:, 0].min()))
    print("  latent    SNR:  mean {:6.2f} dB   min {:6.2f} dB".format(
        a[:, 1].mean(), a[:, 1].min()))
    print("  whole     SNR:  mean {:6.2f} dB   (inflated by the logvar half)".format(
        a[:, 2].mean()))
    print("  nan+inf: {}".format(bad))
    print("  paper deploy SNR target for VAE Enc: {} dB  -> {}".format(
        TARGET, "MET" if a[:, 0].mean() >= TARGET else "NOT met"))

    # trap #8 guard: identical outputs from different inputs is the classic
    # "everything looks fine" failure of a mis-parsed input list.
    g0 = np.fromfile(d / "Result_0" / "moments.raw", dtype=np.float32)
    g1 = np.fromfile(d / "Result_1" / "moments.raw", dtype=np.float32)
    print("  distinct outputs across cases 0/1: max|d| = {:.4f}  {}".format(
        np.abs(g0 - g1).max(),
        "OK" if np.abs(g0 - g1).max() > 1e-3 else "FAILED -- inputs not reaching graph"))


if __name__ == "__main__":
    main()
