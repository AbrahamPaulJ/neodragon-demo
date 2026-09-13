"""Float vs W8A16 VAE decoder on device, held-out real latents."""

from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
SHAPE = (1, 3, 1, 320, 512)
N = 6


def report(tag, sub):
    d = W / sub
    snrs, errs = [], []
    for i in range(N):
        ref = np.fromfile(W / "vio_real" / f"vref_{i}.raw", dtype=np.float32).reshape(SHAPE)
        got = np.fromfile(d / f"Result_{i}" / "video.raw", dtype=np.float32).reshape(SHAPE)
        diff = ref - got
        snrs.append(20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(diff)))
        errs.append(np.abs(diff).max())
    return np.array(snrs), np.array(errs)


def main():
    print("=" * 72)
    print("VAE decoder on S25 Ultra HTP V79 -- held-out REAL latents (0050-0055)")
    print("=" * 72)
    print(f"\n  {'case':>4} {'float SNR':>11} {'W8A16 SNR':>11} {'float max|d|':>13} "
          f"{'W8A16 max|d|':>13}")
    print("  " + "-" * 56)
    fs, fe = report("float", "vout_F")
    qs, qe = report("w8a16", "vout_Q")
    for i in range(N):
        print(f"  {i:>4} {fs[i]:>11.2f} {qs[i]:>11.2f} {fe[i]:>13.4e} {qe[i]:>13.4e}")
    print(f"\n  float  mean {fs.mean():6.2f} dB   min {fs.min():6.2f} dB")
    print(f"  W8A16  mean {qs.mean():6.2f} dB   min {qs.min():6.2f} dB")
    print(f"  paper deploy SNR target for VAE Dec: 35 dB  -> "
          f"{'MET' if qs.mean() >= 35 else 'NOT met'}")

    # pixel-domain
    ref0 = np.fromfile(W / "vio_real" / "vref_0.raw", dtype=np.float32)
    q0 = np.fromfile(W / "vout_Q" / "Result_0" / "video.raw", dtype=np.float32)
    e8 = np.abs(ref0 - q0) * 127.5
    print(f"\n  W8A16 pixel error: mean {e8.mean():.2f}, p99 {np.percentile(e8,99):.2f}, "
          f"max {e8.max():.2f} levels of 255")

    # sanity
    a = np.fromfile(W / "vout_Q" / "Result_0" / "video.raw", dtype=np.float32)
    b = np.fromfile(W / "vout_Q" / "Result_1" / "video.raw", dtype=np.float32)
    print(f"  [sanity] Q case0 vs case1 differ: max|d|={np.abs(a-b).max():.4f}")


if __name__ == "__main__":
    main()
