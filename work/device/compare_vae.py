"""Compare TAEHV decoder device output against the fp32 onnxruntime reference."""

from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
SHAPE = (1, 3, 1, 320, 512)


def main():
    print("=" * 70)
    print("TAEHV decoder T=1 on S25 Ultra HTP V79 vs fp32 reference")
    print("=" * 70)
    print(f"\n  {'case':>4} {'SNR dB':>9} {'max|diff|':>12} {'rel L2':>10} {'nan+inf':>9}")
    print("  " + "-" * 48)
    snrs = []
    for i in range(3):
        ref = np.fromfile(W / "vio" / f"vref_{i}.raw", dtype=np.float32).reshape(SHAPE)
        got = np.fromfile(W / "outv" / f"Result_{i}" / "video.raw",
                          dtype=np.float32).reshape(SHAPE)
        bad = int(np.isnan(got).sum() + np.isinf(got).sum())
        d = ref - got
        snr = 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(d))
        rel = np.linalg.norm(d) / np.linalg.norm(ref)
        snrs.append(snr)
        print(f"  {i:>4} {snr:>9.2f} {np.abs(d).max():>12.4e} {rel:>10.3e} {bad:>9}")
    print(f"\n  mean SNR {np.mean(snrs):.2f} dB")
    print("  paper deploy SNR for VAE Dec (W8A16): 35 dB -- this build is FLOAT")

    r0 = np.fromfile(W / "outv" / "Result_0" / "video.raw", dtype=np.float32)
    r1 = np.fromfile(W / "outv" / "Result_1" / "video.raw", dtype=np.float32)
    print(f"  [sanity] case0 vs case1 differ: max|d|={np.abs(r0 - r1).max():.4f}")

    # pixel-domain view: output is in roughly [-1, 1]
    ref0 = np.fromfile(W / "vio" / "vref_0.raw", dtype=np.float32)
    got0 = r0
    print(f"  [range]  ref [{ref0.min():.3f}, {ref0.max():.3f}]  "
          f"device [{got0.min():.3f}, {got0.max():.3f}]")
    err8 = np.abs(ref0 - got0) * 127.5
    print(f"  [8-bit]  mean abs err {err8.mean():.3f} levels, "
          f"p99 {np.percentile(err8, 99):.3f}, max {err8.max():.3f} (of 255)")


if __name__ == "__main__":
    main()
