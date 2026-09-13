"""Compare device output against the fp32 reference, for both builds."""

from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
SHAPE = (1, 128, 4096)


def report(label, subdir):
    d = W / subdir
    if not d.exists():
        print(f"\n{label}: {subdir} missing")
        return
    print(f"\n{label}")
    print(f"  {'case':>4} {'SNR dB':>9} {'max|diff|':>11} {'rel L2':>10} "
          f"{'cosine':>10} {'nan+inf':>9}")
    print("  " + "-" * 58)
    snrs = []
    for i in range(3):
        ref = np.fromfile(W / "io" / f"ref_{i}.raw", dtype=np.float32).reshape(SHAPE)
        got = np.fromfile(d / f"Result_{i}" / "prompt_embeds.raw",
                          dtype=np.float32).reshape(SHAPE)
        bad = int(np.isnan(got).sum() + np.isinf(got).sum())
        diff = ref - got
        if np.isfinite(diff).all():
            snr = 20 * np.log10(np.linalg.norm(ref) / np.linalg.norm(diff))
            rel = np.linalg.norm(diff) / np.linalg.norm(ref)
            a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            mx = np.abs(diff).max()
        else:
            snr = rel = cos = mx = float("nan")
        snrs.append(snr)
        print(f"  {i:>4} {snr:>9.2f} {mx:>11.3e} {rel:>10.3e} {cos:>10.6f} {bad:>9}")
    if np.isfinite(snrs).all():
        print(f"  mean SNR {np.mean(snrs):.2f} dB   min {np.min(snrs):.2f} dB")

    r0 = np.fromfile(d / "Result_0" / "prompt_embeds.raw", dtype=np.float32)
    r1 = np.fromfile(d / "Result_1" / "prompt_embeds.raw", dtype=np.float32)
    print(f"  [sanity] case0 vs case1 differ: max|d|={np.abs(r0 - r1).max():.4f}")


def main():
    print("=" * 72)
    print("DistilT5 on S25 Ultra (SM8750) HTP V79 -- device vs fp32 reference")
    print("=" * 72)
    report("BUILD A -- float, no residual scaling", "out")
    report("BUILD B -- residual scaling S=16", "outs")
    print("\nPaper deploy-SNR targets for scale (brief Table 9, W8A16):")
    print("  VAE dec 35 dB | SSD1B UNet 33 dB | MMDiT 22-29 dB | QuickSRNet 48 dB")
    print("Both builds here are FLOAT (HTP runs them fp16), not W8A16.")


if __name__ == "__main__":
    main()
