"""Compare on-device NPU output against the fp32 onnxruntime reference.

Reports SNR in dB, which is the metric the paper uses in its Table 9
quantisation summary, so these numbers are directly comparable to the
targets in the research brief.
"""

from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
SHAPE = (1, 128, 4096)


def snr_db(ref, got):
    noise = ref - got
    n = np.linalg.norm(noise)
    if n == 0:
        return float("inf")
    return 20.0 * np.log10(np.linalg.norm(ref) / n)


def main():
    print("=" * 76)
    print("DistilT5 on S25 Ultra HTP V79 -- device vs fp32 reference")
    print("=" * 76)
    print(f"\n{'case':>5}  {'SNR dB':>9}  {'max|diff|':>11}  {'rel L2':>10}  "
          f"{'cosine':>9}  {'ref max':>8}")
    print("-" * 66)

    snrs = []
    for i in range(3):
        ref = np.fromfile(W / "io" / f"ref_{i}.raw", dtype=np.float32).reshape(SHAPE)
        got = np.fromfile(W / "out" / f"Result_{i}" / "prompt_embeds.raw",
                          dtype=np.float32).reshape(SHAPE)
        a, b = ref.ravel().astype(np.float64), got.ravel().astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        s = snr_db(ref, got)
        snrs.append(s)
        print(f"{i:>5}  {s:>9.2f}  {np.abs(ref-got).max():>11.3e}  "
              f"{np.linalg.norm(ref-got)/np.linalg.norm(ref):>10.3e}  "
              f"{cos:>9.6f}  {np.abs(ref).max():>8.4f}")

    print(f"\n  mean SNR {np.mean(snrs):.2f} dB   min {np.min(snrs):.2f} dB")
    print("\n  For scale, the paper's deploy SNR targets (brief Table 9, W8A16):")
    print("    VAE dec 35 dB | SSD1B UNet 33 dB | MMDiT stages 22-29 dB | QuickSRNet 48 dB")
    print("  This build is FLOAT, not W8A16, so a high number here only confirms the")
    print("  conversion+runtime chain is faithful -- it is not a quantisation result.")

    # sanity: outputs must differ between prompts, or we are reading a stale buffer
    r0 = np.fromfile(W / "out" / "Result_0" / "prompt_embeds.raw", dtype=np.float32)
    r1 = np.fromfile(W / "out" / "Result_1" / "prompt_embeds.raw", dtype=np.float32)
    print(f"\n  [sanity] case0 vs case1 on device differ: "
          f"max|d|={np.abs(r0-r1).max():.4f} (must be non-zero)")


if __name__ == "__main__":
    main()
