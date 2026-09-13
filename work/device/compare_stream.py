"""Streaming TAEHV decoder on device vs fp32 ONNX, held-out real latents.

Counterpart to compare_vae_q.py for the explicit-state graph. Two differences:

  * the output is [1, 24, 320, 512] -- 8 video frames packed on channels -- so
    per-frame SNR is reported as well as the whole-tensor figure. The old graph
    emitted one frame per invocation, so its SNR was necessarily whole-tensor.
  * 9 state outputs also come back. They are not scored here, but a state that
    has gone NaN or saturated is the first thing that would break a real
    multi-frame decode, so they are range-checked.

  usage: py -3.10 work/device/compare_stream.py [outdir]
"""

import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
SHAPE = (1, 24, 320, 512)
N_CASES, N_FRAMES, N_STATES = 6, 8, 9


def snr(ref, got):
    d = ref - got
    n = np.linalg.norm(d)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "sout_Q"
    d = W / sub
    if not d.is_dir():
        sys.exit(f"no such output dir: {d}\nrun the device step first")

    print("=" * 74)
    print(f"Streaming VAE decoder on S25 Ultra HTP V79 -- held-out real latents "
          f"(0050-0055)")
    print(f"outputs: {d}")
    print("=" * 74)

    snrs, errs, bad = [], [], 0
    print(f"\n  {'case':>4} {'SNR dB':>9} {'max|d|':>11} {'worst frame':>12} "
          f"{'state |max|':>12}")
    print("  " + "-" * 54)
    for c in range(N_CASES):
        ref = np.fromfile(W / "vio_stream" / f"vref_{c}.raw",
                          dtype=np.float32).reshape(SHAPE)
        got = np.fromfile(d / f"Result_{c}" / "video.raw",
                          dtype=np.float32).reshape(SHAPE)
        bad += int(np.count_nonzero(~np.isfinite(got)))
        s = snr(ref, got)
        per_frame = [snr(ref[:, i*3:(i+1)*3], got[:, i*3:(i+1)*3])
                     for i in range(N_FRAMES)]
        smax = 0.0
        for k in range(N_STATES):
            f = d / f"Result_{c}" / f"nstate_{k}.raw"
            if f.exists():
                a = np.fromfile(f, dtype=np.float32)
                bad += int(np.count_nonzero(~np.isfinite(a)))
                smax = max(smax, float(np.abs(a).max()))
        snrs.append(s)
        errs.append(float(np.abs(ref - got).max()))
        print(f"  {c:>4} {s:>9.2f} {errs[-1]:>11.4e} {min(per_frame):>12.2f} "
              f"{smax:>12.3f}")

    snrs = np.array(snrs)
    print(f"\n  mean {snrs.mean():6.2f} dB   min {snrs.min():6.2f} dB")
    print(f"  nan+inf across video and states: {bad}")
    print(f"  paper deploy SNR target for VAE Dec: 35 dB  -> "
          f"{'MET' if snrs.mean() >= 35 else 'NOT met'}")
    print(f"  T=1 baseline for comparison (docs/phase4-vae-decoder.md): 31.55 dB")

    ref0 = np.fromfile(W / "vio_stream" / "vref_0.raw", dtype=np.float32)
    got0 = np.fromfile(d / "Result_0" / "video.raw", dtype=np.float32)
    e8 = np.abs(ref0 - got0) * 127.5
    print(f"\n  pixel error: mean {e8.mean():.2f}, p99 {np.percentile(e8, 99):.2f}, "
          f"max {e8.max():.2f} levels of 255")


if __name__ == "__main__":
    main()
