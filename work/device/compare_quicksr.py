#!/usr/bin/env python3
"""Deploy SNR for QuickSRNet: device output vs the fp32 model, on HELD-OUT frames.

Paper Table 9 puts QuickSRNet at **48 dB**, uniquely using W8A16 + **AdaRound** (worth
"7+ dB SQNR" by their own account). Our conversion applies `--algorithms cle` but not
AdaRound proper, so landing short of 48 dB is the expected outcome and identifies the
AdaRound gap rather than a broken conversion.

The reference is computed here in fp32 from the same frames the device was given, and
those frames are **video 8, excluded from calibration** -- scoring on calibration data
would flatter the result and could not fail.

  usage: py -3.10 work/device/compare_quicksr.py [--outdir dout_quicksr]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work" / "device"
CAL = ROOT / "work" / "calib" / "quicksr"

H, WD, S = 320, 512, 2


def snr(ref, got):
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="dout_quicksr")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    d = W / a.outdir
    if not d.is_dir():
        sys.exit(f"no such output dir: {d}\nrun the device step first")

    tests = [ln.split(":=")[1] for ln in
             (CAL / "test_list_host.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if a.limit:
        tests = tests[:a.limit]

    # fp32 reference, same checkpoint the ONNX came from.
    import torch
    from qai_hub_models.utils.asset_loaders import always_answer_prompts
    with always_answer_prompts(True):
        from qai_hub_models.models.quicksrnetmedium.model import QuickSRNetMedium
        m = QuickSRNetMedium.from_pretrained(scale_factor=S).eval()

    res = sorted(d.glob("Result_*"), key=lambda p: int(p.name.split("_")[1]))
    assert res, f"no Result_* in {d}"
    print(f"{len(res)} device results, {len(tests)} held-out frames\n")
    print(f"{'i':>4}{'SNR dB':>10}{'max|d|':>12}{'mean px/255':>14}")
    print("-" * 40)

    vals = []
    for i, (rp, tf) in enumerate(zip(res, tests)):
        # WSL path in the list -> Windows path here
        p = tf
        if p.startswith("/mnt/"):
            p = p[5] + ":" + p[6:]
        x = np.fromfile(p, dtype=np.float32).reshape(1, 3, H, WD)
        with torch.no_grad():
            ref = m(torch.from_numpy(x)).numpy()
        outs = list(rp.glob("*.raw"))
        assert len(outs) == 1, [o.name for o in outs]
        got = np.fromfile(outs[0], dtype=np.float32).reshape(ref.shape)
        s = snr(ref, got)
        vals.append(s)
        print(f"{i:>4}{s:>10.2f}{np.abs(ref-got).max():>12.4e}"
              f"{np.abs(ref-got).mean()*255:>14.3f}")

    print(f"\n  SNR: mean {np.mean(vals):6.2f} dB   min {np.min(vals):6.2f} dB")
    print(f"  paper deploy SNR for QuickSRNet: 48.0 dB (W8A16 + AdaRound)")
    print(f"  -> {'MET' if np.mean(vals) >= 48 else 'NOT met'}"
          f"   (we apply CLE, not AdaRound -- the paper credits AdaRound with 7+ dB)")


if __name__ == "__main__":
    main()
