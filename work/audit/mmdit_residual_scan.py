"""Where does the MMDiT's activation range come from? Per-block, real vs padded tokens.

`analyze_encodings.py` on the stage-0 build found activation encodings with a range of
55,130 and a step of 0.84 against a graph median of 24.47 -- i.e. ~3-5 effective bits on
the late residual adds, which is trap #3 and almost certainly the 19.47 dB vs 29 dB gap.

This says which positions produce that range. Two candidate culprits, and they need
different fixes:

  * genuine residual growth through 18 blocks, on REAL tokens
        -> percentile / MSE activation calibration, or residual scaling (trap #3's fix)
  * garbage on PADDED tokens, which are isolated by the mask and whose values therefore
    do not affect the output at all
        -> force them small; exact, and free

Padded positions are isolated by construction: `merge_input` gives padded text ids 0 and
everything real id 1, so no real query ever attends to a padded key, and the output slice
keeps only the trailing real image tokens. So whatever they hold is dead weight -- but it
still shares a per-tensor encoding with the real tokens.

  usage: py -3.10 work/audit/mmdit_residual_scan.py [--unit 1] [--stage 0]
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work" / "export"))
sys.path.insert(0, str(ROOT / "work" / "audit"))
sys.path.insert(0, str(ROOT / "src" / "neodragon"))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(ROOT / "src" / "neodragon" / "neodragon")]
sys.modules["neodragon"] = _pkg

CALLS = ROOT / "work" / "calib" / "mmdit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--unit", type=int, default=1)
    ap.add_argument("--video", type=int, default=50)
    args = ap.parse_args()

    from export_mmdit_stage import (StageMMDiT, TEXT_TOKENS, envelope_for,
                                    pad_latents, stage_conditioning_padded)
    from mmdit_shapes import tokens
    from neodragon.pyramid_mmdit import PyramidMMDiT

    # the AR loop emits calls in (unit, stage) order: call index = (unit-1)*3 + stage
    call = (args.unit - 1) * 3 + args.stage
    f = CALLS / "call_{:03d}_{:02d}.npz".format(args.video, call)
    assert f.exists(), f
    d = np.load(f)
    n_lat = int(d["n_latents"])
    lats = [torch.from_numpy(d["latent_{}".format(i)]).float() for i in range(n_lat)]
    real_shapes = [(int(t.shape[2]), int(t.shape[3]), int(t.shape[4])) for t in lats]
    env = envelope_for(args.stage)

    print("=" * 84)
    print("MMDiT residual scan -- video {} unit {} stage {}".format(
        args.video, args.unit, args.stage))
    print("  real {}   envelope {}".format(
        " ".join("{}x{}x{}".format(*s) for s in real_shapes),
        " ".join("{}x{}x{}".format(*s) for s in env)))
    print("=" * 84)

    dit = PyramidMMDiT.from_pretrained(
        str(ROOT / "work" / "models" / "neodragon" / "diffusion_transformer_320p"),
        torch_dtype=torch.float32).eval()
    net = StageMMDiT(dit, env).eval()

    eam = torch.from_numpy(d["encoder_attention_mask"]).float()
    n_text_real = int(eam.sum())
    mask, cos, sin = stage_conditioning_padded(dit, real_shapes, env, eam)
    padded = pad_latents(lats, real_shapes, env)

    n_img_real = sum(tokens(s) for s in real_shapes)
    n_img_pad = net.n_img - n_img_real
    print("")
    print("  text: {} real + {} padded    image: {} real + {} padded".format(
        n_text_real, TEXT_TOKENS - n_text_real, n_img_real, n_img_pad))
    print("")

    # image tokens sit at the END of the image block (padding is at the front)
    def stats(x, lo, hi):
        if hi <= lo:
            return float("nan")
        return float(x[:, lo:hi].abs().max())

    rows = []
    hidden_hist, enc_hist = [], []

    orig_block = net._block

    def spy(block, hidden, enc_hidden, temb, cos_, sin_, mask_):
        enc_out, hid_out = orig_block(block, hidden, enc_hidden, temb, cos_, sin_, mask_)
        hidden_hist.append(hid_out.detach())
        enc_hist.append(enc_hidden.detach() if enc_out is None else enc_out.detach())
        return enc_out, hid_out

    net._block = spy
    with torch.no_grad():
        net(torch.from_numpy(d["encoder_hidden_states"]).float(),
            torch.from_numpy(d["pooled_projections"]).float(),
            torch.from_numpy(d["timestep_ratio"]).float(),
            mask, cos, sin, *padded)

    print("  {:>5} {:>12} {:>12} {:>12} {:>12}".format(
        "block", "img real", "img PAD", "text real", "text PAD"))
    print("  " + "-" * 58)
    for i, (h, e) in enumerate(zip(hidden_hist, enc_hist)):
        ir = stats(h, n_img_pad, net.n_img)
        ip = stats(h, 0, n_img_pad)
        tr = stats(e, 0, n_text_real)
        tp = stats(e, n_text_real, TEXT_TOKENS)
        rows.append((ir, ip, tr, tp))
        print("  {:>5} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(i, ir, ip, tr, tp))

    # --- how much of the range is outlier? -------------------------------
    # This is what decides the fix. If absmax >> p99.99 the range is set by a handful
    # of values and percentile/MSE activation calibration reclaims the difference
    # directly. If they are close, the distribution is genuinely wide and calibration
    # will not help -- residual scaling would be the only lever.
    print("")
    print("  outlier structure of the residual stream (real tokens only):")
    print("  {:>5} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "block", "which", "absmax", "p99.99", "p99.9", "std", "max/p99.99"))
    print("  " + "-" * 64)
    for i in (0, 8, 13, 16, 17):
        for lbl, x, lo, hi in (("img", hidden_hist[i], n_img_pad, net.n_img),
                               ("text", enc_hist[i], 0, n_text_real)):
            v = x[:, lo:hi].abs().flatten().numpy()
            if v.size == 0:
                continue
            mx = float(v.max())
            p9999 = float(np.percentile(v, 99.99))
            p999 = float(np.percentile(v, 99.9))
            print("  {:>5} {:>6} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.2f} {:>8.1f}x".format(
                i, lbl, mx, p9999, p999, float(v.std()), mx / max(p9999, 1e-9)))

    a = np.array(rows)
    print("")
    print("  peak over all blocks:  img real {:.1f}   img PAD {:.1f}   "
          "text real {:.1f}   text PAD {:.1f}".format(
              np.nanmax(a[:, 0]), np.nanmax(a[:, 1]),
              np.nanmax(a[:, 2]), np.nanmax(a[:, 3])))
    worst = max(np.nanmax(a[:, 1]), np.nanmax(a[:, 3]))
    real = max(np.nanmax(a[:, 0]), np.nanmax(a[:, 2]))
    print("  padded / real peak ratio: {:.2f}x".format(worst / real))
    if worst > 2 * real:
        print("  -> PADDED tokens dominate the range. They are masked out of every real")
        print("     token's attention and discarded at the output, so forcing them small")
        print("     is exact and costs nothing.")
    else:
        print("  -> real tokens set the range: this is genuine residual growth (trap #3).")
        print("     Fix is percentile/MSE activation calibration or residual scaling.")


if __name__ == "__main__":
    main()
