"""Do the PADDED tokens set the encodings that the REAL tokens then have to live with?

Padding was ruled out as a cause of the session-5 gap, and correctly: it is transparent
in fp32 (123-128 dB across all 18 shapes). But transparency in fp32 says nothing about
quantisation, because a padded row still flows through every tensor in the graph and
still participates in the min/max that calibration derives. If padded rows are *larger*
than real ones, every real token pays for their range in effective bits.

The session-6 device result is what raises the question: with the AdaLN modulation fixed,
per-case SNR lines up almost monotonically with how much padding the case carries.

    case  unit  padded tokens   SNR
       0     1        200      24.47
       1     2        160      27.48
       5     6          0      28.50
       6     1        200      25.98
      11     6          0      29.84

This runs the fp32 envelope graph on one case and reports, at every block boundary,
max|x| over the padded image rows against max|x| over the real ones.

  usage: py -3.10 work/device/pad_ranges.py --case 0 [--stage 0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parent
ND = W.parent.parent
sys.path.insert(0, str(ND / "work" / "export"))
sys.path.insert(0, str(ND / "work" / "audit"))

TEXT_TOKENS = 128
LAT_C = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--io", default="")
    a = ap.parse_args()

    from export_mmdit_stage import StageMMDiT, envelope_for, MODEL
    from mmdit_shapes import tokens
    from neodragon.pyramid_mmdit import PyramidMMDiT

    io_dir = W / (a.io or "mio_s{}f".format(a.stage))
    env = envelope_for(a.stage)
    n_img = sum(tokens(s) for s in env)
    seq = TEXT_TOKENS + n_img

    def raw(nm, shp):
        p = io_dir / "{}_{:04d}.raw".format(nm, a.case)
        return torch.from_numpy(np.fromfile(p, np.float32).reshape(shp).copy())

    lats = [raw("latent_{}".format(i), (1, LAT_C, t, h, w))
            for i, (t, h, w) in enumerate(env)]

    # Which image rows are padding? A slot is padding wherever its latent frame is
    # exactly zero -- pad_latents() writes literal zeros and real latents never are.
    pad_rows, off = [], 0
    for (t, h, w), lat in zip(env, lats):
        n_spatial = (h // 2) * (w // 2)
        for f in range(t):
            is_pad = bool(torch.all(lat[0, :, f] == 0))
            pad_rows.extend(range(off, off + n_spatial) if is_pad else [])
            off += n_spatial
    pad = torch.zeros(n_img, dtype=torch.bool)
    pad[list(pad_rows)] = True
    print("case {}: {} image tokens, {} padded ({:.0%})".format(
        a.case, n_img, int(pad.sum()), float(pad.float().mean())))

    print("[load] DiT ...")
    dit = PyramidMMDiT.from_pretrained(str(MODEL / "diffusion_transformer_320p"),
                                       torch_dtype=torch.float32).eval()
    net = StageMMDiT(dit, env).eval()

    args_t = (raw("encoder_hidden_states", (1, TEXT_TOKENS, 1536)),
              raw("temb_act", (1, 1536)),
              raw("attn_mask", (1, seq, seq)),
              raw("rope_cos", (seq, 1, 32)),
              raw("rope_sin", (seq, 1, 32)),
              *lats)

    rows = []

    # PRE-hooks on the LayerNorms, not forward hooks on the blocks: StageMMDiT calls
    # `self._block(i, block, ...)` directly rather than `block(...)`, so a block-level
    # forward hook never fires. `block.norm1.norm` and `block.norm2` are the two places
    # the residual stream is handed to an nn.Module by __call__, and the tensor entering
    # `norm1.norm` IS the block's input boundary.
    def hook(tag):
        def f(_m, inp):
            x = inp[0]
            if not torch.is_tensor(x) or x.dim() != 3 or x.shape[1] != n_img:
                return
            v = x[0].detach()
            rows.append((tag, float(v[pad].abs().max()), float(v[~pad].abs().max())))
        return f

    hs = [b.norm1.norm.register_forward_pre_hook(hook("block {:2d} in".format(i)))
          for i, b in enumerate(dit.transformer_blocks)]
    hs += [b.norm2.register_forward_pre_hook(hook("block {:2d} mid".format(i)))
           for i, b in enumerate(dit.transformer_blocks)]
    hs.append(dit.norm_out.norm.register_forward_pre_hook(hook("norm_out   ")))
    with torch.no_grad():
        net(*args_t)
    for h in hs:
        h.remove()

    print("")
    print("  {:<14} {:>12} {:>12} {:>10}".format(
        "boundary", "max|pad|", "max|real|", "pad/real"))
    print("  " + "-" * 52)
    worst = 0.0
    rows.sort(key=lambda r: (r[0].split()[1], r[0].split()[-1] != "in") if r[0].startswith("block") else ("99", 1))
    for tag, mp, mr in rows:
        r = mp / mr if mr else float("inf")
        worst = max(worst, r)
        print("  {:<14} {:>12.3f} {:>12.3f} {:>10.2f}".format(tag, mp, mr, r))
    print("")
    print("  worst pad/real ratio: {:.2f}x".format(worst))
    print("  > 1 means the padded rows, which the output slice DISCARDS, are what the")
    print("    calibrated encoding is sized on -- and the real tokens pay for it.")


if __name__ == "__main__":
    main()
