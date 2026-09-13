"""Are the converter's calibrated encoding ranges much wider than the real activations?

Context: a host simulation of W8 weights + A16 activations reproduces only 43 dB, while
the device measures 19.48 dB (quant_ablate.py). So ordinary quantisation does not explain
the gap. Two candidates remain:

  (1) the CALIBRATED ranges are far wider than the activations actually seen, so the real
      quantum is much coarser than a per-sample min-max simulation assumes;
  (2) HTP-specific numerics -- e.g. trap #3's fp16 accumulation inside the norms, which
      the residual stream's growth to ~28,000 would make dangerous.

This discriminates. For every block-boundary tensor it prints the net.json encoding width
against the width actually observed on real data, so (1) is either confirmed or removed.

  usage: py -3.10 work/device/enc_vs_actual.py --stage 0 [--case 0]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W.parent / "export"))

from export_mmdit_stage import (                     # noqa: E402
    StageMMDiT, envelope_for, TEXT_TOKENS, MODEL,
)
from neodragon.pyramid_mmdit import PyramidMMDiT     # noqa: E402
from find_block_io import boundaries                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args()

    net_json = W / "mmdit_s{}_net.json".format(a.stage)
    tensors = json.load(open(net_json))["graph"]["tensors"]
    b, streams = boundaries(str(net_json))
    img_names = b[streams[0]][1::2]
    txt_names = b[streams[1]][1::2]

    def enc(nm):
        t = tensors[nm]
        q = t["quant_params"]["scale_offset"]
        bits = int("{:02x}".format(t["data_type"] & 0xFF))
        s, o = q["scale"], q["offset"]
        return s * o, s * (o + 2 ** bits - 1), s

    io_dir = W / "mio_s{}".format(a.stage)

    def raw(nm, shape):
        p = io_dir / "{}_{:04d}.raw".format(nm, a.case)
        return torch.from_numpy(np.fromfile(p, np.float32).reshape(shape).copy())

    env = envelope_for(a.stage)
    print("[load] diffusion_transformer_320p (fp32, CPU) ...")
    dit = PyramidMMDiT.from_pretrained(
        str(MODEL / "diffusion_transformer_320p"), torch_dtype=torch.float32).eval()
    net = StageMMDiT(dit, env).eval()
    S = TEXT_TOKENS + net.n_img

    args_t = (raw("encoder_hidden_states", (1, TEXT_TOKENS, 1536)),
              raw("pooled_projections", (1, 2048)),
              raw("timestep_ratio", (1,)),
              raw("attn_mask", (1, S, S)),
              raw("rope_cos", (S, 1, 32)),
              raw("rope_sin", (S, 1, 32)),
              *[raw("latent_{}".format(i), (1, 16, t, h, w))
                for i, (t, h, w) in enumerate(env)])

    caught = []
    orig = net._block

    def spy(block, hidden, enc_hidden, temb, c, s, m):
        e, h = orig(block, hidden, enc_hidden, temb, c, s, m)
        caught.append((h.detach(), None if e is None else e.detach()))
        return e, h

    net._block = spy
    with torch.no_grad():
        net(*args_t)

    print("")
    print("=" * 86)
    print("Encoded range vs ACTUAL range, per block output (stage {}, case {})".format(
        a.stage, a.case))
    print("=" * 86)
    print("  {:>5} {:>13} {:>13} {:>8}   {:>13} {:>13} {:>8}".format(
        "block", "enc width", "act width", "ratio", "enc width", "act width", "ratio"))
    print("  {:>5} {:>36}   {:>36}".format("", "--- image stream ---", "--- text stream ---"))
    print("  " + "-" * 82)

    ratios = []
    for k in range(len(caught)):
        eo = enc(img_names[k])
        h = caught[k][0]
        ew, aw = eo[1] - eo[0], float(h.max() - h.min())
        r = ew / aw if aw else float("nan")
        ratios.append(r)
        row = "  {:>5} {:>13.1f} {:>13.1f} {:>7.2f}x".format(k, ew, aw, r)
        if k < len(txt_names) and caught[k][1] is not None:
            et = enc(txt_names[k])
            e2 = caught[k][1]
            ew2, aw2 = et[1] - et[0], float(e2.max() - e2.min())
            row += "   {:>13.1f} {:>13.1f} {:>7.2f}x".format(
                ew2, aw2, ew2 / aw2 if aw2 else float("nan"))
        print(row)

    print("")
    print("  image-stream enc/act ratio: min {:.2f}x  median {:.2f}x  max {:.2f}x".format(
        min(ratios), float(np.median(ratios)), max(ratios)))
    print("")
    print("  A ratio near 1 means calibration is TIGHT and hypothesis (1) is dead --")
    print("  the coarseness is real, not an artefact of over-wide ranges.")


if __name__ == "__main__":
    main()
