"""Reproduce the MMDiT's quantisation error on the HOST, and attribute it.

Motivation: hypotheses for the stage-0 accuracy gap kept dying by measurement, and the
obvious next step -- a per-block dump from the device -- turned out to need an
aarch64-android model lib (qnn-net-run's --debug and --set_output_tensors both refuse to
run against a context binary, and the host CPU backend rejects the rank-5 latent
Transpose). That is a detour.

It is not needed. Every tensor's quantisation encoding is already in the converter's
`*_net.json`. So the residual stream's quantisation can be replayed exactly, in fp32
torch, with no device and no NDK -- and switched on and off per group, which a device
dump cannot do at all.

The question this answers: how much of the 19.48 dB is explained by quantising the
RESIDUAL STREAM alone (35 tensors, two per block) versus everything else?

  * reproduces ~19 dB  -> the residual chain is the cause. The fix is trap #3's residual
    scaling folded into weights, and better calibration is beside the point.
  * barely degrades     -> the error is inside the blocks; go find it there.

QNN encoding convention, verified against the printed ranges in net.json:
    real = scale * (q + offset),  q integer in [0, 2**bits - 1]
so  q    = clamp(round(real/scale) - offset, 0, 2**bits - 1).

  usage: py -3.10 work/device/resid_ablate.py --stage 0 [--case 0]
"""

import argparse
import json
import re
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
from find_block_io import boundaries, idx            # noqa: E402


def snr(ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def fq(x, scale, offset, bits):
    """Fake-quantise with QNN's asymmetric scale/offset encoding."""
    n = 2 ** bits - 1
    q = torch.clamp(torch.round(x / scale) - offset, 0, n)
    return scale * (q + offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args()

    net_json = W / "mmdit_s{}_net.json".format(a.stage)
    tensors = json.load(open(net_json))["graph"]["tensors"]

    # the block-output tensors, in block order, image stream then text stream
    b, streams = boundaries(str(net_json))
    per_stream = []
    for shape in streams:
        hits = b[shape]
        per_stream.append([p for p in hits[1::2]])       # post-FFN == block output
    img_names, txt_names = per_stream[0], per_stream[1]

    def enc(nm):
        t = tensors[nm]
        q = t["quant_params"]["scale_offset"]
        return q["scale"], q["offset"], int("{:02x}".format(t["data_type"] & 0xFF))

    img_enc = [enc(n) for n in img_names]
    txt_enc = [enc(n) for n in txt_names]

    print("=" * 78)
    print("Residual-stream quantisation ablation, stage {} case {}".format(a.stage, a.case))
    print("=" * 78)
    print("  {} image block outputs, {} text block outputs".format(
        len(img_enc), len(txt_enc)))
    print("  image step: block 0 {:.4g} -> block {} {:.4g}  ({:.1f}x coarser)".format(
        img_enc[0][0], len(img_enc) - 1, img_enc[-1][0], img_enc[-1][0] / img_enc[0][0]))

    io_dir = W / "mio_s{}".format(a.stage)

    def raw(nm, shape):
        p = io_dir / "{}_{:04d}.raw".format(nm, a.case)
        return torch.from_numpy(np.fromfile(p, np.float32).reshape(shape).copy())

    env = envelope_for(a.stage)
    print("")
    print("[load] diffusion_transformer_320p (fp32, CPU) ...")
    dit = PyramidMMDiT.from_pretrained(
        str(MODEL / "diffusion_transformer_320p"), torch_dtype=torch.float32).eval()
    net = StageMMDiT(dit, env).eval()
    n_img = net.n_img
    S = TEXT_TOKENS + n_img

    args_t = (raw("encoder_hidden_states", (1, TEXT_TOKENS, 1536)),
              raw("pooled_projections", (1, 2048)),
              raw("timestep_ratio", (1,)),
              raw("attn_mask", (1, S, S)),
              raw("rope_cos", (S, 1, 32)),
              raw("rope_sin", (S, 1, 32)),
              *[raw("latent_{}".format(i), (1, 16, t, h, w))
                for i, (t, h, w) in enumerate(env)])

    orig = net._block

    def run(quantise):
        state = {"i": 0}

        def spy(block, hidden, enc_hidden, temb, c, s, m):
            e, h = orig(block, hidden, enc_hidden, temb, c, s, m)
            if quantise:
                k = state["i"]
                if k < len(img_enc):
                    h = fq(h, *img_enc[k])
                if e is not None and k < len(txt_enc):
                    e = fq(e, *txt_enc[k])
            state["i"] += 1
            return e, h

        net._block = spy if quantise else orig
        with torch.no_grad():
            return net(*args_t).numpy()

    print("[run]  fp32 reference ...")
    ref = run(False)
    print("[run]  residual stream fake-quantised ...")
    got = run(True)

    dev = np.fromfile(io_dir / "nref_{:04d}.raw".format(a.case), np.float32)
    print("")
    print("  sanity: fp32 rerun vs stored nref  = {:.2f} dB".format(
        snr(dev.reshape(ref.shape), ref)))
    print("")
    print("  " + "=" * 66)
    print("  RESIDUAL-ONLY quantisation  ->  {:.2f} dB".format(snr(ref, got)))
    print("  device, everything quantised ->  19.48 dB   (16-case mean)")
    print("  " + "=" * 66)
    print("")
    print("  If these are close, the residual chain IS the cause and the fix is")
    print("  residual scaling folded into weights (trap #3), not calibration.")


if __name__ == "__main__":
    main()
