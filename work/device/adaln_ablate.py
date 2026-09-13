"""Ablate the AdaLN / modulation elementwise chain -- the largest untested group.

Where this sits in the chain of evidence (docs/phase5-mmdit-accuracy.md):

    residual boundaries only  -> 53.33 dB
    W8 weights only           -> 43.14 dB
    weights+linear+softmax    -> 43.11 dB
    device, everything        -> 19.48 dB

and the graph has 1247 genuine requantisation points, of which Eltwise_Binary is 629 --
the biggest group by a factor of 2.5 over FullyConnected, and almost entirely untested.
Those 629 are the AdaLN modulation arithmetic: per block, per stream,

    norm(x) * (1 + scale) + shift        <- 2 requantisations
    x + gate * branch                    <- 2 requantisations

on tensors whose dynamic range grows 10x across the 18 blocks.

This reruns the block body with a fake-quantise after every one of those steps. Ranges
are per-tensor min-max on the live tensor, which is OPTIMISTIC (the converter calibrates
over 300 samples, measured 1.21x wider -- enc_vs_actual.py). So whatever degradation this
shows is a LOWER BOUND on the real contribution.

  usage: py -3.10 work/device/adaln_ablate.py --stage 0 [--case 0] [--bits 16]
"""

import argparse
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

BITS = 16
COUNT = [0]


def q(x):
    """A16 asymmetric min-max, the converter's default activation quantiser."""
    n = 2 ** BITS - 1
    lo, hi = float(x.min()), float(x.max())
    COUNT[0] += 1
    if hi <= lo:
        return x
    s = (hi - lo) / n
    return torch.round((x - lo) / s) * s + lo


def block_q(self, block, hidden, enc_hidden, temb, cos, sin, mask):
    """StageMMDiT._block with the modulation chain requantised, as QNN does."""
    norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hidden, emb=temb)
    norm_h = q(norm_h)

    if block.context_pre_only:
        norm_e = q(block.norm1_context(enc_hidden, temb))
        c_gate_msa = c_shift_mlp = c_scale_mlp = c_gate_mlp = None
    else:
        (norm_e, c_gate_msa, c_shift_mlp,
         c_scale_mlp, c_gate_mlp) = block.norm1_context(enc_hidden, emb=temb)
        norm_e = q(norm_e)

    attn_out, ctx_attn_out = self._attention(block, norm_h, norm_e, cos, sin, mask)

    hidden = q(hidden + q(gate_msa[:, None] * attn_out))
    norm_h = q(q(block.norm2(hidden) * (1 + scale_mlp[:, None])) + shift_mlp[:, None])
    hidden = q(hidden + q(gate_mlp[:, None] * block.ff(norm_h)))

    if block.context_pre_only:
        return None, hidden

    enc_hidden = q(enc_hidden + q(c_gate_msa[:, None] * ctx_attn_out))
    norm_e = q(q(block.norm2_context(enc_hidden) * (1 + c_scale_mlp[:, None]))
               + c_shift_mlp[:, None])
    enc_hidden = q(enc_hidden + q(c_gate_mlp[:, None] * block.ff_context(norm_e)))
    return enc_hidden, hidden


def snr(ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def main():
    global BITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--bits", type=int, default=16)
    a = ap.parse_args()
    BITS = a.bits

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

    orig = net._block
    with torch.no_grad():
        ref = net(*args_t).numpy()

    COUNT[0] = 0
    net._block = lambda *args: block_q(net, *args)
    with torch.no_grad():
        got = net(*args_t).numpy()
    net._block = orig

    print("")
    print("=" * 68)
    print("  AdaLN / modulation chain at A{}  ->  {:.2f} dB".format(
        a.bits, snr(ref, got)))
    print("  ({} requantisation points exercised)".format(COUNT[0]))
    print("")
    print("  residual boundaries only      -> 53.33 dB")
    print("  W8 weights only               -> 43.14 dB")
    print("  weights + linear + softmax    -> 43.11 dB")
    print("  device, everything quantised  -> 19.48 dB")
    print("=" * 68)


if __name__ == "__main__":
    main()
