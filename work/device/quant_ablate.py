"""Attribute the MMDiT's quantisation error to a GROUP, on the host, in fp32 torch.

Companion to resid_ablate.py, generalised. Runs the stage graph with one group of
tensors fake-quantised and everything else exact, so each group's contribution to the
deploy SNR can be read off independently. No device, no NDK, no conversion.

Groups:
  weights   W8 per-output-channel on every nn.Linear in the blocks. This is the
            "W8" of W8A16 and covers 255 FullyConnected ops -- by far the largest
            parameter mass in the graph (1512 M params).
  residual  the 35 block-boundary tensors, using their real net.json encodings
            (this is resid_ablate.py; measured 53.33 dB, i.e. not the cause)
  linear    A16 on every nn.Linear OUTPUT, min-max per tensor
  softmax   A16 on every attention weight matrix, encoded over [0,1] as the
            converter does

Weight quantisation uses per-output-channel symmetric min-max, matching
--use_per_channel_quantization --weights_bitwidth 8. It is applied to a COPY; the
model is restored afterwards.

  usage: py -3.10 work/device/quant_ablate.py --group weights [--stage 0] [--case 0]
         py -3.10 work/device/quant_ablate.py --group all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W.parent / "export"))

from export_mmdit_stage import (                     # noqa: E402
    StageMMDiT, envelope_for, TEXT_TOKENS, MODEL,
)
from neodragon.pyramid_mmdit import PyramidMMDiT     # noqa: E402


def snr(ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def q_perchannel_sym(w, bits=8):
    """Symmetric per-output-channel min-max, as --use_per_channel_quantization does."""
    n = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    s = flat.abs().amax(dim=1) / n
    s = torch.where(s == 0, torch.ones_like(s), s)
    shape = (-1,) + (1,) * (w.dim() - 1)
    return (torch.round(w / s.reshape(shape)).clamp(-n - 1, n) * s.reshape(shape))


def q_minmax(x, bits=16):
    n = 2 ** bits - 1
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return x
    s = (hi - lo) / n
    return torch.round((x - lo) / s) * s + lo


def q_unit(x, bits=16):
    """A16 over a fixed [0,1] range -- how the converter encodes softmax outputs."""
    n = 2 ** bits - 1
    return torch.round(x.clamp(0, 1) * n) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--group", default="weights",
                    choices=["weights", "linear", "softmax", "all", "none"])
    ap.add_argument("--name", default="",
                    help="variant, e.g. mmdit_s0f (its io dir supplies temb_act)")
    a = ap.parse_args()

    suf = (a.name[6:] if a.name.startswith("mmdit_") else a.name) or "s{}".format(a.stage)
    io_dir = W / "mio_{}".format(suf)
    host_temb = (io_dir / "temb_act_0000.raw").exists()

    def raw(nm, shape):
        p = io_dir / "{}_{:04d}.raw".format(nm, a.case)
        return torch.from_numpy(np.fromfile(p, np.float32).reshape(shape).copy())

    env = envelope_for(a.stage)
    print("[load] diffusion_transformer_320p (fp32, CPU) ...")
    dit = PyramidMMDiT.from_pretrained(
        str(MODEL / "diffusion_transformer_320p"), torch_dtype=torch.float32).eval()
    net = StageMMDiT(dit, env, temb_host=host_temb).eval()
    S = TEXT_TOKENS + net.n_img

    cond = ((raw("temb_act", (1, 1536)),) if host_temb else
            (raw("pooled_projections", (1, 2048)), raw("timestep_ratio", (1,))))
    args_t = (raw("encoder_hidden_states", (1, TEXT_TOKENS, 1536)),
              *cond,
              raw("attn_mask", (1, S, S)),
              raw("rope_cos", (S, 1, 32)),
              raw("rope_sin", (S, 1, 32)),
              *[raw("latent_{}".format(i), (1, 16, t, h, w))
                for i, (t, h, w) in enumerate(env)])

    with torch.no_grad():
        ref = net(*args_t).numpy()

    # Every nn.Linear the GRAPH actually contains. `dit.transformer_blocks` alone is
    # wrong for the session-6 build: the 214 split AdaLN modulation Linears hang off
    # `net` (see SplitMod), and `dit.time_text_embed` is no longer in the graph at all
    # once it is hoisted -- so quantising by module tree, not by hand.
    skip = set()
    if host_temb:
        skip |= {id(m) for m in dit.time_text_embed.modules()}
    seen, linears = set(), []
    for m in net.modules():
        if isinstance(m, nn.Linear) and id(m) not in skip and id(m.weight) not in seen:
            seen.add(id(m.weight))
            linears.append(m)
    print("[info] {} nn.Linear in the graph, {:.1f} M weight params".format(
        len(linears), sum(m.weight.numel() for m in linears) / 1e6))

    handles, saved = [], []
    g = a.group

    if g in ("weights", "all"):
        with torch.no_grad():
            for m in linears:
                saved.append((m, m.weight.detach().clone()))
                m.weight.copy_(q_perchannel_sym(m.weight, 8))
    if g in ("linear", "all"):
        handles += [m.register_forward_hook(lambda _m, _i, o: q_minmax(o, 16))
                    for m in linears]
    if g in ("softmax", "all"):
        orig_sm = torch.softmax

        def patched(x, dim=-1, **kw):
            return q_unit(orig_sm(x, dim=dim, **kw), 16)
        torch.softmax = patched

    with torch.no_grad():
        got = net(*args_t).numpy()

    if g in ("softmax", "all"):
        torch.softmax = orig_sm
    for h in handles:
        h.remove()
    with torch.no_grad():
        for m, w in saved:
            m.weight.copy_(w)

    print("")
    print("=" * 66)
    print("  group '{}'  ->  {:.2f} dB".format(g, snr(ref, got)))
    print("  device, session-5 baseline mmdit_s0   ->  19.48 dB")
    print("  device, session-6 fix    mmdit_s0f      ->  27.89 dB")
    print("  residual-only (resid_ablate.py) -> 53.33 dB")
    print("=" * 66)


if __name__ == "__main__":
    main()
