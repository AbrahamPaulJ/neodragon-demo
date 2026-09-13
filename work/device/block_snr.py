"""Localise WHERE quantisation error enters the MMDiT, block by block.

Six hypotheses for the stage-0 accuracy gap have now died by measurement (padding,
outlier ranges, 8-bit K/V, the -100 mask -- see docs/phase5-mmdit-accuracy.md). Every
one of them was a plausible *mechanism* argued from static evidence. This script stops
arguing and measures the residual stream directly.

It reads the per-block residual tensors the device dumped via
`qnn-net-run --set_output_tensors=...` (the names come from find_block_io.py, which
derives them from the graph topology because the converter flattens all scope info),
runs the same input through the fp32 torch model, and prints SNR per block.

The shape of the answer tells you what to fix:

  * SNR degrades smoothly, a few dB per block  -> error ACCUMULATES down the residual
    chain. That is trap #3's argument, and the fix is residual scaling folded into
    weights, not better calibration.
  * SNR is flat and then falls off a cliff at block N -> one op in block N is
    responsible. Dump that block's interior and find it.
  * SNR is already bad after block 0 -> the problem is in patchify / conditioning,
    not the blocks at all.

  usage: py -3.10 work/device/block_snr.py --stage 0 [--case 0] [--dump dblk_s0]

Run AFTER: py -3.10 work/device/find_block_io.py work/device/mmdit_s0_net.json --emit
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W.parent / "export"))

# importing export_mmdit_stage also wires up sys.path for the upstream repo package
from export_mmdit_stage import (                     # noqa: E402
    StageMMDiT, envelope_for, TEXT_TOKENS, MODEL,
)
from neodragon.pyramid_mmdit import PyramidMMDiT     # noqa: E402


def load_dit(dtype=torch.float32):
    return PyramidMMDiT.from_pretrained(
        str(MODEL / "diffusion_transformer_320p"), torch_dtype=dtype).eval()


def snr(ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    n = np.linalg.norm(ref - got)
    return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(ref) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--dump", default=None, help="dir holding the device raws")
    ap.add_argument("--names", default=None,
                    help="comma-separated tensor names, in find_block_io order")
    a = ap.parse_args()

    io_dir = W / "mio_s{}".format(a.stage)
    dump = Path(a.dump) if a.dump else W / "dblk_s{}".format(a.stage)

    # ---- inputs: exactly the raws the device was fed --------------------------
    def raw(nm, shape):
        p = io_dir / "{}_{:04d}.raw".format(nm, a.case)
        return torch.from_numpy(np.fromfile(p, np.float32).reshape(shape).copy())

    env = envelope_for(a.stage)
    dtype = torch.float32
    dit = load_dit(dtype)
    net = StageMMDiT(dit, env).eval()
    n_img = net.n_img

    ehs = raw("encoder_hidden_states", (1, TEXT_TOKENS, 1536))
    pooled = raw("pooled_projections", (1, 2048))
    tstep = raw("timestep_ratio", (1,))
    mask = raw("attn_mask", (1, TEXT_TOKENS + n_img, TEXT_TOKENS + n_img))
    cos = raw("rope_cos", (TEXT_TOKENS + n_img, 1, 32))
    sin = raw("rope_sin", (TEXT_TOKENS + n_img, 1, 32))
    lats = [raw("latent_{}".format(i), (1, 16, t, h, w))
            for i, (t, h, w) in enumerate(env)]

    # ---- capture the residual stream after every block ------------------------
    caught = []
    orig = net._block

    def spy(block, hidden, enc_hidden, temb, c, s, m):
        enc, hid = orig(block, hidden, enc_hidden, temb, c, s, m)
        caught.append((hid.detach().clone(),
                       None if enc is None else enc.detach().clone()))
        return enc, hid

    net._block = spy
    with torch.no_grad():
        net(ehs, pooled, tstep, mask, cos, sin, *lats)

    n_blocks = len(caught)
    print("=" * 74)
    print("MMDiT stage {} -- per-block residual SNR, device vs fp32 (case {})".format(
        a.stage, a.case))
    print("  {} blocks, image stream [1,{},1536], text stream [1,{},1536]".format(
        n_blocks, n_img, TEXT_TOKENS))
    print("=" * 74)

    if not dump.is_dir():
        sys.exit("no device dump at {}\n"
                 "  1) py -3.10 work/device/find_block_io.py "
                 "work/device/mmdit_s{}_net.json --emit\n"
                 "  2) qnn-net-run --set_output_tensors=<that list> ...\n"
                 "  3) adb pull the result dir here".format(dump, a.stage))

    names = a.names.split(",") if a.names else None
    if not names:
        sys.exit("pass --names with the find_block_io.py --emit output so the raws "
                 "can be matched to blocks")

    img_names = names[:n_blocks]
    txt_names = names[n_blocks:]

    print("")
    print("  {:>5} {:>12} {:>12}   {}".format("block", "image dB", "text dB", "note"))
    print("  " + "-" * 56)
    prev = None
    for b in range(n_blocks):
        f = dump / (img_names[b] + ".raw")
        if not f.exists():
            print("  {:>5}  MISSING {}".format(b, f.name))
            continue
        got = np.fromfile(f, np.float32).reshape(1, n_img, 1536)
        s_img = snr(caught[b][0].numpy(), got)
        s_txt = float("nan")
        if b < len(txt_names) and caught[b][1] is not None:
            ft = dump / (txt_names[b] + ".raw")
            if ft.exists():
                gt = np.fromfile(ft, np.float32).reshape(1, TEXT_TOKENS, 1536)
                s_txt = snr(caught[b][1].numpy(), gt)
        note = "" if prev is None else "{:+.2f} dB".format(s_img - prev)
        prev = s_img
        print("  {:>5} {:>12.2f} {:>12.2f}   {}".format(b, s_img, s_txt, note))


if __name__ == "__main__":
    main()
