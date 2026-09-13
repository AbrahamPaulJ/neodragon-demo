"""fp32 references for the SSD1B UNet's held-out test set.

`capture_unet_calib.py --split test` stores the graph's INPUTS; this runs the exported
graph's fp32 equivalent on them and writes `uref_XXXX.raw`, so the device output has
something to be scored against. Same role `nref_*.raw` plays for the MMDiT.

The reference comes from HoistedUNet -- the same rewrite that is exported -- not from the
stock module, so the comparison measures quantisation alone. That rewrite is bit-exact
against the stock model anyway (max|diff| 0.000e+00, verified in export_ssd1b_unet.py).

  usage: py -3.10 work/device/make_unet_refs.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

W = Path(__file__).resolve().parent
ROOT = W.parent.parent
sys.path.insert(0, str(ROOT / "work" / "export"))

SHAPES = {"sample": (1, 4, 80, 128), "t_emb": (1, 320),
          "aug_emb": (1, 1280), "encoder_hidden_states": (1, 77, 2048)}


def main():
    from export_ssd1b_unet import HoistedUNet, MODEL
    from diffusers import UNet2DConditionModel

    io = W / "io_ssd1bunet"
    n = len(list(io.glob("sample_*.raw")))
    assert n, "no test inputs -- run capture_unet_calib.py --split test"
    print("[load] ssd_1b_unet (fp32, CPU) ...")
    unet = UNet2DConditionModel.from_pretrained(
        str(MODEL / "ssd_1b_unet"), torch_dtype=torch.float32).eval()
    net = HoistedUNet(unet).eval()

    for c in range(n):
        args = [torch.from_numpy(
            np.fromfile(io / "{}_{:04d}.raw".format(k, c), np.float32).reshape(s).copy())
            for k, s in SHAPES.items()]
        with torch.no_grad():
            r = net(*args)
        np.ascontiguousarray(r.numpy(), dtype=np.float32).tofile(
            io / "uref_{:04d}.raw".format(c))
        print("  case {:2d}  out {}  rms {:.4f}".format(
            c, tuple(r.shape), float(r.std())))
    print("")
    print("wrote {} references -> {}".format(n, io))


if __name__ == "__main__":
    main()
