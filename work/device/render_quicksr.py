#!/usr/bin/env python3
"""Render the QuickSRNet result so it can be LOOKED AT, not just scored.

28.06 dB against the fp32 model sounds bad next to the paper's 48 dB, but SNR against a
reference is not the same question as "does the upscale look right". There is no ground
truth at 640x1024 here -- the frames are natively 320x512 -- so the useful comparison is
what a viewer would actually see:

    source (320x512, nearest-upscaled so it fills the same box)
    bilinear 2x            -- the free alternative; if QuickSRNet does not beat this,
                              377 KB and ~200 ms per video buy nothing
    QuickSRNet fp32        -- what the model is capable of
    QuickSRNet on device   -- what W8A16 + CLE actually ships

  usage: py -3.10 work/device/render_quicksr.py [--frame 0] [--crop 256]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work" / "device"
H, WD, S = 320, 512, 2


def to_img(arr):
    """[3,H,W] float in [0,1] -> PIL RGB."""
    a = np.clip(arr, 0, 1)
    return Image.fromarray((a.transpose(1, 2, 0) * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--crop", type=int, default=0,
                    help="side of a centre crop taken at OUTPUT scale; 0 = whole frame")
    ap.add_argument("--out", default="quicksr_compare.png")
    a = ap.parse_args()

    src = np.fromfile(W / "mio_quicksr" / f"frame_{a.frame:04d}.raw",
                      dtype=np.float32).reshape(3, H, WD)

    res = W / "dout_quicksr" / f"Result_{a.frame}"
    dev = np.fromfile(next(res.glob("*.raw")), dtype=np.float32).reshape(3, H * S, WD * S)

    import torch
    from qai_hub_models.utils.asset_loaders import always_answer_prompts
    with always_answer_prompts(True):
        from qai_hub_models.models.quicksrnetmedium.model import QuickSRNetMedium
        m = QuickSRNetMedium.from_pretrained(scale_factor=S).eval()
    with torch.no_grad():
        ref = m(torch.from_numpy(src[None])).numpy()[0]

    src_img = to_img(src)
    panels = [
        ("source 320x512 (nearest)", src_img.resize((WD * S, H * S), Image.NEAREST)),
        ("bilinear 2x", src_img.resize((WD * S, H * S), Image.BILINEAR)),
        ("QuickSRNet fp32", to_img(ref)),
        ("QuickSRNet on device (W8A16)", to_img(dev)),
    ]

    if a.crop:
        c = a.crop
        cx, cy = (WD * S - c) // 2, (H * S - c) // 2
        panels = [(t, im.crop((cx, cy, cx + c, cy + c))) for t, im in panels]

    pw, ph = panels[0][1].size
    pad, bar = 8, 22
    sheet = Image.new("RGB", (pw * len(panels) + pad * (len(panels) + 1),
                              ph + bar + pad * 2), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    for i, (title, im) in enumerate(panels):
        x = pad + i * (pw + pad)
        sheet.paste(im, (x, pad + bar))
        d.text((x + 2, pad + 4), title, fill=(220, 220, 230))
    out = W / a.out
    sheet.save(out)
    print(f"wrote {out}  ({sheet.size[0]}x{sheet.size[1]})")

    # Numbers alongside the picture, so the visual and the metric are read together.
    def snr(x, y):
        n = np.linalg.norm(x - y)
        return float("inf") if n == 0 else 20 * np.log10(np.linalg.norm(x) / n)

    bil = np.asarray(src_img.resize((WD * S, H * S), Image.BILINEAR),
                     dtype=np.float32).transpose(2, 0, 1) / 255.0
    print(f"  device vs fp32 QuickSRNet : {snr(ref, dev):6.2f} dB")
    print(f"  bilinear vs fp32 QuickSRNet: {snr(ref, bil):6.2f} dB "
          f"  <- how far the free alternative is from the model")
    print(f"  device vs bilinear         : {snr(bil, dev):6.2f} dB")


if __name__ == "__main__":
    main()
