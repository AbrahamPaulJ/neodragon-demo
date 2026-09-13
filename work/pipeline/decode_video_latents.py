"""Decode saved video latents to actual frames, so the end-to-end number can be SEEN.

`run_npu_video.py` runs with `output_type="latent"` and saves the tensor, which makes the
NPU-vs-reference comparison exact but leaves nothing to look at. This decodes both saved
latent tensors with the same host fp32 VAE and writes frames plus a side-by-side strip, so
"7.81 dB end to end, 4.41 dB by the last frame" becomes something visual.

Uses `_decode_latent`'s exact scale/shift convention -- the first latent frame and the
video frames use DIFFERENT scale and shift factors, and getting that wrong washes the
whole video out with nothing to indicate why.

The decode runs on the HOST in fp32 for both inputs, so any difference in the output is
the MMDiT's, not the decoder's.

  usage: py -3.10 work/pipeline/decode_video_latents.py
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / "src" / "neodragon"
MODEL = ROOT / "work" / "models" / "neodragon"
E2E = ROOT / "work" / "device" / "e2e"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg


def main():
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.utils.generation_utils import (VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR,
                                                  VAE_VIDEO_SCALE_FACTOR,
                                                  VAE_VIDEO_SHIFT_FACTOR)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[load] causal video VAE (fp32) on {} ...".format(dev))
    vae = AsymmetricCausalVideoVAE.from_pretrained(
        str(MODEL / "causal_video_vae"), torch_dtype=torch.float32).to(dev).eval()

    frames = {}
    for tag in ("ref", "npu"):
        f = E2E / "video_latents_{}.npy".format(tag)
        if not f.exists():
            print("  (no {})".format(f))
            continue
        lat = torch.from_numpy(np.load(f)).float().to(dev)
        # exactly _decode_latent: frame 0 and frames 1.. use DIFFERENT factors
        if lat.shape[2] == 1:
            lat = (lat / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
        else:
            lat[:, :, :1] = (lat[:, :, :1] / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
            lat[:, :, 1:] = (lat[:, :, 1:] / VAE_VIDEO_SCALE_FACTOR) + VAE_VIDEO_SHIFT_FACTOR
        with torch.no_grad():
            vid = vae.decode(lat).sample          # [B, C, T, H, W]
        v = vid[0].permute(1, 2, 3, 0).float().cpu().numpy()   # [T, H, W, C]
        v = np.clip((v + 1.0) * 127.5, 0, 255).astype(np.uint8)
        frames[tag] = v
        print("  {:<4} decoded {} frames  {}x{}".format(tag, v.shape[0],
                                                        v.shape[2], v.shape[1]))

    if not frames:
        sys.exit("no latents found -- run run_npu_video.py first")

    from PIL import Image
    outdir = E2E / "frames"
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, v in frames.items():
        for i in range(v.shape[0]):
            Image.fromarray(v[i]).save(outdir / "{}_{:02d}.png".format(tag, i))
        Image.fromarray(v[0]).save(E2E / "{}_first.png".format(tag))
        Image.fromarray(v[-1]).save(E2E / "{}_last.png".format(tag))

    # side-by-side strip: reference on top, NPU below, a few frames across the video
    if "ref" in frames and "npu" in frames:
        r, g = frames["ref"], frames["npu"]
        n = min(r.shape[0], g.shape[0])
        pick = list(range(n))
        h, w = r.shape[1], r.shape[2]
        strip = np.zeros((h * 2 + 8, w * len(pick) + 4 * (len(pick) - 1), 3), np.uint8)
        strip[:] = 24
        for j, i in enumerate(pick):
            x = j * (w + 4)
            strip[:h, x:x + w] = r[i]
            strip[h + 8:h + 8 + h, x:x + w] = g[i]
        Image.fromarray(strip).save(E2E / "compare_strip.png")
        print("")
        print("  wrote {}  (top = fp32 reference, bottom = NPU, {} frames)".format(
            E2E / "compare_strip.png", len(pick)))

        # per-frame pixel difference, the visual analogue of the dB table
        print("")
        print("  {:>6} {:>14} {:>12}".format("frame", "mean |d| /255", "max |d|"))
        for i in range(n):
            d = np.abs(r[i].astype(np.int16) - g[i].astype(np.int16))
            print("  {:>6} {:>14.2f} {:>12}".format(i, d.mean(), int(d.max())))

    # also an animated GIF of each, easiest thing to eyeball
    for tag, v in frames.items():
        ims = [Image.fromarray(x) for x in v]
        ims[0].save(E2E / "video_{}.gif".format(tag), save_all=True,
                    append_images=ims[1:], duration=200, loop=0)
        print("  wrote {}".format(E2E / "video_{}.gif".format(tag)))


if __name__ == "__main__":
    main()
