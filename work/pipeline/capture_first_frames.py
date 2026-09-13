"""Capture real VAE-encoder inputs: SSD1B first frames, exactly as generate() sees them.

Phase 4b calibration. Trap #4 says calibration must use real pipeline tensors,
so this reproduces the exact preprocessing chain from generation_utils.py:526-530:

    first_frame = ssd1b(prompt + DEFAULT_PROMPT_MODIFIER).images[0]   # 1024x640 PIL
    image = image.resize((width, height), resample=Image.LANCZOS)     # 512x320
    image = _pil_to_numpy(image)                                      # /255 * 2 - 1
    image = rearrange(image, "(b t) h w c -> b c t h w", b=1, t=1)    # [1,3,1,320,512]

The DiT is NOT needed: the encoder's input depends only on SSD1B and the resize,
so this skips the 35 s/video AR loop entirely and runs ~2 s/image.

Two disjoint sets so the deploy number is honest:
  --split calib  vbench_prompts.txt     -> work/calib/vae_enc/
  --split test   showcase_prompts.txt   -> work/device/eio/

  usage: py -3.10 work/pipeline/capture_first_frames.py --split calib --num 64
"""

import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
LOCAL = ROOT / "work" / "models" / "neodragon"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

HEIGHT, WIDTH = 320, 512

SPLITS = {
    "calib": ("vbench_prompts.txt", ROOT / "work" / "calib" / "vae_enc"),
    "test": ("showcase_prompts.txt", ROOT / "work" / "device" / "eio"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="calib", choices=sorted(SPLITS))
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--png", action="store_true", help="also write PNGs for eyeballing")
    args = ap.parse_args()

    prompt_file, out = SPLITS[args.split]
    out.mkdir(parents=True, exist_ok=True)

    from run_reference import build_ssd1b
    from neodragon.utils.generation_utils import (DEFAULT_PROMPT_MODIFIER,
                                                  _pil_to_numpy)

    prompts = [ln.strip() for ln in
               (REPO / "prompts" / prompt_file).read_text(encoding="utf-8").splitlines()
               if ln.strip()]
    print("=" * 74)
    print("VAE ENCODER CALIBRATION CAPTURE  split={}  {} prompts available".format(
        args.split, len(prompts)))
    print("=" * 74)

    dev = torch.device("cuda")
    ffg = build_ssd1b(torch.float16)
    ffg.enable_model_cpu_offload(device=dev)

    n = 0
    for i in range(args.num):
        prompt = prompts[i % len(prompts)]
        gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
        with torch.no_grad():
            pil = ffg(prompt=prompt + DEFAULT_PROMPT_MODIFIER,
                      num_images_per_prompt=1, generator=gen).images[0]

        pil = pil.resize((WIDTH, HEIGHT), resample=Image.LANCZOS)
        arr = _pil_to_numpy(pil)                       # [1, H, W, 3] in [-1, 1]
        x = np.ascontiguousarray(arr.transpose(0, 3, 1, 2)).astype(np.float32)
        assert x.shape == (1, 3, HEIGHT, WIDTH), x.shape
        assert -1.0001 <= x.min() and x.max() <= 1.0001, (x.min(), x.max())
        x.tofile(out / "image_{:04d}.raw".format(n))
        if args.png:
            pil.save(out / "image_{:04d}.png".format(n))
        n += 1
        if n % 8 == 0 or n == args.num:
            print("  [{:3d}/{}] {:.55s}  min={:.3f} max={:.3f} std={:.3f}".format(
                n, args.num, prompt, x.min(), x.max(), x.std()))

    print("")
    print("wrote {} first-frame tensors to {}".format(n, out))
    print("  each [1, 3, {}, {}] float32 = {:.2f} MB".format(
        HEIGHT, WIDTH, 3 * HEIGHT * WIDTH * 4 / 1e6))


if __name__ == "__main__":
    main()
