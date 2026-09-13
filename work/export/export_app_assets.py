"""Export the host-side weights and tables the Android app needs.

Everything the phone must compute OUTSIDE a converted graph lives here. That is a short
list on purpose: the sinusoidal projections were deliberately hoisted out of the exported
graphs (leaving them in cost the MMDiT 8.41 dB -- quantising a `Sin` at its argument,
which is encoded over [0,1000], caps you at ~47 dB), and the tokenizer and scheduler were
never in a graph to begin with.

For the first-frame path that leaves exactly one weighted module on the host: SSD1B's
`add_embedding`, the little MLP that folds CLIP G's pooled text embedding together with
the six sinusoidally-projected `time_ids` into the UNet's `aug_emb`. `time_proj` and
`add_time_proj` carry no weights at all -- they are pure sinusoids, computed in Kotlin.

Output is a flat `.ndw` blob, read by AppAssets.kt:

    "NDW1" | uint32 count | { uint32 nameLen, name, uint32 ndim, dims..., float32 data }

  usage: py -3.10 work/export/export_app_assets.py
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "work" / "models" / "neodragon"
ASSETS = ROOT / "work" / "android" / "app" / "src" / "main" / "assets"


def write_ndw(path, tensors):
    """tensors: dict name -> np.ndarray (float32)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"NDW1")
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))
            f.write(arr.tobytes())
    mb = path.stat().st_size / 1048576
    print("  wrote {}  ({:.2f} MB, {} tensors)".format(path.name, mb, len(tensors)))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("[1/3] SSD1B add_embedding ...")
    from diffusers import UNet2DConditionModel
    unet_dir = MODEL / "ssd_1b_unet"
    variant = "fp16" if (unet_dir / "diffusion_pytorch_model.fp16.safetensors").exists() \
        else None
    unet = UNet2DConditionModel.from_pretrained(
        str(unet_dir), torch_dtype=torch.float32, variant=variant).eval()

    ten = {}
    for i, lin in ((1, unet.add_embedding.linear_1), (2, unet.add_embedding.linear_2)):
        ten["add_embedding.linear_{}.weight".format(i)] = lin.weight.detach().numpy()
        ten["add_embedding.linear_{}.bias".format(i)] = lin.bias.detach().numpy()
    for k, v in ten.items():
        print("      {:38s} {}".format(k, tuple(v.shape)))
    write_ndw(ASSETS / "ssd1b_addembed.ndw", ten)
    del unet

    print("[2/3] tokenizer (both SDXL tokenizers share one vocab -- verified by md5) ...")
    for src, dst in ((MODEL / "ssd_1b_tokenizer" / "vocab.json", "clip_vocab.json"),
                     (MODEL / "ssd_1b_tokenizer" / "merges.txt", "clip_merges.txt")):
        data = src.read_bytes()
        (ASSETS / dst).write_bytes(data)
        print("  wrote {}  ({:.2f} MB)".format(dst, len(data) / 1048576))

    print("[3/3] constants ...")
    vcfg = json.loads((MODEL / "ssd_1b_vae" / "config.json").read_text())
    ucfg = json.loads((unet_dir / "config.json").read_text())
    cfg = {
        # LCMScheduler(set_alpha_to_one=False, original_inference_steps=4,
        #              steps_offset=1) -- the reference's EXACT construction. A default
        # LCMScheduler denoises differently and the difference is silent.
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "timestep_scaling": 10.0,
        "sigma_data": 0.5,
        "set_alpha_to_one": False,
        "steps_offset": 1,
        "timesteps": [999, 749, 499, 249],
        # UNet sinusoid parameters
        "time_proj_dim": ucfg["block_out_channels"][0],          # 320
        "add_time_proj_dim": ucfg["addition_time_embed_dim"],    # 256
        "flip_sin_to_cos": ucfg.get("flip_sin_to_cos", True),
        "freq_shift": ucfg.get("freq_shift", 0),
        "vae_scaling_factor": vcfg["scaling_factor"],
        "height": 640,
        "width": 1024,
        "tokens": 77,
        "pad_id_clip_l": 49407,   # <|endoftext|>
        "pad_id_clip_g": 0,       # "!"  -- SDXL's tokenizer_2 pads with id 0, NOT eos
        "bos_id": 49406,
        "eos_id": 49407,
    }
    (ASSETS / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))
    print("  wrote pipeline_config.json")

    total = sum(p.stat().st_size for p in ASSETS.iterdir() if p.is_file())
    print("")
    print("assets total {:.2f} MB in {}".format(total / 1048576, ASSETS))


if __name__ == "__main__":
    sys.exit(main())
