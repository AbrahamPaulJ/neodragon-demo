"""Run the reference Neodragon hybrid pipeline on a 6 GB GPU and capture the
real latents that reach the VAE decoder.

Why not the stock script: scripts/inference_neodragon.py does a blanket
`.to(device)` on the whole pipeline (~5.4 GB fp16 for the main modules plus
~4.5 GB of SSD1B), which will not fit 6 GB. And NeodragonPipeline.from_pretrained
insists on snapshot_download into an HF cache layout, re-fetching the 10 GB we
already have under work/models/neodragon. So this builds the components directly
from the local directory and manages residency itself.

Residency plan (hybrid runs SSD1B once, then the DiT loop, then decode):
  resident on GPU : dit + context_adapter + vae   (~3.5 GB fp16)
  offloaded       : text_encoder_bundle, SSD1B pipeline (used once each)

Capture point: a forward pre-hook on vae.post_quant_conv. decode() is
`post_quant_conv(z) -> decoder(z)`, and our exported ONNX graph is exactly that
pair, so the hook input is precisely the graph's `latent` input -- already
scale/shift-corrected by _decode_latent.
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

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "src" / "neodragon"
LOCAL = ROOT / "work" / "models" / "neodragon"
OUT = ROOT / "work" / "calib" / "vae_dec"
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg


def vram(tag):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.max_memory_reserved() / 1024**3
        print(f"    [vram] {tag:<28} alloc {a:5.2f} GiB   peak reserved {r:5.2f} GiB")


def build_ssd1b(dtype):
    """Assemble the SSD1B first-frame pipeline with the right variant per part."""
    from diffusers import AutoencoderKL, LCMScheduler, UNet2DConditionModel
    from neodragon.first_frame_gen import (DEFAULT_INFERENCE_PARAMS,
                                           SSD1B_FirstFrameGeneratorPipeline)
    from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                              CLIPTokenizer)

    def pick(sub, stem):
        """fp16 twin if we have it, else the plain file."""
        d = LOCAL / sub
        return {"variant": "fp16"} if (d / f"{stem}.fp16.safetensors").exists() else {}

    vae = AutoencoderKL.from_pretrained(
        LOCAL / "ssd_1b_vae", torch_dtype=dtype,
        **pick("ssd_1b_vae", "diffusion_pytorch_model"))
    unet = UNet2DConditionModel.from_pretrained(
        LOCAL / "ssd_1b_unet", torch_dtype=dtype,
        **pick("ssd_1b_unet", "diffusion_pytorch_model"))
    text_encoder = CLIPTextModel.from_pretrained(
        LOCAL / "ssd_1b_text_encoder", torch_dtype=dtype,
        **pick("ssd_1b_text_encoder", "model"))
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        LOCAL / "ssd_1b_text_encoder_2", torch_dtype=dtype,
        **pick("ssd_1b_text_encoder_2", "model"))
    tokenizer = CLIPTokenizer.from_pretrained(LOCAL / "ssd_1b_tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(LOCAL / "ssd_1b_tokenizer_2")
    scheduler = LCMScheduler(
        set_alpha_to_one=False,
        original_inference_steps=len(DEFAULT_INFERENCE_PARAMS["timesteps"]),
        steps_offset=1)

    return SSD1B_FirstFrameGeneratorPipeline(
        vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, unet=unet,
        scheduler=scheduler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--offload", default="model", choices=["model", "sequential", "none"])
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.context_adapter import ContextAdapter
    from neodragon.first_frame_gen import SSD1B_FirstFrameGeneratorPipeline
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import generate_hybrid

    print("=" * 74)
    print(f"Reference hybrid pipeline  dtype={args.dtype}  offload={args.offload}")
    print("=" * 74)

    print("\n[load] components from", LOCAL)
    teb = TextEncoderBundle.from_pretrained(str(LOCAL), torch_dtype=dtype)
    ca = ContextAdapter.from_pretrained(str(LOCAL / "context_adapter"), torch_dtype=dtype)
    dit = PyramidMMDiT.from_pretrained(str(LOCAL / "diffusion_transformer_320p"),
                                       torch_dtype=dtype)
    vae = AsymmetricCausalVideoVAE.from_pretrained(str(LOCAL / "causal_video_vae"),
                                                   torch_dtype=dtype)
    sched = PyramidFlowMatchEulerDiscreteScheduler()

    # SSD1B_FirstFrameGeneratorPipeline.from_pretrained forwards one **kwargs to
    # every component, but only vae / text_encoder / text_encoder_2 ship a
    # .fp16.safetensors twin -- ssd_1b_unet has none. Passing variant="fp16"
    # uniformly would fail on the UNet, so build it component-wise instead
    # (cheaper than downloading the 3.6 GB of fp32 twins we skipped).
    ffg = build_ssd1b(dtype)

    for name, m in (("text_encoder_bundle", teb), ("context_adapter", ca),
                    ("dit", dit), ("vae", vae)):
        n = sum(p.numel() for p in m.parameters())
        print(f"    {name:<20} {n/1e6:8.1f} M params  "
              f"{n*dtype.itemsize/1024**3:5.2f} GiB @ {args.dtype}")

    # ---- residency ----------------------------------------------------------
    if args.offload == "none":
        for m in (teb, ca, dit, vae):
            m.to(dev)
        ffg.to(dev)
    else:
        # keep the DiT loop resident, page the once-per-video modules
        ca.to(dev)
        dit.to(dev)
        vae.to(dev)
        if args.offload == "sequential":
            from accelerate import cpu_offload
            cpu_offload(teb, execution_device=dev)
            for sub in ("unet", "text_encoder", "text_encoder_2", "vae"):
                if hasattr(ffg, sub) and getattr(ffg, sub) is not None:
                    cpu_offload(getattr(ffg, sub), execution_device=dev)
        else:
            ffg.enable_model_cpu_offload(device=dev)
            from accelerate import cpu_offload
            cpu_offload(teb, execution_device=dev)
    vram("after residency setup")

    # ---- capture ------------------------------------------------------------
    # Ask for output_type="latent" so the decoder never runs: decoding the full
    # [1,16,7,40,64] needs a 1.17 GiB fp16 intermediate on a card already
    # holding the DiT, and we only want its *input* anyway. Apply the same
    # scale/shift _decode_latent does, so what we save is exactly the tensor the
    # exported ONNX graph takes as `latent`.
    from neodragon.utils.generation_utils import (VAE_SCALE_FACTOR, VAE_SHIFT_FACTOR,
                                                  VAE_VIDEO_SCALE_FACTOR,
                                                  VAE_VIDEO_SHIFT_FACTOR)

    def to_decoder_input(lat):
        lat = lat.detach().float().clone()
        if lat.shape[2] == 1:
            return (lat / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
        lat[:, :, :1] = (lat[:, :, :1] / VAE_SCALE_FACTOR) + VAE_SHIFT_FACTOR
        lat[:, :, 1:] = (lat[:, :, 1:] / VAE_VIDEO_SCALE_FACTOR) + VAE_VIDEO_SHIFT_FACTOR
        return lat

    prompts = [ln.strip() for ln in
               (REPO / "prompts" / "showcase_prompts.txt").read_text(encoding="utf-8")
               .splitlines() if ln.strip()][: args.num_prompts]

    n_saved = 0
    for i, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:64]}")
        try:
            with torch.no_grad():
                latents = generate_hybrid(
                    first_frame_gen_pipeline=ffg,
                    text_encoder_bundle=teb,
                    dit=dit,
                    context_adapter=ca,
                    vae=vae,
                    scheduler=sched,
                    prompt=prompt,
                    height=320, width=512, num_frames=49,
                    num_inference_steps=[1, 1, 1],
                    video_num_inference_steps=[1, 1, 1],
                    do_classifier_free_guidance=False,
                    guidance_scale=0.0, video_guidance_scale=0.0,
                    frames_per_unit=1, num_stages=3,
                    output_type="latent", profile=False,
                    device=dev, dtype=dtype,
                )
        except torch.cuda.OutOfMemoryError as e:
            print(f"    OOM: {e}")
            print("    retry with --offload sequential")
            break
        vram("after generation")

        z = to_decoder_input(latents).cpu()
        print(f"    latents {tuple(z.shape)}  "
              f"min={z.min():.3f} max={z.max():.3f} std={z.std():.3f}")
        # split along T so each sample matches the exported graph's
        # [1,16,1,40,64] input
        for t in range(z.shape[2]):
            s = z[:, :, t : t + 1].contiguous().numpy().astype(np.float32)
            s.tofile(OUT / f"latent_{n_saved:04d}.raw")
            n_saved += 1
        print(f"    -> {n_saved} samples total")
        del latents, z
        torch.cuda.empty_cache()

    print(f"\nwrote {n_saved} calibration latents to {OUT}")
    if n_saved:
        arr = np.fromfile(OUT / "latent_0000.raw", dtype=np.float32)
        print(f"  sample shape {arr.size} floats = {arr.size/(16*40*64):.0f} x [1,16,1,40,64]")


if __name__ == "__main__":
    main()
