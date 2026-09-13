"""First frame, end to end on the NPU: prompt -> CLIP L/G -> SSD1B UNet x4 -> VAE dec.

This is the first thing in the project that runs a WHOLE path on the phone rather than
one module at a time, and it is the natural first half of the E2E because every module it
needs is converted and measured:

    CLIP L        60.75 dB   FP16
    CLIP G        (fp16 14.71 dB / W8A16 rebuilding)
    SSD1B UNet    32.51 dB   W8A16, 4 denoising steps
    SSD1B VAE dec 32.43 dB   W8A16, 1.57/255 mean pixel error

Everything that is NOT a converted module stays on the host and comes from the reference
implementation, unchanged: tokenisation, the LCM scheduler's sigma/step maths, the
`time_ids`/`text_embeds` added conditioning, and the two sinusoidal projections that were
deliberately hoisted out of the UNet graph (`export_ssd1b_unet.host_embeds`).

The point is a CORRECTNESS check, not a speed one: it answers "does the converted
first-frame path produce the right image", and `--compare` scores it against the fp32
reference pipeline so the answer is a number, not an impression. Per-module latency is
measured properly elsewhere with min-of-N in one thermal session (trap #10).

  usage: py -3.10 work/pipeline/run_npu_firstframe.py --prompt "a red fox in snow"
         py -3.10 work/pipeline/run_npu_firstframe.py --compare      # vs fp32 reference
"""

import argparse
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
OUTDIR = ROOT / "work" / "device" / "e2e"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work" / "export"))
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg

from npu_graph import Adb, NpuGraph                                  # noqa: E402

TOKENS = 77
HEIGHT, WIDTH = 640, 1024
LAT_H, LAT_W = HEIGHT // 8, WIDTH // 8
TIMESTEPS = [999, 749, 499, 249]


def build_conditioning(prompt, clipg_tag):
    """Tokenise and run both CLIP encoders on the NPU.

    SDXL uses the PENULTIMATE hidden state of each encoder and concatenates them on the
    channel axis to [1,77,2048]; CLIP G's pooled `text_embeds` becomes the UNet's
    `text_embeds` added condition. The concat is done HERE, on the host, so the
    ~320-row multi-input concat trap never applies.
    """
    from transformers import CLIPTokenizer
    adb = Adb()
    tok_l = CLIPTokenizer.from_pretrained(str(MODEL / "ssd_1b_tokenizer"))
    tok_g = CLIPTokenizer.from_pretrained(str(MODEL / "ssd_1b_tokenizer_2"))

    def ids(tok):
        return tok(prompt, padding="max_length", max_length=TOKENS,
                   truncation=True, return_tensors="np").input_ids.astype(np.int32)

    # FP16 graph + integer input -> --use_native_input_files (trap #8).
    # The W8A16 clipg build is quantised, so it must NOT get that flag.
    gl = NpuGraph("clipl", ["input_ids"], ["hidden"], adb=adb, native=True,
                  tag="io_clipl")
    quantised = clipg_tag.endswith("q")
    gg = NpuGraph(clipg_tag, ["input_ids"], ["hidden", "text_embeds"], adb=adb,
                  native=not quantised, tag="io_clipg")

    hl = gl.run(input_ids=ids(tok_l))["hidden"].reshape(1, TOKENS, 768)
    og = gg.run(input_ids=ids(tok_g))
    hg = og["hidden"].reshape(1, TOKENS, 1280)
    pooled = og["text_embeds"].reshape(1, 1280)

    ehs = np.concatenate([hl, hg], axis=-1)          # [1,77,2048]
    return torch.from_numpy(ehs), torch.from_numpy(pooled), adb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red fox walking through fresh snow, "
                                        "cinematic, highly detailed")
    ap.add_argument("--clipg", default="clipgq",
                    help="clipg (FP16) or clipgq (W8A16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true",
                    help="also run the fp32 reference and score the NPU image against it")
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    from diffusers import LCMScheduler, UNet2DConditionModel
    from export_ssd1b_unet import host_embeds

    print("=" * 74)
    print("FIRST FRAME ON THE NPU  --  {}".format(a.prompt[:52]))
    print("=" * 74)

    print("")
    print("[1/4] CLIP L + CLIP G ({}) ...".format(a.clipg))
    ehs, pooled, adb = build_conditioning(a.prompt, a.clipg)
    print("      encoder_hidden_states {}  pooled {}".format(
        tuple(ehs.shape), tuple(pooled.shape)))

    # The UNet's time/added-cond sinusoids were hoisted to the host on purpose -- that is
    # the trap that cost the MMDiT 8.41 dB. Only the tiny embedding MLPs are needed here.
    print("[2/4] host: sinusoidal embeddings for {} timesteps ...".format(len(TIMESTEPS)))
    unet_cpu = UNet2DConditionModel.from_pretrained(
        str(MODEL / "ssd_1b_unet"), torch_dtype=torch.float32,
        variant="fp16" if (MODEL / "ssd_1b_unet" /
                           "diffusion_pytorch_model.fp16.safetensors").exists() else None
    ).eval()
    time_ids = torch.tensor([[HEIGHT, WIDTH, 0.0, 0.0, HEIGHT, WIDTH]])
    embeds = {t: host_embeds(unet_cpu, t, pooled, time_ids) for t in TIMESTEPS}
    del unet_cpu

    print("[3/4] SSD1B UNet x{} on the NPU ...".format(len(TIMESTEPS)))
    # EXACTLY the reference construction (run_reference.py:77, first_frame_gen.py:97):
    # set_alpha_to_one=False and steps_offset=1 change the sigma schedule, so a default
    # LCMScheduler would silently denoise differently from the fp32 reference.
    sched = LCMScheduler(set_alpha_to_one=False,
                         original_inference_steps=len(TIMESTEPS),
                         steps_offset=1)
    sched.set_timesteps(timesteps=TIMESTEPS, device="cpu")
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn(1, 4, LAT_H, LAT_W, generator=g) * sched.init_noise_sigma

    unet = NpuGraph("ssd1bunet",
                    ["sample", "t_emb", "aug_emb", "encoder_hidden_states"],
                    ["noise_pred"], adb=adb, tag="io_ssd1bunet")
    for i, t in enumerate(TIMESTEPS):
        t_emb, aug = embeds[t]
        out = unet.run(sample=lat.numpy(), t_emb=t_emb.numpy(),
                       aug_emb=aug.numpy(), encoder_hidden_states=ehs.numpy())
        noise = torch.from_numpy(out["noise_pred"]).reshape(1, 4, LAT_H, LAT_W)
        step = sched.step(noise, t, lat, generator=g)
        lat = step.prev_sample
        print("      step {}/{}  t={:4d}  latent rms {:.4f}".format(
            i + 1, len(TIMESTEPS), t, float(lat.std())))
    final = step.denoised if hasattr(step, "denoised") and step.denoised is not None else lat

    print("[4/4] SSD1B VAE decoder on the NPU ...")
    from diffusers import AutoencoderKL
    vcfg = AutoencoderKL.load_config(str(MODEL / "ssd_1b_vae"))
    sf = vcfg["scaling_factor"]
    dec = NpuGraph("ssd1bvaedec", ["latent"], ["image"], adb=adb,
                   tag="io_ssd1bvaedec")
    img = dec.run(latent=(final / sf).numpy())["image"].reshape(1, 3, HEIGHT, WIDTH)

    arr = np.clip((img[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    png = OUTDIR / "firstframe_npu.png"
    try:
        from PIL import Image
        Image.fromarray(arr).save(png)
        print("")
        print("  wrote {}  ({}x{})".format(png, arr.shape[1], arr.shape[0]))
    except Exception as e:                                    # pragma: no cover
        np.save(OUTDIR / "firstframe_npu.npy", arr)
        print("  PIL unavailable ({}), wrote .npy instead".format(e))

    print("  image range [{:.3f}, {:.3f}]  mean {:.3f}".format(
        float(img.min()), float(img.max()), float(img.mean())))
    if float(img.std()) < 1e-3:
        print("  WARNING: image is nearly constant -- the pipeline produced nothing")

    if a.compare:
        print("")
        print("[cmp] fp32 reference pipeline on the host ...")
        from run_reference import build_ssd1b
        from neodragon.utils.generation_utils import DEFAULT_PROMPT_MODIFIER
        ffg = build_ssd1b(torch.float16)
        ffg.enable_model_cpu_offload(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            pil = ffg(prompt=a.prompt + DEFAULT_PROMPT_MODIFIER,
                      num_images_per_prompt=1,
                      generator=torch.Generator(device="cpu").manual_seed(a.seed)).images[0]
        ref = np.asarray(pil).astype(np.float32)
        got = arr.astype(np.float32)
        if ref.shape == got.shape:
            e = np.linalg.norm(ref - got)
            snr = float("inf") if e == 0 else 20 * np.log10(np.linalg.norm(ref) / e)
            print("  NPU vs fp32 reference: {:.2f} dB   mean |d| {:.2f}/255".format(
                snr, float(np.abs(ref - got).mean())))
        try:
            from PIL import Image
            pil.save(OUTDIR / "firstframe_ref.png")
            print("  wrote {}".format(OUTDIR / "firstframe_ref.png"))
        except Exception:
            pass


if __name__ == "__main__":
    main()
