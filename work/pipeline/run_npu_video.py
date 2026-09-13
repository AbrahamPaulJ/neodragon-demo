"""49-frame video with the MMDiT running on the NPU.

`generate_hybrid()` takes every heavy module as an argument, so the AR loop, the pyramid
scheduler, the history pyramid and the per-stage sigmas all stay EXACTLY as the reference
implements them -- only the modules are swapped. That keeps the first video honest: any
difference from the reference is quantisation, not a reimplemented scheduler.

Swapped in by default:
    dit             -> NpuPyramidMMDiT  (18 calls per video, ~90% of the compute)
    context_adapter -> NpuContextAdapter (FP16, 39.36 dB)

Left on the host for now, each behind a flag:
    text_encoder_bundle  DistilT5 is converted (49.04 dB) but tiny -- 12.84 ms
    first_frame_gen      the SSD1B path already runs on the NPU via
                         run_npu_firstframe.py; wiring it into generate_hybrid needs a
                         pipeline-shaped shim, which is not what this script is proving
    vae                  encode seeds the loop; decode is the streaming graph with 9
                         MemBlock state tensors (see NpuVaeDecoder)

So this answers the question the per-module numbers cannot: **what does quantisation do
across 18 chained MMDiT calls, where each unit conditions on the previous one?** Every
stage SNR so far was measured against clean fp32 conditioning.

  usage: py -3.10 work/pipeline/run_npu_video.py --prompt "..." --frames 49
         py -3.10 work/pipeline/run_npu_video.py --reference   # fp32, for comparison
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
sys.path.insert(0, str(ROOT / "work" / "audit"))
sys.path.insert(0, str(REPO))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a red fox walking through fresh snow, cinematic")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", action="store_true",
                    help="run everything in fp32 on the host (no NPU) for comparison")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = a.tag or ("ref" if a.reference else "npu")

    from run_reference import build_ssd1b
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import generate_hybrid

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if not torch.cuda.is_available() else torch.float16

    print("=" * 76)
    print("{} VIDEO  --  {}".format("REFERENCE fp32" if a.reference else "NPU MMDiT",
                                    a.prompt[:48]))
    print("=" * 76)

    print("[load] host modules ...")
    teb = TextEncoderBundle.from_pretrained(str(MODEL), torch_dtype=dtype)
    ca = ContextAdapter.from_pretrained(str(MODEL / "context_adapter"), torch_dtype=dtype)
    dit = PyramidMMDiT.from_pretrained(str(MODEL / "diffusion_transformer_320p"),
                                       torch_dtype=dtype)
    vae = AsymmetricCausalVideoVAE.from_pretrained(str(MODEL / "causal_video_vae"),
                                                   torch_dtype=dtype)
    sched = PyramidFlowMatchEulerDiscreteScheduler()
    ffg = build_ssd1b(torch.float16)

    if a.reference:
        for m in (teb, ca, dit, vae):
            m.to(dev)
        ffg.enable_model_cpu_offload(device=dev)
        dit_mod, ca_mod = dit, ca
    else:
        from npu_modules import NpuContextAdapter, NpuPyramidMMDiT
        # the DiT stays on the CPU: the adapter uses it ONLY for the host-side preamble
        # (_prepare_temporal_rope_ids, temp_rope_embed, time_text_embed), never for the
        # blocks -- those run on the phone.
        dit_cpu = dit.to("cpu").float()
        teb.to(dev)
        vae.to(dev)
        ffg.enable_model_cpu_offload(device=dev)
        dit_mod = NpuPyramidMMDiT(dit_cpu)
        ca_mod = NpuContextAdapter()
        print("[npu]  MMDiT -> mmdit_s0g / mmdit_s1f / mmdit_s2f")
        print("[npu]  ContextAdapter -> ctxadaptfp16")

    # generate_hybrid() takes no `generator`; the first-frame noise AND the video noise
    # both come from the GLOBAL torch RNG. Without seeding it here, two runs produce two
    # DIFFERENT videos and any comparison between them is meaningless -- the first
    # attempt scored 0.11 dB for exactly that reason, with the host-computed first frame
    # already at 2.68 dB when it should have been identical.
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    import time
    t0 = time.time()
    with torch.no_grad():
        latents = generate_hybrid(
            first_frame_gen_pipeline=ffg, text_encoder_bundle=teb,
            dit=dit_mod, context_adapter=ca_mod, vae=vae, scheduler=sched,
            prompt=a.prompt, height=320, width=512, num_frames=a.frames,
            num_inference_steps=[1, 1, 1], video_num_inference_steps=[1, 1, 1],
            do_classifier_free_guidance=False, guidance_scale=0.0,
            video_guidance_scale=0.0, frames_per_unit=1, num_stages=3,
            output_type="latent", profile=False, device=dev, dtype=dtype,
        )
    el = time.time() - t0

    lat = latents if torch.is_tensor(latents) else latents[0]
    out = OUTDIR / "video_latents_{}.npy".format(tag)
    np.save(out, lat.detach().float().cpu().numpy())
    print("")
    print("  latents {}  rms {:.4f}  -> {}".format(
        tuple(lat.shape), float(lat.float().std()), out))
    print("  wall clock: {:.1f} s".format(el))
    if not a.reference:
        print("  MMDiT NPU calls: {}".format(dit_mod.calls))

    ref = OUTDIR / "video_latents_ref.npy"
    if not a.reference and ref.exists():
        r = np.load(ref)
        g = lat.detach().float().cpu().numpy()
        if r.shape == g.shape:
            e = np.linalg.norm(r - g)
            snr = float("inf") if e == 0 else 20 * np.log10(np.linalg.norm(r) / e)
            print("")
            print("  ==> NPU vs fp32 reference latents: {:.2f} dB".format(snr))
            print("      (this is the END-TO-END number -- 18 chained MMDiT calls,")
            print("       each unit conditioning on the previous one)")
        else:
            print("  reference has shape {}, cannot compare".format(r.shape))


if __name__ == "__main__":
    main()
