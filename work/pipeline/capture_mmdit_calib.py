"""Capture real MMDiT inputs from the reference pipeline. Phase 5 calibration.

Trap #4, twice over: calibration must use real pipeline tensors, and for a graph with
explicit conditioning inputs the conditioning must be real too. The DiT's inputs are the
deepest in the pipeline -- they depend on DistilT5, the context adapter, the SSD1B first
frame, the VAE encoder, and every previous AR step -- so nothing synthetic will do.

One reference run yields 18 DiT calls (6 units x 3 stages, 1 step per stage), i.e. 6
samples per stage. Paper Table 9 wants 300 per MMDiT row, so 50 videos covers all three
stages at once.

What is stored is the RAW call, not the derived graph inputs: the latent list, the
context-adapter output, the prompt mask, the pooled projection and the timestep. `attn_mask`
/ `rope_cos` / `rope_sin` are then rebuilt by `export_mmdit_stage.stage_conditioning_padded`
at list-build time, so a change to the padding scheme does not mean re-running the GPU.

  usage: py -3.10 work/pipeline/capture_mmdit_calib.py --num-prompts 50
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
OUT = ROOT / "work" / "calib" / "mmdit"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
_pkg = types.ModuleType("neodragon")
_pkg.__path__ = [str(REPO / "neodragon")]
sys.modules["neodragon"] = _pkg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--prompts", default="vbench_prompts.txt")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16
    dev = torch.device("cuda")

    from run_reference import build_ssd1b
    from neodragon.asymmetric_causal_video_vae import AsymmetricCausalVideoVAE
    from neodragon.context_adapter import ContextAdapter
    from neodragon.pyramid_mmdit import PyramidMMDiT
    from neodragon.pyramid_scheduler import PyramidFlowMatchEulerDiscreteScheduler
    from neodragon.text_encoder_bundle import TextEncoderBundle
    from neodragon.utils.generation_utils import generate_hybrid

    print("=" * 76)
    print("MMDiT CALIBRATION CAPTURE -- {} videos -> {} DiT calls".format(
        args.num_prompts, args.num_prompts * 18))
    print("=" * 76)

    print("")
    print("[load] components from", LOCAL)
    teb = TextEncoderBundle.from_pretrained(str(LOCAL), torch_dtype=dtype)
    ca = ContextAdapter.from_pretrained(str(LOCAL / "context_adapter"), torch_dtype=dtype)
    dit = PyramidMMDiT.from_pretrained(str(LOCAL / "diffusion_transformer_320p"),
                                       torch_dtype=dtype)
    vae = AsymmetricCausalVideoVAE.from_pretrained(str(LOCAL / "causal_video_vae"),
                                                   torch_dtype=dtype)
    sched = PyramidFlowMatchEulerDiscreteScheduler()
    ffg = build_ssd1b(dtype)

    ca.to(dev)
    dit.to(dev)
    vae.to(dev)
    ffg.enable_model_cpu_offload(device=dev)
    from accelerate import cpu_offload
    cpu_offload(teb, execution_device=dev)

    # ---- hook every DiT call ------------------------------------------------
    state = {"video": 0, "call": 0, "saved": 0}
    real_forward = dit.forward

    def spy(sample, encoder_hidden_states, encoder_attention_mask,
            pooled_projections, timestep_ratio, **kw):
        lats = sample[0]
        rec = {
            "encoder_hidden_states": encoder_hidden_states.detach().float().cpu().numpy(),
            "encoder_attention_mask": encoder_attention_mask.detach().float().cpu().numpy(),
            "pooled_projections": pooled_projections.detach().float().cpu().numpy(),
            "timestep_ratio": timestep_ratio.detach().float().cpu().numpy(),
            "n_latents": np.int64(len(lats)),
        }
        for i, la in enumerate(lats):
            rec["latent_{}".format(i)] = la.detach().float().cpu().numpy()
        np.savez_compressed(
            OUT / "call_{:03d}_{:02d}.npz".format(state["video"], state["call"]), **rec)
        state["call"] += 1
        state["saved"] += 1
        return real_forward(sample=sample, encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            pooled_projections=pooled_projections,
                            timestep_ratio=timestep_ratio, **kw)

    dit.forward = spy

    prompts = [ln.strip() for ln in
               (REPO / "prompts" / args.prompts).read_text(encoding="utf-8").splitlines()
               if ln.strip()]

    for i in range(args.start, args.start + args.num_prompts):
        prompt = prompts[i % len(prompts)]
        state["video"], state["call"] = i, 0
        torch.cuda.reset_peak_memory_stats()
        print("")
        print("[{:3d}/{}] {:.60s}".format(i + 1 - args.start, args.num_prompts, prompt))
        try:
            with torch.no_grad():
                generate_hybrid(
                    first_frame_gen_pipeline=ffg, text_encoder_bundle=teb, dit=dit,
                    context_adapter=ca, vae=vae, scheduler=sched, prompt=prompt,
                    height=320, width=512, num_frames=49,
                    num_inference_steps=[1, 1, 1], video_num_inference_steps=[1, 1, 1],
                    do_classifier_free_guidance=False,
                    guidance_scale=0.0, video_guidance_scale=0.0,
                    frames_per_unit=1, num_stages=3,
                    output_type="latent", profile=False, device=dev, dtype=dtype)
        except torch.cuda.OutOfMemoryError as e:
            print("    OOM: {}".format(e))
            break
        print("    {} DiT calls, {} saved total, peak {:.2f} GiB".format(
            state["call"], state["saved"],
            torch.cuda.max_memory_reserved() / 1024 ** 3))
        torch.cuda.empty_cache()

    print("")
    print("wrote {} DiT call records to {}".format(state["saved"], OUT))
    tot = sum(f.stat().st_size for f in OUT.glob("*.npz"))
    print("  {:.2f} GB on disk".format(tot / 1024 ** 3))


if __name__ == "__main__":
    main()
